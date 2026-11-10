// Example application: detect faces in a still image and optionally
// recognise them against an enrolled set.
//
//   face-detection-app --image=photo.jpg
//   face-detection-app --image=photo.jpg --enroll=42
//   face-detection-app --image=photo.jpg --recognize
//
// The face database is either a file or a PostgreSQL table; the target is
// picked by the --db option.
//
//   face-detection-app --db=faces.db --db-list
//   face-detection-app --db=postgres://user@host/db --db-list

#include "face-detection/face/database.hpp"
#include "face-detection/face/detector.hpp"
#include "face-detection/face/recognizer.hpp"
#include "face-detection/face/store.hpp"
#include "face-detection/image.hpp"
#include "face-detection/json.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#ifdef FD_HAVE_FACE_STACK
#include <opencv2/imgcodecs.hpp>
#endif

namespace {

// Options that are switches: they never take a value.
bool isBooleanOption(const std::string& name) {
    return name == "recognize" || name == "db-list" || name == "help";
}

bool isKnownOption(const std::string& name) {
    static const char* const kOptions[] = {
        "image", "models", "threshold", "db", "enroll", "recognize",
        "db-list", "db-remove", "enroll-name", "enroll-metadata", "help",
    };
    for (const char* known : kOptions) {
        if (name == known) {
            return true;
        }
    }
    return false;
}

// Tiny argument helper: returns the value of --<name>= or a fallback.
std::string arg(int argc, char** argv, const std::string& name,
                const std::string& fallback) {
    const std::string prefix = "--" + name + "=";
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a.rfind(prefix, 0) == 0) {
            return a.substr(prefix.size());
        }
    }
    return fallback;
}

bool flag(int argc, char** argv, const std::string& name) {
    const std::string needle = "--" + name;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == needle) {
            return true;
        }
    }
    return false;
}

#ifdef FD_HAVE_FACE_STACK
// Encodes the detected face as a JPEG, so that a human can recognise the
// person later on. Returns an empty vector when the face could not be
// encoded.
std::vector<std::uint8_t> encodeFace(const cv::Mat& source,
                                      const fd::FaceBox& face) {
    // A little padding around the box: the edges of a face carry
    // information as well.
    const int pad = std::max(face.width, face.height) / 4;
    const int x = std::max(0, face.x - pad);
    const int y = std::max(0, face.y - pad);
    const int width = std::min(source.cols - x, face.width + 2 * pad);
    const int height = std::min(source.rows - y, face.height + 2 * pad);
    if (width <= 0 || height <= 0) {
        return {};
    }

    const cv::Mat crop = source(cv::Rect(x, y, width, height));
    std::vector<uchar> buffer;
    if (!cv::imencode(".jpg", crop, buffer, {cv::IMWRITE_JPEG_QUALITY, 85})) {
        return {};
    }
    return std::vector<std::uint8_t>(buffer.begin(), buffer.end());
}
#endif

// ---------------------------------------------------------------------------
void printHelp() {
    std::cout
        << "face-detection-app -- detect faces in an image\n"
           "\n"
           "Usage:\n"
           "  face-detection-app --image=<path> [options]\n"
           "  face-detection-app --db=<target> --db-list\n"
           "\n"
           "Options:\n"
           "  --image=<path>       Input image file (required unless only the\n"
           "                       database is touched).\n"
           "  --models=<dir>       Directory holding the ONNX models\n"
           "                       (default: ./models or $FACE_DETECTION_MODELS).\n"
           "  --threshold=<f>      Minimum detection confidence, 0..1\n"
           "                       (default: 0.6).\n"
           "  --db=<target>        Face database. A file path, or a\n"
           "                       PostgreSQL connection string starting with\n"
           "                       postgres:// or postgresql://.\n"
           "                       Loaded if it exists and written back when\n"
           "                       --enroll is used.\n"
           "  --db-list            List the enrolled people and exit.\n"
           "                       Needs --db.\n"
           "  --db-remove=<id>     Remove every sample of a person and exit.\n"
           "                       Needs --db.\n"
           "  --enroll=<id>        Enrol the first detected face under this\n"
           "                       person id instead of just reporting it.\n"
           "  --enroll-name=<name> The name of the person being enrolled. It\n"
           "                       is stored with the sample and shown when\n"
           "                       the person is recognised.\n"
           "  --enroll-metadata=<json>\n"
           "                       Extra properties of the person, as a JSON\n"
           "                       document (e.g. {\"role\":\"admin\"}).\n"
           "  --recognize          Compare detected faces against the enrolled\n"
           "                       set and report the closest identity.\n"
           "  -h, --help           Print this text and exit.\n"
           "\n"
           "Exit codes:\n"
           "  0  success\n"
           "  1  runtime error (model or image missing, no faces found)\n"
           "  2  bad usage\n";
}

bool validateArgs(int argc, char** argv) {
    bool ok = true;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "-h" || a == "--help") {
            continue;
        }
        if (a.rfind("--", 0) != 0) {
            std::cerr << "Unexpected argument '" << a << "'\n";
            ok = false;
            continue;
        }
        const auto eq = a.find('=');
        const std::string name =
            (eq == std::string::npos) ? a.substr(2) : a.substr(2, eq - 2);
        if (!isKnownOption(name)) {
            std::cerr << "Unknown option '" << a << "'\n";
            ok = false;
        } else if (eq != std::string::npos && isBooleanOption(name)) {
            std::cerr << "Option '--" << name << "' is a switch and takes no value.\n";
            ok = false;
        }
    }
    return ok;
}

// Opens the face database and prints what it holds.
//
// Returns false when the database could not be opened, so the caller can
// stop before doing any work.
bool openDatabase(const std::string& target, std::unique_ptr<fd::FaceStore>& store,
                  fd::FaceDatabase& db) {
    std::string error;
    store = fd::makeFaceStore(target, error);
    if (!store) {
        std::cerr << error << '\n';
        return false;
    }
    if (!store->open(target, error)) {
        std::cerr << "Could not open the face database '" << target
                  << "': " << error << '\n';
        return false;
    }

    std::vector<fd::FaceEntry> entries;
    if (!store->load(entries, error)) {
        std::cout << "No enrolments in '" << target << "' yet (" << error
                  << ").\n";
        return true;
    }

    for (const fd::FaceEntry& entry : entries) {
        // The whole entry is added, so that the name, the picture and the
        // metadata of the person survive the reload.
        db.add(entry);
    }
    std::cout << "Loaded " << entries.size() << " enrolment(s) from " << target
              << '\n';
    return true;
}

} // namespace

int main(int argc, char** argv) {
    if (flag(argc, argv, "help") ||
        (argc > 1 && std::string(argv[1]) == "-h")) {
        printHelp();
        return 0;
    }
    if (!validateArgs(argc, argv)) {
        return 2;
    }

    const std::string image_path = arg(argc, argv, "image", "");
    const std::string db_target = arg(argc, argv, "db", "");
    const std::string enroll_str = arg(argc, argv, "enroll", "");
    const std::string enroll_name = arg(argc, argv, "enroll-name", "");
    const std::string enroll_metadata = arg(argc, argv, "enroll-metadata", "");
    const std::string remove_str = arg(argc, argv, "db-remove", "");
    const bool list_only = flag(argc, argv, "db-list");
    const bool do_recognize = flag(argc, argv, "recognize");

    // ---- Database-only operations -----------------------------------------
    // Listing and removing people do not need an image, so they are handled
    // before the detector is set up.
    if (list_only || !remove_str.empty()) {
        if (db_target.empty()) {
            std::cerr << "--db-list and --db-remove need --db=<target>.\n";
            return 2;
        }

        std::unique_ptr<fd::FaceStore> store;
        fd::FaceDatabase db;
        if (!openDatabase(db_target, store, db)) {
            return 1;
        }

        std::string error;
        if (list_only) {
            std::vector<fd::PersonSummary> people;
            if (!store->list(people, error)) {
                std::cerr << "Could not list the face database: " << error
                          << '\n';
                return 1;
            }
            if (people.empty()) {
                std::cout << "Nobody is enrolled in " << db_target << ".\n";
            }
            for (const fd::PersonSummary& person : people) {
                std::cout << "  person " << person.person_id << ": "
                          << person.samples << " sample(s)";
                if (!person.user_name.empty()) {
                    std::cout << " (" << person.user_name << ")";
                }
                std::cout << '\n';
            }
            return 0;
        }

        int person_id = 0;
        try {
            person_id = std::stoi(remove_str);
        } catch (const std::exception&) {
            std::cerr << "Invalid --db-remove value '" << remove_str << "'\n";
            return 2;
        }
        const int removed = store->remove(person_id, error);
        if (removed < 0) {
            std::cerr << "Could not remove person " << person_id << ": "
                      << error << '\n';
            return 1;
        }
        std::cout << "Removed " << removed << " sample(s) of person "
                  << person_id << ".\n";
        return 0;
    }

    if (image_path.empty()) {
        std::cerr << "Missing required option --image=<path>.\n";
        return 2;
    }

    // The metadata is stored in a JSON column, so it has to be a JSON
    // document. It is checked here, because the error of the server is a
    // lot harder to make sense of.
    if (!enroll_metadata.empty() && !fd::isValidJson(enroll_metadata)) {
        std::cerr << "Invalid --enroll-metadata value '" << enroll_metadata
                  << "': not a JSON document.\n";
        return 2;
    }

    std::string models_dir = arg(argc, argv, "models", "");
    if (models_dir.empty()) {
        if (const char* env = std::getenv("FACE_DETECTION_MODELS")) {
            models_dir = env;
        } else {
            models_dir = "models";
        }
    }

    const std::string threshold_str = arg(argc, argv, "threshold", "0.6");
    float threshold = 0.6f;
    try {
        threshold = std::stof(threshold_str);
    } catch (const std::exception&) {
        std::cerr << "Invalid --threshold value '" << threshold_str << "'.\n";
        return 2;
    }

#ifdef FD_HAVE_FACE_STACK
    const std::string detector_model =
        models_dir + "/face_detection_yunet_2023mar.onnx";
    const std::string recognizer_model =
        models_dir + "/face_recognition_sface_2021dec.onnx";

    fd::FaceDetector detector;
    fd::FaceDetector::Options detector_options;
    detector_options.model_path = detector_model;
    detector_options.score_threshold = threshold;

    try {
        if (!detector.open(detector_options)) {
            std::cerr << "Could not load the detector model '" << detector_model
                      << "'.\nRun models/fetch.sh to download it.\n";
            return 1;
        }
    } catch (const cv::Exception& e) {
        std::cerr << "Could not load the detector model: " << e.what() << '\n';
        return 1;
    }

    cv::Mat source = cv::imread(image_path, cv::IMREAD_COLOR);
    if (source.empty()) {
        std::cerr << "Could not read the image '" << image_path << "'.\n";
        return 1;
    }

    // cv::Mat (BGR) -> fd::Image. One copy, unavoidable here because OpenCV
    // owns the decoded buffer.
    fd::Image image;
    image.width = source.cols;
    image.height = source.rows;
    image.channels = 3;
    image.data.assign(source.datastart, source.dataend);

    const std::vector<fd::FaceBox> faces = detector.detect(image);
    std::cout << "Detected " << faces.size() << " face(s) in " << image.width << "x"
              << image.height << '\n';

    for (std::size_t i = 0; i < faces.size(); ++i) {
        const fd::FaceBox& box = faces[i];
        std::cout << "  face " << i << ": box=(" << box.x << "," << box.y << ") "
                  << box.width << "x" << box.height << " score=" << box.score
                  << '\n';
    }

    // The database may be loaded even without --enroll/--recognize, so that
    // its contents can be inspected.
    std::unique_ptr<fd::FaceStore> store;
    fd::FaceDatabase db;
    if (!db_target.empty() && !openDatabase(db_target, store, db)) {
        return 1;
    }

    if (!enroll_str.empty() || do_recognize) {
        fd::FaceRecognizer recognizer;
        fd::FaceRecognizer::Options recognizer_options;
        recognizer_options.model_path = recognizer_model;
        try {
            if (!recognizer.open(recognizer_options)) {
                std::cerr << "Could not load the recognizer model '"
                          << recognizer_model << "'.\n";
                return 1;
            }
        } catch (const cv::Exception& e) {
            std::cerr << "Could not load the recognizer model: " << e.what() << '\n';
            return 1;
        }

        if (!enroll_str.empty()) {
            int person_id = 0;
            try {
                person_id = std::stoi(enroll_str);
            } catch (const std::exception&) {
                std::cerr << "Invalid --enroll value '" << enroll_str << "'.\n";
                return 2;
            }
            if (faces.empty()) {
                std::cerr << "No face detected, nothing to enrol.\n";
                return 1;
            }
            fd::Embedding embedding;
            if (!recognizer.embed(image, faces.front(), embedding)) {
                std::cerr << "Could not compute an embedding for the face.\n";
                return 1;
            }

            // The properties of the person: the name that is shown when
            // the person is recognised, the picture of the face and the
            // metadata.
            fd::EnrolmentInfo info;
            info.user_name = enroll_name;
            info.metadata = enroll_metadata;
            info.photo = encodeFace(source, faces.front());

            fd::FaceEntry entry;
            if (!db.enroll(person_id, embedding, info, &entry)) {
                std::cerr << "Could not compute an embedding for the face.\n";
                return 1;
            }
            std::cout << "Enrolled the first face under person id " << person_id
                      << " (database now holds " << db.size() << ").\n";
            if (!info.photo.empty()) {
                std::cout << "Stored a " << info.photo.size()
                          << " byte picture of the face.\n";
            }

            // The store appends the new sample on its own, so enrolling one
            // person never rewrites the samples of the others.
            if (store) {
                std::string error;
                if (!store->add(entry, error)) {
                    std::cerr << "Could not write the enrolment to "
                              << db_target << ": " << error << '\n';
                    return 1;
                }
                std::cout << "Stored the enrolment in " << db_target << ".\n";
            }
            return 0;
        }

        if (do_recognize) {
            if (faces.empty()) {
                std::cerr << "No face detected, nothing to recognise.\n";
                return 1;
            }
            fd::Embedding embedding;
            if (!recognizer.embed(image, faces.front(), embedding)) {
                std::cerr << "Could not compute an embedding for the face.\n";
                return 1;
            }
            if (db.empty()) {
                std::cerr << "The face database is empty; nothing to match "
                             "against.\n";
                return 1;
            }
            const fd::FaceDatabase::Match result = db.match(embedding);
            std::cout << "Best match: person=" << result.person_id;
            if (!result.user_name.empty()) {
                std::cout << " (" << result.user_name << ")";
            }
            std::cout << " distance=" << result.distance
                      << " matched=" << (result.matched ? "yes" : "no") << '\n';
            return result.matched ? 0 : 1;
        }
    }

    return faces.empty() ? 1 : 0;
#else
    (void)threshold;
    (void)enroll_str;
    (void)do_recognize;
    (void)db_target;
    std::cerr << "This build has no OpenCV face stack; rebuild with OpenCV "
                 "(objdetect + dnn) available.\n";
    return 1;
#endif
}
