// Example application: detect faces in a still image and optionally
// recognise them against an enrolled set.
//
//   face-detection-app --image=photo.jpg
//   face-detection-app --image=photo.jpg --enroll=42
//   face-detection-app --image=photo.jpg --recognize

#include "face-detection/face/database.hpp"
#include "face-detection/face/detector.hpp"
#include "face-detection/face/recognizer.hpp"
#include "face-detection/image.hpp"

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#ifdef FD_HAVE_FACE_STACK
#include <opencv2/imgcodecs.hpp>
#endif

namespace {

// Options that are switches: they never take a value.
bool isBooleanOption(const std::string& name) {
    return name == "recognize" || name == "help";
}

bool isKnownOption(const std::string& name) {
    static const char* const kOptions[] = {
        "image", "models", "threshold", "db", "enroll", "recognize", "help",
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

// ---------------------------------------------------------------------------
void printHelp() {
    std::cout
        << "face-detection-app -- detect faces in an image\n"
           "\n"
           "Usage:\n"
           "  face-detection-app --image=<path> [options]\n"
           "\n"
           "Options:\n"
           "  --image=<path>       Input image file (required).\n"
           "  --models=<dir>       Directory holding the ONNX models\n"
           "                       (default: ./models or $FACE_DETECTION_MODELS).\n"
           "  --threshold=<f>      Minimum detection confidence, 0..1\n"
           "                       (default: 0.6).\n"
           "  --db=<path>          Face database file. Loaded if it exists and\n"
           "                       written back when --enroll is used.\n"
           "  --enroll=<id>        Enrol the first detected face under this\n"
           "                       person id instead of just reporting it.\n"
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
            std::cerr << "Unknown option '" << a << "'.\n";
            ok = false;
        } else if (eq != std::string::npos && isBooleanOption(name)) {
            std::cerr << "Option '--" << name << "' is a switch and takes no value.\n";
            ok = false;
        }
    }
    return ok;
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
    if (image_path.empty()) {
        std::cerr << "Missing required option --image=<path>.\n";
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

    const std::string db_path = arg(argc, argv, "db", "");
    const std::string enroll_str = arg(argc, argv, "enroll", "");
    const bool do_recognize = flag(argc, argv, "recognize");

    // A database may be loaded even without --enroll/--recognize, so that
    // its contents can be inspected.
    fd::FaceDatabase db;
    if (!db_path.empty() && db.load(db_path)) {
        std::cout << "Loaded " << db.size() << " enrolment(s) from " << db_path
                  << '\n';
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
            db.enroll(person_id, embedding);
            std::cout << "Enrolled the first face under person id " << person_id
                      << " (database now holds " << db.size() << ").\n";
            if (!db_path.empty() && !db.save(db_path)) {
                std::cerr << "Could not write the database to " << db_path << '\n';
                return 1;
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
            std::cout << "Best match: person=" << result.person_id
                      << " distance=" << result.distance
                      << " matched=" << (result.matched ? "yes" : "no") << '\n';
            return result.matched ? 0 : 1;
        }
    }

    return faces.empty() ? 1 : 0;
#else
    (void)threshold;
    std::cerr << "This build has no OpenCV face stack; rebuild with OpenCV "
                 "(objdetect + dnn) available.\n";
    return 1;
#endif
}
