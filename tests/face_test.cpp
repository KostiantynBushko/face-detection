// Smoke tests for face-detection. No external test framework required.
//
// The model-backed checks are skipped automatically when the ONNX weights are
// not present, so the suite still passes on a machine without the models.

#include "face-detection/face/database.hpp"
#include "face-detection/face/detector.hpp"
#include "face-detection/face/recognizer.hpp"
#include "face-detection/image.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#ifndef FD_MODELS_DIR
#define FD_MODELS_DIR "models"
#endif

namespace {

int failures = 0;

void check(bool condition, const char* what) {
    if (!condition) {
        std::cerr << "FAIL: " << what << '\n';
        ++failures;
    }
}

bool fileExists(const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) {
        return false;
    }
    std::fclose(f);
    return true;
}

/// Builds a unit-length embedding whose first component is @p direction.
fd::Embedding makeEmbedding(float direction) {
    fd::Embedding e;
    e.values[0] = direction;
    e.values[1] = std::sqrt(std::max(0.0f, 1.0f - direction * direction));
    e.valid = true;
    return e;
}

/// Builds the unit vector along @p axis. Two different axes are perfectly
/// orthogonal, which makes it easy to construct genuinely distant pairs.
fd::Embedding makeEmbeddingOnAxis(std::size_t axis) {
    fd::Embedding e;
    e.values[axis] = 1.0f;
    e.valid = true;
    return e;
}

} // namespace

int main() {
    using fd::Embedding;
    using fd::FaceBox;
    using fd::FaceDatabase;
    using fd::FaceDetector;
    using fd::FaceRecognizer;
    using fd::Image;

    // ---- Image ------------------------------------------------------------
    {
        Image empty;
        check(empty.empty(), "default image is empty");
        check(!empty.consistent(), "default image is not consistent");

        Image good;
        good.width = 4;
        good.height = 2;
        good.channels = 3;
        good.data.assign(4 * 2 * 3, 0);
        check(!good.empty(), "populated image is not empty");
        check(good.consistent(), "populated image is consistent");

        Image bad = good;
        bad.data.pop_back();
        check(!bad.consistent(), "short buffer is not consistent");
    }

    // ---- Embedding geometry ----------------------------------------------
    {
        const Embedding a = makeEmbeddingOnAxis(0);
        check(a.valid, "embedding is valid");

        const float self = a.cosineDistance(a);
        check(std::fabs(self) < 1e-5f, "cosine distance to self is ~0");

        const Embedding b = makeEmbeddingOnAxis(1);
        const float orthogonal = a.cosineDistance(b);
        check(std::fabs(orthogonal - 1.0f) < 1e-5f,
              "cosine distance to an orthogonal vector is ~1");

        Embedding invalid;
        invalid.valid = false;
        check(std::fabs(invalid.cosineDistance(a) - 1.0f) < 1e-5f,
              "distance to an invalid embedding is 1");
    }

    // ---- FaceDatabase -----------------------------------------------------
    {
        FaceDatabase db;
        check(db.empty(), "new database is empty");
        check(db.match(makeEmbedding(1.0f)).person_id == -1,
              "matching an empty database finds nothing");

        check(db.enroll(7, makeEmbeddingOnAxis(0)),
              "enroll accepts a valid embedding");
        check(db.enroll(8, makeEmbeddingOnAxis(1)),
              "enroll accepts a second person");
        check(db.size() == 2, "database holds two entries");

        // Slightly perturbed version of person 7's axis.
        const auto hit = db.match(makeEmbedding(0.99f), 0.4f);
        check(hit.matched, "a close embedding matches");
        check(hit.person_id == 7, "the closest person is reported");
        check(hit.distance < 0.4f, "the reported distance is within the limit");

        // Axis 2 is orthogonal to both enrolled vectors, so the closest
        // distance is 1.0 -- far outside any sane threshold.
        const auto miss = db.match(makeEmbeddingOnAxis(2), 0.05f);
        check(!miss.matched, "a distant embedding does not match a tight limit");
        check(miss.person_id >= 0, "a nearest entry is still reported");
        check(miss.distance > 0.05f, "the reported distance is above the limit");

        check(db.remove(7) == 1, "remove drops one entry");
        check(db.size() == 1, "one entry remains");
        db.clear();
        check(db.empty(), "clear empties the database");

        Embedding invalid;
        check(!db.enroll(1, invalid), "enroll rejects an invalid embedding");
    }

    // ---- Model-backed checks ----------------------------------------------
    const std::string models_dir = FD_MODELS_DIR;
    const std::string detector_model =
        models_dir + "/face_detection_yunet_2023mar.onnx";
    const std::string recognizer_model =
        models_dir + "/face_recognition_sface_2021dec.onnx";

    if (!fileExists(detector_model)) {
        std::cout << "SKIP: detector model not found at " << detector_model
                  << " (run models/fetch.sh)\n";
    } else {
        FaceDetector detector;
        FaceDetector::Options options;
        options.model_path = detector_model;
        check(detector.open(options), "detector opens with a valid model");
        check(detector.isOpen(), "detector reports open");

        // A blank frame must not report any faces.
        Image blank;
        blank.width = 320;
        blank.height = 240;
        blank.channels = 3;
        blank.data.assign(320 * 240 * 3, 0);
        check(detector.detect(blank).empty(), "a blank frame yields no faces");

        // A malformed frame must be rejected rather than crash.
        Image malformed;
        malformed.width = 100;
        malformed.height = 100;
        malformed.channels = 3;
        malformed.data.assign(10, 0);
        check(detector.detect(malformed).empty(),
              "a malformed frame yields no faces");

        detector.close();
        check(!detector.isOpen(), "detector reports closed after close");
    }

    if (!fileExists(recognizer_model)) {
        std::cout << "SKIP: recognizer model not found at " << recognizer_model
                  << " (run models/fetch.sh)\n";
    } else {
        FaceRecognizer recognizer;
        FaceRecognizer::Options options;
        options.model_path = recognizer_model;
        check(recognizer.open(options), "recognizer opens with a valid model");
        check(recognizer.isOpen(), "recognizer reports open");

        // Embedding a synthetic face box on a blank frame must still yield a
        // valid, unit-length embedding (the pixels are meaningless, but the
        // pipeline has to produce a well-formed vector).
        Image blank;
        blank.width = 320;
        blank.height = 240;
        blank.channels = 3;
        blank.data.assign(320 * 240 * 3, 128);

        FaceBox box;
        box.x = 100;
        box.y = 60;
        box.width = 80;
        box.height = 80;
        box.landmarks[0] = {124.0f, 84.0f};
        box.landmarks[1] = {156.0f, 84.0f};
        box.landmarks[2] = {140.0f, 104.0f};
        box.landmarks[3] = {126.0f, 124.0f};
        box.landmarks[4] = {154.0f, 124.0f};

        Embedding embedding;
        check(recognizer.embed(blank, box, embedding),
              "recognizer produces an embedding");
        check(embedding.valid, "the embedding is marked valid");

        float norm = 0.0f;
        for (const float v : embedding.values) {
            norm += v * v;
        }
        check(std::fabs(std::sqrt(norm) - 1.0f) < 1e-4f,
              "the embedding is unit length");

        recognizer.close();
        check(!recognizer.isOpen(), "recognizer reports closed after close");
    }

    if (failures == 0) {
        std::cout << "face_test: all checks passed.\n";
        return 0;
    }
    std::cerr << "face_test: " << failures << " check(s) failed.\n";
    return 1;
}
