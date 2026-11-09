#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <string>

#include "face-detection/face/detector.hpp"
#include "face-detection/image.hpp"

namespace fd {

/// A face embedding: a fixed-length, L2-normalised feature vector.
///
/// The values are unit length, so the cosine *distance* between two
/// embeddings is `1 - dot(a, b)`, which lies in [0, 2] and is 0 for
/// identical faces.
struct Embedding {
    /// Dimensionality of the embedding produced by SFace.
    static constexpr std::size_t kDimension = 128;

    std::array<float, kDimension> values{};
    bool valid = false;

    /// Cosine distance to @p other. Lower means more similar.
    /// Returns 1.0 if either embedding is invalid.
    [[nodiscard]] float cosineDistance(const Embedding& other) const;
};

/// Computes face embeddings for detected faces.
///
/// Backed by SFace (a compact face-recognition CNN) running through
/// OpenCV's DNN module. Each face is first *aligned* to a canonical
/// 112x112 crop using the detector's five landmarks, which is what makes
/// the embedding stable across pose and scale changes.
class FaceRecognizer {
public:
    struct Options {
        /// Path to `face_recognition_sface_*.onnx`.
        std::string model_path;
    };

    FaceRecognizer();
    ~FaceRecognizer();

    FaceRecognizer(const FaceRecognizer&) = delete;
    FaceRecognizer& operator=(const FaceRecognizer&) = delete;
    FaceRecognizer(FaceRecognizer&&) noexcept;
    FaceRecognizer& operator=(FaceRecognizer&&) noexcept;

    /// Loads the model. Returns false if the file is missing or invalid.
    bool open(const Options& options);

    /// Releases the model. Safe to call repeatedly.
    void close();

    [[nodiscard]] bool isOpen() const;

    /// Computes the embedding of the face at @p box inside @p image.
    ///
    /// @p box must come from a FaceDetector run on the *same* image, so
    /// that its landmarks are in the same coordinate space.
    /// Returns false if the model is closed or the inputs are invalid.
    bool embed(const Image& image, const FaceBox& box, Embedding& out) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace fd
