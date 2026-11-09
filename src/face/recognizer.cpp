#include "face-detection/face/recognizer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <utility>

#ifdef FD_HAVE_FACE_STACK
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect/face.hpp>
#endif

namespace fd {

namespace {
#ifdef FD_HAVE_FACE_STACK

// A detection row as expected by cv::FaceRecognizerSF::alignCrop:
// 15 floats laid out as
//   [0..3]   x, y, width, height
//   [4..13]  five landmarks (right eye, left eye, nose, mouth R, mouth L)
// The landmarks are what the alignment transform is computed from, so the
// full row must be supplied -- a bare bounding box is not enough.
constexpr int kRowLength = 15;
constexpr int kLandmarkOffset = 4;

cv::Mat buildRow(const FaceBox& box) {
    cv::Mat row(1, kRowLength, CV_32FC1);
    row.at<float>(0, 0) = static_cast<float>(box.x);
    row.at<float>(0, 1) = static_cast<float>(box.y);
    row.at<float>(0, 2) = static_cast<float>(box.width);
    row.at<float>(0, 3) = static_cast<float>(box.height);
    for (std::size_t lm = 0; lm < kLandmarkCount; ++lm) {
        const int base = kLandmarkOffset + static_cast<int>(lm) * 2;
        row.at<float>(0, base) = box.landmarks[lm][0];
        row.at<float>(0, base + 1) = box.landmarks[lm][1];
    }
    return row;
}

#endif // FD_HAVE_FACE_STACK
} // namespace

// ---------------------------------------------------------------------------
// Embedding
// ---------------------------------------------------------------------------
float Embedding::cosineDistance(const Embedding& other) const {
    if (!valid || !other.valid) {
        return 1.0f;
    }
    float dot = 0.0f;
    for (std::size_t i = 0; i < kDimension; ++i) {
        dot += values[i] * other.values[i];
    }
    // Both vectors are unit length, so the cosine similarity is the dot
    // product and the cosine distance is 1 - similarity.
    return 1.0f - dot;
}

// ---------------------------------------------------------------------------
// FaceRecognizer::Impl
// ---------------------------------------------------------------------------
struct FaceRecognizer::Impl {
#ifdef FD_HAVE_FACE_STACK
    cv::Ptr<cv::FaceRecognizerSF> recognizer;

    cv::Mat converted;  // grayscale -> BGR scratch
    cv::Mat feature;

    static cv::Mat view(const Image& image) {
        const int type = image.channels == 1 ? CV_8UC1 : CV_8UC3;
        return cv::Mat(image.height, image.width, type,
                       const_cast<std::uint8_t*>(image.data.data()));
    }
#endif
};

// ---------------------------------------------------------------------------
// FaceRecognizer
// ---------------------------------------------------------------------------
FaceRecognizer::FaceRecognizer() : impl_(std::make_unique<Impl>()) {}

FaceRecognizer::~FaceRecognizer() = default;

FaceRecognizer::FaceRecognizer(FaceRecognizer&&) noexcept = default;

FaceRecognizer& FaceRecognizer::operator=(FaceRecognizer&&) noexcept = default;

bool FaceRecognizer::open(const Options& options) {
#ifdef FD_HAVE_FACE_STACK
    impl_->recognizer = cv::FaceRecognizerSF::create(options.model_path, "");
    return impl_->recognizer != nullptr;
#else
    (void)options;
    return false;
#endif
}

void FaceRecognizer::close() {
#ifdef FD_HAVE_FACE_STACK
    impl_->recognizer.reset();
#endif
}

bool FaceRecognizer::isOpen() const {
#ifdef FD_HAVE_FACE_STACK
    return impl_->recognizer != nullptr;
#else
    return false;
#endif
}

bool FaceRecognizer::embed(const Image& image, const FaceBox& box,
                            Embedding& out) const {
    out = Embedding{};
#ifdef FD_HAVE_FACE_STACK
    if (impl_->recognizer == nullptr || !image.consistent()) {
        return false;
    }
    if (image.channels != 1 && image.channels != 3) {
        return false;
    }

    cv::Mat source = Impl::view(image);
    cv::Mat input;
    if (source.channels() == 1) {
        cv::cvtColor(source, impl_->converted, cv::COLOR_GRAY2BGR);
        input = impl_->converted;
    } else {
        input = source;
    }

    // Align the face to the canonical 112x112 pose using the landmarks.
    const cv::Mat row = buildRow(box);
    cv::Mat aligned;
    impl_->recognizer->alignCrop(input, row, aligned);

    cv::Mat feature;
    impl_->recognizer->feature(aligned, feature);
    if (feature.empty()) {
        return false;
    }

    // Copy into the fixed-size array and L2-normalise so that the cosine
    // distance is simply 1 - dot(a, b).
    float norm = 0.0f;
    const int count = std::min<int>(feature.cols,
                                     static_cast<int>(Embedding::kDimension));
    for (int i = 0; i < count; ++i) {
        const float v = feature.at<float>(0, i);
        out.values[static_cast<std::size_t>(i)] = v;
        norm += v * v;
    }
    if (count != static_cast<int>(Embedding::kDimension) || norm <= 0.0f) {
        out = Embedding{};
        return false;
    }

    norm = std::sqrt(norm);
    for (std::size_t i = 0; i < Embedding::kDimension; ++i) {
        out.values[i] /= norm;
    }
    out.valid = true;
    return true;
#else
    (void)image;
    (void)box;
    return false;
#endif
}

} // namespace fd
