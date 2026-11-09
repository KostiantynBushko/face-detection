#include "face-detection/face/detector.hpp"

#include <algorithm>
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

// Layout of one detection row produced by cv::FaceDetectorYN:
//   [0..3]   x, y, width, height
//   [4..13]  five landmarks (right eye, left eye, nose, mouth R, mouth L)
//   [14]     confidence score
constexpr int kBoxX = 0;
constexpr int kBoxY = 1;
constexpr int kBoxW = 2;
constexpr int kBoxH = 3;
constexpr int kLandmarkOffset = 4;
constexpr int kScoreIndex = 14;

#endif // FD_HAVE_FACE_STACK
} // namespace

// ---------------------------------------------------------------------------
// FaceDetector::Impl
// ---------------------------------------------------------------------------
struct FaceDetector::Impl {
#ifdef FD_HAVE_FACE_STACK
    cv::Ptr<cv::FaceDetectorYN> detector;
    Options options;

    // Scratch buffers reused between frames to avoid per-frame allocation.
    cv::Mat scaled;
    cv::Mat grayOrBgr;
    cv::Mat detections; // [N x 15] float rows

    /// Wraps @p image in a cv::Mat without copying. The returned matrix
    /// aliases the caller's buffer and must not outlive it.
    static cv::Mat view(const Image& image) {
        const int type = image.channels == 1 ? CV_8UC1 : CV_8UC3;
        return cv::Mat(image.height, image.width, type,
                       const_cast<std::uint8_t*>(image.data.data()));
    }
#endif
};

// ---------------------------------------------------------------------------
// FaceDetector
// ---------------------------------------------------------------------------
FaceDetector::FaceDetector() : impl_(std::make_unique<Impl>()) {}

FaceDetector::~FaceDetector() = default;

FaceDetector::FaceDetector(FaceDetector&&) noexcept = default;

FaceDetector& FaceDetector::operator=(FaceDetector&&) noexcept = default;

bool FaceDetector::open(const Options& options) {
#ifdef FD_HAVE_FACE_STACK
    impl_->options = options;
    impl_->detector = cv::FaceDetectorYN::create(
        options.model_path, "",
        cv::Size(320, 320), // overwritten by setInputSize() before every run
        options.score_threshold, options.nms_threshold, options.top_k);
    return impl_->detector != nullptr;
#else
    (void)options;
    return false;
#endif
}

void FaceDetector::close() {
#ifdef FD_HAVE_FACE_STACK
    impl_->detector.reset();
#endif
}

bool FaceDetector::isOpen() const {
#ifdef FD_HAVE_FACE_STACK
    return impl_->detector != nullptr;
#else
    return false;
#endif
}

std::vector<FaceBox> FaceDetector::detect(const Image& image) const {
    std::vector<FaceBox> faces;
#ifdef FD_HAVE_FACE_STACK
    if (impl_->detector == nullptr || !image.consistent()) {
        return faces;
    }
    if (image.channels != 1 && image.channels != 3) {
        return faces;
    }

    cv::Mat source = Impl::view(image);

    // YuNet needs a 3-channel input.
    cv::Mat input;
    if (source.channels() == 1) {
        cv::cvtColor(source, impl_->grayOrBgr, cv::COLOR_GRAY2BGR);
        input = impl_->grayOrBgr;
    } else {
        input = source;
    }

    // Run the detector on a downscaled copy: the cost of YuNet grows
    // steeply with the input area, while accuracy is largely preserved.
    double scale_x = 1.0;
    double scale_y = 1.0;
    const int target_width = impl_->options.detection_width;

    if (target_width > 0 && input.cols > target_width) {
        const int target_height = std::max(
            1, static_cast<int>(static_cast<double>(input.rows) * target_width /
                                static_cast<double>(input.cols)));
        cv::resize(input, impl_->scaled, cv::Size(target_width, target_height));
        scale_x = static_cast<double>(input.cols) / static_cast<double>(target_width);
        scale_y = static_cast<double>(input.rows) / static_cast<double>(target_height);
        input = impl_->scaled;
    }

    // The detector rejects any input whose size differs from the configured
    // one, so this must be called before every run.
    impl_->detector->setInputSize(input.size());

    // NOTE: cv::FaceDetectorYN::detect() returns a hard-coded 1 in OpenCV
    // 4.10 rather than the number of faces. The real count is the number of
    // rows in the output matrix.
    impl_->detections.release();
    impl_->detector->detect(input, impl_->detections);
    if (impl_->detections.empty()) {
        return faces;
    }

    const int count = impl_->detections.rows;
    faces.reserve(static_cast<std::size_t>(count));

    for (int i = 0; i < count; ++i) {
        const cv::Mat row = impl_->detections.row(i);

        FaceBox box;
        box.x = static_cast<int>(row.at<float>(kBoxX) * scale_x);
        box.y = static_cast<int>(row.at<float>(kBoxY) * scale_y);
        box.width = static_cast<int>(row.at<float>(kBoxW) * scale_x);
        box.height = static_cast<int>(row.at<float>(kBoxH) * scale_y);
        box.score = row.at<float>(kScoreIndex);

        for (std::size_t lm = 0; lm < kLandmarkCount; ++lm) {
            const int base = kLandmarkOffset + static_cast<int>(lm) * 2;
            box.landmarks[lm][0] = row.at<float>(base) * static_cast<float>(scale_x);
            box.landmarks[lm][1] = row.at<float>(base + 1) * static_cast<float>(scale_y);
        }

        faces.push_back(box);
    }
#endif
    return faces;
}

} // namespace fd
