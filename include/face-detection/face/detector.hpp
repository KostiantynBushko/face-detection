#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "face-detection/image.hpp"

namespace fd {

/// Number of facial landmarks produced by the detector.
inline constexpr std::size_t kLandmarkCount = 5;

/// Landmark indices, in the order used by the underlying model.
enum class Landmark : std::size_t {
    RightEye = 0,   ///< Subject's right eye (left side of the image).
    LeftEye = 1,    ///< Subject's left eye.
    Nose = 2,       ///< Nose tip.
    MouthRight = 3, ///< Right corner of the mouth.
    MouthLeft = 4,  ///< Left corner of the mouth.
};

/// A detected face: bounding box, confidence and five landmarks.
struct FaceBox {
    int x = 0;
    int y = 0;
    int width = 0;
    int height = 0;

    /// Detector confidence in [0, 1].
    float score = 0.0f;

    /// Landmark positions in *full-resolution* image coordinates.
    std::array<std::array<float, 2>, kLandmarkCount> landmarks{};

    [[nodiscard]] int right() const { return x + width; }
    [[nodiscard]] int bottom() const { return y + height; }
};

/// Detects faces in image frames.
///
/// Backed by YuNet (a compact CNN) running through OpenCV's DNN module.
/// Detection is performed on a downscaled copy of the frame by default,
/// because YuNet's cost grows steeply with the input area; the resulting
/// boxes and landmarks are mapped back to full-resolution coordinates.
class FaceDetector {
public:
    struct Options {
        /// Path to `face_detection_yunet_*.onnx`.
        std::string model_path;

        /// Minimum confidence for a detection to be reported.
        float score_threshold = 0.6f;

        /// Non-maximum-suppression IoU threshold.
        float nms_threshold = 0.3f;

        /// Maximum number of candidate boxes kept before NMS.
        int top_k = 5000;

        /// Width of the image the detector actually runs on. The height is
        /// derived from the frame's aspect ratio, so the frame is never
        /// distorted. Set to 0 to run at native resolution.
        ///
        /// 960 is a good default: on a 1080p source it costs ~7 ms/frame
        /// (~134 FPS, single-threaded) instead of ~112 ms at full size.
        int detection_width = 960;
    };

    FaceDetector();
    ~FaceDetector();

    FaceDetector(const FaceDetector&) = delete;
    FaceDetector& operator=(const FaceDetector&) = delete;
    FaceDetector(FaceDetector&&) noexcept;
    FaceDetector& operator=(FaceDetector&&) noexcept;

    /// Loads the model. Returns false if the file is missing or invalid.
    bool open(const Options& options);

    /// Releases the model. Safe to call repeatedly.
    void close();

    [[nodiscard]] bool isOpen() const;

    /// Detects every face in @p image.
    ///
    /// Returns an empty vector if the detector is closed or the image is
    /// malformed. Boxes and landmarks are in full-resolution coordinates.
    std::vector<FaceBox> detect(const Image& image) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace fd
