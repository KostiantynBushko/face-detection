#pragma once

#include <cstdint>
#include <vector>

namespace fd {

/// A decoded image frame in dense (row-major) order.
///
/// The layout deliberately mirrors objective-control's `oc::Frame` so that a
/// frame can be handed between the two libraries without copying a single
/// pixel: both are `std::vector<std::uint8_t>` in BGR (3 channels) or
/// grayscale (1 channel) order.
///
/// The public API deliberately does NOT expose `cv::Mat`, so consumers are
/// not forced to have the OpenCV headers on their include path.
struct Image {
    int width = 0;
    int height = 0;
    int channels = 0; ///< 1 (grayscale) or 3 (BGR).
    std::vector<std::uint8_t> data;

    [[nodiscard]] bool empty() const {
        return width <= 0 || height <= 0 || data.empty();
    }

    /// True when the buffer is large enough for the declared geometry.
    [[nodiscard]] bool consistent() const {
        const auto expected = static_cast<std::size_t>(width) *
                              static_cast<std::size_t>(height) *
                              static_cast<std::size_t>(channels);
        return !empty() && data.size() == expected;
    }
};

} // namespace fd
