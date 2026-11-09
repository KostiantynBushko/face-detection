# face-detection

Face **detection** and **recognition** for camera streams, as a small,
self-contained C++20 library.

Built on OpenCV's DNN module with the compact **YuNet** detector and the
**SFace** recognition model — no heavyweight extra dependencies.

```
Frame (BGR) ──► FaceDetector ──► FaceBox[] ──► FaceRecognizer ──► Embedding (128-d)
                                                                  └──► FaceDatabase ──► identity
```

> **Naming.** The project is currently called `face-detection`. It is a
> standalone git repository that is intended to become a **submodule** of
> `objective-control`. Renaming it later is straightforward — see
> [Renaming](#renaming).

---

## Requirements

- CMake ≥ 3.20, a C++20 compiler (GCC 11+/Clang 14+/MSVC).
- [OpenCV](https://opencv.org/) ≥ 4.5.4 with the **`objdetect`** and **`dnn`**
  modules — both ship with the *main* OpenCV repository, so no `opencv_contrib`
  package is required.
  ```sh
  sudo apt install libopencv-dev     # Debian/Ubuntu
  ```
- The ONNX model weights (see [Models](#models)).

## Build

```sh
cmake --preset debug          # or: release, asan
cmake --build --preset debug
ctest --preset default
```

Binaries land in `build/<preset>/bin/`. The `asan` preset builds into
`build/asan` — it does **not** reconfigure `build/debug`.

### Options

| Option                          | Default | Purpose                                  |
|---------------------------------|---------|------------------------------------------|
| `FACE_DETECTION_ENABLE_OPENCV`   | `ON`    | Use the OpenCV-backed implementation     |
| `FACE_DETECTION_BUILD_APPS`      | `ON`\*  | Build the example application             |
| `FACE_DETECTION_BUILD_TESTS`     | `ON`\*  | Build the unit tests                     |
| `FACE_DETECTION_ENABLE_LEGACY`   | `OFF`   | Enable the contrib `face` module (LBPH)  |
| `FACE_DETECTION_ENABLE_SANITIZERS` | `OFF` | ASan/UBSan in debug builds               |

\* `ON` only when this is the top-level project; `OFF` when it is pulled in
via `add_subdirectory()`.

## Models

The weights are **not** committed. Fetch them once:

```sh
./models/fetch.sh            # FP32: YuNet 233 KB + SFace 38.7 MB
./models/fetch.sh --int8     # INT8: YuNet 100 KB + SFace 9.9 MB
```

The script verifies SHA-256 checksums. See [`models/README.md`](models/README.md)
for the runtime lookup order and licensing details.

## Library API

Everything lives in namespace `fd`.

```cpp
#include "face-detection/face/detector.hpp"
#include "face-detection/face/recognizer.hpp"
#include "face-detection/face/database.hpp"

fd::Image image;                 // width / height / channels / data
image.width  = 1920;
image.height = 1080;
image.channels = 3;              // BGR
image.data.resize(1920 * 1080 * 3);

fd::FaceDetector detector;
fd::FaceDetector::Options detector_options;
detector_options.model_path = "models/face_detection_yunet_2023mar.onnx";
if (!detector.open(detector_options)) { /* handle */ }

const std::vector<fd::FaceBox> faces = detector.detect(image);
for (const fd::FaceBox& face : faces) {
    // face.x, face.y, face.width, face.height, face.score
    // face.landmarks[fd::Landmark::RightEye] -> {x, y}, ...
}

fd::FaceRecognizer recognizer;
fd::FaceRecognizer::Options recognizer_options;
recognizer_options.model_path = "models/face_recognition_sface_2021dec.onnx";
recognizer.open(recognizer_options);

fd::Embedding embedding;
recognizer.embed(image, faces.front(), embedding);   // 128-d, unit length

fd::FaceDatabase database;
database.enroll(/*person_id=*/42, image, faces.front(), recognizer);

const fd::FaceDatabase::Match match = database.match(embedding);
if (match.matched) {
    // match.person_id == 42
}
```

### Key types

| Type              | Notes                                                                    |
|-------------------|--------------------------------------------------------------------------|
| `fd::Image`       | Dense row-major BGR (3ch) or grayscale (1ch). Layout-compatible with `oc::Frame`. |
| `fd::FaceBox`     | Bounding box, confidence in [0,1], and five landmarks.                   |
| `fd::Embedding`   | 128-dimensional, L2-normalised feature vector.                          |
| `fd::FaceDatabase`| Enrolled embeddings keyed by an application-defined person id.           |

### Distance semantics

`Embedding::cosineDistance()` returns a **distance**: `0` means identical,
larger means more different. It is computed as `1 - dot(a, b)` on unit-length
vectors, so it lies in `[0, 2]`.

Typical SFace thresholds:

| Distance | Interpretation          |
|----------|-------------------------|
| `< 0.30` | very strong match       |
| `< 0.40` | same person (default)   |
| `> 0.60` | different people        |

Tune against your own camera, lens and lighting.

## Design notes

### The public API does not expose `cv::Mat`

`fd::Image` mirrors the layout of `objective-control`'s `oc::Frame`, so frames
move between the two libraries with **no pixel copy**. Consumers are not
forced to have the OpenCV headers on their include path.

### Detection runs on a downscaled frame

YuNet's cost grows steeply with the input area. Measured on an Intel
i9-13900H, single-threaded, on a 1080p source:

| Detection width | ms/frame | FPS    |
|-----------------|-----------|--------|
| 1920 (native)   | 111.8     | 8.9    |
| 1280            | 21.7      | 46.2   |
| **960**         | **7.5**   | **134** |
| 640             | 5.7       | 175    |

`FaceDetector::Options::detection_width` defaults to `960`. The height is
derived from the frame's aspect ratio, so the frame is never distorted, and
boxes and landmarks are mapped back to full-resolution coordinates. Set it to
`0` to run at native resolution.

### OpenCV quirks this library works around

* `cv::FaceDetectorYN::detect()` returns a **hard-coded `1`** in OpenCV 4.10
  rather than the number of faces. This library uses the number of rows in
  the output matrix instead.
* `cv::FaceDetectorYN::setInputSize()` must be called before every run — the
  detector rejects any input whose size differs from the configured one.
* `cv::FaceRecognizerSF::alignCrop()` needs the **full 1x15 detection row**
  (it reads the landmarks at indices 4..13); a bare bounding box is not
  enough. `fd::FaceBox` carries the landmarks so the row can be rebuilt.
* `cv::FaceRecognizerSF::match()` with `FR_COSINE` returns a **similarity**
  (1.0 = identical), not a distance.

### The OpenCV face stack is probed separately

`find_package(OpenCV COMPONENTS ...)` fails *as a whole* when a single
requested component is missing. The contrib `face` module is absent from many
distributions, so adding it to the same component list as the core modules
would silently disable **all** OpenCV support on those machines.
`cmake/Dependencies.cmake` therefore probes the core modules and the face
stack in two independent steps.

## Command-line app

```sh
./build/debug/bin/face-detection-app --image=photo.jpg
./build/debug/bin/face-detection-app --image=photo.jpg --db=faces.db --enroll=42
./build/debug/bin/face-detection-app --image=photo.jpg --db=faces.db --recognize
```

| Option            | Default              | Meaning                                     |
|-------------------|----------------------|---------------------------------------------|
| `--image=<path>`  | *(required)*         | Input image                                 |
| `--models=<dir>`  | `$FACE_DETECTION_MODELS` or `models` | Model directory       |
| `--threshold=<f>` | `0.6`                | Minimum detection confidence               |
| `--db=<path>`     | *(none)*             | Face database file (load + save)            |
| `--enroll=<id>`   | *(none)*             | Enrol the first detected face under this id |
| `--recognize`     | *(off)*              | Match against the enrolled set             |
| `-h`, `--help`    | —                    | Print the option list                       |

Exit codes: `0` success, `1` runtime error, `2` bad usage.

## Layout

```
include/face-detection/image.hpp            Public image type
include/face-detection/face/detector.hpp     Face detection
include/face-detection/face/recognizer.hpp   Face embeddings
include/face-detection/face/database.hpp     Enrolment / lookup
src/                                          Implementation
apps/face_demo.cpp                            Example CLI application
tests/face_test.cpp                           Smoke tests (no external test deps)
models/fetch.sh                               Model downloader with checksums
cmake/                                        Dependency + warning helpers
CMakePresets.json                             debug / release / asan presets
```

## Integration with `objective-control`

This project is currently an in-tree standalone repository at
`objective-control/third_party/face-detection`. It is intended to become a
git submodule.

To consume it as a subdirectory today:

```cmake
add_subdirectory(third_party/face-detection)
target_link_libraries(objective-control PUBLIC face-detection::face_detection)
```

When `objective-control` is the top-level project, `FACE_DETECTION_BUILD_APPS`
and `FACE_DETECTION_BUILD_TESTS` default to `OFF`, so the submodule does not
build its own app or tests inside the parent build.

## Renaming

Renaming is straightforward and is expected to happen:

1. Rename the CMake project and the exported target:
   ```cmake
   project(<new-name> ...)
   add_library(<new-name>::<new-name> ALIAS <target>)
   ```
2. Rename the include directory and update the `#include` paths.
3. Rename the namespace (`fd` → your choice) and the root include guard style.
4. Rename the git repository and update `origin`.

Because the library is a submodule, the rename is contained: the parent only
sees the gitlink and the target name.

## License

See the upstream project for licensing decisions. The bundled model weights
are MIT (YuNet) and Apache 2.0 (SFace) — see [`models/README.md`](models/README.md).
