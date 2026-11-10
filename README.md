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
> standalone git repository consumed by `objective-control` as a **git
> submodule** at `third_party/face-detection`. Renaming it later is
> straightforward — see [Renaming](#renaming).

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
- Optional: the PostgreSQL client library (libpq), for the
  [PostgreSQL face store](#where-the-enrolments-live).
  ```sh
  sudo apt install libpq-dev     # Debian/Ubuntu
  ```

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
| `FACE_DETECTION_ENABLE_POSTGRESQL` | `ON`  | Store the enrolments in PostgreSQL       |
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

// The properties of the person: the name that is shown when the person
// is recognised, a picture of the face and a JSON document.
fd::EnrolmentInfo info;
info.user_name = "Ada Lovelace";
info.metadata = "{\"role\":\"admin\"}";
database.enroll(/*person_id=*/42, image, faces.front(), recognizer, info);

const fd::FaceDatabase::Match match = database.match(embedding);
if (match.matched) {
    // match.person_id == 42
}

// Persist the enrolments. The file format is shared with
// objective-control.
database.save("faces.db");
database.load("faces.db");
```

### Where the enrolments live

`FaceDatabase` itself is in-memory: it holds the samples that the matching
runs against. *Where* those samples are kept is a separate concern, handled
by a `FaceStore`:

```cpp
#include "face-detection/face/store.hpp"

std::string error;
auto store = fd::makeFaceStore("postgres://user@localhost/faces", error);
if (!store) { /* the build has no PostgreSQL support */ }
if (!store->open("postgres://user@localhost/faces", error)) { /* handle */ }

std::vector<fd::FaceEntry> entries;
store->load(entries, error);          // read every sample

fd::FaceEntry entry;                  // append one sample
entry.person_id = 42;
entry.embedding = embedding;
entry.info.user_name = "Ada Lovelace";
entry.info.metadata = "{\"role\":\"admin\"}";
store->add(entry, error);             // cheap: one INSERT

store->remove(/*person_id=*/42, error);
```

Two implementations ship with the library:

| Store               | Target                                    | Notes                                        |
|---------------------|-------------------------------------------|----------------------------------------------|
| `FileFaceStore`     | a file path                               | The flat `FDB2` binary format                |
| `PostgresFaceStore` | `postgres://…` / `postgresql://…`         | A PostgreSQL table; needs libpq              |

`makeFaceStore()` picks the backend from the target: a target that starts
with `postgres://` or `postgresql://` is a PostgreSQL one, everything else is
a file.

The file store writes the `FDB2` format:

```text
magic "FDB2" | uint32 count | entry * count
entry := int32 person_id | float[128] | uint32 name_len | name bytes
       | uint32 photo_len | photo bytes | uint32 meta_len | meta bytes
```

The older `FDB1` format -- the person id and the embedding, nothing else
-- is still read, so a database written by an earlier version keeps
working. It is never written any more.

The PostgreSQL store keeps one row per sample:

```sql
CREATE TABLE face_enrollments (
    id          bigserial   PRIMARY KEY,
    person_id   integer     NOT NULL,
    embedding   real[]      NOT NULL,
    user_name   text,
    photo       bytea,
    metadata    jsonb       NOT NULL DEFAULT '{}'::jsonb,
    enrolled_at timestamptz NOT NULL DEFAULT now()
);
```

The three optional columns hold the properties of the person a sample
belongs to:

| Column      | Type    | What it holds                                              |
|-------------|---------|------------------------------------------------------------|
| `user_name` | `text`  | The name that is shown on screen when the person is detected |
| `photo`     | `bytea` | A picture of the face, as the bytes of an encoded image      |
| `metadata`  | `jsonb` | An arbitrary JSON document with whatever the application wants to remember |

The properties are stored with **every** sample of a person, so a store
can answer "who is this?" without a second lookup. The columns were added
after the table was introduced, so a table that predates them is migrated
when the store creates the schema -- and a table that cannot be altered
is used as it is, without the properties.

`PostgresFaceStore::missingPropertyColumns()` names the columns that the
table does not have, so that an application can tell the user that the
name, the picture and the metadata are not stored.
`PostgresFaceStore::createSchema()` -- and `FaceStore::prepareSchema()`,
which every store has -- report every problem they run into, which is
what a deployment that prepares the schema wants. `open()` only reports
what it cannot work with at all.

The table is created when the store connects; [`sql/schema.sql`](sql/schema.sql)
holds the same DDL for deployments that prepare the schema separately. The
connection uses libpq, so `PGHOST`, `PGPORT`, `PGDATABASE`, `PGUSER` and
`PGPASSWORD` apply whenever the connection string does not spell them out.

The picture is optional: a sample without one still matches perfectly
well, and the table does not have to hold biometric source material when
nobody looks at it.

The metadata is stored in a `jsonb` column, so the server checks it and
reformats it -- a document that was written as `{"role":"admin"}` comes
back as `{"role": "admin"}`. The library checks the documents itself as
well (`fd::isValidJson`), so an invalid one is reported with a readable
message instead of a server error.

### Key types

| Type              | Notes                                                                    |
|-------------------|--------------------------------------------------------------------------|
| `fd::Image`       | Dense row-major BGR (3ch) or grayscale (1ch). Layout-compatible with `oc::Frame`. |
| `fd::FaceBox`     | Bounding box, confidence in [0,1], and five landmarks.                   |
| `fd::Embedding`   | 128-dimensional, L2-normalised feature vector.                          |
| `fd::FaceDatabase`| Enrolled embeddings keyed by an application-defined person id. `save`/`load` persist them. |
| `fd::FaceStore`    | Where the enrolled samples are kept — a file or a PostgreSQL table. |
| `fd::FaceEntry`    | One enrolled sample: a person id, its embedding and the properties of the person. |

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

## Tests

```sh
ctest --preset default
```

The suite runs without a database. The PostgreSQL tests are skipped unless a
server is named in the environment:

```sh
FACE_DETECTION_TEST_DSN="host=127.0.0.1 dbname=faces user=faces" \
    ctest --preset default
```

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

# ... with a name and some metadata
./build/debug/bin/face-detection-app --image=photo.jpg --db=faces.db \
    --enroll=42 --enroll-name="Ada Lovelace" --enroll-metadata='{"role":"admin"}'

# The same, but with the enrolments in PostgreSQL
./build/debug/bin/face-detection-app --image=photo.jpg \
    --db=postgres://user@localhost/faces --enroll=42

# List and remove the enrolled people (no image needed)
./build/debug/bin/face-detection-app --db=postgres://user@localhost/faces --db-list
./build/debug/bin/face-detection-app --db=postgres://user@localhost/faces --db-remove=42
```

| Option            | Default              | Meaning                                     |
|-------------------|----------------------|---------------------------------------------|
| `--image=<path>`  | *(required)*         | Input image                                 |
| `--models=<dir>`  | `$FACE_DETECTION_MODELS` or `models` | Model directory       |
| `--threshold=<f>` | `0.6`                | Minimum detection confidence               |
| `--db=<target>`   | *(none)*             | Face database: a file path or a `postgres://…` connection string (load + save) |
| `--db-list`       | *(off)*              | List the enrolled people and exit           |
| `--db-remove=<id>`| *(none)*             | Remove every sample of a person and exit    |
| `--enroll=<id>`   | *(none)*             | Enrol the first detected face under this id |
| `--enroll-name=<name>` | *(none)*       | The name of the person being enrolled; stored with the sample and shown when the person is recognised |
| `--enroll-metadata=<json>` | *(none)* | Extra properties of the person, as a JSON document |
| `--recognize`     | *(off)*              | Match against the enrolled set             |
| `-h`, `--help`    | —                    | Print the option list                       |

Exit codes: `0` success, `1` runtime error, `2` bad usage.

### Locating the model files

The model directory is resolved **relative to the current working directory**,
not relative to the executable. The lookup order is:

1. `--models=<dir>`
2. the `FACE_DETECTION_MODELS` environment variable
3. `./models`, relative to the working directory

Launching the app from `build/debug/bin/` therefore fails with:

```text
Could not load the detector model: OpenCV(4.10.0) ./modules/dnn/src/onnx/onnx_importer.cpp:277:
error: (-5:Bad argument) Can't read ONNX file: models/face_detection_yunet_2023mar.onnx
in function 'ONNXImporter'
```

Give the app the model directory explicitly:

```sh
./build/debug/bin/face-detection-app --image=photo.jpg \
    --models=/path/to/face-detection/models
```

Or export it once for the current shell session:

```sh
export FACE_DETECTION_MODELS=/path/to/face-detection/models
```

Running the app from the repository root also works, because `models/` sits
next to the sources.

## Layout

```
include/face-detection/image.hpp            Public image type
include/face-detection/face/detector.hpp     Face detection
include/face-detection/face/recognizer.hpp   Face embeddings
include/face-detection/face/database.hpp     Enrolment / lookup
include/face-detection/face/store.hpp        Where the enrolments are kept
include/face-detection/face/postgres_store.hpp  PostgreSQL backend
src/                                          Implementation
sql/schema.sql                                DDL of the PostgreSQL table
apps/face_demo.cpp                            Example CLI application
tests/face_test.cpp                           Smoke tests (no external test deps)
models/fetch.sh                               Model downloader with checksums
cmake/                                        Dependency + warning helpers
CMakePresets.json                             debug / release / asan presets
```

## Integration with `objective-control`

This repository is consumed by `objective-control` as a git submodule:

```
objective-control/
|-- .gitmodules
|-- third_party/
|   `-- face-detection/      <- this repository (gitlink)
```

The parent wires it up with:

```cmake
add_subdirectory(third_party/face-detection)
target_link_libraries(objective-control PUBLIC face-detection::face_detection)
```

When `objective-control` is the top-level project, `FACE_DETECTION_BUILD_APPS`
and `FACE_DETECTION_BUILD_TESTS` default to `OFF`, so the submodule does not
build its own app or tests inside the parent build.

### Cloning

```sh
git clone --recurse-submodules <objective-control-url>
```

or, for an existing clone:

```sh
git submodule update --init --recursive
```

The submodule's `origin` currently points at a local stand-in repository
(`/home/kbush/milproj/face-detection.git`). Point it at a real hosted remote
when one exists:

```sh
cd third_party/face-detection
git remote set-url origin <real-url>
git push -u origin master
```

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
