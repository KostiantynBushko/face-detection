# Models

The ONNX weights are **not** committed to this repository. Fetch them with:

```sh
./models/fetch.sh            # FP32: YuNet 233 KB + SFace 38.7 MB
./models/fetch.sh --int8     # INT8: YuNet 100 KB + SFace 9.9 MB
```

The script verifies SHA-256 checksums and installs into this directory (or
`$FACE_DETECTION_MODELS`).

## What the models are

| File | Purpose | Source | License |
|---|---|---|---|
| `face_detection_yunet_2023mar.onnx` | Face detector (YuNet) | [opencv_zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) | MIT |
| `face_recognition_sface_2021dec.onnx` | Face embeddings (SFace, 128-d) | [opencv_zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface) | Apache 2.0 |

Both are also available in INT8 variants (`--int8`), which are noticeably
faster on CPU at a small cost in accuracy.

## Runtime lookup order

The library does not hard-code a path. The example application resolves the
model directory in this order:

1. `--models=<dir>` command-line option
2. `$FACE_DETECTION_MODELS` environment variable
3. `./models` relative to the working directory

Step 3 is relative to the **working directory**, not to the executable, so the
app has to be launched from the directory that holds `models/` — or given an
explicit path. Otherwise OpenCV reports:

```text
Can't read ONNX file: models/face_detection_yunet_2023mar.onnx
```

## Note on the SFace training data

SFace was trained on VGGFace2. The *weights* are Apache 2.0, but if you ship
a commercial product you should confirm that your intended use complies with
the dataset's terms as well.
