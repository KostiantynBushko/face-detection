# Dependencies.cmake
#
# Central place for discovering optional third-party dependencies.
#
# IMPORTANT — why the face stack is probed separately
# ---------------------------------------------------
# find_package(OpenCV COMPONENTS ...) fails *as a whole* when a single
# requested component is missing, even if every other component is present.
# The `face` module lives in opencv_contrib and is therefore absent from
# many distributions.
#
# If we added `face`/`dnn`/`objdetect` to the same component list used for
# the camera backends, a machine without opencv_contrib would silently lose
# ALL OpenCV support. So we probe in two independent steps:
#
#   1. core OpenCV  -> required by the face stack itself
#   2. face stack   -> objdetect + dnn (main OpenCV) and, optionally, the
#                      contrib `face` module
#
# Each probe sets its own *_FOUND switch and its own imported targets.

# ---------------------------------------------------------------------------
# Step 1: core OpenCV.
# ---------------------------------------------------------------------------
set(FD_HAVE_OPENCV FALSE)
set(FD_OPENCV_TARGETS "")

if(FACE_DETECTION_ENABLE_OPENCV)
    find_package(OpenCV QUIET COMPONENTS core imgproc imgcodecs)

    if(OpenCV_FOUND)
        set(FD_HAVE_OPENCV TRUE)
        set(FD_OPENCV_TARGETS ${OpenCV_LIBS})
        message(STATUS "OpenCV ${OpenCV_VERSION} found (config package)")
    else()
        find_package(PkgConfig QUIET)
        if(PkgConfig_FOUND)
            pkg_check_modules(OpenCV IMPORTED_TARGET QUIET opencv4)
            if(OpenCV_FOUND)
                set(FD_HAVE_OPENCV TRUE)
                set(FD_OPENCV_TARGETS PkgConfig::OpenCV)
                message(STATUS "OpenCV found via pkg-config (opencv4)")
            endif()
        endif()
    endif()

    if(NOT FD_HAVE_OPENCV)
        message(WARNING
            "OpenCV not found. face-detection will build without the "
            "OpenCV-backed implementation. Install e.g. "
            "'sudo apt install libopencv-dev'.")
    endif()
endif()

# ---------------------------------------------------------------------------
# Step 2a: the DNN face stack (objdetect + dnn).
#
# Both modules ship with the *main* OpenCV repository, so this is available
# on virtually every distribution. It provides:
#   * cv::FaceDetectorYN   (YuNet)  - detector
#   * cv::FaceRecognizerSF (SFace) - 128-d face embeddings
#
# NOTE: the probe is deliberately separate from step 1 so that a missing
# contrib module can never take down the core OpenCV support.
# ---------------------------------------------------------------------------
set(FD_HAVE_FACE_STACK FALSE)
set(FD_FACE_STACK_TARGETS "")

if(FD_HAVE_OPENCV)
    find_package(OpenCV QUIET COMPONENTS objdetect dnn)
    if(OpenCV_FOUND)
        set(FD_HAVE_FACE_STACK TRUE)
        set(FD_FACE_STACK_TARGETS ${OpenCV_LIBS})
        message(STATUS "OpenCV face stack available (objdetect + dnn)")
    else()
        message(WARNING
            "OpenCV objdetect/dnn not found; face detection/recognition "
            "will be disabled. Install the OpenCV development package.")
    endif()
endif()

# ---------------------------------------------------------------------------
# Step 2b: the optional contrib 'face' module (LBPH / Eigenfaces / Fisher).
#
# Purely optional: it enables the classical, model-free recognisers as a
# fallback for systems where the DNN stack is unavailable.
# ---------------------------------------------------------------------------
set(FD_HAVE_LEGACY_FACE FALSE)
set(FD_LEGACY_FACE_TARGETS "")

if(FACE_DETECTION_ENABLE_LEGACY)
    if(FD_HAVE_OPENCV)
        find_package(OpenCV QUIET COMPONENTS face)
        if(OpenCV_FOUND)
            set(FD_HAVE_LEGACY_FACE TRUE)
            set(FD_LEGACY_FACE_TARGETS ${OpenCV_LIBS})
            message(STATUS "OpenCV contrib 'face' module available (LBPH fallback)")
        else()
            message(STATUS
                "OpenCV contrib 'face' module not found; "
                "skipping the LBPH fallback.")
        endif()
    endif()
endif()

# ---------------------------------------------------------------------------
# Step 3: PostgreSQL (libpq) -- optional backend for the enrolled faces.
#
# The face store keeps the enrolled embeddings either in a flat file or in
# a PostgreSQL table. The database backend needs libpq, the C client
# library that ships with PostgreSQL; libpqxx is deliberately not used, so
# the only extra build dependency is the one PostgreSQL already provides.
# ---------------------------------------------------------------------------
set(FD_HAVE_POSTGRES FALSE)
set(FD_POSTGRES_TARGETS "")

if(FACE_DETECTION_ENABLE_POSTGRESQL)
    find_package(PostgreSQL QUIET)

    if(PostgreSQL_FOUND)
        set(FD_HAVE_POSTGRES TRUE)
        set(FD_POSTGRES_TARGETS PostgreSQL::PostgreSQL)
        message(STATUS "PostgreSQL ${PostgreSQL_VERSION} found (libpq)")
    else()
        find_package(PkgConfig QUIET)
        if(PkgConfig_FOUND)
            pkg_check_modules(PostgreSQL IMPORTED_TARGET QUIET libpq)
            if(PostgreSQL_FOUND)
                set(FD_HAVE_POSTGRES TRUE)
                set(FD_POSTGRES_TARGETS PkgConfig::PostgreSQL)
                message(STATUS "PostgreSQL found via pkg-config (libpq)")
            endif()
        endif()
    endif()

    if(NOT FD_HAVE_POSTGRES)
        message(WARNING
            "PostgreSQL client library (libpq) not found; the PostgreSQL "
            "face store is disabled. Install e.g. "
            "'sudo apt install libpq-dev'.")
    endif()
endif()

# ---------------------------------------------------------------------------
# Sanitizers (debug builds only).
# ---------------------------------------------------------------------------
if(FACE_DETECTION_ENABLE_SANITIZERS)
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        message(FATAL_ERROR "Sanitizers are only supported on GCC/Clang.")
    endif()
    add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address,undefined)
endif()
