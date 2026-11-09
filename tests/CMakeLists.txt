# Minimal, dependency-free smoke test (no GTest required).

add_executable(face_test
    face_test.cpp
)

target_link_libraries(face_test PRIVATE face_detection)

target_compile_definitions(face_test PRIVATE
    FD_MODELS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../models"
)

warn_on(face_test)

add_test(NAME face_test COMMAND face_test)
