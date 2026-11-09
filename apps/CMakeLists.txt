# Example application.

add_executable(face-detection-app
    face_demo.cpp
)

target_link_libraries(face-detection-app PRIVATE face_detection)
warn_on(face-detection-app)
