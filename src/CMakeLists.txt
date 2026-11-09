# Object library: face_detection

add_library(face_detection
    face/detector.cpp
    face/recognizer.cpp
    face/database.cpp
)

add_library(face-detection::face_detection ALIAS face_detection)

target_include_directories(face_detection
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
        $<INSTALL_INTERFACE:include>
)

if(FD_HAVE_FACE_STACK)
    target_compile_definitions(face_detection PUBLIC FD_HAVE_FACE_STACK=1)
    target_link_libraries(face_detection PUBLIC ${FD_FACE_STACK_TARGETS})
endif()

if(FD_HAVE_LEGACY_FACE)
    target_compile_definitions(face_detection PUBLIC FD_HAVE_LEGACY_FACE=1)
    target_link_libraries(face_detection PUBLIC ${FD_LEGACY_FACE_TARGETS})
endif()

warn_on(face_detection)
