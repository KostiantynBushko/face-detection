# Object library: face_detection

add_library(face_detection
    face/detector.cpp
    face/recognizer.cpp
    face/database.cpp
    face/store.cpp
    face/postgres_store.cpp
    face/json.cpp
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

# The PostgreSQL store needs libpq. It is PRIVATE on purpose: the public
# headers do not include any libpq header, so consumers never have to have
# the PostgreSQL development files on their include path.
if(FD_HAVE_POSTGRES)
    target_compile_definitions(face_detection PRIVATE FD_HAVE_POSTGRES=1)
    target_link_libraries(face_detection PRIVATE ${FD_POSTGRES_TARGETS})
endif()

warn_on(face_detection)
