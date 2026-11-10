# Minimal, dependency-free smoke test (no GTest required).

add_executable(face_test
    face_test.cpp
)

target_link_libraries(face_test PRIVATE face_detection)

# The PostgreSQL checks talk to the server directly to confirm that the
# schema really is in place, so they need libpq as well.
if(FD_HAVE_POSTGRES)
    target_compile_definitions(face_test PRIVATE FD_HAVE_POSTGRES=1)
    target_link_libraries(face_test PRIVATE ${FD_POSTGRES_TARGETS})
endif()

target_compile_definitions(face_test PRIVATE
    FD_MODELS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../models"
)

warn_on(face_test)

add_test(NAME face_test COMMAND face_test)
