// Smoke tests for face-detection. No external test framework required.
//
// The model-backed checks are skipped automatically when the ONNX weights are
// not present, so the suite still passes on a machine without the models.

#include "face-detection/face/database.hpp"
#include "face-detection/face/detector.hpp"
#include "face-detection/face/postgres_store.hpp"
#include "face-detection/face/recognizer.hpp"
#include "face-detection/face/store.hpp"
#include "face-detection/image.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#ifdef FD_HAVE_POSTGRES
#include <libpq-fe.h>
#endif

#ifndef FD_MODELS_DIR
#define FD_MODELS_DIR "models"
#endif

namespace {

int failures = 0;

void check(bool condition, const char* what) {
    if (!condition) {
        std::cerr << "FAIL: " << what << '\n';
        ++failures;
    }
}

bool fileExists(const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) {
        return false;
    }
    std::fclose(f);
    return true;
}

/// Builds a unit-length embedding whose first component is @p direction.
fd::Embedding makeEmbedding(float direction) {
    fd::Embedding e;
    e.values[0] = direction;
    e.values[1] = std::sqrt(std::max(0.0f, 1.0f - direction * direction));
    e.valid = true;
    return e;
}

/// Builds the unit vector along @p axis. Two different axes are perfectly
/// orthogonal, which makes it easy to construct genuinely distant pairs.
fd::Embedding makeEmbeddingOnAxis(std::size_t axis) {
    fd::Embedding e;
    e.values[axis] = 1.0f;
    e.valid = true;
    return e;
}

#ifdef FD_HAVE_POSTGRES

/// Runs @p sql with @p param bound to $1 and returns the first column of
/// the first row as a string. Returns an empty string when the query
/// fails.
std::string queryScalar(const std::string& dsn, const std::string& sql,
                        const std::string& param) {
    PGconn* conn = PQconnectdb(dsn.c_str());
    if (conn == nullptr) {
        return {};
    }
    const char* values[1] = {param.c_str()};
    PGresult* result = PQexecParams(conn, sql.c_str(), 1, nullptr, values,
                                     nullptr, nullptr, 0);
    std::string value;
    if (result != nullptr && PQresultStatus(result) == PGRES_TUPLES_OK &&
        PQntuples(result) > 0 && !PQgetisnull(result, 0, 0)) {
        value = PQgetvalue(result, 0, 0);
    }
    if (result != nullptr) {
        PQclear(result);
    }
    PQfinish(conn);
    return value;
}

/// True when the table @p table has an index on its person_id column.
bool hasPersonIdIndex(const std::string& dsn, const std::string& table) {
    const std::string sql =
        "SELECT EXISTS (SELECT 1 FROM pg_index i "
        "JOIN pg_class c ON c.oid = i.indrelid "
        "JOIN pg_namespace n ON n.oid = c.relnamespace "
        "JOIN pg_attribute a ON a.attrelid = i.indrelid "
        "AND a.attnum = i.indkey[0] "
        "WHERE c.relname = $1 AND n.nspname = current_schema() "
        "AND a.attname = 'person_id')";
    return queryScalar(dsn, sql, table) == "t";
}

/// Runs a statement that has no parameters.
bool runCommand(const std::string& dsn, const std::string& sql) {
    PGconn* conn = PQconnectdb(dsn.c_str());
    if (conn == nullptr) {
        return false;
    }
    PGresult* result = PQexec(conn, sql.c_str());
    const bool ok = result != nullptr &&
                    PQresultStatus(result) == PGRES_COMMAND_OK;
    if (result != nullptr) {
        PQclear(result);
    }
    PQfinish(conn);
    return ok;
}

/// Drops the table @p table, if it is there.
void dropTable(const std::string& dsn, const std::string& table) {
    // The names used here only hold letters, digits and underscores, so
    // quoting them is enough to keep them apart from the SQL.
    runCommand(dsn, "DROP TABLE IF EXISTS \"" + table + "\" CASCADE");
}

#endif // FD_HAVE_POSTGRES

} // namespace

int main() {
    using fd::Embedding;
    using fd::FaceBox;
    using fd::FaceDatabase;
    using fd::FaceDetector;
    using fd::FaceRecognizer;
    using fd::Image;

    // ---- Image ------------------------------------------------------------
    {
        Image empty;
        check(empty.empty(), "default image is empty");
        check(!empty.consistent(), "default image is not consistent");

        Image good;
        good.width = 4;
        good.height = 2;
        good.channels = 3;
        good.data.assign(4 * 2 * 3, 0);
        check(!good.empty(), "populated image is not empty");
        check(good.consistent(), "populated image is consistent");

        Image bad = good;
        bad.data.pop_back();
        check(!bad.consistent(), "short buffer is not consistent");
    }

    // ---- Embedding geometry ----------------------------------------------
    {
        const Embedding a = makeEmbeddingOnAxis(0);
        check(a.valid, "embedding is valid");

        const float self = a.cosineDistance(a);
        check(std::fabs(self) < 1e-5f, "cosine distance to self is ~0");

        const Embedding b = makeEmbeddingOnAxis(1);
        const float orthogonal = a.cosineDistance(b);
        check(std::fabs(orthogonal - 1.0f) < 1e-5f,
              "cosine distance to an orthogonal vector is ~1");

        Embedding invalid;
        invalid.valid = false;
        check(std::fabs(invalid.cosineDistance(a) - 1.0f) < 1e-5f,
              "distance to an invalid embedding is 1");
    }

    // ---- FaceDatabase -----------------------------------------------------
    {
        FaceDatabase db;
        check(db.empty(), "new database is empty");
        check(db.match(makeEmbedding(1.0f)).person_id == -1,
              "matching an empty database finds nothing");

        check(db.enroll(7, makeEmbeddingOnAxis(0)),
              "enroll accepts a valid embedding");
        check(db.enroll(8, makeEmbeddingOnAxis(1)),
              "enroll accepts a second person");
        check(db.size() == 2, "database holds two entries");

        // Slightly perturbed version of person 7's axis.
        const auto hit = db.match(makeEmbedding(0.99f), 0.4f);
        check(hit.matched, "a close embedding matches");
        check(hit.person_id == 7, "the closest person is reported");
        check(hit.distance < 0.4f, "the reported distance is within the limit");

        // Axis 2 is orthogonal to both enrolled vectors, so the closest
        // distance is 1.0 -- far outside any sane threshold.
        const auto miss = db.match(makeEmbeddingOnAxis(2), 0.05f);
        check(!miss.matched, "a distant embedding does not match a tight limit");
        check(miss.person_id >= 0, "a nearest entry is still reported");
        check(miss.distance > 0.05f, "the reported distance is above the limit");

        check(db.remove(7) == 1, "remove drops one entry");
        check(db.size() == 1, "one entry remains");
        db.clear();
        check(db.empty(), "clear empties the database");

        Embedding invalid;
        check(!db.enroll(1, invalid), "enroll rejects an invalid embedding");
    }

    // ---- FaceStore --------------------------------------------------------
    {
        using fd::FaceEntry;
        using fd::FileFaceStore;
        using fd::PersonSummary;

        // The target decides which backend is used.
        check(fd::isPostgresTarget("postgres://host/db"),
              "postgres:// is a PostgreSQL target");
        check(fd::isPostgresTarget("postgresql://host/db"),
              "postgresql:// is a PostgreSQL target");
        check(!fd::isPostgresTarget("faces.db"),
              "a file path is not a PostgreSQL target");
        check(!fd::isPostgresTarget(""),
              "an empty target is not a PostgreSQL target");

        std::string error;
        auto store = fd::makeFaceStore("faces.db", error);
        check(static_cast<bool>(store), "a file target yields a store");
        check(dynamic_cast<FileFaceStore*>(store.get()) != nullptr,
              "a file target yields a file store");

        auto pg = fd::makeFaceStore("postgres://host/db", error);
        check(static_cast<bool>(pg) == fd::postgresAvailable(),
              "a PostgreSQL target yields a store only when PostgreSQL is "
              "available");

        // ---- the file store round-trips the samples ----
        const std::string path = "face_store_test.db";
        std::remove(path.c_str());
        {
            FileFaceStore file_store;
            check(file_store.open(path, error), "the file store opens");
            check(file_store.isOpen(), "the file store reports open");

            std::vector<FaceEntry> loaded;
            check(!file_store.load(loaded, error),
                  "loading a missing file fails");
            check(loaded.empty(), "a failed load leaves nothing behind");

            FaceEntry first;
            first.person_id = 11;
            first.embedding = makeEmbeddingOnAxis(0);
            check(file_store.add(first, error), "the file store adds a sample");

            FaceEntry second;
            second.person_id = 12;
            second.embedding = makeEmbeddingOnAxis(1);
            check(file_store.add(second, error),
                  "the file store adds a second sample");

            std::vector<PersonSummary> people;
            check(file_store.list(people, error),
                  "the file store lists the enrolled people");
            check(people.size() == 2, "two people are listed");
            if (people.size() == 2) {
                check(people[0].person_id == 11 && people[0].samples == 1,
                      "the first person has one sample");
                check(people[1].person_id == 12 && people[1].samples == 1,
                      "the second person has one sample");
            }

            check(file_store.remove(11, error) == 1,
                  "the file store removes one sample");
            check(file_store.remove(11, error) == 0,
                  "removing an unknown person removes nothing");

            file_store.close();
            check(!file_store.isOpen(), "the file store reports closed");
        }

        // The samples survived the round trip through the file.
        {
            FileFaceStore file_store;
            check(file_store.open(path, error), "the file store reopens");
            std::vector<FaceEntry> loaded;
            check(file_store.load(loaded, error), "the file store loads");
            check(loaded.size() == 1, "one sample is left in the file");
            if (loaded.size() == 1) {
                check(loaded[0].person_id == 12,
                      "the remaining sample kept its person id");
                check(loaded[0].embedding.valid,
                      "the remaining sample kept its embedding");
            }
        }
        std::remove(path.c_str());
    }

    // ---- PostgreSQL store --------------------------------------------------
    // The database tests need a server, so they only run when one is named
    // in the environment. Everything else stays database-free.
    {
        const char* dsn = std::getenv("FACE_DETECTION_TEST_DSN");
        if (dsn == nullptr || *dsn == '\0') {
            std::cout << "SKIP: PostgreSQL store test (set "
                         "FACE_DETECTION_TEST_DSN to run it)\n";
        } else if (!fd::postgresAvailable()) {
            std::cout << "SKIP: PostgreSQL store test (this build has no "
                         "PostgreSQL support)\n";
        } else {
            using fd::FaceEntry;
            using fd::PersonSummary;
            using fd::PostgresFaceStore;

            std::string error;
            PostgresFaceStore pg;
            const bool opened = pg.open(dsn, error);
            check(opened, ("the PostgreSQL store opens: " + error).c_str());
            check(pg.isOpen(), "the PostgreSQL store reports open");

            // Start from a known state.
            pg.remove(9001, error);
            pg.remove(9002, error);

            FaceEntry first;
            first.person_id = 9001;
            first.embedding = makeEmbeddingOnAxis(0);
            const bool added = pg.add(first, error);
            check(added, "the PostgreSQL store adds a sample");

            FaceEntry again;
            again.person_id = 9001;
            again.embedding = makeEmbeddingOnAxis(0);
            const bool addedAgain = pg.add(again, error);
            check(addedAgain,
                  "the PostgreSQL store adds a second sample of the person");

            std::vector<FaceEntry> loaded;
            const bool loadedOk = pg.load(loaded, error);
            check(loadedOk, ("the PostgreSQL store loads: " + error).c_str());

            int found = 0;
            bool exact = true;
            for (const FaceEntry& entry : loaded) {
                if (entry.person_id != 9001) {
                    continue;
                }
                ++found;
                for (std::size_t i = 0; i < Embedding::kDimension; ++i) {
                    if (entry.embedding.values[i] != first.embedding.values[i]) {
                        exact = false;
                    }
                }
            }
            check(found == 2, "both samples of the person come back");
            check(exact, "the embedding survives the round trip unchanged");

            std::vector<PersonSummary> people;
            const bool listedOk = pg.list(people, error);
            check(listedOk, "the PostgreSQL store lists");
            bool listed = false;
            for (const PersonSummary& person : people) {
                if (person.person_id == 9001 && person.samples == 2) {
                    listed = true;
                }
            }
            check(listed, "the person is listed with both of its samples");

            check(pg.remove(9001, error) == 2,
                  "the PostgreSQL store removes both samples");
            check(pg.remove(9001, error) == 0,
                  "removing an unknown person removes nothing");

            pg.close();
            check(!pg.isOpen(), "the PostgreSQL store reports closed");
        }
    }

    // ---- PostgreSQL store: table names at the identifier limit --------
    // PostgreSQL truncates identifiers to 63 bytes. A table name at that
    // limit used to make the name of the index collapse onto the name of
    // the table, so the index was silently never created.
#ifdef FD_HAVE_POSTGRES
    {
        const char* dsn = std::getenv("FACE_DETECTION_TEST_DSN");
        if (dsn != nullptr && *dsn != '\0' && fd::postgresAvailable()) {
            using fd::PostgresFaceStore;

            // The longest name PostgreSQL accepts, and one that already
            // ends in the suffix the index name is built from.
            std::vector<std::string> tables;
            tables.push_back(std::string(63, 'a'));
            tables.push_back(std::string(49, 'b') + "_person_id_idx");

            for (const std::string& table : tables) {
                PostgresFaceStore named;
                PostgresFaceStore::Options namedOptions;
                namedOptions.table = table;
                named.setOptions(namedOptions);

                std::string error;
                const bool opened = named.open(dsn, error);
                check(opened,
                      ("the store opens with a " +
                       std::to_string(table.size()) +
                       " character table name: " + error).c_str());
                check(hasPersonIdIndex(dsn, table),
                      "the person_id index is created for the table");

                dropTable(dsn, table);
            }
        }
    }
#endif

    // ---- Model-backed checks ----------------------------------------------
    const std::string models_dir = FD_MODELS_DIR;
    const std::string detector_model =
        models_dir + "/face_detection_yunet_2023mar.onnx";
    const std::string recognizer_model =
        models_dir + "/face_recognition_sface_2021dec.onnx";

    if (!fileExists(detector_model)) {
        std::cout << "SKIP: detector model not found at " << detector_model
                  << " (run models/fetch.sh)\n";
    } else {
        FaceDetector detector;
        FaceDetector::Options options;
        options.model_path = detector_model;
        check(detector.open(options), "detector opens with a valid model");
        check(detector.isOpen(), "detector reports open");

        // A blank frame must not report any faces.
        Image blank;
        blank.width = 320;
        blank.height = 240;
        blank.channels = 3;
        blank.data.assign(320 * 240 * 3, 0);
        check(detector.detect(blank).empty(), "a blank frame yields no faces");

        // A malformed frame must be rejected rather than crash.
        Image malformed;
        malformed.width = 100;
        malformed.height = 100;
        malformed.channels = 3;
        malformed.data.assign(10, 0);
        check(detector.detect(malformed).empty(),
              "a malformed frame yields no faces");

        detector.close();
        check(!detector.isOpen(), "detector reports closed after close");
    }

    if (!fileExists(recognizer_model)) {
        std::cout << "SKIP: recognizer model not found at " << recognizer_model
                  << " (run models/fetch.sh)\n";
    } else {
        FaceRecognizer recognizer;
        FaceRecognizer::Options options;
        options.model_path = recognizer_model;
        check(recognizer.open(options), "recognizer opens with a valid model");
        check(recognizer.isOpen(), "recognizer reports open");

        // Embedding a synthetic face box on a blank frame must still yield a
        // valid, unit-length embedding (the pixels are meaningless, but the
        // pipeline has to produce a well-formed vector).
        Image blank;
        blank.width = 320;
        blank.height = 240;
        blank.channels = 3;
        blank.data.assign(320 * 240 * 3, 128);

        FaceBox box;
        box.x = 100;
        box.y = 60;
        box.width = 80;
        box.height = 80;
        box.landmarks[0] = {124.0f, 84.0f};
        box.landmarks[1] = {156.0f, 84.0f};
        box.landmarks[2] = {140.0f, 104.0f};
        box.landmarks[3] = {126.0f, 124.0f};
        box.landmarks[4] = {154.0f, 124.0f};

        Embedding embedding;
        check(recognizer.embed(blank, box, embedding),
              "recognizer produces an embedding");
        check(embedding.valid, "the embedding is marked valid");

        float norm = 0.0f;
        for (const float v : embedding.values) {
            norm += v * v;
        }
        check(std::fabs(std::sqrt(norm) - 1.0f) < 1e-4f,
              "the embedding is unit length");

        recognizer.close();
        check(!recognizer.isOpen(), "recognizer reports closed after close");
    }

    if (failures == 0) {
        std::cout << "face_test: all checks passed.\n";
        return 0;
    }
    std::cerr << "face_test: " << failures << " check(s) failed.\n";
    return 1;
}
