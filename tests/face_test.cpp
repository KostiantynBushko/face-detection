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
#include "face-detection/json.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
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

/// Removes the whitespace of a JSON document.
///
/// The server reformats the metadata it stores -- a jsonb column comes
/// back as "{\"role\": \"admin\"}" whatever was written -- so the two
/// documents are compared without their whitespace.
std::string withoutSpaces(const std::string& text) {
    std::string out;
    for (const char c : text) {
        if (c != ' ' && c != '\n' && c != '\t' && c != '\r') {
            out.push_back(c);
        }
    }
    return out;
}

/// Writes a face database in the format of an earlier version: the
/// person id and the embedding, nothing else.
bool writeLegacyFile(const std::string& path, int person_id,
                     const fd::Embedding& embedding) {
    std::FILE* file = std::fopen(path.c_str(), "wb");
    if (file == nullptr) {
        return false;
    }

    const char magic[4] = {'F', 'D', 'B', '1'};
    const std::uint32_t count = 1;
    const auto id = static_cast<std::int32_t>(person_id);

    const bool ok =
        std::fwrite(magic, 1, sizeof(magic), file) == sizeof(magic) &&
        std::fwrite(&count, sizeof(count), 1, file) == 1 &&
        std::fwrite(&id, sizeof(id), 1, file) == 1 &&
        std::fwrite(embedding.values.data(), sizeof(float),
                    fd::Embedding::kDimension, file) ==
            fd::Embedding::kDimension;
    std::fclose(file);
    return ok;
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

/// Runs @p sql and returns the first column of the first row as a string.
/// Returns an empty string when the query fails.
std::string queryScalarNoParam(const std::string& dsn, const std::string& sql) {
    PGconn* conn = PQconnectdb(dsn.c_str());
    if (conn == nullptr) {
        return {};
    }
    PGresult* result = PQexec(conn, sql.c_str());
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

/// Builds a connection string for another user of the same database.
///
/// Both the keyword form ("host=... dbname=...") and the URL form
/// ("postgres://user:pass@host/db") are understood. The name of the
/// database is always spelled out, because it would otherwise default to
/// the name of the user.
std::string dsnForUser(const std::string& dsn, const std::string& user,
                       const std::string& password,
                       const std::string& database) {
    const std::size_t scheme = dsn.find("://");
    if (scheme != std::string::npos) {
        // The URL form: everything behind the "@" is the host.
        std::string authority = dsn.substr(scheme + 3);
        std::string tail;
        const std::size_t slash = authority.find('/');
        if (slash != std::string::npos) {
            tail = authority.substr(slash);
            authority = authority.substr(0, slash);
        }
        const std::size_t at = authority.find('@');
        if (at != std::string::npos) {
            authority = authority.substr(at + 1);
        }
        // The query string of the original DSN is kept.
        const std::size_t question = tail.find('?');
        const std::string query =
            (question == std::string::npos) ? std::string()
                                             : tail.substr(question);
        return dsn.substr(0, scheme + 3) + user + ":" + password + "@" +
               authority + "/" + database + query;
    }

    // The keyword form: the tokens are copied, except for the ones that
    // are replaced.
    std::string out;
    std::size_t start = 0;
    while (start <= dsn.size()) {
        std::size_t end = dsn.find(' ', start);
        if (end == std::string::npos) {
            end = dsn.size();
        }
        const std::string token = dsn.substr(start, end - start);
        if (!token.empty()) {
            const bool replaced =
                token.rfind("user=", 0) == 0 ||
                token.rfind("password=", 0) == 0 ||
                token.rfind("dbname=", 0) == 0;
            if (!replaced) {
                if (!out.empty()) {
                    out += " ";
                }
                out += token;
            }
        }
        if (end == dsn.size()) {
            break;
        }
        start = end + 1;
    }
    if (!out.empty()) {
        out += " ";
    }
    out += "user=" + user + " password=" + password + " dbname=" + database;
    return out;
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

    // ---- the name of a person comes back with the match -------------------
    {
        using fd::EnrolmentInfo;
        using fd::FaceEntry;

        FaceDatabase db;

        EnrolmentInfo info;
        info.user_name = "Ada Lovelace";
        info.metadata = "{\"role\":\"admin\"}";
        check(db.enroll(42, makeEmbeddingOnAxis(0), info),
              "enroll accepts the properties of a person");

        const auto hit = db.match(makeEmbeddingOnAxis(0), 0.4f);
        check(hit.matched, "the enrolled person is matched");
        check(hit.person_id == 42, "the person id is reported");
        check(hit.user_name == "Ada Lovelace",
              "the name of the person comes back with the match");

        // A sample that was read from a store keeps its properties.
        FaceEntry entry;
        entry.person_id = 43;
        entry.embedding = makeEmbeddingOnAxis(1);
        entry.info.user_name = "Alan Turing";
        check(db.add(entry), "a sample of a store is added");
        const auto other = db.match(makeEmbeddingOnAxis(1), 0.4f);
        check(other.user_name == "Alan Turing",
              "the name of a reloaded sample comes back as well");
    }

    // ---- the JSON validator -----------------------------------------------
    {
        static const char* const kValid[] = {
            "{}",
            "[]",
            "null",
            "true",
            "false",
            "0",
            "-1",
            "1.5",
            "-2.5e10",
            "\"text\"",
            "{\"a\":1}",
            "[1,2,3]",
            "{\"a\":{\"b\":[1,{\"c\":null}]}}",
            "{\"a\":\"quote \\\" and backslash \\\\\"}",
        };
        for (const char* document : kValid) {
            check(fd::isValidJson(document),
                  (std::string("a valid JSON document is accepted: ") +
                   document).c_str());
        }

        static const char* const kInvalid[] = {
            "",
            " ",
            "{",
            "}",
            "[1,]",
            "{\"a\":1,}",
            "{a:1}",
            "{\"a\"}",
            "{\"a\":}",
            "tru",
            "01",
            "\"unterminated",
            "{'a':1}",
            "{} trailing",
        };
        for (const char* document : kInvalid) {
            check(!fd::isValidJson(document),
                  (std::string("an invalid JSON document is rejected: ") +
                   document).c_str());
        }
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

    // ---- the file store keeps the properties of a person -------------------
    {
        using fd::FaceEntry;
        using fd::FileFaceStore;
        using fd::PersonSummary;

        std::string error;
        const std::string path = "face_store_properties_test.db";
        std::remove(path.c_str());
        {
            FileFaceStore file_store;
            check(file_store.open(path, error), "the file store opens");

            FaceEntry entry;
            entry.person_id = 21;
            entry.embedding = makeEmbeddingOnAxis(0);
            entry.info.user_name = "Ada Lovelace";
            entry.info.metadata = "{\"role\":\"admin\"}";
            entry.info.photo = {0xff, 0xd8, 0xff, 0x00, 0x7f};
            check(file_store.add(entry, error),
                  "the file store adds a sample with properties");

            // A sample without any property is fine as well.
            FaceEntry plain;
            plain.person_id = 22;
            plain.embedding = makeEmbeddingOnAxis(1);
            check(file_store.add(plain, error),
                  "the file store adds a sample without properties");
        }
        {
            FileFaceStore file_store;
            check(file_store.open(path, error), "the file store reopens");
            std::vector<FaceEntry> loaded;
            check(file_store.load(loaded, error), "the file store loads");
            check(loaded.size() == 2, "both samples are in the file");
            for (const FaceEntry& entry : loaded) {
                if (entry.person_id == 21) {
                    check(entry.info.user_name == "Ada Lovelace",
                          "the name of the person survived the round trip");
                    check(withoutSpaces(entry.info.metadata) ==
                              withoutSpaces("{\"role\":\"admin\"}"),
                          "the metadata of the person survived the round trip");
                    check(entry.info.photo.size() == 5 &&
                              entry.info.photo[0] == 0xff &&
                              entry.info.photo[4] == 0x7f,
                          "the picture of the person survived the round trip");
                } else if (entry.person_id == 22) {
                    check(entry.info.user_name.empty(),
                          "a sample without a name stays without one");
                    check(entry.info.metadata.empty(),
                          "a sample without metadata stays without one");
                    check(entry.info.photo.empty(),
                          "a sample without a picture stays without one");
                }
            }

            std::vector<PersonSummary> people;
            check(file_store.list(people, error), "the file store lists");
            check(people.size() == 2, "two people are listed");
            for (const PersonSummary& person : people) {
                if (person.person_id == 21) {
                    check(person.user_name == "Ada Lovelace",
                          "the list carries the name of the person");
                }
            }
        }
        std::remove(path.c_str());
    }

    // ---- the file store reads the format of an earlier version -------------
    {
        using fd::FaceEntry;
        using fd::FileFaceStore;

        std::string error;
        const std::string path = "face_store_legacy_test.db";
        std::remove(path.c_str());
        const fd::Embedding embedding = makeEmbeddingOnAxis(0);
        check(writeLegacyFile(path, 31, embedding),
              "a database of an earlier version is written");

        FileFaceStore file_store;
        check(file_store.open(path, error), "the file store opens");
        std::vector<FaceEntry> loaded;
        check(file_store.load(loaded, error),
              "the file store reads a database of an earlier version");
        check(loaded.size() == 1, "the old sample is read");
        if (loaded.size() == 1) {
            check(loaded[0].person_id == 31,
                  "the old sample kept its person id");
            check(loaded[0].embedding.valid,
                  "the old sample kept its embedding");
            check(loaded[0].info.user_name.empty(),
                  "the old sample has no name");
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

            // The properties of a person travel with the sample.
            FaceEntry named;
            named.person_id = 9003;
            named.embedding = makeEmbeddingOnAxis(2);
            named.info.user_name = "Grace Hopper";
            named.info.metadata = "{\"role\":\"admin\"}";
            named.info.photo = {0xff, 0xd8, 0xff, 0x00, 0x7f};
            check(pg.add(named, error),
                  "the PostgreSQL store adds a sample with properties");

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

            // The properties of the person come back as well.
            bool properties = false;
            for (const FaceEntry& entry : loaded) {
                if (entry.person_id != 9003) {
                    continue;
                }
                properties = entry.info.user_name == "Grace Hopper" &&
                             withoutSpaces(entry.info.metadata) ==
                                 withoutSpaces("{\"role\":\"admin\"}") &&
                             entry.info.photo.size() == 5 &&
                             entry.info.photo[0] == 0xff &&
                             entry.info.photo[4] == 0x7f;
            }
            check(properties,
                  "the name, the picture and the metadata survive the round "
                  "trip");

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
            check(pg.remove(9003, error) == 1,
                  "the PostgreSQL store removes the sample with properties");

            pg.close();
            check(!pg.isOpen(), "the PostgreSQL store reports closed");
        }
    }

    // ---- PostgreSQL store: a table of an earlier version is migrated ----
#ifdef FD_HAVE_POSTGRES
    {
        const char* dsn = std::getenv("FACE_DETECTION_TEST_DSN");
        if (dsn != nullptr && *dsn != '\0' && fd::postgresAvailable()) {
            using fd::FaceEntry;
            using fd::PostgresFaceStore;

            const std::string table = "face_enrollments_legacy_test";
            dropTable(dsn, table);

            // A table of the shape the store used to create: no name, no
            // picture and no metadata.
            runCommand(dsn,
                       "CREATE TABLE \"" + table + "\" ("
                       "id bigserial PRIMARY KEY, "
                       "person_id integer NOT NULL, "
                       "embedding real[] NOT NULL, "
                       "enrolled_at timestamptz NOT NULL DEFAULT now())");

            PostgresFaceStore pg;
            PostgresFaceStore::Options options;
            options.table = table;
            pg.setOptions(options);

            std::string error;
            check(pg.open(dsn, error),
                  ("the store opens on a table of an earlier version: " +
                   error).c_str());

            check(queryScalar(dsn,
                              "SELECT count(*) FROM "
                              "information_schema.columns WHERE "
                              "table_name = $1 AND column_name = 'user_name'",
                              table) == "1",
                  "the name column is added to the old table");
            check(queryScalar(dsn,
                              "SELECT count(*) FROM "
                              "information_schema.columns WHERE "
                              "table_name = $1 AND column_name = 'photo'",
                              table) == "1",
                  "the picture column is added to the old table");
            check(queryScalar(dsn,
                              "SELECT count(*) FROM "
                              "information_schema.columns WHERE "
                              "table_name = $1 AND column_name = 'metadata'",
                              table) == "1",
                  "the metadata column is added to the old table");

            // The migrated table holds the samples with their properties.
            FaceEntry entry;
            entry.person_id = 41;
            entry.embedding = makeEmbeddingOnAxis(0);
            entry.info.user_name = "Alan Turing";
            entry.info.metadata = "{\"role\":\"admin\"}";
            check(pg.add(entry, error),
                  ("the migrated table accepts a sample: " + error).c_str());

            std::vector<FaceEntry> loaded;
            check(pg.load(loaded, error),
                  ("the migrated table loads: " + error).c_str());
            bool found = false;
            for (const FaceEntry& sample : loaded) {
                if (sample.person_id == 41) {
                    found = sample.info.user_name == "Alan Turing";
                }
            }
            check(found, "the migrated table keeps the name of the person");

            dropTable(dsn, table);
        }
    }
#endif

    // ---- PostgreSQL store: a table that may not be altered ---------------
#ifdef FD_HAVE_POSTGRES
    {
        const char* dsn = std::getenv("FACE_DETECTION_TEST_DSN");
        if (dsn != nullptr && *dsn != '\0' && fd::postgresAvailable()) {
            // The scenario needs a second role that may not alter the
            // table, so it only runs when the test connects as a
            // superuser.
            const bool superuser = queryScalarNoParam(
                                       dsn,
                                       "SELECT rolsuper FROM pg_roles "
                                       "WHERE rolname = current_user") == "t";
            if (superuser) {
                using fd::FaceEntry;
                using fd::PostgresFaceStore;

                const std::string role = "fd_test_limited";
                const std::string password = "fd_test_limited_pw";
                const std::string table = "face_enrollments_unalterable_test";
                const std::string database = queryScalarNoParam(
                    dsn, "SELECT current_database()");

                dropTable(dsn, table);
                runCommand(dsn, "DROP ROLE IF EXISTS " + role);
                runCommand(dsn, "CREATE ROLE " + role + " LOGIN PASSWORD '" +
                                    password + "'");

                // A table of the shape the store used to create: no name,
                // no picture and no metadata. It belongs to the test
                // user, so the limited role may not alter it.
                runCommand(dsn,
                           "CREATE TABLE \"" + table + "\" ("
                           "id bigserial PRIMARY KEY, "
                           "person_id integer NOT NULL, "
                           "embedding real[] NOT NULL, "
                           "enrolled_at timestamptz NOT NULL DEFAULT now())");
                runCommand(dsn, "GRANT USAGE ON SCHEMA public TO " + role);
                runCommand(dsn, "GRANT SELECT, INSERT, UPDATE, DELETE ON \"" +
                                    table + "\" TO " + role);
                runCommand(dsn, "GRANT USAGE, SELECT ON SEQUENCE \"" + table +
                                    "_id_seq\" TO " + role);

                const std::string limited =
                    dsnForUser(dsn, role, password, database);

                PostgresFaceStore pg;
                PostgresFaceStore::Options options;
                options.table = table;
                pg.setOptions(options);

                std::string error;

                // The table cannot be altered, so the columns cannot be
                // added -- but the store still opens and is used as the
                // table is.
                check(pg.open(limited, error),
                      ("the store opens on a table it may not alter: " +
                       error).c_str());
                check(pg.isOpen(),
                      "the store on an unalterable table is open");

                const std::vector<std::string> missing =
                    pg.missingPropertyColumns();
                check(missing.size() == 3,
                      "the three properties are reported as missing");
                bool has_name = false;
                for (const std::string& column : missing) {
                    has_name = has_name || (column == "user_name");
                }
                check(has_name, "the missing name column is reported");

                // A sample is stored, without the properties.
                FaceEntry entry;
                entry.person_id = 51;
                entry.embedding = makeEmbeddingOnAxis(0);
                entry.info.user_name = "Alan Turing";
                check(pg.add(entry, error),
                      ("the unalterable table accepts a sample: " + error)
                          .c_str());

                std::vector<FaceEntry> loaded;
                check(pg.load(loaded, error),
                      ("the unalterable table loads: " + error).c_str());
                check(loaded.size() == 1, "the sample is in the table");
                if (!loaded.empty()) {
                    check(loaded[0].person_id == 51,
                          "the sample kept its person id");
                    check(loaded[0].info.user_name.empty(),
                          "the name of the person was not stored");
                }

                // Preparing the schema on purpose still reports the
                // problem, so that a deployment finds out.
                check(!pg.prepareSchema(error),
                      "preparing the schema reports the problem");
                check(!error.empty(),
                      "the problem of the schema comes with a reason");

                pg.close();
                dropTable(dsn, table);
                runCommand(dsn, "DROP ROLE IF EXISTS " + role);
            }
        }
    }
#endif

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
