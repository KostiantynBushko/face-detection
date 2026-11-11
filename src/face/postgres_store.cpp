#include "face-detection/face/postgres_store.hpp"

#include "face-detection/json.hpp"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#ifdef FD_HAVE_POSTGRES
#include <libpq-fe.h>
#endif

namespace fd {

#ifdef FD_HAVE_POSTGRES

namespace {

// ---------------------------------------------------------------------------
// Small helpers around libpq
// ---------------------------------------------------------------------------

/// Drops the trailing newline libpq puts at the end of its messages.
std::string trimMessage(const char* message) {
    if (message == nullptr) {
        return {};
    }
    std::string text(message);
    while (!text.empty() &&
           (text.back() == '\n' || text.back() == '\r')) {
        text.pop_back();
    }
    return text;
}

/// The longest identifier PostgreSQL accepts (NAMEDATALEN - 1).
///
/// A longer name is not an error: the server silently truncates it to
/// this length. That is why every identifier this file builds is cut to
/// fit *before* it is used -- a name that only the server shortens no
/// longer matches the name the code believes it created.
constexpr std::size_t kMaxIdentifierLength = 63;

/// True when @p name is a plain SQL identifier we are willing to use.
bool isPlainIdentifier(const std::string& name) {
    if (name.empty() || name.size() > kMaxIdentifierLength) {
        return false;
    }
    if (!(std::isalpha(static_cast<unsigned char>(name[0])) != 0 ||
          name[0] == '_')) {
        return false;
    }
    for (const char c : name) {
        const bool ok = (std::isalnum(static_cast<unsigned char>(c)) != 0) ||
                        c == '_';
        if (!ok) {
            return false;
        }
    }
    return true;
}

/// Quotes an identifier so it can be spliced into a statement.
///
/// The name has already been checked by isPlainIdentifier(), so the only
/// character that needs escaping is the double quote itself. Doing it by
/// hand keeps the code independent of the libpq version: PQescapeIdentifier
/// changed its signature in PostgreSQL 18.
std::string quoteIdentifier(const std::string& name) {
    std::string quoted;
    quoted.reserve(name.size() + 2);
    quoted.push_back('"');
    for (const char c : name) {
        if (c == '"') {
            quoted.push_back('"');
        }
        quoted.push_back(c);
    }
    quoted.push_back('"');
    return quoted;
}

/// Builds the name of the index that covers the person_id column.
///
/// The name has to fit into kMaxIdentifierLength, because a longer one is
/// truncated by the server -- and a truncated name is not necessarily
/// unique. The table part is shortened first, so the index still ends in
/// the recognisable "_person_id_idx".
///
/// The result is never equal to the table name. A table whose own name
/// already ends in the suffix would otherwise collide with its index, and
/// the existence check would then find the table where it looked for the
/// index -- and skip creating the index.
std::string deriveIndexName(const std::string& table) {
    const std::string suffix = "_person_id_idx";

    std::string name;
    if (table.size() + suffix.size() <= kMaxIdentifierLength) {
        name = table + suffix;
    } else {
        name = table.substr(0, kMaxIdentifierLength - suffix.size()) + suffix;
    }

    if (name == table) {
        // The table name already is the index name: drop one character
        // from the table part so the two cannot be confused.
        name = table.substr(0, kMaxIdentifierLength - suffix.size() - 1) +
               suffix;
    }
    return name;
}

/// Runs a statement and reports the server's message on failure.
bool execCommand(PGconn* conn, const std::string& sql, std::string& error) {
    PGresult* result = PQexec(conn, sql.c_str());
    if (result == nullptr) {
        error = trimMessage(PQerrorMessage(conn));
        if (error.empty()) {
            error = "the PostgreSQL connection was lost";
        }
        return false;
    }

    const ExecStatusType status = PQresultStatus(result);
    if (status != PGRES_COMMAND_OK && status != PGRES_TUPLES_OK) {
        error = trimMessage(PQresultErrorMessage(result));
        if (error.empty()) {
            error = trimMessage(PQerrorMessage(conn));
        }
        PQclear(result);
        return false;
    }

    PQclear(result);
    return true;
}

/// Runs a statement with bound parameters.
///
/// @p formats and @p lengths are optional. A parameter in binary format
/// needs its length, because the bytes it holds may contain a zero byte.
bool execParams(PGconn* conn, const std::string& sql, const char* const* values,
                int count, std::string& error, const int* formats = nullptr,
                const int* lengths = nullptr) {
    // The order of the two arrays matters: libpq wants the lengths
    // first and the formats second.
    PGresult* result = PQexecParams(conn, sql.c_str(), count, nullptr, values,
                                     lengths, formats, 0);
    if (result == nullptr) {
        error = trimMessage(PQerrorMessage(conn));
        if (error.empty()) {
            error = "the PostgreSQL connection was lost";
        }
        return false;
    }

    const ExecStatusType status = PQresultStatus(result);
    if (status != PGRES_COMMAND_OK && status != PGRES_TUPLES_OK) {
        error = trimMessage(PQresultErrorMessage(result));
        if (error.empty()) {
            error = trimMessage(PQerrorMessage(conn));
        }
        PQclear(result);
        return false;
    }

    PQclear(result);
    return true;
}

/// Formats an embedding as a PostgreSQL real[] array literal.
///
/// The shortest round-trip representation of each float is used, so the
/// value that comes back out of the database is bit-for-bit the one that
/// went in. Returns an empty string when the embedding holds a value that
/// PostgreSQL cannot represent.
std::string toArrayLiteral(const Embedding& embedding) {
    std::string literal;
    literal.reserve(Embedding::kDimension * 12 + 2);
    literal.push_back('{');

    for (std::size_t i = 0; i < Embedding::kDimension; ++i) {
        const float value = embedding.values[i];
        if (!std::isfinite(value)) {
            return {};
        }
        if (i != 0) {
            literal.push_back(',');
        }
        char buffer[32];
        const auto result =
            std::to_chars(buffer, buffer + sizeof(buffer), value);
        if (result.ec != std::errc()) {
            return {};
        }
        literal.append(buffer, result.ptr);
    }

    literal.push_back('}');
    return literal;
}

/// Parses a PostgreSQL array literal such as "{0.1,-0.2}" into floats.
///
/// Quoted elements and the "NULL" marker are handled, so a malformed or
/// partially NULL array is rejected instead of silently producing garbage.
bool parseArrayLiteral(const std::string& text, std::vector<float>& out) {
    out.clear();

    std::size_t i = 0;
    const auto skipSpace = [&]() {
        while (i < text.size() &&
               std::isspace(static_cast<unsigned char>(text[i])) != 0) {
            ++i;
        }
    };

    skipSpace();
    if (i >= text.size() || text[i] != '{') {
        return false;
    }
    ++i;
    skipSpace();

    // The empty array "{}" is legal but never holds an embedding.
    if (i < text.size() && text[i] == '}') {
        return true;
    }

    while (i < text.size()) {
        std::string element;

        if (text[i] == '"') {
            ++i;
            bool closed = false;
            while (i < text.size()) {
                if (text[i] == '"') {
                    if (i + 1 < text.size() && text[i + 1] == '"') {
                        element.push_back('"');
                        i += 2;
                        continue;
                    }
                    ++i;
                    closed = true;
                    break;
                }
                element.push_back(text[i]);
                ++i;
            }
            if (!closed) {
                return false;
            }
        } else {
            while (i < text.size() && text[i] != ',' && text[i] != '}') {
                element.push_back(text[i]);
                ++i;
            }
        }

        // Trim the whitespace around an unquoted element.
        if (element.empty() || element.front() != '"') {
            const auto begin = element.find_first_not_of(" \t\n\r");
            const auto end = element.find_last_not_of(" \t\n\r");
            element = (begin == std::string::npos)
                          ? std::string()
                          : element.substr(begin, end - begin + 1);
        }

        if (element.empty() || element == "NULL") {
            return false;
        }

        char* end = nullptr;
        const float value = std::strtof(element.c_str(), &end);
        if (end == element.c_str() || !std::isfinite(value)) {
            return false;
        }
        out.push_back(value);

        skipSpace();
        if (i < text.size() && text[i] == ',') {
            ++i;
            continue;
        }
        if (i < text.size() && text[i] == '}') {
            ++i;
            break;
        }
        return false;
    }

    skipSpace();
    return i == text.size();
}

/// The value of one hexadecimal digit, or -1 when @p c is not one.
int hexValue(char c) {
    if (c >= '0' && c <= '9') {
        return c - '0';
    }
    if (c >= 'a' && c <= 'f') {
        return c - 'a' + 10;
    }
    if (c >= 'A' && c <= 'F') {
        return c - 'A' + 10;
    }
    return -1;
}

/// Decodes a hexadecimal string into bytes.
///
/// The photo is read with encode(photo, 'hex'), so the server hands it
/// over as plain hexadecimal no matter what its bytea_output setting
/// says.
bool decodeHex(const std::string& text, std::vector<std::uint8_t>& out) {
    out.clear();
    if (text.size() % 2 != 0) {
        return false;
    }

    out.reserve(text.size() / 2);
    for (std::size_t i = 0; i < text.size(); i += 2) {
        const int high = hexValue(text[i]);
        const int low = hexValue(text[i + 1]);
        if (high < 0 || low < 0) {
            out.clear();
            return false;
        }
        out.push_back(static_cast<std::uint8_t>((high << 4) | low));
    }
    return true;
}

/// The parameters of one INSERT.
///
/// The strings live in the struct, because libpq reads them while the
/// statement is executed.
struct InsertParams {
    std::string person_id;
    std::string embedding;
    std::string user_name;
    std::string metadata;
    std::vector<const char*> values;
    std::vector<int> formats;
    std::vector<int> lengths;
};

/// Builds the INSERT that stores one sample.
///
/// The columns that the table does not have are left out, so that a table
/// that was never migrated still works: the store then simply has no
/// place to keep the properties.
bool buildInsert(const std::string& table, bool has_user_name, bool has_photo,
                 bool has_metadata, const FaceEntry& entry,
                 InsertParams& params, std::string& sql, std::string& error) {
    params.embedding = toArrayLiteral(entry.embedding);
    if (params.embedding.empty()) {
        error = "the embedding holds a value PostgreSQL cannot store";
        return false;
    }

    // The metadata is kept in a jsonb column, so the server checks it.
    // It is checked here as well, because the message of the server is a
    // lot harder to make sense of.
    if (entry.info.metadata.empty()) {
        params.metadata = "{}";
    } else if (!isValidJson(entry.info.metadata)) {
        error = "the metadata of person " + std::to_string(entry.person_id) +
                " is not a valid JSON document";
        return false;
    } else {
        params.metadata = entry.info.metadata;
    }

    params.person_id = std::to_string(entry.person_id);
    params.user_name = entry.info.user_name;

    std::string columns = "person_id, embedding";
    std::string placeholders = "$1::integer, $2::real[]";

    params.values.clear();
    params.formats.clear();
    params.lengths.clear();

    const auto add = [&params](const char* value, int format, int length) {
        params.values.push_back(value);
        params.formats.push_back(format);
        params.lengths.push_back(length);
    };

    add(params.person_id.c_str(), 0, static_cast<int>(params.person_id.size()));
    add(params.embedding.c_str(), 0, static_cast<int>(params.embedding.size()));

    // The placeholder of the next parameter.
    const auto next = [&params]() {
        return "$" + std::to_string(params.values.size() + 1);
    };

    if (has_user_name) {
        columns += ", user_name";
        placeholders += ", " + next() + "::text";
        add(params.user_name.c_str(), 0,
            static_cast<int>(params.user_name.size()));
    }

    if (has_photo) {
        columns += ", photo";
        placeholders += ", " + next() + "::bytea";
        if (entry.info.photo.empty()) {
            // A NULL is a parameter whose value is a null pointer; the
            // format stays binary, the length does not matter then.
            add(nullptr, 1, 0);
        } else {
            add(reinterpret_cast<const char*>(entry.info.photo.data()), 1,
                static_cast<int>(entry.info.photo.size()));
        }
    }

    if (has_metadata) {
        columns += ", metadata";
        placeholders += ", " + next() + "::jsonb";
        add(params.metadata.c_str(), 0,
            static_cast<int>(params.metadata.size()));
    }

    sql = "INSERT INTO " + quoteIdentifier(table) + " (" + columns +
          ") VALUES (" + placeholders + ")";
    return true;
}

/// Finds out which of the optional columns the table has.
bool detectColumns(PGconn* conn, const std::string& table, bool& has_user_name,
                   bool& has_photo, bool& has_metadata, std::string& error) {
    const std::string sql =
        "SELECT a.attname FROM pg_attribute a "
        "JOIN pg_class c ON c.oid = a.attrelid "
        "JOIN pg_namespace n ON n.oid = c.relnamespace "
        "WHERE c.relname = $1 AND n.nspname = current_schema() "
        "AND a.attnum > 0 AND NOT a.attisdropped "
        "AND a.attname IN ('user_name', 'photo', 'metadata')";

    const char* values[1] = {table.c_str()};
    PGresult* result = PQexecParams(conn, sql.c_str(), 1, nullptr, values,
                                     nullptr, nullptr, 0);
    if (result == nullptr) {
        error = trimMessage(PQerrorMessage(conn));
        return false;
    }
    if (PQresultStatus(result) != PGRES_TUPLES_OK) {
        error = trimMessage(PQresultErrorMessage(result));
        PQclear(result);
        return false;
    }

    has_user_name = false;
    has_photo = false;
    has_metadata = false;

    const int rows = PQntuples(result);
    for (int row = 0; row < rows; ++row) {
        const std::string name = PQgetvalue(result, row, 0);
        if (name == "user_name") {
            has_user_name = true;
        } else if (name == "photo") {
            has_photo = true;
        } else if (name == "metadata") {
            has_metadata = true;
        }
    }

    PQclear(result);
    return true;
}

/// The number of rows a command touched.
long affectedRows(PGconn* conn, PGresult* result) {
    const char* tuples = PQcmdTuples(result);
    if (tuples == nullptr || *tuples == '\0') {
        return 0;
    }
    (void)conn;
    return std::strtol(tuples, nullptr, 10);
}

} // namespace

// ---------------------------------------------------------------------------
// PostgresFaceStore::Impl
// ---------------------------------------------------------------------------

struct PostgresFaceStore::Impl {
    PGconn* conn = nullptr;
    Options options;
    std::string table = "face_enrollments";
    std::string index = "face_enrollments_person_id_idx";

    /// Which of the optional columns the table has.
    ///
    /// They were added after the table was introduced, so an installation
    /// that predates them still has the old shape -- and one that was
    /// never migrated (the runtime user is not allowed to alter the
    /// table) keeps it as well. The store works with whatever is there.
    bool has_user_name = true;
    bool has_photo = true;
    bool has_metadata = true;

    /// The qualified name of the table, quoted for use in a statement.
    [[nodiscard]] std::string quotedTable() const {
        return quoteIdentifier(table);
    }
};

PostgresFaceStore::PostgresFaceStore() : impl_(std::make_unique<Impl>()) {}

PostgresFaceStore::~PostgresFaceStore() {
    close();
}

PostgresFaceStore::PostgresFaceStore(PostgresFaceStore&&) noexcept = default;

PostgresFaceStore& PostgresFaceStore::operator=(
    PostgresFaceStore&&) noexcept = default;

void PostgresFaceStore::setOptions(const Options& options) {
    impl_->options = options;
}

const PostgresFaceStore::Options& PostgresFaceStore::options() const {
    return impl_->options;
}

bool PostgresFaceStore::open(const std::string& dsn, std::string& error) {
    close();

    if (!isPlainIdentifier(impl_->options.table)) {
        error = "'" + impl_->options.table +
                "' is not a valid table name (letters, digits and "
                "underscores only)";
        return false;
    }

    impl_->table = impl_->options.table;
    impl_->index = deriveIndexName(impl_->table);

    // An empty connection string makes libpq use the PG* environment
    // variables, which is what a deployment usually wants.
    impl_->conn = PQconnectdb(dsn.c_str());
    if (impl_->conn == nullptr) {
        error = "could not allocate a PostgreSQL connection";
        return false;
    }
    if (PQstatus(impl_->conn) != CONNECTION_OK) {
        error = trimMessage(PQerrorMessage(impl_->conn));
        if (error.empty()) {
            error = "could not connect to PostgreSQL";
        }
        PQfinish(impl_->conn);
        impl_->conn = nullptr;
        return false;
    }

    // The schema is prepared in a tolerant way: a table that is already
    // there and that this process is not allowed to alter is used as it
    // is, without the properties of a person. Only a table that cannot be
    // used at all stops the store from opening.
    if (impl_->options.create_schema && !ensureSchema(error, true)) {
        close();
        return false;
    }

    // The optional columns are not always there. A table that was created
    // before they existed and that this process is not allowed to alter
    // keeps the old shape, and the store has to work with that.
    //
    // A failure here is not fatal: the statements below report a clear
    // error when a column really is missing.
    std::string ignored;
    detectColumns(impl_->conn, impl_->table, impl_->has_user_name,
                  impl_->has_photo, impl_->has_metadata, ignored);
    return true;
}

bool PostgresFaceStore::createSchema(std::string& error) {
    // The strict variant: every problem is reported, so that a deployment
    // that prepares the schema finds out when it did not work.
    return ensureSchema(error, false);
}

bool PostgresFaceStore::ensureSchema(std::string& error, bool tolerant) {
    if (impl_->conn == nullptr) {
        error = "the PostgreSQL store is not open";
        return false;
    }

    // Tells us whether a relation with the given name is already there.
    // The schema is only created when it is missing: CREATE ... IF NOT
    // EXISTS would do the same, but it prints a notice every time it runs,
    // and those notices would end up on the application's stderr.
    //
    // The name is compared against pg_class.relname instead of being looked
    // up with to_regclass(), because to_regclass() truncates a name that is
    // too long -- the same way the server truncated the name when it was
    // created. Comparing the stored name directly matches exactly what is
    // in the database.
    //
    // @p kind is the pg_class.relkind the relation has to have: 'r' for a
    // table, 'i' for an index. Looking at the kind keeps a table from
    // passing for an index of the same name.
    const auto exists = [this](const std::string& name,
                                const char* kind) -> bool {
        const char* values[1] = {name.c_str()};
        const std::string probe =
            std::string("SELECT EXISTS (SELECT 1 FROM pg_class c "
                        "JOIN pg_namespace n ON n.oid = c.relnamespace "
                        "WHERE c.relname = $1 "
                        "AND n.nspname = current_schema() "
                        "AND c.relkind = '") +
            kind + "')";
        PGresult* result = PQexecParams(impl_->conn, probe.c_str(), 1, nullptr,
                                         values, nullptr, nullptr, 0);
        if (result == nullptr) {
            return false;
        }
        if (PQresultStatus(result) != PGRES_TUPLES_OK) {
            PQclear(result);
            return false;
        }
        const bool found = PQntuples(result) > 0 &&
                           std::string(PQgetvalue(result, 0, 0)) == "t";
        PQclear(result);
        return found;
    };

    if (!exists(impl_->table, "r")) {
        const std::string create =
            "CREATE TABLE " + impl_->quotedTable() + " ("
            "id bigserial PRIMARY KEY, "
            "person_id integer NOT NULL, "
            "embedding real[] NOT NULL, "
            "user_name text, "
            "photo bytea, "
            "metadata jsonb NOT NULL DEFAULT '{}'::jsonb, "
            "enrolled_at timestamptz NOT NULL DEFAULT now())";
        if (!execCommand(impl_->conn, create, error)) {
            return false;
        }
    } else {
        // The table is already there, but it may have been created before
        // the properties of a person were added to it. The columns are
        // looked up first, because ALTER TABLE ... ADD COLUMN IF NOT
        // EXISTS prints a notice every time it runs, and those notices
        // would end up on the application's stderr.
        const auto addColumn = [this](const std::string& name,
                                      const std::string& definition,
                                      std::string& reason) -> bool {
            const char* values[2] = {impl_->table.c_str(), name.c_str()};
            const std::string probe =
                "SELECT EXISTS (SELECT 1 FROM pg_attribute a "
                "JOIN pg_class c ON c.oid = a.attrelid "
                "JOIN pg_namespace n ON n.oid = c.relnamespace "
                "WHERE c.relname = $1 AND n.nspname = current_schema() "
                "AND a.attname = $2 AND a.attnum > 0 AND NOT a.attisdropped)";
            PGresult* result =
                PQexecParams(impl_->conn, probe.c_str(), 2, nullptr, values,
                             nullptr, nullptr, 0);
            if (result == nullptr) {
                return false;
            }
            const bool found = PQresultStatus(result) == PGRES_TUPLES_OK &&
                               PQntuples(result) > 0 &&
                               std::string(PQgetvalue(result, 0, 0)) == "t";
            PQclear(result);
            if (found) {
                return true;
            }

            const std::string alter = "ALTER TABLE " + impl_->quotedTable() +
                                      " ADD COLUMN " + name + " " + definition;
            if (!execCommand(impl_->conn, alter, reason)) {
                return false;
            }
            return true;
        };

        // Every column is added on its own, so that a table that has
        // some of them is completed instead of being left alone.
        std::string reason;
        bool complete = true;
        complete = addColumn("user_name", "text", reason) && complete;
        complete = addColumn("photo", "bytea", reason) && complete;
        complete =
            addColumn("metadata", "jsonb NOT NULL DEFAULT '{}'::jsonb",
                      reason) &&
            complete;

        if (!complete && !tolerant) {
            error = "could not add the new columns to the table '" +
                    impl_->table + "': " + reason +
                    " (the database user is probably not allowed to "
                    "alter it)";
            return false;
        }
    }

    if (!exists(impl_->index, "i")) {
        const std::string create_index =
            "CREATE INDEX " + quoteIdentifier(impl_->index) + " ON " +
            impl_->quotedTable() + " (person_id)";
        std::string reason;
        if (!execCommand(impl_->conn, create_index, reason)) {
            if (!tolerant) {
                error = "could not create the index '" + impl_->index +
                        "': " + reason;
                return false;
            }
            // The index only makes the lookups by person id fast. Without
            // it the store still works, so a table that may not be
            // altered is used as it is.
        }
    }

    // Check that the table is really there. PostgreSQL truncates
    // identifiers that are too long, so a name that was not shortened
    // beforehand can end up as something else entirely -- and the store
    // would then work on a table without its index, without anyone
    // noticing.
    if (!exists(impl_->table, "r")) {
        error = "the table '" + impl_->table + "' does not exist";
        return false;
    }
    if (!tolerant && !exists(impl_->index, "i")) {
        error = "the index '" + impl_->index + "' does not exist";
        return false;
    }
    return true;
}

bool PostgresFaceStore::isOpen() const {
    return impl_ && impl_->conn != nullptr;
}

void PostgresFaceStore::close() {
    if (impl_ && impl_->conn != nullptr) {
        PQfinish(impl_->conn);
        impl_->conn = nullptr;
    }
}

bool PostgresFaceStore::load(std::vector<FaceEntry>& out, std::string& error) {
    out.clear();
    if (!isOpen()) {
        error = "the PostgreSQL store is not open";
        return false;
    }

    // The photo is hex-encoded by the server, so that it comes back the
    // same way whatever the bytea_output setting of the server says.
    //
    // The columns that are read are collected in a list, because the
    // index of a column is its position in that list -- and counting the
    // commas of the statement would not do, as one of the expressions
    // holds a comma of its own.
    std::vector<std::string> columns{"person_id", "embedding"};
    int user_name_column = -1;
    int photo_column = -1;
    int metadata_column = -1;
    if (impl_->has_user_name) {
        user_name_column = static_cast<int>(columns.size());
        columns.push_back("user_name");
    }
    if (impl_->has_photo) {
        photo_column = static_cast<int>(columns.size());
        columns.push_back("encode(photo, 'hex')");
    }
    if (impl_->has_metadata) {
        metadata_column = static_cast<int>(columns.size());
        columns.push_back("metadata");
    }

    std::string selection;
    for (std::size_t i = 0; i < columns.size(); ++i) {
        if (i != 0) {
            selection += ", ";
        }
        selection += columns[i];
    }

    const std::string sql = "SELECT " + selection + " FROM " +
                            impl_->quotedTable() + " ORDER BY id";
    PGresult* result = PQexec(impl_->conn, sql.c_str());
    if (result == nullptr) {
        error = trimMessage(PQerrorMessage(impl_->conn));
        return false;
    }
    if (PQresultStatus(result) != PGRES_TUPLES_OK) {
        error = trimMessage(PQresultErrorMessage(result));
        if (error.empty()) {
            error = trimMessage(PQerrorMessage(impl_->conn));
        }
        PQclear(result);
        return false;
    }

    const int rows = PQntuples(result);
    for (int row = 0; row < rows; ++row) {
        if (PQgetisnull(result, row, 0) || PQgetisnull(result, row, 1)) {
            continue;
        }

        const char* person_text = PQgetvalue(result, row, 0);
        const char* embedding_text = PQgetvalue(result, row, 1);

        std::vector<float> values;
        if (!parseArrayLiteral(embedding_text, values) ||
            values.size() != Embedding::kDimension) {
            error = "row " + std::to_string(row) +
                    " does not hold a valid embedding";
            PQclear(result);
            out.clear();
            return false;
        }

        FaceEntry entry;
        entry.person_id = std::atoi(person_text);
        std::copy(values.begin(), values.end(),
                  entry.embedding.values.begin());
        entry.embedding.valid = true;

        if (user_name_column >= 0 &&
            !PQgetisnull(result, row, user_name_column)) {
            entry.info.user_name = PQgetvalue(result, row, user_name_column);
        }
        if (photo_column >= 0 && !PQgetisnull(result, row, photo_column)) {
            const std::string hex = PQgetvalue(result, row, photo_column);
            if (!decodeHex(hex, entry.info.photo)) {
                error = "row " + std::to_string(row) +
                        " holds a photo that is not hexadecimal";
                PQclear(result);
                out.clear();
                return false;
            }
        }
        if (metadata_column >= 0 &&
            !PQgetisnull(result, row, metadata_column)) {
            const std::string text = PQgetvalue(result, row, metadata_column);
            // An empty document is stored as '{}'; it comes back as the
            // empty string, the way it was handed in.
            entry.info.metadata = (text == "{}") ? std::string() : text;
        }

        out.push_back(std::move(entry));
    }

    PQclear(result);
    return true;
}

bool PostgresFaceStore::add(const FaceEntry& entry, std::string& error) {
    if (!isOpen()) {
        error = "the PostgreSQL store is not open";
        return false;
    }
    if (!entry.embedding.valid) {
        error = "the embedding is not valid";
        return false;
    }

    InsertParams params;
    std::string sql;
    if (!buildInsert(impl_->table, impl_->has_user_name, impl_->has_photo,
                     impl_->has_metadata, entry, params, sql, error)) {
        return false;
    }

    return execParams(impl_->conn, sql, params.values.data(),
                      static_cast<int>(params.values.size()), error,
                      params.formats.data(), params.lengths.data());
}

bool PostgresFaceStore::addAll(const std::vector<FaceEntry>& entries,
                                std::string& error) {
    if (!isOpen()) {
        error = "the PostgreSQL store is not open";
        return false;
    }
    if (entries.empty()) {
        return true;
    }

    // The samples of one person land in one transaction: the person is
    // either stored with all of its views or not at all.
    if (!execCommand(impl_->conn, "BEGIN", error)) {
        return false;
    }

    for (const FaceEntry& entry : entries) {
        if (!entry.embedding.valid) {
            error = "the database holds an invalid embedding";
            std::string ignored;
            execCommand(impl_->conn, "ROLLBACK", ignored);
            return false;
        }

        InsertParams params;
        std::string sql;
        if (!buildInsert(impl_->table, impl_->has_user_name,
                         impl_->has_photo, impl_->has_metadata, entry, params,
                         sql, error)) {
            std::string ignored;
            execCommand(impl_->conn, "ROLLBACK", ignored);
            return false;
        }

        if (!execParams(impl_->conn, sql, params.values.data(),
                        static_cast<int>(params.values.size()), error,
                        params.formats.data(), params.lengths.data())) {
            std::string ignored;
            execCommand(impl_->conn, "ROLLBACK", ignored);
            return false;
        }
    }

    return execCommand(impl_->conn, "COMMIT", error);
}

bool PostgresFaceStore::save(const std::vector<FaceEntry>& entries,
                             std::string& error) {
    if (!isOpen()) {
        error = "the PostgreSQL store is not open";
        return false;
    }

    if (!execCommand(impl_->conn, "BEGIN", error)) {
        return false;
    }

    const std::string table = impl_->quotedTable();
    if (!execCommand(impl_->conn, "TRUNCATE TABLE " + table, error)) {
        execCommand(impl_->conn, "ROLLBACK", error);
        return false;
    }

    for (const FaceEntry& entry : entries) {
        if (!entry.embedding.valid) {
            error = "the database holds an invalid embedding";
            execCommand(impl_->conn, "ROLLBACK", error);
            return false;
        }

        InsertParams params;
        std::string sql;
        if (!buildInsert(impl_->table, impl_->has_user_name,
                         impl_->has_photo, impl_->has_metadata, entry, params,
                         sql, error)) {
            execCommand(impl_->conn, "ROLLBACK", error);
            return false;
        }

        if (!execParams(impl_->conn, sql, params.values.data(),
                        static_cast<int>(params.values.size()), error,
                        params.formats.data(), params.lengths.data())) {
            execCommand(impl_->conn, "ROLLBACK", error);
            return false;
        }
    }

    return execCommand(impl_->conn, "COMMIT", error);
}

int PostgresFaceStore::remove(int person_id, std::string& error) {
    if (!isOpen()) {
        error = "the PostgreSQL store is not open";
        return -1;
    }

    const std::string value = std::to_string(person_id);
    const char* values[1] = {value.c_str()};
    const std::string sql =
        "DELETE FROM " + impl_->quotedTable() + " WHERE person_id = $1::integer";

    PGresult* result = PQexecParams(impl_->conn, sql.c_str(), 1, nullptr, values,
                                     nullptr, nullptr, 0);
    if (result == nullptr) {
        error = trimMessage(PQerrorMessage(impl_->conn));
        return -1;
    }
    if (PQresultStatus(result) != PGRES_COMMAND_OK) {
        error = trimMessage(PQresultErrorMessage(result));
        if (error.empty()) {
            error = trimMessage(PQerrorMessage(impl_->conn));
        }
        PQclear(result);
        return -1;
    }

    const long removed = affectedRows(impl_->conn, result);
    PQclear(result);
    return static_cast<int>(removed);
}

std::vector<std::string> PostgresFaceStore::missingPropertyColumns() const {
    std::vector<std::string> missing;
    if (!impl_->has_user_name) {
        missing.push_back("user_name");
    }
    if (!impl_->has_photo) {
        missing.push_back("photo");
    }
    if (!impl_->has_metadata) {
        missing.push_back("metadata");
    }
    return missing;
}

bool PostgresFaceStore::prepareSchema(std::string& error) {
    return createSchema(error);
}

bool PostgresFaceStore::list(std::vector<PersonSummary>& people,
                             std::string& error) {
    people.clear();
    if (!isOpen()) {
        error = "the PostgreSQL store is not open";
        return false;
    }

    // The name is stored with every sample of a person, so any of them
    // will do. max() skips the samples that have no name at all.
    const bool with_name = impl_->has_user_name;
    const std::string sql =
        "SELECT person_id, count(*)" +
        std::string(with_name ? ", max(user_name)" : "") +
        " FROM " + impl_->quotedTable() +
        " GROUP BY person_id ORDER BY person_id";
    PGresult* result = PQexec(impl_->conn, sql.c_str());
    if (result == nullptr) {
        error = trimMessage(PQerrorMessage(impl_->conn));
        return false;
    }
    if (PQresultStatus(result) != PGRES_TUPLES_OK) {
        error = trimMessage(PQresultErrorMessage(result));
        if (error.empty()) {
            error = trimMessage(PQerrorMessage(impl_->conn));
        }
        PQclear(result);
        return false;
    }

    const int rows = PQntuples(result);
    for (int row = 0; row < rows; ++row) {
        if (PQgetisnull(result, row, 0)) {
            continue;
        }
        PersonSummary summary;
        summary.person_id = std::atoi(PQgetvalue(result, row, 0));
        summary.samples = PQgetisnull(result, row, 1)
                              ? 0
                              : std::atoi(PQgetvalue(result, row, 1));
        if (with_name && !PQgetisnull(result, row, 2)) {
            summary.user_name = PQgetvalue(result, row, 2);
        }
        people.push_back(summary);
    }

    PQclear(result);
    return true;
}

#else // !FD_HAVE_POSTGRES

// ---------------------------------------------------------------------------
// Stub build: the API exists, but every call reports that the library was
// built without PostgreSQL support.
// ---------------------------------------------------------------------------

struct PostgresFaceStore::Impl {
    Options options;
};

PostgresFaceStore::PostgresFaceStore() : impl_(std::make_unique<Impl>()) {}

PostgresFaceStore::~PostgresFaceStore() = default;

PostgresFaceStore::PostgresFaceStore(PostgresFaceStore&&) noexcept = default;

PostgresFaceStore& PostgresFaceStore::operator=(
    PostgresFaceStore&&) noexcept = default;

void PostgresFaceStore::setOptions(const Options& options) {
    impl_->options = options;
}

const PostgresFaceStore::Options& PostgresFaceStore::options() const {
    return impl_->options;
}

bool PostgresFaceStore::open(const std::string&, std::string& error) {
    error = "this build has no PostgreSQL support";
    return false;
}

bool PostgresFaceStore::createSchema(std::string& error) {
    error = "this build has no PostgreSQL support";
    return false;
}

bool PostgresFaceStore::isOpen() const {
    return false;
}

void PostgresFaceStore::close() {}

bool PostgresFaceStore::load(std::vector<FaceEntry>& out, std::string& error) {
    out.clear();
    error = "this build has no PostgreSQL support";
    return false;
}

bool PostgresFaceStore::save(const std::vector<FaceEntry>&, std::string& error) {
    error = "this build has no PostgreSQL support";
    return false;
}

bool PostgresFaceStore::add(const FaceEntry&, std::string& error) {
    error = "this build has no PostgreSQL support";
    return false;
}

bool PostgresFaceStore::addAll(const std::vector<FaceEntry>&,
                                std::string& error) {
    error = "this build has no PostgreSQL support";
    return false;
}

int PostgresFaceStore::remove(int, std::string& error) {
    error = "this build has no PostgreSQL support";
    return -1;
}

std::vector<std::string> PostgresFaceStore::missingPropertyColumns() const {
    return {"user_name", "photo", "metadata"};
}

bool PostgresFaceStore::prepareSchema(std::string& error) {
    error = "this build has no PostgreSQL support";
    return false;
}

bool PostgresFaceStore::list(std::vector<PersonSummary>& people,
                             std::string& error) {
    people.clear();
    error = "this build has no PostgreSQL support";
    return false;
}

#endif // FD_HAVE_POSTGRES

} // namespace fd
