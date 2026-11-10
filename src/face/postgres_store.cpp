#include "face-detection/face/postgres_store.hpp"

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
bool execParams(PGconn* conn, const std::string& sql, const char* const* values,
                int count, std::string& error) {
    PGresult* result = PQexecParams(conn, sql.c_str(), count, nullptr, values,
                                     nullptr, nullptr, 0);
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

    if (impl_->options.create_schema && !createSchema(error)) {
        close();
        return false;
    }
    return true;
}

bool PostgresFaceStore::createSchema(std::string& error) {
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
            "enrolled_at timestamptz NOT NULL DEFAULT now())";
        if (!execCommand(impl_->conn, create, error)) {
            return false;
        }
    }

    if (!exists(impl_->index, "i")) {
        const std::string create_index =
            "CREATE INDEX " + quoteIdentifier(impl_->index) + " ON " +
            impl_->quotedTable() + " (person_id)";
        if (!execCommand(impl_->conn, create_index, error)) {
            return false;
        }
    }

    // Check that both are really there. PostgreSQL truncates identifiers
    // that are too long, so a name that was not shortened beforehand can
    // end up as something else entirely -- and the store would then work
    // on a table without its index, without anyone noticing.
    if (!exists(impl_->table, "r")) {
        error = "the table '" + impl_->table + "' does not exist";
        return false;
    }
    if (!exists(impl_->index, "i")) {
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

    const std::string sql =
        "SELECT person_id, embedding FROM " + impl_->quotedTable() +
        " ORDER BY id";
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

    const std::string literal = toArrayLiteral(entry.embedding);
    if (literal.empty()) {
        error = "the embedding holds a value PostgreSQL cannot store";
        return false;
    }

    const std::string person_id = std::to_string(entry.person_id);
    const char* values[2] = {person_id.c_str(), literal.c_str()};
    const std::string sql = "INSERT INTO " + impl_->quotedTable() +
                            " (person_id, embedding) "
                            "VALUES ($1::integer, $2::real[])";
    return execParams(impl_->conn, sql, values, 2, error);
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

    const std::string sql = "INSERT INTO " + table +
                            " (person_id, embedding) "
                            "VALUES ($1::integer, $2::real[])";
    for (const FaceEntry& entry : entries) {
        if (!entry.embedding.valid) {
            error = "the database holds an invalid embedding";
            execCommand(impl_->conn, "ROLLBACK", error);
            return false;
        }
        const std::string literal = toArrayLiteral(entry.embedding);
        if (literal.empty()) {
            error = "an embedding holds a value PostgreSQL cannot store";
            execCommand(impl_->conn, "ROLLBACK", error);
            return false;
        }
        const std::string person_id = std::to_string(entry.person_id);
        const char* values[2] = {person_id.c_str(), literal.c_str()};
        if (!execParams(impl_->conn, sql, values, 2, error)) {
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

bool PostgresFaceStore::list(std::vector<PersonSummary>& people,
                             std::string& error) {
    people.clear();
    if (!isOpen()) {
        error = "the PostgreSQL store is not open";
        return false;
    }

    const std::string sql =
        "SELECT person_id, count(*) FROM " + impl_->quotedTable() +
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

int PostgresFaceStore::remove(int, std::string& error) {
    error = "this build has no PostgreSQL support";
    return -1;
}

bool PostgresFaceStore::list(std::vector<PersonSummary>& people,
                             std::string& error) {
    people.clear();
    error = "this build has no PostgreSQL support";
    return false;
}

#endif // FD_HAVE_POSTGRES

} // namespace fd
