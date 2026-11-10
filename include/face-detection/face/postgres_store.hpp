#pragma once

#include <memory>
#include <string>
#include <vector>

#include "face-detection/face/store.hpp"

namespace fd {

/// A store backed by a PostgreSQL table.
///
/// The table holds one row per enrolled sample:
///
///   CREATE TABLE face_enrollments (
///       id          bigserial   PRIMARY KEY,
///       person_id   integer     NOT NULL,
///       embedding   real[]      NOT NULL,
///       user_name   text,
///       photo       bytea,
///       metadata    jsonb       NOT NULL DEFAULT '{}'::jsonb,
///       enrolled_at timestamptz NOT NULL DEFAULT now()
///   );
///
/// The embedding is a 128-element array of `real` (single-precision float),
/// which is exactly what SFace produces.
///
/// The three optional columns hold the properties of the person a sample
/// belongs to: the name that is shown on screen when the person is
/// detected, a picture of the face and a JSON document with whatever the
/// application wants to remember. They were added after the table was
/// introduced, so a table that predates them is migrated when the store
/// creates the schema -- and a table that cannot be altered is used as it
/// is, without the properties.
///
/// The connection is made with libpq, so the usual environment variables
/// (PGHOST, PGPORT, PGDATABASE, PGUSER, PGPASSWORD) apply whenever the
/// connection string does not spell them out.
class PostgresFaceStore : public FaceStore {
public:
    struct Options {
        /// Name of the table that holds the samples.
        std::string table = "face_enrollments";

        /// Create the table and its index when they do not exist yet.
        bool create_schema = true;
    };

    PostgresFaceStore();
    ~PostgresFaceStore() override;

    PostgresFaceStore(const PostgresFaceStore&) = delete;
    PostgresFaceStore& operator=(const PostgresFaceStore&) = delete;
    PostgresFaceStore(PostgresFaceStore&&) noexcept;
    PostgresFaceStore& operator=(PostgresFaceStore&&) noexcept;

    /// Sets the options. Has to be called before open().
    void setOptions(const Options& options);

    /// The options in use.
    [[nodiscard]] const Options& options() const;

    /// Connects to @p dsn and, unless disabled, creates the schema.
    ///
    /// @p dsn is a libpq connection string. An empty string makes libpq
    /// fall back to the PG* environment variables.
    bool open(const std::string& dsn, std::string& error) override;

    [[nodiscard]] bool isOpen() const override;
    void close() override;

    bool load(std::vector<FaceEntry>& out, std::string& error) override;
    bool save(const std::vector<FaceEntry>& entries,
              std::string& error) override;

    /// Appends a single sample. Unlike save() this never rewrites the rows
    /// of the other people, so enrolling one person leaves the rest alone.
    bool add(const FaceEntry& entry, std::string& error) override;

    int remove(int person_id, std::string& error) override;
    bool list(std::vector<PersonSummary>& people, std::string& error) override;

    /// The properties of a person that the table cannot keep.
    ///
    /// The names of the missing columns are returned: "user_name",
    /// "photo" and "metadata". A table that predates them and that the
    /// runtime user is not allowed to alter is used as it is, so the
    /// properties are dropped on the way in.
    [[nodiscard]] std::vector<std::string>
    missingPropertyColumns() const override;

    /// Creates the table and its index if they are missing, and adds the
    /// columns that were introduced later to a table that is already
    /// there.
    ///
    /// Called by open() unless Options::create_schema is false. Exposed on
    /// its own so that a deployment can prepare the schema with a user that
    /// is not allowed to create tables at run time.
    ///
    /// Unlike open(), this reports every problem it runs into: a column
    /// that could not be added is an error here, while open() goes on
    /// without the properties of a person.
    bool createSchema(std::string& error);

    /// Prepares the schema and reports every problem it runs into.
    bool prepareSchema(std::string& error) override;

private:
    /// Creates the table, its index and the columns that were introduced
    /// later.
    ///
    /// @p tolerant makes the store work with a table that it is not
    /// allowed to alter: the properties of a person are then not stored.
    /// The table itself has to be there in either case.
    bool ensureSchema(std::string& error, bool tolerant);

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace fd
