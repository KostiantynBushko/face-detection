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
///       enrolled_at timestamptz NOT NULL DEFAULT now()
///   );
///
/// The embedding is a 128-element array of `real` (single-precision float),
/// which is exactly what SFace produces. The photograph the embedding was
/// computed from is never stored.
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

    /// Creates the table and its index if they are missing.
    ///
    /// Called by open() unless Options::create_schema is false. Exposed on
    /// its own so that a deployment can prepare the schema with a user that
    /// is not allowed to create tables at run time.
    bool createSchema(std::string& error);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace fd
