#pragma once

#include <memory>
#include <string>
#include <vector>

#include "face-detection/face/recognizer.hpp"

namespace fd {

/// One enrolled face sample: a person id and the embedding computed for it.
///
/// Only the embedding is kept -- never the photograph it was computed from --
/// so a face store stays small and does not duplicate biometric source
/// material.
struct FaceEntry {
    int person_id = -1;
    Embedding embedding;
};

/// An enrolled person and how many samples of that person are stored.
struct PersonSummary {
    int person_id = -1;
    int samples = 0;
};

/// A place where enrolled faces are kept.
///
/// The library ships two implementations:
///   * FileFaceStore     -- the flat "FDB1" binary file format
///   * PostgresFaceStore -- a PostgreSQL table (see postgres_store.hpp)
///
/// A store is opened on a *target*, whose meaning depends on the backend:
/// a file path for the file store, a libpq connection string for
/// PostgreSQL. Every method reports what went wrong through an `error`
/// string rather than by throwing.
class FaceStore {
public:
    virtual ~FaceStore() = default;

    /// Opens the store.
    ///
    /// @p error receives a human-readable reason when this returns false.
    virtual bool open(const std::string& target, std::string& error) = 0;

    /// True while the store is usable.
    [[nodiscard]] virtual bool isOpen() const = 0;

    /// Releases the store. Safe to call more than once.
    virtual void close() = 0;

    /// Reads every stored sample into @p out, replacing its contents.
    virtual bool load(std::vector<FaceEntry>& out, std::string& error) = 0;

    /// Replaces the whole content of the store with @p entries.
    virtual bool save(const std::vector<FaceEntry>& entries,
                      std::string& error) = 0;

    /// Stores one extra sample for an already-enrolled person.
    ///
    /// The generic implementation rewrites the whole store (load + save).
    /// Backends that can append a single row cheaply -- PostgreSQL -- override
    /// it, so enrolling a person never rewrites the other people's samples.
    virtual bool add(const FaceEntry& entry, std::string& error);

    /// Removes every sample of @p person_id.
    /// Returns the number of removed samples, or -1 on error.
    virtual int remove(int person_id, std::string& error);

    /// Lists the enrolled people and how many samples each of them has.
    virtual bool list(std::vector<PersonSummary>& people, std::string& error);
};

/// A store backed by the flat binary file format.
///
/// The on-disk layout is deliberately trivial:
///   magic "FDB1" | uint32 count | { int32 person_id, float[128] } * count
class FileFaceStore : public FaceStore {
public:
    bool open(const std::string& path, std::string& error) override;
    [[nodiscard]] bool isOpen() const override;
    void close() override;
    bool load(std::vector<FaceEntry>& out, std::string& error) override;
    bool save(const std::vector<FaceEntry>& entries,
              std::string& error) override;

    /// Appends a sample. A file that does not exist yet is created.
    bool add(const FaceEntry& entry, std::string& error) override;

    /// Removes every sample of @p person_id.
    int remove(int person_id, std::string& error) override;

    /// Lists the enrolled people and how many samples each of them has.
    bool list(std::vector<PersonSummary>& people, std::string& error) override;

    /// The path this store was opened with.
    [[nodiscard]] const std::string& path() const { return path_; }

private:
    /// The outcome of reading the file.
    enum class LoadResult {
        Ok,      ///< The file was read.
        Missing, ///< There is no file yet.
        Invalid, ///< The file exists but is not a face database.
    };

    LoadResult readEntries(std::vector<FaceEntry>& out, std::string& error);

    std::string path_;
};

/// True when @p target names a PostgreSQL database rather than a file.
///
/// A target counts as PostgreSQL when it starts with "postgres://" or
/// "postgresql://".
[[nodiscard]] bool isPostgresTarget(const std::string& target);

/// True when this build of the library can talk to PostgreSQL.
[[nodiscard]] bool postgresAvailable();

/// Creates the store that matches @p target.
///
/// Returns nullptr when the target is empty or when it needs a backend that
/// this build does not have -- a PostgreSQL target in a build without
/// PostgreSQL support, for instance. @p error then explains why.
[[nodiscard]] std::unique_ptr<FaceStore> makeFaceStore(
    const std::string& target, std::string& error);

} // namespace fd
