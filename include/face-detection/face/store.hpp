#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "face-detection/face/recognizer.hpp"

namespace fd {

/// The properties of a person that are not part of the embedding.
///
/// They are stored next to every sample of that person, so that a store
/// can answer "who is this?" without a second lookup.
struct EnrolmentInfo {
    /// The name of the person. It is what an application shows on screen
    /// when the person is detected. May be empty.
    std::string user_name;

    /// A picture of the face, as the bytes of an encoded image (a JPEG,
    /// for instance). It is optional: a sample without one still matches
    /// perfectly well, the picture is only there for a human to look at.
    std::vector<std::uint8_t> photo;

    /// Arbitrary properties of the person, as a JSON document. May be
    /// empty, which means "no metadata".
    std::string metadata;
};

/// One enrolled face sample: a person id, the embedding computed for it
/// and the properties of the person it belongs to.
///
/// The embedding is what the matching runs on. The properties -- the name
/// of the person, a picture of the face and a JSON document with whatever
/// the application wants to remember -- travel with the sample, so that a
/// store hands them back together with the embedding.
struct FaceEntry {
    /// The id of the row that holds this sample, when the store gives the
    /// rows an id (PostgreSQL does).
    ///
    /// It is filled in by load() and is 0 for a sample that was not read
    /// from a store. A store that keeps the id can update a sample in
    /// place, so that the identity of the row and the time the face was
    /// enrolled survive a change of the sample.
    long row_id = 0;

    int person_id = -1;
    Embedding embedding;
    EnrolmentInfo info;
};

/// An enrolled person and how many samples of that person are stored.
struct PersonSummary {
    int person_id = -1;
    int samples = 0;

    /// The name of the person, when the store knows one.
    std::string user_name;
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

    /// Updates the samples of the store.
    ///
    /// The entries are the ones the store holds -- the ones load() returned
    /// -- with the changes that are to be written. A sample that carries
    /// the id of a row is updated in place, so that the row keeps its
    /// identity and the time the face was enrolled. A sample without an id
    /// is stored as a new one.
    ///
    /// The default implementation rewrites the whole store, which gives
    /// the rows new ids. A backend that can update a row in place --
    /// PostgreSQL -- overrides it.
    virtual bool update(const std::vector<FaceEntry>& entries,
                        std::string& error);

    /// Stores one extra sample for an already-enrolled person.
    ///
    /// The generic implementation rewrites the whole store (load + save).
    /// Backends that can append a single row cheaply -- PostgreSQL -- override
    /// it, so enrolling a person never rewrites the other people's samples.
    virtual bool add(const FaceEntry& entry, std::string& error);

    /// Stores several samples at once.
    ///
    /// This is what an application needs when it enrols a person with
    /// several views of the face: the samples of one person belong
    /// together, so they are written in one go.
    ///
    /// The generic implementation loads the store, appends the samples and
    /// saves it again, so the file is rewritten once and not once per
    /// sample. PostgreSQL overrides it and writes the samples in a single
    /// transaction, so that a person is either stored with all of its
    /// samples or not at all.
    ///
    /// An empty list is a no-op and succeeds.
    virtual bool addAll(const std::vector<FaceEntry>& entries,
                        std::string& error);

    /// Removes every sample of @p person_id.
    /// Returns the number of removed samples, or -1 on error.
    virtual int remove(int person_id, std::string& error);

    /// Lists the enrolled people and how many samples each of them has.
    virtual bool list(std::vector<PersonSummary>& people, std::string& error);

    /// The properties of a person that this store cannot keep.
    ///
    /// The names of the missing ones are returned: "user_name", "photo"
    /// and "metadata". A store that keeps everything -- the file store --
    /// returns an empty list.
    ///
    /// A PostgreSQL table that predates the properties and that the
    /// runtime user is not allowed to alter is used as it is, so the
    /// properties are dropped on the way in. An application can ask for
    /// the missing ones and tell the user about it.
    [[nodiscard]] virtual std::vector<std::string>
    missingPropertyColumns() const;

    /// Prepares the storage behind the store.
    ///
    /// The PostgreSQL store creates its table and adds the columns that
    /// were introduced later to a table that is already there; the file
    /// store has nothing to prepare. Returns false and explains itself
    /// when the storage could not be prepared.
    virtual bool prepareSchema(std::string& error);
};

/// A store backed by the flat binary file format.
///
/// The on-disk layout is deliberately trivial:
///   magic "FDB2" | uint32 count | entry * count
///   entry := int32 person_id | float[128] | uint32 name_len | name bytes
///          | uint32 photo_len | photo bytes | uint32 meta_len | meta bytes
///
/// The older "FDB1" format -- the person id and the embedding, nothing
/// else -- is still read, so a database written by an earlier version
/// keeps working. It is never written any more.
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

    /// Appends several samples. The file is rewritten once, not once per
    /// sample. A file that does not exist yet is created.
    bool addAll(const std::vector<FaceEntry>& entries,
                std::string& error) override;

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
