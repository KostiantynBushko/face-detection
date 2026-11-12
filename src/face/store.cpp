#include "face-detection/face/store.hpp"

#include "face-detection/face/postgres_store.hpp"
#include "face-detection/json.hpp"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <limits>
#include <string>
#include <utility>

namespace fd {

namespace {

// The on-disk format is deliberately trivial:
//   magic "FDB2" | uint32 count | entry * count
//   entry := int32 person_id | float[128] | uint32 name_len | name bytes
//          | uint32 photo_len | photo bytes | uint32 meta_len | meta bytes
constexpr char kDbMagic[4] = {'F', 'D', 'B', '2'};

// The format before the properties were added. It only held the person
// id and the embedding, and is still read so that a database written by
// an earlier version keeps working.
constexpr char kLegacyDbMagic[4] = {'F', 'D', 'B', '1'};

// The largest blob a file may hold. A corrupt length would otherwise
// make the reader allocate gigabytes.
constexpr std::uint32_t kMaxBlobSize = 64u * 1024u * 1024u;

/// Reads a length-prefixed blob.
bool readBlob(std::istream& in, std::string& blob) {
    std::uint32_t size = 0;
    in.read(reinterpret_cast<char*>(&size), sizeof(size));
    if (!in) {
        return false;
    }
    if (size > kMaxBlobSize) {
        return false;
    }
    blob.assign(size, '\0');
    if (size == 0) {
        return true;
    }
    in.read(blob.data(), static_cast<std::streamsize>(size));
    return static_cast<bool>(in);
}

/// Reads a length-prefixed blob of bytes.
bool readBlob(std::istream& in, std::vector<std::uint8_t>& blob) {
    std::string bytes;
    if (!readBlob(in, bytes)) {
        return false;
    }
    blob.assign(bytes.begin(), bytes.end());
    return true;
}

/// Writes a length-prefixed blob.
bool writeBlob(std::ostream& out, const std::string& blob) {
    const auto size = static_cast<std::uint32_t>(blob.size());
    out.write(reinterpret_cast<const char*>(&size), sizeof(size));
    if (size > 0) {
        out.write(blob.data(), static_cast<std::streamsize>(size));
    }
    return static_cast<bool>(out);
}

/// Writes a length-prefixed blob of bytes.
bool writeBlob(std::ostream& out, const std::vector<std::uint8_t>& blob) {
    const auto size = static_cast<std::uint32_t>(blob.size());
    out.write(reinterpret_cast<const char*>(&size), sizeof(size));
    if (size > 0) {
        out.write(reinterpret_cast<const char*>(blob.data()),
                  static_cast<std::streamsize>(size));
    }
    return static_cast<bool>(out);
}

} // namespace

// ---------------------------------------------------------------------------
// FaceStore: the generic implementations
// ---------------------------------------------------------------------------

bool FaceStore::update(const std::vector<FaceEntry>& entries,
                       std::string& error) {
    // A backend that cannot update a row in place rewrites the whole
    // store. The rows then get new ids and a new time of enrolment, which
    // is what the file store does anyway.
    return save(entries, error);
}

bool FaceStore::add(const FaceEntry& entry, std::string& error) {
    std::vector<FaceEntry> entries;
    if (!load(entries, error)) {
        return false;
    }
    entries.push_back(entry);
    return save(entries, error);
}

bool FaceStore::addAll(const std::vector<FaceEntry>& entries,
                        std::string& error) {
    if (entries.empty()) {
        return true;
    }
    // The samples are appended to the ones that are already there, and the
    // whole store is written once: a person that is enrolled with several
    // views does not rewrite the file once per view.
    std::vector<FaceEntry> all;
    if (!load(all, error)) {
        return false;
    }
    all.insert(all.end(), entries.begin(), entries.end());
    return save(all, error);
}

int FaceStore::remove(int person_id, std::string& error) {
    std::vector<FaceEntry> entries;
    if (!load(entries, error)) {
        return -1;
    }

    const auto before = entries.size();
    entries.erase(std::remove_if(entries.begin(), entries.end(),
                                 [person_id](const FaceEntry& entry) {
                                     return entry.person_id == person_id;
                                 }),
                  entries.end());
    const auto removed = before - entries.size();
    if (removed == 0) {
        return 0;
    }
    if (!save(entries, error)) {
        return -1;
    }
    return static_cast<int>(removed);
}

bool FaceStore::list(std::vector<PersonSummary>& people, std::string& error) {
    std::vector<FaceEntry> entries;
    if (!load(entries, error)) {
        return false;
    }

    people.clear();
    for (const FaceEntry& entry : entries) {
        auto it = std::find_if(people.begin(), people.end(),
                               [&entry](const PersonSummary& summary) {
                                   return summary.person_id == entry.person_id;
                               });
        if (it == people.end()) {
            people.push_back(
                PersonSummary{entry.person_id, 1, entry.info.user_name});
        } else {
            ++it->samples;
            if (it->user_name.empty()) {
                it->user_name = entry.info.user_name;
            }
        }
    }
    std::sort(people.begin(), people.end(),
              [](const PersonSummary& a, const PersonSummary& b) {
                  return a.person_id < b.person_id;
              });
    return true;
}

std::vector<std::string> FaceStore::missingPropertyColumns() const {
    // The file store keeps every property of a person, so there is never
    // one that is missing.
    return {};
}

bool FaceStore::prepareSchema(std::string& error) {
    // The file store is a plain file: there is no schema to prepare.
    (void)error;
    return true;
}

// ---------------------------------------------------------------------------
// FileFaceStore
// ---------------------------------------------------------------------------

bool FileFaceStore::open(const std::string& path, std::string& error) {
    if (path.empty()) {
        error = "the face database path is empty";
        return false;
    }
    path_ = path;
    return true;
}

bool FileFaceStore::isOpen() const {
    return !path_.empty();
}

void FileFaceStore::close() {
    path_.clear();
}

FileFaceStore::LoadResult FileFaceStore::readEntries(
    std::vector<FaceEntry>& out, std::string& error) {
    out.clear();

    std::ifstream in(path_, std::ios::binary);
    if (!in) {
        error = "no such face database file";
        return LoadResult::Missing;
    }

    char magic[sizeof(kDbMagic)] = {};
    in.read(magic, sizeof(magic));
    if (!in) {
        error = "truncated face database file";
        return LoadResult::Invalid;
    }
    const bool legacy = std::string(magic, sizeof(magic)) ==
                        std::string(kLegacyDbMagic, sizeof(kLegacyDbMagic));
    if (!legacy &&
        std::string(magic, sizeof(magic)) !=
            std::string(kDbMagic, sizeof(kDbMagic))) {
        error = "not a face database file (bad magic)";
        return LoadResult::Invalid;
    }

    std::uint32_t count = 0;
    in.read(reinterpret_cast<char*>(&count), sizeof(count));
    if (!in) {
        error = "truncated face database file";
        return LoadResult::Invalid;
    }

    for (std::uint32_t i = 0; i < count; ++i) {
        std::int32_t person_id = 0;
        in.read(reinterpret_cast<char*>(&person_id), sizeof(person_id));
        Embedding embedding;
        in.read(reinterpret_cast<char*>(embedding.values.data()),
                static_cast<std::streamsize>(sizeof(float) *
                                             Embedding::kDimension));
        if (!in) {
            error = "truncated face database file";
            out.clear();
            return LoadResult::Invalid;
        }
        embedding.valid = true;

        FaceEntry entry;
        entry.person_id = person_id;
        entry.embedding = embedding;

        if (!legacy) {
            // The properties of the person come after the embedding.
            if (!readBlob(in, entry.info.user_name) ||
                !readBlob(in, entry.info.photo) ||
                !readBlob(in, entry.info.metadata)) {
                error = "truncated face database file";
                out.clear();
                return LoadResult::Invalid;
            }
        }

        out.push_back(std::move(entry));
    }
    return LoadResult::Ok;
}

bool FileFaceStore::load(std::vector<FaceEntry>& out, std::string& error) {
    return readEntries(out, error) == LoadResult::Ok;
}

bool FileFaceStore::add(const FaceEntry& entry, std::string& error) {
    std::vector<FaceEntry> entries;
    const LoadResult result = readEntries(entries, error);
    if (result == LoadResult::Invalid) {
        return false;
    }
    // A missing file is not a problem here: it just means that nobody has
    // been enrolled yet.
    entries.push_back(entry);
    return save(entries, error);
}

bool FileFaceStore::addAll(const std::vector<FaceEntry>& entries,
                            std::string& error) {
    std::vector<FaceEntry> all;
    const LoadResult result = readEntries(all, error);
    if (result == LoadResult::Invalid) {
        return false;
    }
    // A missing file is not a problem here: it just means that nobody has
    // been enrolled yet.
    all.insert(all.end(), entries.begin(), entries.end());
    return save(all, error);
}

int FileFaceStore::remove(int person_id, std::string& error) {
    std::vector<FaceEntry> entries;
    const LoadResult result = readEntries(entries, error);
    if (result == LoadResult::Invalid) {
        return -1;
    }
    if (result == LoadResult::Missing) {
        return 0;
    }

    const auto before = entries.size();
    entries.erase(std::remove_if(entries.begin(), entries.end(),
                                 [person_id](const FaceEntry& entry) {
                                     return entry.person_id == person_id;
                                 }),
                  entries.end());
    const auto removed = before - entries.size();
    if (removed == 0) {
        return 0;
    }
    if (!save(entries, error)) {
        return -1;
    }
    return static_cast<int>(removed);
}

bool FileFaceStore::list(std::vector<PersonSummary>& people,
                         std::string& error) {
    people.clear();

    std::vector<FaceEntry> entries;
    const LoadResult result = readEntries(entries, error);
    if (result == LoadResult::Invalid) {
        return false;
    }
    if (result == LoadResult::Missing) {
        return true;
    }

    for (const FaceEntry& entry : entries) {
        auto it = std::find_if(people.begin(), people.end(),
                               [&entry](const PersonSummary& summary) {
                                   return summary.person_id == entry.person_id;
                               });
        if (it == people.end()) {
            people.push_back(
                PersonSummary{entry.person_id, 1, entry.info.user_name});
        } else {
            ++it->samples;
            if (it->user_name.empty()) {
                it->user_name = entry.info.user_name;
            }
        }
    }
    std::sort(people.begin(), people.end(),
              [](const PersonSummary& a, const PersonSummary& b) {
                  return a.person_id < b.person_id;
              });
    return true;
}

bool FileFaceStore::save(const std::vector<FaceEntry>& entries,
                         std::string& error) {
    std::ofstream out(path_, std::ios::binary | std::ios::trunc);
    if (!out) {
        error = "could not open the face database file for writing";
        return false;
    }

    // The metadata has to be a JSON document: the PostgreSQL store would
    // refuse anything else, and the two stores should not disagree about
    // what they accept.
    for (const FaceEntry& entry : entries) {
        if (!entry.info.metadata.empty() &&
            !isValidJson(entry.info.metadata)) {
            error = "the metadata of person " +
                    std::to_string(entry.person_id) +
                    " is not a valid JSON document";
            return false;
        }
    }

    out.write(kDbMagic, sizeof(kDbMagic));
    const auto count = static_cast<std::uint32_t>(entries.size());
    out.write(reinterpret_cast<const char*>(&count), sizeof(count));
    for (const FaceEntry& entry : entries) {
        const auto person_id = static_cast<std::int32_t>(entry.person_id);
        out.write(reinterpret_cast<const char*>(&person_id), sizeof(person_id));
        out.write(reinterpret_cast<const char*>(entry.embedding.values.data()),
                  static_cast<std::streamsize>(sizeof(float) *
                                               Embedding::kDimension));
        if (!writeBlob(out, entry.info.user_name) ||
            !writeBlob(out, entry.info.photo) ||
            !writeBlob(out, entry.info.metadata)) {
            error = "could not write the face database file";
            return false;
        }
    }
    if (!out.good()) {
        error = "could not write the face database file";
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Store factory
// ---------------------------------------------------------------------------

bool isPostgresTarget(const std::string& target) {
    static const char* const kPrefixes[] = {"postgres://", "postgresql://"};
    for (const char* prefix : kPrefixes) {
        const std::string p(prefix);
        if (target.size() >= p.size() &&
            target.compare(0, p.size(), p) == 0) {
            return true;
        }
    }
    return false;
}

bool postgresAvailable() {
#ifdef FD_HAVE_POSTGRES
    return true;
#else
    return false;
#endif
}

std::unique_ptr<FaceStore> makeFaceStore(const std::string& target,
                                          std::string& error) {
    if (target.empty()) {
        error = "the face database target is empty";
        return nullptr;
    }

    if (isPostgresTarget(target)) {
        if (!postgresAvailable()) {
            error =
                "this build has no PostgreSQL support; rebuild with "
                "FACE_DETECTION_ENABLE_POSTGRESQL=ON";
            return nullptr;
        }
        return std::make_unique<PostgresFaceStore>();
    }

    return std::make_unique<FileFaceStore>();
}

} // namespace fd
