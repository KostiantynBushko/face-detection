#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "face-detection/face/detector.hpp"
#include "face-detection/face/recognizer.hpp"
#include "face-detection/face/store.hpp"
#include "face-detection/image.hpp"

namespace fd {

/// A registry of known faces, keyed by an application-defined person id.
///
/// The database holds the embeddings the matching runs on, together with
/// the properties of the person each sample belongs to -- a name, a
/// picture and a JSON document. The picture is optional, so a database
/// that does not need one does not have to store any.
///
/// The database itself is in-memory: it holds the samples that the matching
/// runs against. Where those samples are *kept* is a separate concern,
/// handled by a FaceStore (a file, a PostgreSQL table, ...).
class FaceDatabase {
public:
    /// The outcome of a lookup.
    struct Match {
        /// Id of the best matching person, or -1 when nothing matched.
        int person_id = -1;

        /// Cosine distance to the matched entry. Lower is better.
        float distance = 1.0f;

        /// True when a match was found within the accepted threshold.
        bool matched = false;

        /// The name of the matched person, when one of its samples carries
        /// one. It is what an application shows on screen.
        ///
        /// The name belongs to the person, so it is looked up among all of
        /// its samples: the closest sample is not necessarily the one that
        /// carries the name.
        std::string user_name;
    };

    /// An enrolled face.
    using Entry = FaceEntry;

    /// Typical distance thresholds for SFace cosine distance:
    /// ~0.4 is a reasonable "same person" cut-off; below ~0.3 is a very
    /// strong match. Tune against your own camera and lighting.
    static constexpr float kDefaultMaxDistance = 0.4f;

    /// Adds one face sample for @p person_id.
    ///
    /// @p box must come from a FaceDetector run on the same @p image.
    /// Returns false if the embedding could not be computed.
    ///
    /// @p info carries the properties of the person -- the name, the
    /// picture and the metadata. They are stored with the sample.
    ///
    /// When @p out is given, it receives the stored entry, so that the
    /// caller can persist exactly what was added.
    bool enroll(int person_id, const Image& image, const FaceBox& box,
                const FaceRecognizer& recognizer,
                const EnrolmentInfo& info = {}, Entry* out = nullptr);

    /// Adds an already-computed embedding.
    bool enroll(int person_id, const Embedding& embedding,
                const EnrolmentInfo& info = {}, Entry* out = nullptr);

    /// Adds a sample that was read from a store.
    ///
    /// The entry keeps its embedding and its properties, so a name that
    /// was stored with the sample is still there after a reload.
    /// Returns false when the entry holds an invalid embedding.
    bool add(const FaceEntry& entry);

    /// Finds the closest entry for @p embedding.
    ///
    /// @p max_distance is the largest cosine distance still considered a
    /// match. A value <= 0 uses kDefaultMaxDistance.
    [[nodiscard]] Match match(const Embedding& embedding,
                              float max_distance = 0.0f) const;

    /// Removes every sample belonging to @p person_id.
    /// Returns the number of removed entries.
    std::size_t remove(int person_id);

    /// Removes every sample.
    void clear();

    [[nodiscard]] std::size_t size() const { return entries_.size(); }
    [[nodiscard]] bool empty() const { return entries_.empty(); }

    /// All enrolled entries, in insertion order.
    [[nodiscard]] const std::vector<Entry>& entries() const { return entries_; }

    /// Writes the database to @p store.
    ///
    /// The store is expected to be open. Returns false if the store could
    /// not be written.
    bool save(FaceStore& store) const;

    /// Reads the samples held by @p store and appends them to the database.
    ///
    /// The store is expected to be open. Returns false if the store could
    /// not be read; the database is left untouched in that case.
    bool load(FaceStore& store);

    /// Persists the database to @p path.
    ///
    /// The on-disk format is deliberately trivial:
    ///   magic "FDB2" | uint32 count | entry * count
    ///   entry := int32 person_id | float[128] | uint32 name_len | name bytes
    ///          | uint32 photo_len | photo bytes | uint32 meta_len | meta bytes
    /// The older "FDB1" format is still read.
    ///
    /// Returns false if the file could not be written.
    bool save(const std::string& path) const;

    /// Loads the entries stored in @p path and appends them to the database.
    ///
    /// Returns false if the file does not exist or is not a valid database.
    /// The database is left untouched in that case.
    bool load(const std::string& path);

private:
    std::vector<Entry> entries_;
};

} // namespace fd
