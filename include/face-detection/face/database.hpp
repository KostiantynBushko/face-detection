#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "face-detection/face/detector.hpp"
#include "face-detection/face/recognizer.hpp"
#include "face-detection/face/store.hpp"
#include "face-detection/image.hpp"

namespace fd {

/// How a person is scored against a probe.
struct PersonMatchOptions {
    /// The largest score that is still a match. A value <= 0 uses
    /// FaceDatabase::kDefaultMaxDistance.
    float max_distance = 0.0f;

    /// How many samples of a person are averaged.
    ///
    /// One makes the score the distance to the closest sample -- the
    /// behaviour of match(). More than one averages the closest samples,
    /// which is steadier. The default of three is a good compromise: it
    /// smooths the noise of one sample without hiding a person behind the
    /// others.
    int samples = 3;

    /// How much better the best person has to be than the second one.
    ///
    /// A value <= 0 accepts any margin. A margin is what keeps a person
    /// with one good sample from stealing a probe that belongs to
    /// somebody who is enrolled with several samples.
    float min_margin = 0.0f;
};

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

    /// The outcome of a lookup against the *people* of the database.
    ///
    /// A person is scored by the agreement of its samples, not by the
    /// closest one of them: the score is the mean distance to the
    /// @c samples samples of the person that are the closest to the probe.
    /// A person with several samples is therefore steadier than a person
    /// with one, which is what a database that is filled by watching
    /// people needs.
    struct PersonMatch {
        /// Id of the best matching person, or -1 when the database is
        /// empty.
        int person_id = -1;

        /// The score of that person. Lower is better.
        float distance = 1.0f;

        /// Id of the second best person, or -1.
        int runner_up_id = -1;

        /// The score of the second best person.
        float runner_up = 1.0f;

        /// How much better the best person is than the second one.
        ///
        /// A small margin means that the probe is not clearly one person
        /// rather than the other, and the caller may want to refuse the
        /// match.
        float margin = 0.0f;

        /// True when the match was accepted: the score is within the
        /// threshold and the margin is large enough.
        bool matched = false;

        /// The name of the matched person, when one of its samples carries
        /// one.
        std::string user_name;
    };

    /// Finds the person that @p embedding belongs to.
    ///
    /// Every person of the database is scored, and the best one is
    /// returned together with the second best one, so that the caller can
    /// tell a clear match from a doubtful one.
    [[nodiscard]] PersonMatch matchPerson(
        const Embedding& embedding,
        const PersonMatchOptions& options = {}) const;

    /// How similar two people are.
    ///
    /// The distance is the mean of the @p samples smallest distances
    /// between the samples of the two people. It is 1.0 when one of them
    /// has no samples at all.
    ///
    /// This is what tells two people that were enrolled twice apart from
    /// two people that merely look alike.
    [[nodiscard]] float personDistance(int a, int b, int samples = 3) const;

    /// The people of the database.
    struct PersonInfo {
        /// The id of the person.
        int person_id = -1;

        /// How many samples the person has.
        std::size_t samples = 0;

        /// The name of the person, when one of its samples carries one.
        std::string user_name;
    };

    /// The people of the database, with the number of their samples.
    [[nodiscard]] std::vector<PersonInfo> people() const;

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
