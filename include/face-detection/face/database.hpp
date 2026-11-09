#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "face-detection/face/detector.hpp"
#include "face-detection/face/recognizer.hpp"
#include "face-detection/image.hpp"

namespace fd {

/// A registry of known faces, keyed by an application-defined person id.
///
/// The database stores embeddings only — never photographs — so it stays
/// small and does not duplicate biometric source material.
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
    };

    /// An enrolled face.
    struct Entry {
        int person_id = -1;
        Embedding embedding;
    };

    /// Typical distance thresholds for SFace cosine distance:
    /// ~0.4 is a reasonable "same person" cut-off; below ~0.3 is a very
    /// strong match. Tune against your own camera and lighting.
    static constexpr float kDefaultMaxDistance = 0.4f;

    /// Adds one face sample for @p person_id.
    ///
    /// @p box must come from a FaceDetector run on the same @p image.
    /// Returns false if the embedding could not be computed.
    bool enroll(int person_id, const Image& image, const FaceBox& box,
                const FaceRecognizer& recognizer);

    /// Adds an already-computed embedding.
    bool enroll(int person_id, const Embedding& embedding);

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

    /// Persists the database to @p path.
    ///
    /// The on-disk format is deliberately trivial:
    ///   magic "FDB1" | uint32 count | { int32 person_id, float[128] } * count
    /// Only embeddings are stored, never photographs.
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
