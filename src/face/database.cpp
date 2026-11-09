#include "face-detection/face/database.hpp"

#include <algorithm>
#include <limits>
#include <utility>

namespace fd {

bool FaceDatabase::enroll(int person_id, const Image& image, const FaceBox& box,
                          const FaceRecognizer& recognizer) {
    Embedding embedding;
    if (!recognizer.embed(image, box, embedding)) {
        return false;
    }
    return enroll(person_id, embedding);
}

bool FaceDatabase::enroll(int person_id, const Embedding& embedding) {
    if (!embedding.valid) {
        return false;
    }
    Entry entry;
    entry.person_id = person_id;
    entry.embedding = embedding;
    entries_.push_back(std::move(entry));
    return true;
}

FaceDatabase::Match FaceDatabase::match(const Embedding& embedding,
                                        float max_distance) const {
    Match best;
    if (!embedding.valid || entries_.empty()) {
        return best;
    }

    const float limit = max_distance > 0.0f ? max_distance : kDefaultMaxDistance;

    float best_distance = std::numeric_limits<float>::max();
    int best_id = -1;

    for (const Entry& entry : entries_) {
        const float distance = embedding.cosineDistance(entry.embedding);
        if (distance < best_distance) {
            best_distance = distance;
            best_id = entry.person_id;
        }
    }

    best.person_id = best_id;
    best.distance = best_distance;
    best.matched = best_id >= 0 && best_distance <= limit;
    return best;
}

std::size_t FaceDatabase::remove(int person_id) {
    const auto before = entries_.size();
    entries_.erase(
        std::remove_if(entries_.begin(), entries_.end(),
                       [person_id](const Entry& entry) {
                           return entry.person_id == person_id;
                       }),
        entries_.end());
    return before - entries_.size();
}

void FaceDatabase::clear() {
    entries_.clear();
}

} // namespace fd
