#include "face-detection/face/database.hpp"

#include <algorithm>
#include <limits>
#include <utility>

namespace fd {

bool FaceDatabase::enroll(int person_id, const Image& image, const FaceBox& box,
                          const FaceRecognizer& recognizer,
                          const EnrolmentInfo& info, Entry* out) {
    Embedding embedding;
    if (!recognizer.embed(image, box, embedding)) {
        return false;
    }
    return enroll(person_id, embedding, info, out);
}

bool FaceDatabase::enroll(int person_id, const Embedding& embedding,
                          const EnrolmentInfo& info, Entry* out) {
    if (!embedding.valid) {
        return false;
    }
    Entry entry;
    entry.person_id = person_id;
    entry.embedding = embedding;
    entry.info = info;
    entries_.push_back(entry);
    if (out != nullptr) {
        *out = std::move(entry);
    }
    return true;
}

bool FaceDatabase::add(const FaceEntry& entry) {
    if (!entry.embedding.valid) {
        return false;
    }
    entries_.push_back(entry);
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

    // The name belongs to the *person*, not to the sample that happened to
    // be the closest one: a person can have several samples, and only
    // some of them carry the name. The name is therefore looked up among
    // all the samples of the matched person, with the same rule the
    // stores use when they list the people -- the largest name wins.
    std::string best_name;
    if (best_id >= 0) {
        for (const Entry& entry : entries_) {
            if (entry.person_id != best_id || entry.info.user_name.empty()) {
                continue;
            }
            if (best_name.empty() || entry.info.user_name > best_name) {
                best_name = entry.info.user_name;
            }
        }
    }

    best.person_id = best_id;
    best.distance = best_distance;
    best.matched = best_id >= 0 && best_distance <= limit;
    best.user_name = best_name;
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

bool FaceDatabase::save(FaceStore& store) const {
    std::string error;
    if (!store.save(entries_, error)) {
        return false;
    }
    return true;
}

bool FaceDatabase::load(FaceStore& store) {
    std::vector<Entry> loaded;
    std::string error;
    if (!store.load(loaded, error)) {
        return false;
    }

    for (const Entry& entry : loaded) {
        entries_.push_back(entry);
    }
    return true;
}

bool FaceDatabase::save(const std::string& path) const {
    FileFaceStore store;
    std::string error;
    if (!store.open(path, error)) {
        return false;
    }
    return save(store);
}

bool FaceDatabase::load(const std::string& path) {
    FileFaceStore store;
    std::string error;
    if (!store.open(path, error)) {
        return false;
    }
    return load(store);
}

} // namespace fd
