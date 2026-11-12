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

namespace {

/// The distances from a probe to the samples of one person, smallest first.
std::vector<float> distancesToPerson(const std::vector<FaceEntry>& entries,
                                     int person_id, const Embedding& probe) {
    std::vector<float> distances;
    for (const FaceEntry& entry : entries) {
        if (entry.person_id == person_id) {
            distances.push_back(probe.cosineDistance(entry.embedding));
        }
    }
    return distances;
}

/// The mean of the @p count smallest distances.
///
/// A person with fewer samples than that is scored by all of them. The
/// distances are sorted first: the caller does not have to.
float scoreOf(std::vector<float> distances, int count) {
    if (distances.empty()) {
        return 1.0f;
    }
    std::sort(distances.begin(), distances.end());
    const std::size_t take = std::min(static_cast<std::size_t>(count),
                                      distances.size());
    float sum = 0.0f;
    for (std::size_t i = 0; i < take; ++i) {
        sum += distances[i];
    }
    return sum / static_cast<float>(take);
}

} // namespace

FaceDatabase::PersonMatch FaceDatabase::matchPerson(
    const Embedding& embedding, const PersonMatchOptions& options) const {
    PersonMatch best;
    if (!embedding.valid || entries_.empty()) {
        return best;
    }

    const float limit =
        options.max_distance > 0.0f ? options.max_distance : kDefaultMaxDistance;
    const int count = options.samples > 0 ? options.samples : 1;

    // The people of the database, in the order they were found.
    std::vector<int> ids;
    for (const Entry& entry : entries_) {
        if (std::find(ids.begin(), ids.end(), entry.person_id) == ids.end()) {
            ids.push_back(entry.person_id);
        }
    }

    float best_distance = std::numeric_limits<float>::max();
    float second_distance = std::numeric_limits<float>::max();
    int best_id = -1;
    int second_id = -1;

    for (const int id : ids) {
        const std::vector<float> distances =
            distancesToPerson(entries_, id, embedding);
        const float score = scoreOf(distances, count);
        if (score < best_distance) {
            second_distance = best_distance;
            second_id = best_id;
            best_distance = score;
            best_id = id;
        } else if (score < second_distance) {
            second_distance = score;
            second_id = id;
        }
    }

    // The name belongs to the *person*, not to the sample that happened to
    // be the closest one: the name is looked up among all the samples of
    // the person, with the same rule the stores use when they list the
    // people -- the largest name wins.
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
    best.runner_up_id = second_id;
    best.runner_up = second_distance;
    best.margin = second_distance - best_distance;
    best.user_name = best_name;
    best.matched = best_id >= 0 && best_distance <= limit &&
                   best.margin >= options.min_margin;
    return best;
}

float FaceDatabase::personDistance(int a, int b, int samples) const {
    std::vector<float> distances;
    for (const Entry& left : entries_) {
        if (left.person_id != a) {
            continue;
        }
        for (const Entry& right : entries_) {
            if (right.person_id != b) {
                continue;
            }
            distances.push_back(left.embedding.cosineDistance(right.embedding));
        }
    }
    const int count = samples > 0 ? samples : 1;
    return scoreOf(distances, count);
}

std::vector<FaceDatabase::PersonInfo> FaceDatabase::people() const {
    std::vector<PersonInfo> result;
    for (const Entry& entry : entries_) {
        auto it = std::find_if(result.begin(), result.end(),
                               [&entry](const PersonInfo& info) {
                                   return info.person_id == entry.person_id;
                               });
        if (it == result.end()) {
            PersonInfo info;
            info.person_id = entry.person_id;
            result.push_back(info);
            it = result.end() - 1;
        }
        ++it->samples;
        if (!entry.info.user_name.empty() &&
            (it->user_name.empty() || entry.info.user_name > it->user_name)) {
            it->user_name = entry.info.user_name;
        }
    }
    return result;
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
