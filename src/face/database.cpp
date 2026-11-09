#include "face-detection/face/database.hpp"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <limits>
#include <utility>

namespace fd {

namespace {

// The on-disk format is deliberately trivial:
//   magic "FDB1" | uint32 count | { int32 person_id, float[128] } * count
constexpr char kDbMagic[4] = {'F', 'D', 'B', '1'};

} // namespace

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

bool FaceDatabase::save(const std::string& path) const {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        return false;
    }

    out.write(kDbMagic, sizeof(kDbMagic));
    const auto count = static_cast<std::uint32_t>(entries_.size());
    out.write(reinterpret_cast<const char*>(&count), sizeof(count));
    for (const Entry& entry : entries_) {
        const auto person_id = static_cast<std::int32_t>(entry.person_id);
        out.write(reinterpret_cast<const char*>(&person_id), sizeof(person_id));
        out.write(reinterpret_cast<const char*>(entry.embedding.values.data()),
                  static_cast<std::streamsize>(sizeof(float) *
                                               Embedding::kDimension));
    }
    return out.good();
}

bool FaceDatabase::load(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return false;
    }

    char magic[sizeof(kDbMagic)] = {};
    in.read(magic, sizeof(magic));
    if (!in ||
        std::string(magic, sizeof(magic)) != std::string(kDbMagic, sizeof(kDbMagic))) {
        return false;
    }

    std::uint32_t count = 0;
    in.read(reinterpret_cast<char*>(&count), sizeof(count));
    if (!in) {
        return false;
    }

    for (std::uint32_t i = 0; i < count; ++i) {
        std::int32_t person_id = 0;
        in.read(reinterpret_cast<char*>(&person_id), sizeof(person_id));
        Embedding embedding;
        in.read(reinterpret_cast<char*>(embedding.values.data()),
                static_cast<std::streamsize>(sizeof(float) *
                                             Embedding::kDimension));
        if (!in) {
            return false;
        }
        embedding.valid = true;
        enroll(person_id, embedding);
    }
    return true;
}

} // namespace fd
