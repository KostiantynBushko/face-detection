#include "face-detection/json.hpp"

#include <cstddef>

namespace fd {

namespace {

// A recursive-descent validator for JSON documents.
//
// The library carries no JSON dependency: the metadata is only stored and
// handed back, so a syntax check is all that is needed. The validator
// never builds a value -- it only decides whether the text is a document
// the server would accept.

// How deep a document may nest. A deeper one is rejected instead of
// blowing the stack.
constexpr int kMaxDepth = 64;

class JsonValidator {
public:
    explicit JsonValidator(std::string_view text) : text_(text) {}

    bool run() {
        skipWhitespace();
        if (!parseValue(0)) {
            return false;
        }
        skipWhitespace();
        // Nothing but whitespace may follow the document.
        return pos_ == text_.size();
    }

private:
    void skipWhitespace() {
        while (pos_ < text_.size()) {
            const char c = text_[pos_];
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
                ++pos_;
            } else {
                break;
            }
        }
    }

    [[nodiscard]] bool atEnd() const { return pos_ >= text_.size(); }

    [[nodiscard]] char peek() const { return text_[pos_]; }

    bool parseValue(int depth) {
        if (depth > kMaxDepth || atEnd()) {
            return false;
        }
        switch (peek()) {
            case '{': return parseObject(depth);
            case '[': return parseArray(depth);
            case '"': return parseString();
            case 't': return parseLiteral("true");
            case 'f': return parseLiteral("false");
            case 'n': return parseLiteral("null");
            default: return parseNumber();
        }
    }

    bool parseObject(int depth) {
        ++pos_;  // '{'
        skipWhitespace();
        if (!atEnd() && peek() == '}') {
            ++pos_;
            return true;
        }

        for (;;) {
            skipWhitespace();
            // The key has to be a string.
            if (atEnd() || peek() != '"' || !parseString()) {
                return false;
            }
            skipWhitespace();
            if (atEnd() || peek() != ':') {
                return false;
            }
            ++pos_;
            skipWhitespace();
            if (!parseValue(depth + 1)) {
                return false;
            }

            skipWhitespace();
            if (atEnd()) {
                return false;
            }
            if (peek() == ',') {
                ++pos_;
                continue;
            }
            if (peek() == '}') {
                ++pos_;
                return true;
            }
            return false;
        }
    }

    bool parseArray(int depth) {
        ++pos_;  // '['
        skipWhitespace();
        if (!atEnd() && peek() == ']') {
            ++pos_;
            return true;
        }

        for (;;) {
            skipWhitespace();
            if (!parseValue(depth + 1)) {
                return false;
            }

            skipWhitespace();
            if (atEnd()) {
                return false;
            }
            if (peek() == ',') {
                ++pos_;
                continue;
            }
            if (peek() == ']') {
                ++pos_;
                return true;
            }
            return false;
        }
    }

    bool parseString() {
        ++pos_;  // '"'
        while (!atEnd()) {
            const unsigned char c = static_cast<unsigned char>(text_[pos_]);
            if (c == '"') {
                ++pos_;
                return true;
            }
            if (c == '\\') {
                ++pos_;
                if (atEnd()) {
                    return false;
                }
                switch (text_[pos_]) {
                    case '"':
                    case '\\':
                    case '/':
                    case 'b':
                    case 'f':
                    case 'n':
                    case 'r':
                    case 't':
                        ++pos_;
                        break;
                    case 'u':
                        ++pos_;
                        if (!parseHexQuad()) {
                            return false;
                        }
                        break;
                    default:
                        return false;
                }
                continue;
            }
            if (c < 0x20) {
                // A control character has to be escaped in a JSON string.
                return false;
            }
            ++pos_;
        }
        return false;  // the string was never closed
    }

    bool parseHexQuad() {
        for (int i = 0; i < 4; ++i) {
            if (atEnd()) {
                return false;
            }
            const char c = text_[pos_];
            const bool hex = (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') ||
                             (c >= 'A' && c <= 'F');
            if (!hex) {
                return false;
            }
            ++pos_;
        }
        return true;
    }

    bool parseLiteral(std::string_view word) {
        if (text_.compare(pos_, word.size(), word) != 0) {
            return false;
        }
        pos_ += word.size();
        return true;
    }

    bool parseNumber() {
        const std::size_t start = pos_;

        if (!atEnd() && peek() == '-') {
            ++pos_;
        }
        if (atEnd()) {
            return false;
        }
        if (peek() == '0') {
            // A leading zero is only allowed on its own.
            ++pos_;
        } else if (peek() >= '1' && peek() <= '9') {
            while (!atEnd() && peek() >= '0' && peek() <= '9') {
                ++pos_;
            }
        } else {
            return false;
        }

        if (!atEnd() && peek() == '.') {
            ++pos_;
            if (atEnd() || peek() < '0' || peek() > '9') {
                return false;
            }
            while (!atEnd() && peek() >= '0' && peek() <= '9') {
                ++pos_;
            }
        }

        if (!atEnd() && (peek() == 'e' || peek() == 'E')) {
            ++pos_;
            if (!atEnd() && (peek() == '+' || peek() == '-')) {
                ++pos_;
            }
            if (atEnd() || peek() < '0' || peek() > '9') {
                return false;
            }
            while (!atEnd() && peek() >= '0' && peek() <= '9') {
                ++pos_;
            }
        }

        return pos_ > start;
    }

    std::string_view text_;
    std::size_t pos_ = 0;
};

} // namespace

bool isValidJson(std::string_view text) {
    JsonValidator validator(text);
    return validator.run();
}

} // namespace fd
