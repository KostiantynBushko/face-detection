#pragma once

#include <string_view>

namespace fd {

/// True when @p text is a well-formed JSON document.
///
/// The check is syntactic and follows RFC 8259: it accepts the six value
/// types, rejects trailing commas and unquoted keys, and understands the
/// escape sequences of a JSON string.
///
/// It is used to give a clear error when metadata is set from the command
/// line, and to keep the file store and the PostgreSQL store from
/// accepting different documents -- the server would refuse an invalid
/// JSON document, so the file store refuses it as well.
[[nodiscard]] bool isValidJson(std::string_view text);

} // namespace fd
