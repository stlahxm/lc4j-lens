package io.lc4jlens;

/**
 * Minimal JSON helpers. Not a general parser/writer — lc4j-lens only ever
 * exchanges two shapes ({@code [{id,text,x,y}]} and {@code {"text": "..."}}),
 * so a full JSON library would be dead weight for one field each way.
 */
final class Json {

    private Json() {}

    static String escape(String s) {
        StringBuilder out = new StringBuilder(s.length() + 8);
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            switch (c) {
                case '"' -> out.append("\\\"");
                case '\\' -> out.append("\\\\");
                case '\n' -> out.append("\\n");
                case '\r' -> out.append("\\r");
                case '\t' -> out.append("\\t");
                default -> {
                    if (c < 0x20) out.append(String.format("\\u%04x", (int) c));
                    else out.append(c);
                }
            }
        }
        return out.toString();
    }

    /** Extracts the string value of a top-level "field" from a small flat JSON object. */
    static String extractStringField(String json, String field) {
        String needle = "\"" + field + "\"";
        int keyIdx = json.indexOf(needle);
        if (keyIdx < 0) throw new IllegalArgumentException("Missing field \"" + field + "\" in request body");
        int colon = json.indexOf(':', keyIdx + needle.length());
        int firstQuote = json.indexOf('"', colon + 1);
        if (colon < 0 || firstQuote < 0) throw new IllegalArgumentException("Malformed value for \"" + field + "\"");
        StringBuilder value = new StringBuilder();
        for (int i = firstQuote + 1; i < json.length(); i++) {
            char c = json.charAt(i);
            if (c == '\\' && i + 1 < json.length()) {
                char next = json.charAt(i + 1);
                switch (next) {
                    case 'n' -> value.append('\n');
                    case 't' -> value.append('\t');
                    case '"' -> value.append('"');
                    case '\\' -> value.append('\\');
                    default -> value.append(next);
                }
                i++;
            } else if (c == '"') {
                return value.toString();
            } else {
                value.append(c);
            }
        }
        throw new IllegalArgumentException("Unterminated string value for \"" + field + "\"");
    }
}
