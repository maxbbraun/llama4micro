#include "heartnano_convert.h"

#include <stddef.h>

#include <algorithm>

#include "flite_phonemes.h"

namespace {

constexpr int kErrorOov = -101;
constexpr int kErrorDropped = -102;

char normalized[NANO_LEX_MAX_CHARS + 1];

// Reject partial conversions rather than silently omitting words or sounds.
int StrictTextToIds(const char* text, int32_t* ids, int capacity) {
  nano_lex_g2p_stats_t local = {};
  int cap = std::min(capacity, NANO_LEX_MAX_TOKENS);
  int result = nano_lex_g2p_text_to_ids(text, ids, cap);
  if (result >= 0) {
    nano_lex_g2p_get_stats(&local);
    if (local.oov_words) {
      result = kErrorOov;
    } else if (local.dropped) {
      result = kErrorDropped;
    }
  }
  return result;
}

// Convert only an explicit set of typography to ASCII. The frontend cannot
// pronounce arbitrary Unicode words; reject those instead of deleting letters.
int Normalize(const char* text, size_t length) {
  size_t out = 0;
  for (size_t pos = 0; pos < length;) {
    const unsigned char lead = static_cast<unsigned char>(text[pos]);
    if (lead < 0x80) {
      normalized[out++] = text[pos++];
      continue;
    }
    size_t width;
    uint32_t cp;
    if (lead >= 0xc2 && lead <= 0xdf) {
      width = 2;
      cp = lead & 0x1f;
    } else if (lead >= 0xe0 && lead <= 0xef) {
      width = 3;
      cp = lead & 0x0f;
    } else if (lead >= 0xf0 && lead <= 0xf4) {
      width = 4;
      cp = lead & 0x07;
    } else {
      return NANO_LEX_E_BAD_UTF8;
    }
    if (pos + width > length) {
      return NANO_LEX_E_BAD_UTF8;
    }
    for (size_t i = 1; i < width; ++i) {
      const unsigned char next = static_cast<unsigned char>(text[pos + i]);
      if ((next & 0xc0) != 0x80) {
        return NANO_LEX_E_BAD_UTF8;
      }
      cp = (cp << 6) | (next & 0x3f);
    }
    if ((width == 3 && cp < 0x800) || (width == 4 && cp < 0x10000) ||
        (cp >= 0xd800 && cp <= 0xdfff) || cp > 0x10ffff) {
      return NANO_LEX_E_BAD_UTF8;
    }
    pos += width;
    char replacement;
    if (cp == 0x2018 || cp == 0x2019 || cp == 0x201a || cp == 0x201b ||
        cp == 0x02bc) {
      replacement = '\'';  // Keep contractions and possessives intact.
    } else if (cp == 0x00ab || cp == 0x00bb || (cp >= 0x201c && cp <= 0x201f) ||
               cp == 0x2039 || cp == 0x203a) {
      replacement = '"';
    } else if ((cp >= 0x2010 && cp <= 0x2015) || cp == 0x0085 || cp == 0x00a0 ||
               cp == 0x1680 || (cp >= 0x2000 && cp <= 0x200a) || cp == 0x2028 ||
               cp == 0x2029 || cp == 0x202f || cp == 0x205f || cp == 0x3000) {
      // A standalone ASCII dash is itself OOV in this frontend. Treat prose
      // dashes as word separators, including when no spaces surround them.
      replacement = ' ';
    } else if (cp == 0x2026) {
      // Preserve a pause and a word boundary in an attached Unicode ellipsis.
      normalized[out++] = '.';
      replacement = ' ';
    } else if (cp == 0x2212) {
      replacement = '-';  // A mathematical minus is not a prose separator.
    } else {
      return kErrorOov;
    }

    // Every mapping is no longer than its UTF-8 input, so this stays bounded.
    normalized[out++] = replacement;
  }
  normalized[out] = '\0';
  return NANO_LEX_OK;
}

bool HasPhoneme(const int32_t* ids, int count) {
  // IDs 13..53 and 56..58 are vowels/consonants in the 62-symbol voice.
  // Exclude framing, whitespace, punctuation, and stress-only markers.
  return std::any_of(ids, ids + count, [](int32_t id) {
    return (id >= 13 && id <= 53) || (id >= 56 && id <= 58);
  });
}

int BoundedError(int error) {
  if (error == NANO_LEX_E_TEXT_LONG || error == NANO_LEX_E_TOKENS ||
      error == NANO_LEX_E_ARENA) {
    return NANO_LEX_E_CAP;
  }
  return error;
}

int Fail(int error, int32_t* ids, int capacity) {
  if (ids && capacity > 0) {
    ids[0] = 0;
  }
  return BoundedError(error);
}

}  // namespace

extern "C" int heartnano_text_to_ids(const char* text, int32_t* ids,
                                     int capacity) {
  if (!text || !ids) {
    return Fail(NANO_LEX_E_NULL_ARG, ids, capacity);
  }
  if (capacity < 2) {
    return Fail(NANO_LEX_E_BAD_CAP, ids, capacity);
  }
  size_t length = 0;
  while (length <= NANO_LEX_MAX_CHARS && text[length]) {
    ++length;
  }
  if (length > NANO_LEX_MAX_CHARS) {
    return Fail(NANO_LEX_E_CAP, ids, capacity);
  }
  const int normalization = Normalize(text, length);
  if (normalization < 0) {
    return Fail(normalization, ids, capacity);
  }
  nano_lex_g2p_set_fallback(flite_word_to_phonemes);
  int result = StrictTextToIds(normalized, ids, capacity);

  // Retry attached quotes as word separators when lookup failed. Successful
  // conversions stay unchanged, and apostrophes retain their meaning.
  if (result == kErrorOov || result == NANO_LEX_E_NO_SYMBOLS) {
    bool changed = false;
    for (char* c = normalized; *c; ++c) {
      if (*c == '"') {
        *c = ' ';
        changed = true;
      }
    }
    if (changed) {
      result = StrictTextToIds(normalized, ids, capacity);
    }
  }
  if (result < 0) {
    return Fail(result, ids, capacity);
  }
  if (!HasPhoneme(ids, result)) {
    return Fail(NANO_LEX_E_NO_SYMBOLS, ids, capacity);
  }
  return result;
}
