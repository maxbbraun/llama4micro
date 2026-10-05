#include "heartnano_convert.h"

#include <stddef.h>
#include <string.h>

namespace {

char rewritten[NANO_LEX_MAX_CHARS + 1];
char normalized[NANO_LEX_MAX_CHARS + 1];
char word[NANO_LEX_MAX_CHARS + 1];
int32_t word_ids[HEARTNANO_MAX_IDS];

// Reject partial conversions so the caller can spell unknown words or split
// text.
int StrictTextToIds(const char* text, int32_t* ids, int capacity) {
  nano_lex_g2p_stats_t local = {};
  int cap = capacity > HEARTNANO_MAX_IDS ? HEARTNANO_MAX_IDS : capacity;
  int result = nano_lex_g2p_text_to_ids(text, ids, cap);
  if (result >= 0) {
    nano_lex_g2p_get_stats(&local);
    if (local.oov_words)
      result = HEARTNANO_E_OOV;
    else if (local.dropped)
      result = HEARTNANO_E_DROPPED;
  }
  if (result < 0 && ids && capacity > 0) ids[0] = 0;
  return result;
}

bool alpha(unsigned char c) {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}

// Convert only an explicit set of typography to ASCII. The dictionary cannot
// pronounce arbitrary Unicode words; reject those instead of deleting letters.
int normalize(const char* text, size_t length) {
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
    } else
      return NANO_LEX_E_BAD_UTF8;
    if (pos + width > length) return NANO_LEX_E_BAD_UTF8;
    for (size_t i = 1; i < width; ++i) {
      const unsigned char next = static_cast<unsigned char>(text[pos + i]);
      if ((next & 0xc0) != 0x80) return NANO_LEX_E_BAD_UTF8;
      cp = (cp << 6) | (next & 0x3f);
    }
    if ((width == 3 && cp < 0x800) || (width == 4 && cp < 0x10000) ||
        (cp >= 0xd800 && cp <= 0xdfff) || cp > 0x10ffff)
      return NANO_LEX_E_BAD_UTF8;
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
      return HEARTNANO_E_OOV;
    }
    // Every mapping is no longer than its UTF-8 input, so this stays bounded.
    normalized[out++] = replacement;
  }
  normalized[out] = '\0';
  return NANO_LEX_OK;
}

bool has_phoneme(const int32_t* ids, int count) {
  // IDs 13..53 and 56..58 are vowels/consonants in the 62-symbol voice.
  // Exclude framing, whitespace, punctuation, and stress-only markers.
  for (int i = 0; i < count; ++i) {
    if ((ids[i] >= 13 && ids[i] <= 53) || (ids[i] >= 56 && ids[i] <= 58))
      return true;
  }
  return false;
}

int bounded_error(int error) {
  if (error == NANO_LEX_E_TEXT_LONG || error == NANO_LEX_E_TOKENS ||
      error == NANO_LEX_E_ARENA)
    return NANO_LEX_E_CAP;
  return error;
}

int fail(int error, int32_t* ids, int capacity) {
  rewritten[0] = '\0';
  if (ids && capacity > 0) ids[0] = 0;
  return bounded_error(error);
}

bool append(char c, size_t* length) {
  if (*length >= NANO_LEX_MAX_CHARS) return false;
  rewritten[(*length)++] = c;
  rewritten[*length] = '\0';
  return true;
}

}  // namespace

extern "C" int heartnano_text_to_ids(const char* text, int32_t* ids,
                                     int capacity) {
  rewritten[0] = '\0';
  if (!text || !ids) return fail(NANO_LEX_E_NULL_ARG, ids, capacity);
  if (capacity < 2) return fail(NANO_LEX_E_BAD_CAP, ids, capacity);
  size_t length = 0;
  while (length <= NANO_LEX_MAX_CHARS && text[length]) ++length;
  if (length > NANO_LEX_MAX_CHARS) return fail(NANO_LEX_E_CAP, ids, capacity);
  const int normalization = normalize(text, length);
  if (normalization < 0) return fail(normalization, ids, capacity);
  text = normalized;
  length = strlen(text);
  const int cap = capacity < HEARTNANO_MAX_IDS ? capacity : HEARTNANO_MAX_IDS;
  int result = StrictTextToIds(text, ids, cap);
  if (result >= 0) {
    if (!has_phoneme(ids, result))
      return fail(NANO_LEX_E_NO_SYMBOLS, ids, capacity);
    return result;
  }
  // An all-unknown phrase may have no symbols, before strict OOV checking.
  if (result != HEARTNANO_E_OOV && result != NANO_LEX_E_NO_SYMBOLS)
    return fail(result, ids, capacity);

  size_t out = 0;
  for (size_t pos = 0; pos < length;) {
    if (!alpha(static_cast<unsigned char>(text[pos]))) {
      // Keep already-supported ASCII text byte-for-byte above. When a quoted
      // token fails lookup (e.g. Hello,"Her), quotes must separate words rather
      // than remain attached to the dictionary key. Do not alter apostrophes.
      const char c = text[pos++];
      if (!append(c == '"' ? ' ' : c, &out))
        return fail(NANO_LEX_E_CAP, ids, capacity);
      continue;
    }
    size_t end = pos + 1;
    while (end < length && (alpha(static_cast<unsigned char>(text[end])) ||
                            (text[end] == '\'' && end + 1 < length &&
                             alpha(static_cast<unsigned char>(text[end + 1])))))
      ++end;
    const size_t n = end - pos;
    memcpy(word, text + pos, n);
    word[n] = '\0';
    const int check = StrictTextToIds(word, word_ids, HEARTNANO_MAX_IDS);
    const bool unknown =
        check == HEARTNANO_E_OOV || check == NANO_LEX_E_NO_SYMBOLS;
    if (check < 0 && !unknown) return fail(check, ids, capacity);
    for (size_t i = pos; i < end; ++i) {
      // Spell the complete unknown word; retain its internal apostrophes.
      if (unknown && i != pos && !append(' ', &out))
        return fail(NANO_LEX_E_CAP, ids, capacity);
      char c = text[i];
      if (unknown && c >= 'a' && c <= 'z') c = static_cast<char>(c - 'a' + 'A');
      if (!append(c, &out)) return fail(NANO_LEX_E_CAP, ids, capacity);
    }
    pos = end;
  }
  result = StrictTextToIds(rewritten, ids, cap);
  if (result < 0) return fail(result, ids, capacity);
  if (!has_phoneme(ids, result))
    return fail(NANO_LEX_E_NO_SYMBOLS, ids, capacity);
  return result;
}
