#include "amy_convert.h"

#include <stddef.h>
#include <string.h>

namespace {

char rewritten[NANO_LEX_MAX_CHARS + 1];
char normalized[NANO_LEX_MAX_CHARS + 1];
char word[NANO_LEX_MAX_CHARS + 1];
int32_t word_ids[AMY_MAX_IDS];

// The shared Piper table contains 154 symbols, but Amy Small's duration
// embedding has only 129 rows (the acoustic embedding has 145).
constexpr int kVocabularySize = 129;
char phonemes[4 * AMY_MAX_IDS + 1];
uint32_t symbols[AMY_MAX_IDS];

bool IsVowel(uint32_t c) {
  static constexpr uint32_t vowels[] = {
      'A',   'I',   'O',   'Q',   'W',   'Y',   'a',
      'i',   'u',   0xe6,  0x250, 0x251, 0x252, 0x254,
      0x259, 0x25b, 0x25c, 0x26a, 0x28a, 0x28c, 0x1d7b};
  for (uint32_t vowel : vowels) {
    if (c == vowel) return true;
  }
  return false;
}

bool IsStress(uint32_t c) { return c == 0x2c8 || c == 0x2cc; }

bool LongI(int pos, int start, int end) {
  if (pos + 1 < end && symbols[pos + 1] == 0x259) return false;
  int stress = -1;
  for (int i = start; i < pos; ++i) {
    if (IsStress(symbols[i])) stress = i;
  }
  if (stress >= 0) {
    for (int i = stress + 1; i < pos; ++i) {
      if (IsVowel(symbols[i])) return false;
    }
    return true;
  }
  for (int i = pos + 1; i < end; ++i) {
    if (IsStress(symbols[i])) return true;
  }
  for (int i = start; i < end; ++i) {
    if (i != pos && IsVowel(symbols[i])) return false;
  }
  return true;
}

int SymbolId(uint32_t c) {
  struct Entry {
    uint32_t symbol;
    int id;
  };
  // Reachable English dictionary symbols after upstream Misaki-to-Piper
  // expansion. All are already NFD; no arbitrary Unicode decomposition is used.
  static constexpr Entry table[] = {
      {' ', 3},     {'!', 4},     {'\'', 5},    {'(', 6},     {')', 7},
      {',', 8},     {'-', 9},     {'.', 10},    {':', 11},    {';', 12},
      {'?', 13},    {'a', 14},    {'b', 15},    {'d', 17},    {'e', 18},
      {'f', 19},    {'h', 20},    {'i', 21},    {'j', 22},    {'k', 23},
      {'l', 24},    {'m', 25},    {'n', 26},    {'o', 27},    {'p', 28},
      {'s', 31},    {'t', 32},    {'u', 33},    {'v', 34},    {'w', 35},
      {'z', 38},    {0xe6, 39},   {0xf0, 41},   {0x14b, 44},  {0x250, 50},
      {0x251, 51},  {0x254, 54},  {0x259, 59},  {0x25a, 60},  {0x25b, 61},
      {0x25c, 62},  {0x261, 66},  {0x26a, 74},  {0x279, 88},  {0x27e, 92},
      {0x283, 96},  {0x28a, 100}, {0x28c, 102}, {0x292, 108}, {0x294, 109},
      {0x2c8, 120}, {0x2cc, 121}, {0x2d0, 122}, {0x3b8, 126}, {0x1d7b, 128}};
  for (const Entry& entry : table) {
    if (c == entry.symbol) return entry.id;
  }
  return AMY_E_DROPPED;
}

int AppendSymbol(uint32_t c, int32_t* ids, int capacity, int* count) {
  const int id = SymbolId(c);
  if (id < 0 || id >= kVocabularySize) return AMY_E_DROPPED;
  if (*count + 3 > capacity) return NANO_LEX_E_CAP;
  ids[(*count)++] = id;
  ids[(*count)++] = 0;
  return NANO_LEX_OK;
}

int MapPhonemes(int32_t* ids, int capacity) {
  int length = 0;
  for (size_t pos = 0; phonemes[pos];) {
    if (length == AMY_MAX_IDS) return NANO_LEX_E_CAP;
    uint32_t c = static_cast<unsigned char>(phonemes[pos++]);
    int following = 0;
    if (c >= 0xc2 && c <= 0xdf) {
      c &= 0x1f;
      following = 1;
    } else if (c >= 0xe0 && c <= 0xef) {
      c &= 0x0f;
      following = 2;
    } else if (c >= 0x80) {
      return AMY_E_DROPPED;
    }
    while (following--) {
      const unsigned char next = phonemes[pos++];
      if ((next & 0xc0) != 0x80) return NANO_LEX_E_BAD_UTF8;
      c = (c << 6) | (next & 0x3f);
    }
    symbols[length++] = c;
  }
  int count = 2;
  ids[0] = 1;
  ids[1] = 0;
  // Match piper_g2p.py's token-local longest-match rules and long-i test.
  // Collapse/trim spaces exactly as its final _MULTI_SPACE operation does.
  for (int start = 0; start < length;) {
    if (symbols[start] == ' ') {
      ++start;
      continue;
    }
    int end = start;
    while (end < length && symbols[end] != ' ') ++end;
    if (count > 2) {
      const int rc = AppendSymbol(' ', ids, capacity, &count);
      if (rc < 0) return rc;
    }
    for (int pos = start; pos < end; ++pos) {
      uint32_t first = symbols[pos];
      uint32_t second = 0;
      if ((first == 0x25c || first == 0x259) && pos + 1 < end &&
          symbols[pos + 1] == 0x279 &&
          (pos + 2 == end || !IsVowel(symbols[pos + 2]))) {
        ++pos;
        if (first == 0x259)
          first = 0x25a;
        else
          second = 0x2d0;
      } else {
        switch (first) {
          case 'A':
            first = 'e';
            second = 0x26a;
            break;
          case 'I':
            first = 'a';
            second = 0x26a;
            break;
          case 'O':
            first = 'o';
            second = 0x28a;
            break;
          case 'W':
            first = 'a';
            second = 0x28a;
            break;
          case 'Y':
            first = 0x254;
            second = 0x26a;
            break;
          case 0x2a4:
            first = 'd';
            second = 0x292;
            break;
          case 0x2a7:
            first = 't';
            second = 0x283;
            break;
          case 'T':
            first = 0x27e;
            break;
          case 0x1d4a:
            first = 0x259;
            break;
          case 'u':
          case 0x251:
          case 0x254:
          case 0x25c:
            second = 0x2d0;
            break;
          case 'i':
            if (LongI(pos, start, end)) second = 0x2d0;
            break;
          case 0x2018:
          case 0x2019:
            first = '\'';
            break;
        }
      }
      int rc = AppendSymbol(first, ids, capacity, &count);
      if (rc < 0) return rc;
      if (second) {
        rc = AppendSymbol(second, ids, capacity, &count);
        if (rc < 0) return rc;
      }
    }
    start = end;
  }
  ids[count++] = 2;
  return count;
}

int StrictTextToIds(const char* text, int32_t* ids, int capacity) {
  int result = nano_lex_g2p_text_to_phonemes(text, phonemes, sizeof(phonemes));
  if (result < 0) return result;
  nano_lex_g2p_stats_t stats = {};
  nano_lex_g2p_get_stats(&stats);
  if (stats.oov_words) return AMY_E_OOV;
  return MapPhonemes(ids, capacity);
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
      if (lead == '"') {
        // Quotation delimiters are not spoken; the shared quote ID150 is
        // outside this voice's actual duration/acoustic vocabularies.
        normalized[out++] = ' ';
        ++pos;
        continue;
      }
      if (lead == '$') {
        size_t next = pos + 1;
        while (next < length && (text[next] == ' ' || text[next] == '\t'))
          ++next;
        if (next == length || text[next] < '0' || text[next] > '9')
          return AMY_E_DROPPED;
      }
      if ((lead < 0x20 && lead != '\t' && lead != '\n' && lead != '\r') ||
          lead == 0x7f || lead == '_' || lead == '^' || lead == '`')
        return AMY_E_DROPPED;
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
      replacement = ' ';  // Non-spoken quotation delimiter.
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
      return AMY_E_OOV;
    }
    // Every mapping is no longer than its UTF-8 input, so this stays bounded.
    normalized[out++] = replacement;
  }
  normalized[out] = '\0';
  return NANO_LEX_OK;
}

bool has_phoneme(const int32_t* ids, int count) {
  // Exclude framing, blanks, punctuation, stress, and length markers.
  for (int i = 0; i < count; ++i) {
    if ((ids[i] >= 14 && ids[i] <= 119) || ids[i] == 126 || ids[i] == 128)
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

extern "C" int amy_text_to_ids(const char* text, int32_t* ids, int capacity) {
  rewritten[0] = '\0';
  if (!text || !ids) return fail(NANO_LEX_E_NULL_ARG, ids, capacity);
  if (capacity < 3) return fail(NANO_LEX_E_BAD_CAP, ids, capacity);
  size_t length = 0;
  while (length <= NANO_LEX_MAX_CHARS && text[length]) ++length;
  if (length > NANO_LEX_MAX_CHARS) return fail(NANO_LEX_E_CAP, ids, capacity);
  const int normalization = normalize(text, length);
  if (normalization < 0) return fail(normalization, ids, capacity);
  text = normalized;
  length = strlen(text);
  const int cap = capacity < AMY_MAX_IDS ? capacity : AMY_MAX_IDS;
  int result = StrictTextToIds(text, ids, cap);
  if (result >= 0) {
    if (!has_phoneme(ids, result))
      return fail(NANO_LEX_E_NO_SYMBOLS, ids, capacity);
    return result;
  }
  // An all-unknown phrase may have no symbols, before strict OOV checking.
  if (result != AMY_E_OOV && result != NANO_LEX_E_NO_SYMBOLS)
    return fail(result, ids, capacity);

  size_t out = 0;
  for (size_t pos = 0; pos < length;) {
    if (!alpha(static_cast<unsigned char>(text[pos]))) {
      // Preserve punctuation and whitespace while rewriting unknown words.
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
    const int check = StrictTextToIds(word, word_ids, AMY_MAX_IDS);
    const bool unknown = check == AMY_E_OOV || check == NANO_LEX_E_NO_SYMBOLS;
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
