#include "flite_phonemes.h"

#include <cstring>

#include "nano_lex_g2p.h"
#include "nano_lex_tables.h"

extern "C" {
#include "cst_lts.h"
extern const cst_lts_model cmu_lts_model[153030];
extern const cst_lts_addr cmu_lts_letter_index[27];
extern const char* const cmu_lts_phone_table[76];
}

namespace {

constexpr size_t kMaxWordChars = 64;
constexpr size_t kNodeBytes = 6;
constexpr size_t kMaxDecisionSteps = 256;
constexpr size_t kModelNodes = sizeof(cmu_lts_model) / kNodeBytes;
constexpr size_t kPhoneRows =
    sizeof(cmu_lts_phone_table) / sizeof(cmu_lts_phone_table[0]) - 1;
static_assert(sizeof(cmu_lts_model) % kNodeBytes == 0,
              "Flite decision nodes must contain six bytes");

struct Phone {
  char name[3];
  uint8_t code;
};

// CMU vowels include a trailing stress digit, processed separately below.
constexpr Phone kPhones[] = {
    {"aa", NLG_PH_0251}, {"ae", NLG_PH_00E6}, {"ah", NLG_PH_028C},
    {"ao", NLG_PH_0254}, {"aw", NLG_PH_0057}, {"ax", NLG_PH_0259},
    {"ay", NLG_PH_0049}, {"b", NLG_PH_0062},  {"ch", NLG_PH_02A7},
    {"d", NLG_PH_0064},  {"dh", NLG_PH_00F0}, {"eh", NLG_PH_025B},
    {"er", NLG_PH_025C}, {"ey", NLG_PH_0041}, {"f", NLG_PH_0066},
    {"g", NLG_PH_0261},  {"hh", NLG_PH_0068}, {"ih", NLG_PH_026A},
    {"iy", NLG_PH_0069}, {"jh", NLG_PH_02A4}, {"k", NLG_PH_006B},
    {"l", NLG_PH_006C},  {"m", NLG_PH_006D},  {"n", NLG_PH_006E},
    {"ng", NLG_PH_014B}, {"ow", NLG_PH_004F}, {"oy", NLG_PH_0059},
    {"p", NLG_PH_0070},  {"r", NLG_PH_0279},  {"s", NLG_PH_0073},
    {"sh", NLG_PH_0283}, {"t", NLG_PH_0074},  {"th", NLG_PH_03B8},
    {"uh", NLG_PH_028A}, {"uw", NLG_PH_0075}, {"v", NLG_PH_0076},
    {"w", NLG_PH_0077},  {"y", NLG_PH_006A},  {"z", NLG_PH_007A},
    {"zh", NLG_PH_0292},
};

uint16_t LowerAscii(uint16_t c) {
  return c >= 'A' && c <= 'Z' ? c - 'A' + 'a' : c;
}

bool IsAsciiLetter(uint16_t c) {
  c = LowerAscii(c);
  return c >= 'a' && c <= 'z';
}

bool Append(uint8_t code, uint8_t* phones, size_t capacity, size_t* count) {
  if (*count >= capacity) return false;
  phones[(*count)++] = code;
  return true;
}

// Decode an upstream phone label such as "er0" or "y-er1" directly.
int AppendPhones(const char* label, uint8_t* phones, size_t capacity,
                 size_t* count, bool* had_stress, uint8_t* last) {
  if (std::strcmp(label, "epsilon") == 0) return NANO_LEX_OK;
  while (*label) {
    const size_t part_length = std::strcspn(label, "-");
    if (part_length == 0) return NANO_LEX_E_INTERNAL;
    const char suffix = label[part_length - 1];
    const bool stressed = suffix == '1';
    const bool unstressed = suffix == '0';
    const size_t length = part_length - (stressed || unstressed ? 1 : 0);
    if (length < 1 || length > 2) return NANO_LEX_E_INTERNAL;
    const Phone* match = nullptr;
    for (const Phone& phone : kPhones) {
      if (phone.name[length] == '\0' &&
          std::strncmp(phone.name, label, length) == 0) {
        match = &phone;
        break;
      }
    }
    if (!match) return NANO_LEX_E_INTERNAL;
    uint8_t code = match->code;
    const bool rhotic_vowel = length == 2 && label[0] == 'e' && label[1] == 'r';
    if (unstressed &&
        (rhotic_vowel || (length == 2 && label[0] == 'a' && label[1] == 'h')))
      code = NLG_PH_0259;
    if (stressed) {
      if (!Append(*had_stress ? NLG_PH_02CC : NLG_PH_02C8, phones, capacity,
                  count))
        return NANO_LEX_E_CAP;
      *had_stress = true;
    }
    if (!Append(code, phones, capacity, count)) return NANO_LEX_E_CAP;
    *last = code;
    if (rhotic_vowel) {
      if (!Append(NLG_PH_0279, phones, capacity, count)) return NANO_LEX_E_CAP;
      *last = NLG_PH_0279;
    }
    label += part_length;
    if (*label == '-') ++label;
  }
  return NANO_LEX_OK;
}

int Fail(int error, uint8_t* phones, size_t capacity) {
  if (phones && capacity) phones[0] = 0;
  return error;
}

}  // namespace

extern "C" int flite_word_to_phonemes(const uint16_t* word, size_t length,
                                      uint8_t* phones, size_t capacity) {
  if (!word || !phones) return Fail(NANO_LEX_E_NULL_ARG, phones, capacity);
  if (!capacity) return NANO_LEX_E_BAD_CAP;
  phones[0] = 0;
  if (length < 2 || length > kMaxWordChars) return 0;
  const bool possessive = length > 2 && word[length - 2] == '\'' &&
                          LowerAscii(word[length - 1]) == 's';
  const size_t root_length = length - (possessive ? 2 : 0);
  char padded[kMaxWordChars + 8];
  for (size_t i = 0; i < root_length; ++i) {
    const uint16_t c = LowerAscii(word[i]);
    const bool apostrophe = c == '\'' && i > 0 && i + 1 < root_length &&
                            IsAsciiLetter(word[i - 1]) &&
                            IsAsciiLetter(word[i + 1]);
    if (!IsAsciiLetter(c) && !apostrophe) return 0;
    padded[i + 4] = static_cast<char>(c);
  }

  // Flite examines four characters on either side of the current letter.
  padded[0] = padded[1] = padded[2] = '0';
  padded[3] = '#';
  padded[root_length + 4] = '#';
  padded[root_length + 5] = padded[root_length + 6] = padded[root_length + 7] =
      '0';
  size_t count = 0;
  bool had_stress = false;
  uint8_t last = 0;
  for (size_t i = 0; i < root_length; ++i) {
    // Apostrophes influence neighboring letters but emit no phones themselves.
    if (padded[i + 4] == '\'') continue;
    uint8_t features[9];
    for (size_t j = 0; j < 4; ++j) {
      features[j] = static_cast<uint8_t>(padded[i + j]);
      features[j + 4] = static_cast<uint8_t>(padded[i + 5 + j]);
    }
    features[8] = '0';
    size_t state = cmu_lts_letter_index[padded[i + 4] - 'a'];
    size_t steps = 0;
    const uint8_t* node;
    for (;;) {
      if (state >= kModelNodes || ++steps > kMaxDecisionSteps)
        return Fail(NANO_LEX_E_INTERNAL, phones, capacity);
      node = cmu_lts_model + state * kNodeBytes;
      if (node[0] == 255) break;
      if (node[0] >= sizeof(features))
        return Fail(NANO_LEX_E_INTERNAL, phones, capacity);
      const size_t branch = features[node[0]] == node[1] ? 2 : 4;
      state = node[branch] | (static_cast<size_t>(node[branch + 1]) << 8);
    }
    if (node[1] >= kPhoneRows)
      return Fail(NANO_LEX_E_INTERNAL, phones, capacity);
    const int result = AppendPhones(cmu_lts_phone_table[node[1]], phones,
                                    capacity, &count, &had_stress, &last);
    if (result < 0) return Fail(result, phones, capacity);
  }

  // Possessive endings are /s/, /z/, or /iz/, depending on the final sound.
  if (possessive && last) {
    uint8_t ending = NLG_PH_007A;
    if (last == NLG_PH_0073 || last == NLG_PH_007A || last == NLG_PH_0283 ||
        last == NLG_PH_0292 || last == NLG_PH_02A7 || last == NLG_PH_02A4) {
      if (!Append(NLG_PH_026A, phones, capacity, &count))
        return Fail(NANO_LEX_E_CAP, phones, capacity);
    } else if (last == NLG_PH_0070 || last == NLG_PH_0074 ||
               last == NLG_PH_006B || last == NLG_PH_0066 ||
               last == NLG_PH_03B8) {
      ending = NLG_PH_0073;
    }
    if (!Append(ending, phones, capacity, &count))
      return Fail(NANO_LEX_E_CAP, phones, capacity);
  }
  return static_cast<int>(count);
}
