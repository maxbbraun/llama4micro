#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Predicts an unknown English word's Heartnano phoneme codes using Flite.
// Accepts up to 64 characters: ASCII letters, internal apostrophes, and 's.
// Returns the code count, zero for unsupported words, or a negative frontend
// error. Discard output on failure; at most four codes per character are
// needed.
int flite_word_to_phonemes(const uint16_t* word, size_t length, uint8_t* phones,
                           size_t capacity);

#ifdef __cplusplus
}
#endif
