#pragma once

#include <stdint.h>

#include "nano_lex_g2p.h"

#define HEARTNANO_MAX_IDS 207
#define HEARTNANO_E_OOV (-101)
#define HEARTNANO_E_DROPPED (-102)

#ifdef __cplusplus
extern "C" {
#endif

// Converts text to Heartnano phoneme IDs.
// - Normalizes supported Unicode punctuation and spacing.
// - Spells unknown ASCII words letter by letter.
// - Preserves IDs for supported ASCII text.
// - Rejects invalid UTF-8, unsupported Unicode, and punctuation-only text.
//
// Limits: 512 UTF-8 bytes before and after rewriting; at most 207 output IDs.
// capacity must be at least 2.
//
// Returns: an ID count or a negative frontend error. NANO_LEX_E_CAP means a
// text, output, or workspace limit was exceeded; split the text and retry.
// On failure, clears ids[0] if ids is non-null and capacity > 0.
//
// Threading: call from one task at a time; conversion uses shared buffers.
int heartnano_text_to_ids(const char* text, int32_t* ids, int capacity);

#ifdef __cplusplus
}
#endif
