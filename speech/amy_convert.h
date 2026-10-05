#pragma once

#include <stdint.h>

#include "nano_lex_g2p.h"

#define AMY_MAX_IDS 265
#define AMY_E_OOV (-101)
#define AMY_E_DROPPED (-102)

#ifdef __cplusplus
extern "C" {
#endif

// Converts English text to Amy Small Piper phoneme IDs with interleaved blanks.
// - Normalizes supported Unicode typography and spacing.
// - Removes non-spoken double quotation delimiters; preserves apostrophes.
// - Spells unknown ASCII words letter by letter.
// - Rejects invalid UTF-8, unsupported symbols, and punctuation-only text.
//
// Limits: 512 UTF-8 bytes before and after rewriting; at most 265 IDs, all
// below 129 to fit both model vocabularies. capacity must be at least 3.
//
// Returns: an ID count or a negative frontend error. NANO_LEX_E_CAP means a
// text, output, or workspace limit was exceeded; split the text and retry.
// On failure, clears ids[0] if ids is non-null and capacity > 0.
//
// Threading: call from one task at a time; conversion uses shared buffers.
int amy_text_to_ids(const char* text, int32_t* ids, int capacity);

#ifdef __cplusplus
}
#endif
