#pragma once

#include <cstddef>
#include <cstdint>

namespace audio_playback {

constexpr uint32_t kSampleRate = 22050;

// Single producer; PIT1 drains a bounded ring into the 12-bit DAC on
// DAC_OUT/A2. WritePcm copies samples and yields only when the ring is full.
// EndUtterance fades the sentence's edges without waiting, so synthesis can run
// ahead.
bool Begin();
bool WritePcm(const float* pcm, size_t samples);
bool EndUtterance();
// Drain all queued audio and leave the DAC biased at midpoint. Reports invalid
// PCM or a stalled timer; no caller-owned sample buffers remain referenced.
bool Finish();

}  // namespace audio_playback
