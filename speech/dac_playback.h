#pragma once

#include <cstddef>
#include <cstdint>

namespace speech {

constexpr uint32_t kSampleRateHz = 24000;

// Single-task playback through J10 pin 9, using PIT1 channel 0 and the 12-bit
// DAC. Overwrites PCM with DAC codes; the caller owns the buffer and must keep
// it alive until return. Yields while playing and fades both ends over 5 ms.
// Returns false on invalid input or playback failure.
bool PlayPcm16(int16_t* pcm, size_t samples);

}  // namespace speech
