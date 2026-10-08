#pragma once

#include <cstddef>
#include <cstdint>

namespace speech {

constexpr uint32_t kSampleRateHz = 24000;

// Single-task playback through J10 pin 9, using PIT1 channel 0 and the 12-bit
// DAC. PCM stays alive until return. Yields while playing and fades both ends
// over 5 ms. Returns false on invalid input, allocation failure, or timeout.
bool PlayPcm16(const int16_t* pcm, size_t samples);

}  // namespace speech
