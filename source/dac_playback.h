#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>

namespace speech {

constexpr uint32_t kSampleRateHz = 24000;

// Single-task playback through J10 pin 9, using PIT1 channel 0 and the 12-bit
// DAC. Overwrites PCM with DAC codes; the caller owns the buffer and must keep
// it alive until return. Yields while playing; normal ends fade over 5 ms.
// Stops on cancellation; returns false if playback did not finish.
bool PlayPcm16(int16_t* pcm, size_t samples,
               const std::atomic<bool>& cancelled);

}  // namespace speech
