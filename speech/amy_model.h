#pragma once

#include <cstddef>
#include <cstdint>

namespace amy_model {

constexpr int kSampleRate = 22050;
constexpr int kMaxIds = 265;
constexpr int kMaxFrames = 2048;

// Load once at startup. The SDK retains pointers into these model buffers for
// its lifetime, including after Stop(); replacing a loaded model is
// unsupported.
bool Load(const char* front_path, const char* prefix_path,
          const char* tail_path);

// Calls must be serialized by the speech worker. Stop before using the camera's
// TPU interpreter. Start retains two interpreters and a low-power TPU context.
bool Start();
void Stop();

// Emits at most 2560 samples per callback, valid only until the callback
// returns. Returning false from emit aborts synthesis. No full waveform is
// retained. Utterances shorter than 32 predicted frames have their token
// durations proportionally extended to 32 before acoustic inference (minimum
// 0.372 s). Longer utterances retain the model's original durations and
// boundary behavior.
bool Synthesize(const int32_t* ids, int count,
                bool (*emit)(const float*, size_t, void*), void* user);

}  // namespace amy_model
