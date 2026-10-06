#pragma once

namespace speech {

// Load the voice before starting speech. Buffers stay resident for playback;
// reloading after the worker has started is not supported.
bool LoadModel(const char* front_path, const char* decoder_path);

// Call from one producer task. Queues two sentences ahead of playback, and
// temporarily lowers the producer priority until Flush. Falls back to
// synchronous speech if the worker cannot be created.
bool BeginAsync();
void Append(const char* piece, void* unused);

// Submit the final partial sentence and wait until all audio has played.
// Restores the producer priority and reports any synthesis/playback failure.
bool Flush();

}  // namespace speech
