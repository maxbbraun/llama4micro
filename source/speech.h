#pragma once

namespace speech {

// Load the voice and create the playback worker before starting speech.
// Returns false if either initialization step fails. Buffers stay resident;
// reloading after successful initialization is not supported.
bool LoadModel(const char* front_path, const char* decoder_path);

// Begin a story after successful LoadModel. Call BeginAsync, Append, and Flush
// from one producer task, with each BeginAsync paired with a Flush. Queues two
// chunks ahead of playback and lowers the producer priority until Flush.
void BeginAsync();

// Request cancellation from another task. Flush must still finish the batch.
void Cancel();

// Whether this story was cancelled; reset by the next BeginAsync.
bool Cancelled();

// Append a tokenizer piece, submitting complete sentences or bounded chunks.
// Copies text before returning; may block while speech catches up.
void Append(const char* piece);

// Submit the final partial sentence and wait for playback or cancellation
// cleanup. Restores the producer priority and reports any synthesis/playback
// failure.
bool Flush();

}  // namespace speech
