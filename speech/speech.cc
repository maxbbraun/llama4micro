#include "speech.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "dac_playback.h"
#include "heartnano_convert.h"
#include "libs/base/filesystem.h"
#include "libs/base/tasks.h"
#include "libs/base/timer.h"
#include "nano_q8_meta.h"
#include "snt_nano.h"
#include "snt_port.h"
#include "third_party/freertos_kernel/include/FreeRTOS.h"
#include "third_party/freertos_kernel/include/queue.h"
#include "third_party/freertos_kernel/include/semphr.h"
#include "third_party/freertos_kernel/include/task.h"

extern "C" void snt_par_run(snt_par_fn fn, int n, void* ctx) { fn(0, n, ctx); }
extern "C" int snt_scratch_id() { return 0; }
extern "C" int64_t snt_now_us() { return coralmicro::TimerMicros(); }

namespace {

constexpr size_t kArenaBytes = 1024 * 1024;
constexpr uint32_t kSampleRateHz = audio_playback::kSampleRateHz;
constexpr size_t kMaxAudioDurationSeconds = 24;
constexpr size_t kMaxSamples = kMaxAudioDurationSeconds * kSampleRateHz;
constexpr size_t kChunkBytes = 120;
constexpr size_t kMaxChunkBytes = 480;
constexpr UBaseType_t kQueueDepth = 2;

// The recursive fallback retains a phoneme array at each depth.
constexpr size_t kWorkerStackWords = 4096;
constexpr UBaseType_t kWorkerPriority = coralmicro::kAppTaskPriority;
static_assert(kWorkerPriority > tskIDLE_PRIORITY,
              "The producer needs a priority below the TTS worker");

struct WorkItem {
  bool barrier;
  char text[kMaxChunkBytes + 1];
};

StaticQueue_t queue_storage;
uint8_t queue_items[kQueueDepth * sizeof(WorkItem)];
StaticSemaphore_t completion_storage;
QueueHandle_t work_queue = nullptr;
SemaphoreHandle_t completion = nullptr;
TaskHandle_t worker_task = nullptr;
TaskHandle_t producer_task = nullptr;
UBaseType_t producer_priority = 0;
bool async_active = false;  // Producer task only.
bool producer_ok = true;    // Producer task only.

// Published by the worker before giving the completion semaphore.
volatile bool completed_ok = true;
std::string pending;
std::vector<uint8_t> front_model_buffer;
std::vector<uint8_t> decoder_model_buffer;
const uint8_t* speech_front = nullptr;
const uint8_t* speech_decoder = nullptr;

bool LoadBlob(const char* path, size_t bytes, std::vector<uint8_t>* buffer,
              const uint8_t** data) {
  printf(">>> Loading speech model %s...\n", path);
  if (coralmicro::LfsSize(path) != static_cast<ssize_t>(bytes)) {
    printf("ERROR: Missing or incorrectly sized speech model: %s\n", path);
    return false;
  }

  // The runtime's weight pointers must be 16-byte aligned.
  buffer->resize(bytes + 15);
  auto* aligned = reinterpret_cast<uint8_t*>(
      (reinterpret_cast<uintptr_t>(buffer->data()) + 15) &
      ~static_cast<uintptr_t>(15));
  if (coralmicro::LfsReadFile(path, aligned, bytes) != bytes) {
    printf("ERROR: Failed to load speech model: %s\n", path);
    return false;
  }
  *data = aligned;
  return true;
}

struct Capture {
  int16_t* pcm;
  size_t count;
};

int CapturePcm(const float* pcm, int n, void* user) {
  auto* c = static_cast<Capture*>(user);
  if (n < 0 || c->count + static_cast<size_t>(n) > kMaxSamples) return 1;
  for (int i = 0; i < n; ++i) {
    if (!std::isfinite(pcm[i])) return 1;
    const float bounded = std::max(-1.0f, std::min(1.0f, pcm[i]));
    c->pcm[c->count++] = static_cast<int16_t>(std::lrintf(bounded * 32767));
  }
  return 0;
}

bool HasWord(const char* s) {
  for (; *s; ++s) {
    if ((*s >= 'A' && *s <= 'Z') || (*s >= 'a' && *s <= 'z') ||
        (*s >= '0' && *s <= '9'))
      return true;
  }
  return false;
}

bool SayChunk(const std::string& text, int depth) {
  if (!speech_front || !speech_decoder) {
    printf("ERROR: Speech model is not loaded.\n");
    return false;
  }
  if (!HasWord(text.c_str())) return true;
  int32_t ids[HEARTNANO_MAX_IDS];
  const int count = heartnano_text_to_ids(text.c_str(), ids, HEARTNANO_MAX_IDS);
  if (count < 0) {
    // A long clause or spelled-out name can exceed the neural model's 207 IDs.
    // Retry smaller word-aligned pieces; never silently truncate the sentence.
    const size_t mid = text.size() / 2;
    size_t split = text.rfind(' ', mid);
    if (split == std::string::npos || split == 0) split = text.find(' ', mid);
    if (depth < 8 && split != std::string::npos && split > 0 &&
        split + 1 < text.size()) {
      const bool first = SayChunk(text.substr(0, split), depth + 1);
      const bool second = SayChunk(text.substr(split + 1), depth + 1);
      return first && second;
    }
    printf("ERROR: Speech frontend failed (%d).\n", count);
    return false;
  }
  void* raw = malloc(kArenaBytes + 15);
  Capture capture{static_cast<int16_t*>(malloc(kMaxSamples * sizeof(int16_t))),
                  0};
  if (!raw || !capture.pcm) {
    free(raw);
    free(capture.pcm);
    printf("ERROR: Could not allocate speech buffers.\n");
    return false;
  }
  snt_nano_config cfg{};
  cfg.front_blob = speech_front;
  cfg.dec_blob = speech_decoder;
  cfg.arena = reinterpret_cast<void*>((reinterpret_cast<uintptr_t>(raw) + 15) &
                                      ~static_cast<uintptr_t>(15));
  cfg.arena_size = kArenaBytes;
  cfg.noise_seed = 2236265385529901705ULL;
  snt_nano_stats synth{};
  const int rc =
      snt_nano_synthesize(&cfg, ids, count, CapturePcm, &capture, &synth);
  free(raw);
  bool ok = rc == 0 && capture.count > 0 &&
            capture.count == static_cast<size_t>(synth.samples);
  if (!ok) {
    printf("ERROR: Speech synthesis failed (%d).\n", rc);
  } else {
    ok = audio_playback::PlayPcm16(capture.pcm, capture.count);
    if (!ok) printf("ERROR: Speech playback failed.\n");
  }
  free(capture.pcm);
  return ok;
}

// A barrier is queued after the final sentence. FIFO order plus the semaphore
// means Flush cannot finish until the DAC has played every preceding sample.
void Worker(void*) {
  WorkItem item;
  bool batch_ok = true;
  while (true) {
    if (xQueueReceive(work_queue, &item, portMAX_DELAY) != pdPASS) continue;
    if (item.barrier) {
      completed_ok = batch_ok;
      batch_ok = true;
      xSemaphoreGive(completion);
    } else {
      const bool ok = SayChunk(item.text, 0);
      batch_ok = ok && batch_ok;
    }
  }
}

bool EnsureWorker() {
  if (worker_task) return true;
  work_queue = xQueueCreateStatic(kQueueDepth, sizeof(WorkItem), queue_items,
                                  &queue_storage);
  completion = xSemaphoreCreateBinaryStatic(&completion_storage);
  if (work_queue && completion &&
      xTaskCreate(Worker, "tts", kWorkerStackWords, nullptr, kWorkerPriority,
                  &worker_task) == pdPASS)
    return true;
  if (work_queue) vQueueDelete(work_queue);
  if (completion) vSemaphoreDelete(completion);
  work_queue = nullptr;
  completion = nullptr;
  worker_task = nullptr;
  printf(
      "ERROR: Could not create speech worker; using synchronous playback.\n");
  return false;
}

void QueueWork(const WorkItem& item) {
  // Bounded backpressure: wait for a slot instead of dropping generated text.
  // With a live queue and portMAX_DELAY this retries only an unexpected RTOS
  // failure, retaining the exact item until it has been accepted.
  while (xQueueSend(work_queue, &item, portMAX_DELAY) != pdPASS) {
    printf("ERROR: Could not queue speech; retrying.\n");
    vTaskDelay(1);
  }
}

void SubmitChunk(const std::string& text) {
  if (text.empty()) return;
  if (!async_active) {
    const bool ok = SayChunk(text, 0);
    producer_ok = ok && producer_ok;
    return;
  }

  // Append bounds each chunk before it gets here; copy into the queue, never
  // retain the tokenizer's temporary piece or a pointer into pending.
  configASSERT(text.size() <= kMaxChunkBytes);
  WorkItem item{};
  std::memcpy(item.text, text.data(), text.size());
  QueueWork(item);
}

void SubmitPending() {
  if (!pending.empty()) {
    SubmitChunk(pending);
    pending.clear();
  }
}

}  // namespace

namespace speech {

bool LoadModel(const char* front_path, const char* decoder_path) {
  if (worker_task) {
    printf("ERROR: Cannot reload the speech model after starting playback.\n");
    return false;
  }
  speech_front = nullptr;
  speech_decoder = nullptr;
  const uint8_t* front = nullptr;
  const uint8_t* decoder = nullptr;
  if (!LoadBlob(front_path, NANO_FRONT_BYTES, &front_model_buffer, &front) ||
      !LoadBlob(decoder_path, NANO_DEC_BYTES, &decoder_model_buffer,
                &decoder)) {
    return false;
  }
  speech_front = front;
  speech_decoder = decoder;
  return true;
}

bool BeginAsync() {
  if (async_active) return true;
  if (!pending.empty()) Flush();
  producer_ok = true;
  if (!EnsureWorker()) return false;
  producer_task = xTaskGetCurrentTaskHandle();
  producer_priority = uxTaskPriorityGet(producer_task);
  async_active = true;

  // Time slicing is disabled in this SDK. Keep USB/PMIC above the worker, and
  // let LLM computation use the CPU whenever playback puts the worker to sleep.
  vTaskPrioritySet(producer_task, kWorkerPriority - 1);
  return true;
}

void Append(const char* piece, void*) {
  if (!piece) return;

  // Decode may return multiple characters in one token. Preserve all of them.
  for (const char* p = piece; *p; ++p) {
    pending += *p;
    if (*p == '.' || *p == '!' || *p == '?' || *p == '\n') {
      SubmitPending();
    } else if (pending.size() >= kChunkBytes) {
      const size_t split = pending.rfind(' ');
      if (split != std::string::npos && split > 0) {
        SubmitChunk(pending.substr(0, split));
        pending.erase(0, split + 1);
      } else if (pending.size() >= kMaxChunkBytes) {
        // Bound pathological unbroken output. SayChunk reports unsupported
        // text.
        SubmitPending();
      }
    }
  }
}

bool Flush() {
  SubmitPending();
  bool ok = producer_ok;
  if (async_active) {
    WorkItem barrier{};
    barrier.barrier = true;
    QueueWork(barrier);
    while (xSemaphoreTake(completion, portMAX_DELAY) != pdTRUE) {
    }
    ok = completed_ok && ok;
    async_active = false;
    vTaskPrioritySet(producer_task, producer_priority);
    producer_task = nullptr;
  }
  producer_ok = true;
  return ok;
}

}  // namespace speech
