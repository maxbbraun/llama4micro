#include "speech.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

#include "amy_convert.h"
#include "amy_model.h"
#include "dac_playback.h"
#include "libs/base/tasks.h"
#include "third_party/freertos_kernel/include/FreeRTOS.h"
#include "third_party/freertos_kernel/include/queue.h"
#include "third_party/freertos_kernel/include/semphr.h"
#include "third_party/freertos_kernel/include/task.h"

namespace {

constexpr size_t kChunkChars = 120;
constexpr size_t kMaxChunkChars = 480;
constexpr UBaseType_t kQueueDepth = 2;
// The recursive fallback retains a phoneme array at each depth.
constexpr size_t kWorkerStackWords = 4096;
constexpr UBaseType_t kWorkerPriority = coralmicro::kAppTaskPriority;
static_assert(kWorkerPriority > tskIDLE_PRIORITY,
              "The producer needs a priority below the TTS worker");

struct WorkItem {
  bool barrier;
  char text[kMaxChunkChars + 1];
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
bool playback_active = false;  // Speech worker, or the synchronous producer.

bool EmitPcm(const float* pcm, size_t count, void*) {
  if (!playback_active) {
    if (!audio_playback::Begin()) return false;
    playback_active = true;
  }
  return audio_playback::WritePcm(pcm, count);
}

bool FinishPlayback() {
  const bool ok = !playback_active || audio_playback::Finish();
  playback_active = false;
  // Release speech's TPU context before the next camera inference.
  amy_model::Stop();
  return ok;
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
  if (!HasWord(text.c_str())) return true;
  int32_t ids[amy_model::kMaxIds];
  const int count = amy_text_to_ids(text.c_str(), ids, amy_model::kMaxIds);
  if (count < 0) {
    // A long clause or spelled-out name can exceed the neural model's 265 IDs.
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
  if (!amy_model::Start()) return false;
  const bool ok = amy_model::Synthesize(ids, count, EmitPcm, nullptr);
  const bool played = !playback_active || audio_playback::EndUtterance();
  if (!ok || !played) printf("ERROR: Speech synthesis or playback failed.\n");
  return ok && played;
}
// A barrier is queued after the final sentence. FIFO order plus the semaphore
// means Flush cannot finish until the DAC has played every preceding sample.
void Worker(void*) {
  WorkItem item;
  bool batch_ok = true;
  while (true) {
    if (xQueueReceive(work_queue, &item, portMAX_DELAY) != pdPASS) continue;
    if (item.barrier) {
      const bool played = FinishPlayback();
      completed_ok = batch_ok && played;
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
  configASSERT(text.size() <= kMaxChunkChars);
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

bool LoadModel(const char* front_path, const char* prefix_path,
               const char* tail_path) {
  if (worker_task) {
    printf("ERROR: Cannot reload the speech model after starting playback.\n");
    return false;
  }
  return amy_model::Load(front_path, prefix_path, tail_path);
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
    } else if (pending.size() >= kChunkChars) {
      const size_t split = pending.rfind(' ');
      if (split != std::string::npos && split > 0) {
        SubmitChunk(pending.substr(0, split));
        pending.erase(0, split + 1);
      } else if (pending.size() >= kMaxChunkChars) {
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
  } else {
    const bool played = FinishPlayback();
    ok = played && ok;
  }
  producer_ok = true;
  return ok;
}

}  // namespace speech
