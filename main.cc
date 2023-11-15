#include <string>

#include "libs/base/filesystem.h"
#include "libs/base/gpio.h"
#include "libs/base/led.h"
#include "libs/base/timer.h"
#include "third_party/freertos_kernel/include/FreeRTOS.h"
#include "third_party/freertos_kernel/include/task.h"

#include "llama2.h"

using namespace coralmicro;

const uint64_t kButtonDebounceUs = 50000;

// Model data paths.
const char* kModelPath = "/llama2.c/stories15M_q80.bin";
const char* kTokenizerPath = "/llama2.c/tokenizer.bin";

// Model inference parameters.
const float kTemperature = 1.0f;
const float kTopP = 0.9f;
const int kSteps = 256;
const char* kPrompt = nullptr;

// Model data.
Transformer transformer;
std::string model_buffer;
int group_size;
int steps = kSteps;
Tokenizer tokenizer;
std::string tokenizer_buffer;
Sampler sampler;

void LoadModel() {
  printf(">>> Loading model %s...\n", kModelPath);

  int64_t load_start = TimerMillis();

  build_transformer(&transformer, kModelPath, &model_buffer, &group_size);
  if (steps == 0 || steps > transformer.config.seq_len) {
    steps = transformer.config.seq_len;
  }

  build_tokenizer(&tokenizer, kTokenizerPath, &tokenizer_buffer,
                  transformer.config.vocab_size);

  unsigned long long rng_seed = xTaskGetTickCount();
  build_sampler(&sampler, transformer.config.vocab_size, kTemperature, kTopP,
                rng_seed);

  int64_t load_end = TimerMillis();
  float load_s = (load_end - load_start) / 1000.0f;

  printf(">>> Model loading took %.2f s\n", load_s);
}

void UnloadModel() {
  printf(">>> Unloading model...\n");

  free_sampler(&sampler);
  free_tokenizer(&tokenizer);
  free_transformer(&transformer);
}

void TellStory() {
  printf(">>> Generating tokens...\n");

  float tokens_s;
  generate(&transformer, &tokenizer, &sampler, kPrompt, steps, group_size,
           &tokens_s);

  printf(">>> Averaged %.2f tokens/s\n", tokens_s);
}

extern "C" [[noreturn]] void app_main(void* param) {
  (void)param;

  // Set up the button interrupt.
  GpioConfigureInterrupt(
      Gpio::kUserButton, GpioInterruptMode::kIntModeFalling,
      [handle = xTaskGetCurrentTaskHandle()]() { xTaskResumeFromISR(handle); },
      kButtonDebounceUs);

  // Load the model while showing the status LED.
  LedSet(Led::kStatus, true);
  LedSet(Led::kUser, false);
  LoadModel();

  while (true) {
    // Wait for a button press while showing the user LED.
    LedSet(Led::kStatus, false);
    LedSet(Led::kUser, true);
    vTaskSuspend(nullptr);
    // Continuing here after the button interrupt.

    // Tell a story while showing the status LED.
    LedSet(Led::kStatus, true);
    LedSet(Led::kUser, false);
    TellStory();
  }

  // Unreachable in regular operation. The model stays in memory.
  UnloadModel();
}
