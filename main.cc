#include <sstream>
#include <string>
#include <vector>

#include "libs/base/filesystem.h"
#include "libs/base/gpio.h"
#include "libs/base/led.h"
#include "libs/base/timer.h"
#include "libs/camera/camera.h"
#include "libs/tensorflow/utils.h"
#include "libs/tpu/edgetpu_manager.h"
#include "libs/tpu/edgetpu_op.h"
#include "third_party/freertos_kernel/include/FreeRTOS.h"
#include "third_party/freertos_kernel/include/task.h"
#include "third_party/tflite-micro/tensorflow/lite/micro/micro_error_reporter.h"
#include "third_party/tflite-micro/tensorflow/lite/micro/micro_interpreter.h"
#include "third_party/tflite-micro/tensorflow/lite/micro/micro_mutable_op_resolver.h"

#include "llama2.h"
#include "yolov5.h"

using namespace coralmicro;
using namespace tflite;

// Llama model data paths.
const char* kLlamaModelPath = "/models/llama2/stories15M_q80.bin";
const char* kLlamaTokenizerPath = "/models/llama2/tokenizer.bin";

// Llama model inference parameters.
const float kTemperature = 1.0f;
const float kTopP = 0.9f;
const int kSteps = 256;
const char* kPromptPattern = "Once upon a time, there was a ";

// Llama model data structures.
Transformer transformer;
std::vector<uint8_t>* llama_model_buffer;
int group_size;
int steps = kSteps;
Tokenizer tokenizer;
std::vector<uint8_t>* llama_tokenizer_buffer;
Sampler sampler;

// Vision model data path.
const char* kVisionModelPath = "/models/yolov5/yolov5n-int8_edgetpu.tflite";
const char* kVisionLabelsPath = "/models/yolov5/coco_labels.txt";

// Vision model data structures.
std::vector<uint8_t>* vision_model_buffer;
std::vector<std::string>* vision_labels;
const size_t kTensorArenaSize = 575 * 1024;
STATIC_TENSOR_ARENA_IN_SDRAM(tensor_arena, kTensorArenaSize);
MicroErrorReporter tf_error_reporter;
MicroMutableOpResolver<4>* tf_resolver;
MicroInterpreter* tf_interpreter;
PerformanceMode kTpuPerformanceMode = PerformanceMode::kLow;  // Fast enough.
std::shared_ptr<EdgeTpuContext> tpu_context;

// Camera and object detection configuration.
CameraFrameFormat frame_format;
const int kDiscardFrames = 30;
const float kLabelConfidenceThreshold = 0.4f;
const float kBboxScoreThreshold = 0.2f;
const float kMinBboxSize = 0.1f;

// Debounce interval for the button interrupt.
const uint64_t kButtonDebounceUs = 50000;

// Loads the Llama model and tokenizer into memory and sets up data structures.
void LoadLlamaModel() {
  int64_t timer_start = TimerMillis();

  printf(">>> Loading Llama model %s...\n", kLlamaModelPath);
  llama_model_buffer = new std::vector<uint8_t>();
  build_transformer(&transformer, kLlamaModelPath, llama_model_buffer,
                    &group_size);
  if (steps == 0 || steps > transformer.config.seq_len) {
    steps = transformer.config.seq_len;
  }

  printf(">>> Loading Llama tokenizer %s...\n", kLlamaTokenizerPath);
  llama_tokenizer_buffer = new std::vector<uint8_t>();
  build_tokenizer(&tokenizer, kLlamaTokenizerPath, llama_tokenizer_buffer,
                  transformer.config.vocab_size);

  unsigned long long rng_seed = xTaskGetTickCount();
  build_sampler(&sampler, transformer.config.vocab_size, kTemperature, kTopP,
                rng_seed);

  int64_t timer_stop = TimerMillis();
  float timer_s = (timer_stop - timer_start) / 1000.0f;
  printf(">>> Llama model loading took %.2f s\n", timer_s);
}

// Frees the memory associated with the Llama model.
void UnloadLlamaModel() {
  printf(">>> Unloading Llama model...\n");

  free_sampler(&sampler);
  free_tokenizer(&tokenizer);
  free_transformer(&transformer);

  delete llama_tokenizer_buffer;
  delete llama_model_buffer;
}

// Loads the vision model and labels into memory.
void LoadVisionModel() {
  int64_t timer_start = TimerMillis();

  // Load the model weights.
  printf(">>> Loading vision model %s...\n", kVisionModelPath);
  vision_model_buffer = new std::vector<uint8_t>();
  if (!LfsReadFile(kVisionModelPath, vision_model_buffer)) {
    printf("ERROR: Failed to load vision model weights: %s\n",
           kVisionModelPath);
    return;
  }

  // Load the model labels.
  printf(">>> Loading vision labels %s...\n", kVisionLabelsPath);
  std::string vision_labels_buffer;
  vision_labels = new std::vector<std::string>();
  if (!LfsReadFile(kVisionLabelsPath, &vision_labels_buffer)) {
    printf("ERROR: Failed to load vision labels: %s\n", kVisionLabelsPath);
    return;
  }
  std::istringstream labels_stream(vision_labels_buffer);
  std::string label;
  while (std::getline(labels_stream, label)) {
    vision_labels->push_back(label);
  }

  // Initialize the TPU.
  // TODO: Investigate why the TPU context can't just live in the TakePicture()
  //       scope (which would turn off the TPU while it's not being used).
  tpu_context = EdgeTpuManager::GetSingleton()->OpenDevice(kTpuPerformanceMode);
  if (!tpu_context) {
    printf("ERROR: Failed to get TPU context\n");
    return;
  }

  // TODO: Can vision_model_buffer be discarded once the data is on the TPU?
  //       Does this enable using the "s" instead of the "n" version of YOLOv5?

  // Initialize the TF Lite interpreter.
  tf_resolver = new MicroMutableOpResolver<4>();
  tf_resolver->AddCustom(kCustomOp, RegisterCustomOp());
  tf_resolver->AddQuantize();
  tf_resolver->AddConcatenation();
  tf_resolver->AddReshape();
  tf_interpreter =
      new MicroInterpreter(GetModel(vision_model_buffer->data()), *tf_resolver,
                           tensor_arena, kTensorArenaSize, &tf_error_reporter);
  if (tf_interpreter->AllocateTensors() != kTfLiteOk) {
    printf("ERROR: Failed to allocate tensors\n");
    return;
  }

  // Initialize the camera capture.
  auto* input_tensor = tf_interpreter->input_tensor(0);
  frame_format.fmt = CameraFormat::kRgb;
  frame_format.filter = CameraFilterMethod::kBilinear;
  frame_format.rotation = CameraRotation::k270;
  frame_format.width = input_tensor->dims->data[1];
  frame_format.height = input_tensor->dims->data[2];
  frame_format.preserve_ratio = false;
  frame_format.buffer = GetTensorData<uint8_t>(input_tensor);
  frame_format.white_balance = true;

  int64_t timer_stop = TimerMillis();
  float timer_s = (timer_stop - timer_start) / 1000.0f;
  printf(">>> Vision model loading took %.2f s\n", timer_s);
}

// Frees the memory associated with the vision model.
void UnloadVisionModel() {
  printf(">>> Unloading vision model...\n");

  delete tf_resolver;
  delete tf_interpreter;
  delete vision_model_buffer;
  delete vision_labels;
}

// Takes a picture and returns the label of the main detected object.
std::string TakePicture() {
  printf(">>> Taking picture...\n");
  int64_t timer_start = TimerMillis();

  // Turn on the camera.
  if (!CameraTask::GetSingleton()->SetPower(true)) {
    printf("ERROR: Failed to power on camera\n");
    return "";
  }
  if (!CameraTask::GetSingleton()->Enable(CameraMode::kStreaming)) {
    printf("ERROR: Failed to enable camera\n");
    return "";
  }

  // Discard some frames to get a recent one and give auto exposure more time.
  CameraTask::GetSingleton()->DiscardFrames(10);
  if (!CameraTask::GetSingleton()->GetFrame({frame_format})) {
    printf("ERROR: Failed to take picture\n");
    return "";
  }

  // Turn off the camera.
  CameraTask::GetSingleton()->Disable();
  CameraTask::GetSingleton()->SetPower(false);

  // Run the object vision model on the image.
  if (tf_interpreter->Invoke() != kTfLiteOk) {
    printf("ERROR: Failed to detect objects\n");
    return "";
  }

  // Process the results.
  auto results = yolo::GetDetectionResults(
      tf_interpreter, kLabelConfidenceThreshold, kBboxScoreThreshold,
      kMinBboxSize, vision_labels);
  if (results.empty()) {
    printf(">>> Found no objects\n");
    return "";
  }
  for (auto result : results) {
    printf(">>> Found %s (%.2f @ %.2f|%.2f %.2fx%.2f)\n", result.label.c_str(),
           result.confidence, result.x, result.y, result.width, result.height);
  }

  int64_t timer_stop = TimerMillis();
  float timer_s = (timer_stop - timer_start) / 1000.0f;
  printf(">>> Picture taking took %.2f s\n", timer_s);

  // Use the top result's label.
  return results[0].label;
}

// Generates a story beginning with the specified prompt.
void TellStory(std::string prompt) {
  printf(">>> Generating tokens...\n");

  float tokens_s;
  generate(&transformer, &tokenizer, &sampler, prompt.c_str(), steps,
           group_size, &tokens_s);

  printf(">>> Averaged %.2f tokens/s\n", tokens_s);
}

extern "C" [[noreturn]] void app_main(void* param) {
  (void)param;

  // Set up the button interrupt.
  GpioConfigureInterrupt(
      Gpio::kUserButton, GpioInterruptMode::kIntModeFalling,
      [handle = xTaskGetCurrentTaskHandle()]() { xTaskResumeFromISR(handle); },
      kButtonDebounceUs);

  // Load the models while showing the status LED.
  LedSet(Led::kStatus, true);
  LedSet(Led::kUser, false);
  LoadLlamaModel();
  LoadVisionModel();

  while (true) {
    // Wait for a button press while showing the user LED.
    LedSet(Led::kStatus, false);
    LedSet(Led::kUser, true);
    vTaskSuspend(nullptr);
    // Continuing here after the button interrupt.

    // Take a picture while (automatically) showing the camera LED. The result
    // is the label of the main detected object.
    LedSet(Led::kStatus, false);
    LedSet(Led::kUser, false);
    std::string label = TakePicture();

    // The label might have multiple comma-separated parts. Pick the first one
    // and combine it with the prompt.
    std::istringstream label_stream(label);
    std::getline(label_stream, label, ',');
    std::string prompt = kPromptPattern;
    prompt += label;

    // Tell a story while showing the status LED.
    LedSet(Led::kStatus, true);
    LedSet(Led::kUser, false);
    TellStory(prompt);
  }

  // Unreachable in regular operation. The models stay in memory.
  UnloadLlamaModel();
  UnloadVisionModel();
}
