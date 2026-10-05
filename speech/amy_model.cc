#include "amy_model.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>

#include "libs/base/filesystem.h"
#include "libs/tpu/edgetpu_manager.h"
#include "libs/tpu/edgetpu_op.h"
#include "snt_front_f32.h"
#include "third_party/tflite-micro/tensorflow/lite/micro/micro_error_reporter.h"
#include "third_party/tflite-micro/tensorflow/lite/micro/micro_interpreter.h"
#include "third_party/tflite-micro/tensorflow/lite/micro/micro_mutable_op_resolver.h"

namespace amy_model {
namespace {

constexpr size_t kArenaBytes = 256 * 1024;
constexpr size_t kFrontBytes = 1417572;
constexpr size_t kFrontMetaBytes = 568;
constexpr int kChannels = 192;
constexpr int kWindowFrames = 32;
constexpr int kChunkFrames = 10;
constexpr int kHop = 256;
constexpr int kPrefixSamples = 2048;
constexpr int kTailSamples = 666;
constexpr int kTailChannels = 40;
constexpr int kOutputSamples = 2664;

struct Free {
  void operator()(void* p) const { free(p); }
};

class Buffer {
 public:
  bool Allocate(size_t size) {
    allocation_.reset(static_cast<uint8_t*>(malloc(size + 31)));
    data_ = allocation_
                ? reinterpret_cast<uint8_t*>(
                      (reinterpret_cast<uintptr_t>(allocation_.get()) + 31) &
                      ~uintptr_t(31))
                : nullptr;
    return data_ != nullptr;
  }
  void Reset() {
    allocation_.reset();
    data_ = nullptr;
  }
  uint8_t* data() const { return data_; }
  float* floats() const { return reinterpret_cast<float*>(data_); }

 private:
  std::unique_ptr<uint8_t, Free> allocation_;
  uint8_t* data_ = nullptr;
};

struct Decoder {
  Buffer model_buffer;
  Buffer arena;
  const tflite::Model* model = nullptr;
  std::unique_ptr<tflite::MicroInterpreter> interpreter;
  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;
};

Buffer front_buffer;
snt_front_model front;
Decoder decoders[2];
std::shared_ptr<coralmicro::EdgeTpuContext> context;
tflite::MicroMutableOpResolver<1> resolver;
tflite::MicroErrorReporter reporter;
bool loaded = false;
bool started = false;
bool resolver_ready = false;

bool Error(const char* message) {
  printf("ERROR: Amy speech %s.\n", message);
  return false;
}

uint32_t Fingerprint(const uint8_t* bytes, size_t size) {
  uint32_t hash = 2166136261u;
  for (size_t i = 0; i < size; ++i) hash = (hash ^ bytes[i]) * 16777619u;
  return hash;
}

bool LoadBlob(const char* path, size_t size, uint32_t fingerprint,
              Buffer* buffer) {
  if (path) printf(">>> Loading speech model %s...\n", path);
  if (!path || coralmicro::LfsSize(path) != static_cast<ssize_t>(size) ||
      !buffer->Allocate(size))
    return Error("model is missing, incorrectly sized, or cannot be allocated");
  if (coralmicro::LfsReadFile(path, buffer->data(), size) != size ||
      Fingerprint(buffer->data(), size) != fingerprint)
    return Error("model read or version check failed");
  return true;
}

bool Shape(const tflite::Tensor* tensor, int time, int channels) {
  if (!tensor || tensor->type() != tflite::TensorType_INT8 ||
      !tensor->shape() || tensor->shape()->size() != 4)
    return false;
  const auto* shape = tensor->shape();
  const auto* quant = tensor->quantization();
  return shape->Get(0) == 1 && shape->Get(1) == 1 && shape->Get(2) == time &&
         shape->Get(3) == channels && quant && quant->scale() &&
         quant->scale()->size() == 1 && std::isfinite(quant->scale()->Get(0)) &&
         quant->scale()->Get(0) > 0 && quant->zero_point() &&
         quant->zero_point()->size() == 1 &&
         quant->zero_point()->Get(0) >= -128 &&
         quant->zero_point()->Get(0) <= 127;
}

bool ValidateDecoder(Decoder* decoder, size_t bytes, int in_time, int in_ch,
                     int out_time, int out_ch) {
  flatbuffers::Verifier verifier(decoder->model_buffer.data(), bytes);
  if (!tflite::VerifyModelBuffer(verifier)) return false;
  const auto* model = tflite::GetModel(decoder->model_buffer.data());
  if (model->version() != TFLITE_SCHEMA_VERSION || !model->subgraphs() ||
      model->subgraphs()->size() != 1)
    return false;
  const auto* graph = model->subgraphs()->Get(0);
  if (!graph->inputs() || graph->inputs()->size() != 1 || !graph->outputs() ||
      graph->outputs()->size() != 1 || !graph->tensors())
    return false;
  const int input = graph->inputs()->Get(0);
  const int output = graph->outputs()->Get(0);
  if (input < 0 || output < 0 ||
      static_cast<size_t>(input) >= graph->tensors()->size() ||
      static_cast<size_t>(output) >= graph->tensors()->size() ||
      !Shape(graph->tensors()->Get(input), in_time, in_ch) ||
      !Shape(graph->tensors()->Get(output), out_time, out_ch))
    return false;
  decoder->model = model;
  return true;
}

bool Shape(const TfLiteTensor* tensor, int time, int channels) {
  return tensor && tensor->type == kTfLiteInt8 && tensor->dims &&
         tensor->dims->size == 4 && tensor->dims->data[0] == 1 &&
         tensor->dims->data[1] == 1 && tensor->dims->data[2] == time &&
         tensor->dims->data[3] == channels &&
         tensor->bytes == static_cast<size_t>(time * channels) &&
         std::isfinite(tensor->params.scale) && tensor->params.scale > 0 &&
         tensor->params.zero_point >= -128 && tensor->params.zero_point <= 127;
}

int8_t Quantize(float value, const TfLiteTensor* input) {
  const float quantized =
      std::nearbyint(value / input->params.scale) + input->params.zero_point;
  return static_cast<int8_t>(std::max(-128.0f, std::min(127.0f, quantized)));
}

// Exact largest-remainder allocation, with stable token-order tie breaking.
// Since the scale is greater than one, no positive duration can become zero.
void ExtendShortDurations(int32_t* durations, int count, int frames) {
  int remainder[kWindowFrames];
  int total = 0;
  for (int i = 0; i < count; ++i) {
    const int numerator = durations[i] * kWindowFrames;
    durations[i] = numerator / frames;
    remainder[i] = numerator % frames;
    total += durations[i];
  }
  while (total < kWindowFrames) {
    int best = 0;
    for (int i = 1; i < count; ++i)
      if (remainder[i] > remainder[best]) best = i;
    ++durations[best];
    remainder[best] = -1;
    ++total;
  }
}

bool Decode(const float* latent, int frames,
            bool (*emit)(const float*, size_t, void*), void* user) {
  Buffer samples;
  if (!samples.Allocate(kChunkFrames * kHop * sizeof(float)))
    return Error("output allocation failed");
  auto& prefix = decoders[0];
  auto& tail = decoders[1];
  int cached = -1;
  for (int first = 0; first < frames; first += kChunkFrames) {
    const int end = std::min(frames, first + kChunkFrames);
    // Shift the final window to the true end; never zero-extend latent frames.
    const int window =
        std::max(0, std::min(first - 11, frames - kWindowFrames));
    if (window != cached) {
      // The SDK XORs int8 input in place. Refill every byte before each Invoke.
      for (int t = 0; t < kWindowFrames; ++t)
        for (int c = 0; c < kChannels; ++c)
          prefix.input->data.int8[t * kChannels + c] =
              Quantize(latent[size_t(c) * frames + window + t], prefix.input);
      if (prefix.interpreter->Invoke() != kTfLiteOk)
        return Error("prefix inference failed");
      cached = window;
    }
    const int crop = std::max(
        0, std::min(64 * (first - window) - 13, kPrefixSamples - kTailSamples));
    const int trim = kHop * (first - window) - 4 * crop;
    const int keep = kHop * (end - first);
    if (trim < 0 || trim + keep > kOutputSamples)
      return Error("decoder window is out of range");
    if (prefix.output->params.scale == tail.input->params.scale &&
        prefix.output->params.zero_point == tail.input->params.zero_point) {
      memcpy(tail.input->data.int8,
             prefix.output->data.int8 + crop * kTailChannels,
             kTailSamples * kTailChannels);
    } else {
      for (int i = 0; i < kTailSamples * kTailChannels; ++i) {
        const float value =
            (int(prefix.output->data.int8[crop * kTailChannels + i]) -
             prefix.output->params.zero_point) *
            prefix.output->params.scale;
        tail.input->data.int8[i] = Quantize(value, tail.input);
      }
    }
    if (tail.interpreter->Invoke() != kTfLiteOk)
      return Error("tail inference failed");
    for (int i = 0; i < keep; ++i) {
      const float value = (int(tail.output->data.int8[trim + i]) -
                           tail.output->params.zero_point) *
                          tail.output->params.scale;
      samples.floats()[i] = std::tanh(value);
    }
    if (!emit(samples.floats(), keep, user))
      return Error("playback was aborted");
  }
  return true;
}

}  // namespace

bool Load(const char* front_path, const char* prefix_path,
          const char* tail_path) {
  if (loaded) return Error("models are already loaded");
  const bool ok =
      LoadBlob(front_path, kFrontBytes, 0x7b66c9a5, &front_buffer) &&
      LoadBlob(prefix_path, 1155648, 0x7800530c, &decoders[0].model_buffer) &&
      LoadBlob(tail_path, 143936, 0x5768004b, &decoders[1].model_buffer) &&
      snt_front_init(
          &front, front_buffer.data(), kFrontMetaBytes,
          reinterpret_cast<const float*>(front_buffer.data() + kFrontMetaBytes),
          (kFrontBytes - kFrontMetaBytes) / sizeof(float)) == 0 &&
      front.d_vocab == 129 && front.a_vocab == 145 &&
      front.a_out == kChannels && front.d_max_tokens == kMaxIds &&
      ValidateDecoder(&decoders[0], 1155648, kWindowFrames, kChannels,
                      kPrefixSamples, kTailChannels) &&
      ValidateDecoder(&decoders[1], 143936, kTailSamples, kTailChannels,
                      kOutputSamples, 1);
  if (!ok) {
    front_buffer.Reset();
    for (auto& decoder : decoders) {
      decoder.model_buffer.Reset();
      decoder.model = nullptr;
    }
    return Error("model validation failed");
  }
  loaded = true;
  return true;
}

bool Start() {
  if (started) return true;
  if (!loaded) return Error("models are not loaded");
  if (!resolver_ready) {
    if (resolver.AddCustom(coralmicro::kCustomOp,
                           coralmicro::RegisterCustomOp()) != kTfLiteOk)
      return Error("operator registration failed");
    resolver_ready = true;
  }
  context = coralmicro::EdgeTpuManager::GetSingleton()->OpenDevice(
      coralmicro::PerformanceMode::kLow);
  if (!context || context.use_count() != 1) {
    Stop();
    return Error("TPU is unavailable or still in use");
  }
  for (int i = 0; i < 2; ++i) {
    auto& decoder = decoders[i];
    if (decoder.arena.Allocate(kArenaBytes)) {
      decoder.interpreter.reset(new (std::nothrow) tflite::MicroInterpreter(
          decoder.model, resolver, decoder.arena.data(), kArenaBytes,
          &reporter));
    }
    if (!decoder.interpreter ||
        decoder.interpreter->AllocateTensors() != kTfLiteOk) {
      Stop();
      return Error("TPU interpreter allocation failed");
    }
    decoder.input = decoder.interpreter->input(0);
    decoder.output = decoder.interpreter->output(0);
    if (!Shape(decoder.input, i ? kTailSamples : kWindowFrames,
               i ? kTailChannels : kChannels) ||
        !Shape(decoder.output, i ? kOutputSamples : kPrefixSamples,
               i ? 1 : kTailChannels)) {
      Stop();
      return Error("TPU tensor validation failed");
    }
  }
  started = true;
  return true;
}

void Stop() {
  started = false;
  for (auto& decoder : decoders) {
    decoder.interpreter.reset();
    decoder.input = nullptr;
    decoder.output = nullptr;
    decoder.arena.Reset();
  }
  context.reset();
}

bool Synthesize(const int32_t* ids, int count,
                bool (*emit)(const float*, size_t, void*), void* user) {
  if (!started || !ids || !emit || count <= 0 || count > kMaxIds)
    return Error("invalid synthesis request");
  for (int i = 0; i < count; ++i)
    if (ids[i] < 0 || ids[i] >= front.d_vocab || ids[i] >= front.a_vocab)
      return Error("phoneme ID is out of range");
  int32_t durations[kMaxIds];
  Buffer scratch;
  const size_t duration_size = snt_front_duration_arena_floats(&front, count);
  if (!scratch.Allocate(duration_size * sizeof(float)))
    return Error("duration allocation failed");
  long frames = snt_front_durations(&front, ids, count, 1.08f, durations,
                                    scratch.floats(), duration_size);
  if (frames < count || frames > kMaxFrames)
    return Error("predicted duration exceeds the synthesis limit");
  if (frames < kWindowFrames) {
    ExtendShortDurations(durations, count, frames);
    frames = kWindowFrames;
  }
  scratch.Reset();
  Buffer latent;
  const size_t acoustic_size =
      snt_front_latent_arena_floats(&front, count, frames);
  if (!latent.Allocate(size_t(kChannels) * frames * sizeof(float)) ||
      !scratch.Allocate(acoustic_size * sizeof(float)))
    return Error("acoustic allocation failed");
  if (snt_front_latent(&front, ids, durations, count, frames, latent.floats(),
                       scratch.floats(), acoustic_size) != 0)
    return Error("acoustic inference failed");
  scratch.Reset();
  for (size_t i = 0; i < size_t(kChannels) * frames; ++i)
    if (!std::isfinite(latent.floats()[i]))
      return Error("acoustic output is not finite");
  return Decode(latent.floats(), frames, emit, user);
}

}  // namespace amy_model
