#include "dac_playback.h"

#include <algorithm>
#include <atomic>
#include <cmath>

#include "libs/base/timer.h"
#include "third_party/freertos_kernel/include/FreeRTOS.h"
#include "third_party/freertos_kernel/include/task.h"
#include "third_party/nxp/rt1176-sdk/devices/MIMXRT1176/drivers/fsl_dac12.h"
#include "third_party/nxp/rt1176-sdk/devices/MIMXRT1176/drivers/fsl_pit.h"

namespace {

constexpr uint32_t kMidpoint = 2048;
constexpr uint32_t kCapacity = 65536;  // Just under three seconds at 22.05 kHz.
constexpr uint32_t kPrefill = 5120;
constexpr size_t kFadeSamples = audio_playback::kSampleRate / 200;
static_assert(std::atomic<uint32_t>::is_always_lock_free,
              "The high-priority DAC ISR cannot take a lock");
__attribute__((section(".sdram_bss"), aligned(32))) uint16_t ring[kCapacity];
std::atomic<uint32_t> read_index{0};
std::atomic<uint32_t> write_index{0};
std::atomic<bool> running{false};
bool initialized = false;
bool active = false;
bool failed = false;
float tail[kFadeSamples];
size_t tail_head = 0;
size_t tail_count = 0;
size_t utterance_samples = 0;

void Init() {
  if (initialized) return;
  dac12_config_t dac{};
  DAC12_GetDefaultConfig(&dac);
  dac.referenceVoltageSource = kDAC12_ReferenceVoltageSourceAlt2;
  dac.enableAnalogBuffer = true;
  dac.speedMode = kDAC12_SpeedHighMode;
  DAC12_Init(DAC, &dac);
  DAC12_SetData(DAC, kMidpoint);
  DAC12_Enable(DAC, true);

  pit_config_t pit{};
  PIT_GetDefaultConfig(&pit);
  PIT_Init(PIT1, &pit);
  // This ISR never calls FreeRTOS. Priority 1 stays responsive while the kernel
  // masks kernel-aware interrupts
  // (configLIBRARY_MAX_SYSCALL_INTERRUPT_PRIORITY=2).
  NVIC_SetPriority(PIT1_IRQn, 1);
  NVIC_ClearPendingIRQ(PIT1_IRQn);
  EnableIRQ(PIT1_IRQn);
  initialized = true;
  vTaskDelay(pdMS_TO_TICKS(10));
}

void StartTimer(bool force) {
  // Serialize the empty-ring stop in the ISR with restarting from the producer.
  DisableIRQ(PIT1_IRQn);
  const uint32_t available = write_index.load() - read_index.load();
  if (!running.load() && available && (force || available >= kPrefill)) {
    PIT_ClearStatusFlags(PIT1, kPIT_Chnl_0, kPIT_TimerFlag);
    NVIC_ClearPendingIRQ(PIT1_IRQn);
    running.store(true);
    PIT_EnableInterrupts(PIT1, kPIT_Chnl_0, kPIT_TimerInterruptEnable);
    PIT_StartTimer(PIT1, kPIT_Chnl_0);
  }
  EnableIRQ(PIT1_IRQn);
}

uint16_t Code(float sample) {
  sample = std::max(-1.0f, std::min(1.0f, sample));
  return static_cast<uint16_t>(kMidpoint + std::lrintf(sample * 2047));
}

bool Push(float sample) {
  const uint32_t next = write_index.load(std::memory_order_relaxed);
  const uint64_t started = coralmicro::TimerMicros();
  while (next - read_index.load(std::memory_order_acquire) == kCapacity) {
    StartTimer(true);
    if (coralmicro::TimerMicros() - started > 2000000) {
      failed = true;
      return false;
    }
    vTaskDelay(1);
  }
  ring[next % kCapacity] = Code(sample);
  write_index.store(next + 1, std::memory_order_release);
  if (!running.load(std::memory_order_relaxed)) StartTimer(false);
  return true;
}

}  // namespace

extern "C" void PIT1_IRQHandler() {
  PIT_ClearStatusFlags(PIT1, kPIT_Chnl_0, kPIT_TimerFlag);
  const uint32_t next = read_index.load(std::memory_order_relaxed);
  if (next != write_index.load(std::memory_order_acquire)) {
    // The DAC DATA register requires the SDK's 32-bit write.
    DAC12_SetData(DAC, ring[next % kCapacity]);
    read_index.store(next + 1, std::memory_order_release);
  } else {
    DAC12_SetData(DAC, kMidpoint);
    PIT_StopTimer(PIT1, kPIT_Chnl_0);
    PIT_DisableInterrupts(PIT1, kPIT_Chnl_0, kPIT_TimerInterruptEnable);
    running.store(false);
  }
  __DSB();
}

namespace audio_playback {

bool Begin() {
  if (active) return false;
  Init();
  const uint32_t bus_hz = CLOCK_GetRootClockFreq(kCLOCK_Root_Bus);
  if (bus_hz < kSampleRate) return false;
  // 22.05 kHz is not an integer divisor of the bus clock. Choose the nearest
  // timer period; at 240 MHz the sample-rate error is less than 37 ppm.
  PIT_SetTimerPeriod(PIT1, kPIT_Chnl_0,
                     (bus_hz + kSampleRate / 2) / kSampleRate);
  read_index.store(0);
  write_index.store(0);
  tail_head = tail_count = utterance_samples = 0;
  failed = false;
  active = true;
  return true;
}

bool WritePcm(const float* pcm, size_t samples) {
  if (!active || failed || (!pcm && samples)) return false;
  for (size_t i = 0; i < samples; ++i) {
    if (!std::isfinite(pcm[i])) {
      failed = true;
      return false;
    }
    float sample = pcm[i];
    if (utterance_samples < kFadeSamples)
      sample *= static_cast<float>(utterance_samples) / kFadeSamples;
    ++utterance_samples;
    // Retain only the last 5 ms so EndUtterance can fade the true end, never a
    // boundary between TPU chunks. Older samples stream into the DAC ring.
    if (tail_count == kFadeSamples) {
      if (!Push(tail[tail_head])) return false;
      tail[tail_head] = sample;
      tail_head = (tail_head + 1) % kFadeSamples;
    } else {
      tail[(tail_head + tail_count++) % kFadeSamples] = sample;
    }
  }
  return true;
}

bool EndUtterance() {
  if (!active) return false;
  bool ok = !failed;
  for (size_t i = 0; ok && i < tail_count; ++i) {
    const float fade = static_cast<float>(tail_count - 1 - i) / kFadeSamples;
    ok = Push(tail[(tail_head + i) % kFadeSamples] * fade);
  }
  tail_head = tail_count = utterance_samples = 0;
  StartTimer(true);
  return ok;
}

bool Finish() {
  if (!active) return false;
  const bool ended = EndUtterance();
  const uint64_t started = coralmicro::TimerMicros();
  while (read_index.load() != write_index.load()) {
    if (coralmicro::TimerMicros() - started > 5000000) {
      failed = true;
      break;
    }
    vTaskDelay(1);
  }
  DisableIRQ(PIT1_IRQn);
  PIT_StopTimer(PIT1, kPIT_Chnl_0);
  PIT_DisableInterrupts(PIT1, kPIT_Chnl_0, kPIT_TimerInterruptEnable);
  running.store(false);
  DAC12_SetData(DAC, kMidpoint);
  PIT_ClearStatusFlags(PIT1, kPIT_Chnl_0, kPIT_TimerFlag);
  NVIC_ClearPendingIRQ(PIT1_IRQn);
  EnableIRQ(PIT1_IRQn);
  active = false;
  return ended && !failed;
}

}  // namespace audio_playback
