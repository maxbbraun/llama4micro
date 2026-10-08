#include "dac_playback.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>

#include "libs/base/timer.h"
#include "third_party/freertos_kernel/include/FreeRTOS.h"
#include "third_party/freertos_kernel/include/task.h"
#include "third_party/nxp/rt1176-sdk/devices/MIMXRT1176/drivers/fsl_dac12.h"
#include "third_party/nxp/rt1176-sdk/devices/MIMXRT1176/drivers/fsl_pit.h"

namespace {

constexpr uint32_t kMidpoint = 2048;
const uint16_t* volatile g_next = nullptr;
volatile size_t g_remaining = 0;
volatile bool g_done = true;
bool g_initialized = false;

void Init() {
  if (g_initialized) return;

  // Keep the same DAC reference selection as coralmicro::DacInit(). Enable the
  // analog output buffer to drive the amplifier input and select fast settling.
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

  // No RTOS APIs in this ISR. Priority 1 remains responsive while FreeRTOS
  // masks kernel-aware interrupts
  // (configLIBRARY_MAX_SYSCALL_INTERRUPT_PRIORITY=2).
  NVIC_SetPriority(PIT1_IRQn, 1);
  NVIC_ClearPendingIRQ(PIT1_IRQn);
  EnableIRQ(PIT1_IRQn);
  g_initialized = true;
  vTaskDelay(pdMS_TO_TICKS(10));
}

bool PlayCodes(const uint16_t* codes, size_t count) {
  constexpr uint32_t kSampleRateHz = dac_playback::kSampleRateHz;
  constexpr uint64_t kPlaybackTimeoutMarginUs = 2000000;
  if (!codes || !count || !g_done) return false;
  Init();
  const uint32_t bus_hz = CLOCK_GetRootClockFreq(kCLOCK_Root_Bus);
  if (!bus_hz || bus_hz % kSampleRateHz != 0) return false;
  PIT_StopTimer(PIT1, kPIT_Chnl_0);
  PIT_DisableInterrupts(PIT1, kPIT_Chnl_0, kPIT_TimerInterruptEnable);
  PIT_ClearStatusFlags(PIT1, kPIT_Chnl_0, kPIT_TimerFlag);
  NVIC_ClearPendingIRQ(PIT1_IRQn);
  PIT_SetTimerPeriod(PIT1, kPIT_Chnl_0, bus_hz / kSampleRateHz);
  g_next = codes;
  g_remaining = count;
  g_done = false;
  __DMB();
  const uint64_t started = coralmicro::TimerMicros();
  const uint64_t timeout_us =
      static_cast<uint64_t>(count) * 1000000 / kSampleRateHz +
      kPlaybackTimeoutMarginUs;
  PIT_EnableInterrupts(PIT1, kPIT_Chnl_0, kPIT_TimerInterruptEnable);
  PIT_StartTimer(PIT1, kPIT_Chnl_0);
  while (!g_done && coralmicro::TimerMicros() - started < timeout_us) {
    vTaskDelay(pdMS_TO_TICKS(1));
  }
  PIT_StopTimer(PIT1, kPIT_Chnl_0);
  PIT_DisableInterrupts(PIT1, kPIT_Chnl_0, kPIT_TimerInterruptEnable);
  DisableIRQ(PIT1_IRQn);
  __DSB();
  const bool finished = g_done;
  g_next = nullptr;
  g_remaining = 0;
  g_done = true;

  // Leave DAC biased at midpoint between utterances to avoid repeated pops.
  DAC12_SetData(DAC, kMidpoint);
  PIT_ClearStatusFlags(PIT1, kPIT_Chnl_0, kPIT_TimerFlag);
  NVIC_ClearPendingIRQ(PIT1_IRQn);
  EnableIRQ(PIT1_IRQn);
  return finished;
}

float Fade(size_t i, size_t count, uint32_t sample_rate_hz) {
  constexpr uint32_t kFadeDurationMs = 5;
  const size_t fade_samples =
      static_cast<uint64_t>(sample_rate_hz) * kFadeDurationMs / 1000;
  const size_t ramp =
      std::max<size_t>(1, std::min<size_t>(fade_samples, count / 2));
  const size_t edge = std::min(i, count - 1 - i);
  return static_cast<float>(std::min(edge, ramp)) / ramp;
}

uint16_t Code(float sample) {
  sample = std::max(-1.0f, std::min(1.0f, sample));
  return static_cast<uint16_t>(2048 + std::lrintf(sample * 2047));
}

}  // namespace

extern "C" void PIT1_IRQHandler() {
  PIT_ClearStatusFlags(PIT1, kPIT_Chnl_0, kPIT_TimerFlag);
  if (g_remaining) {
    // The DAC DATA register requires a 32-bit write; the SDK does this.
    DAC12_SetData(DAC, *g_next++);
    --g_remaining;
  } else {
    DAC12_SetData(DAC, kMidpoint);
    PIT_StopTimer(PIT1, kPIT_Chnl_0);
    PIT_DisableInterrupts(PIT1, kPIT_Chnl_0, kPIT_TimerInterruptEnable);
    g_done = true;
  }

  // Prevent a second entry before the peripheral has observed flag clearing.
  __DSB();
}

namespace dac_playback {

bool PlayPcm16(const int16_t* pcm, size_t samples) {
  if (!pcm || !samples ||
      samples > std::numeric_limits<size_t>::max() / sizeof(uint16_t))
    return false;
  auto* codes = static_cast<uint16_t*>(malloc(samples * sizeof(uint16_t)));
  if (!codes) return false;

  for (size_t i = 0; i < samples; ++i) {
    codes[i] = Code(pcm[i] / 32768.0f * Fade(i, samples, kSampleRateHz));
  }
  const bool ok = PlayCodes(codes, samples);
  free(codes);
  return ok;
}

}  // namespace dac_playback
