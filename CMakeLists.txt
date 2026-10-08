cmake_minimum_required(VERSION 3.13)

set(CMAKE_TOOLCHAIN_FILE
    ${CMAKE_CURRENT_LIST_DIR}/coralmicro/cmake/toolchain-arm-none-eabi-gcc.cmake
)

project(llama4micro)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories(coralmicro)
add_subdirectory(coralmicro)

add_executable_m7(llama4micro
    main.cc
    speech/speech.cc
    speech/dac_playback.cc
    speech/heartnano_convert.cc
    speech/flite_phonemes.cc
    flite/lang/cmulex/cmu_lts_model.c
    flite/lang/cmulex/cmu_lts_rules.c
    sanoTTS/mcu/src/snt_nano.c
    sanoTTS/mcu/src/snt_kernels_ref.c
    sanoTTS/esphome/g2p/lex/nano_lex_g2p.c
    sanoTTS/esphome/g2p/lex/nano_lex_tables.c
    coralmicro/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/drivers/fsl_pit.c
    LINKER_SCRIPT
    ${PROJECT_SOURCE_DIR}/MIMXRT1176xxxxx_cm7_ram.ld
    DATA
    ${PROJECT_SOURCE_DIR}/models/llama2/stories15M_q80.bin
    ${PROJECT_SOURCE_DIR}/models/llama2/tokenizer.bin
    ${PROJECT_SOURCE_DIR}/models/yolov8/yolov8n-int8_edgetpu.tflite
    ${PROJECT_SOURCE_DIR}/models/yolov8/coco_labels.txt
    ${PROJECT_SOURCE_DIR}/models/sanotts/front_q8.bin
    ${PROJECT_SOURCE_DIR}/models/sanotts/model_q8.bin
)

target_link_libraries(llama4micro
    libs_base-m7_freertos
)

target_include_directories(llama4micro PRIVATE
    flite/include
    sanoTTS/mcu/include
    sanoTTS/mcu/models/en_us_e13b
    sanoTTS/esphome/g2p/lex
)

target_compile_options(llama4micro PRIVATE -O3)
