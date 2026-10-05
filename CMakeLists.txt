cmake_minimum_required(VERSION 3.13)

set(CMAKE_TOOLCHAIN_FILE
    ${CMAKE_CURRENT_LIST_DIR}/coralmicro/cmake/toolchain-arm-none-eabi-gcc.cmake
)

project(llama4micro)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories(coralmicro)
add_subdirectory(coralmicro)

include_directories(llama2.c)

add_executable_m7(llama4micro
    main.cc
    speech/speech.cc
    speech/dac_playback.cc
    speech/amy_convert.cc
    speech/amy_model.cc
    sanoTTS/mcu/src/snt_front_f32.c
    sanoTTS/esphome/g2p/lex/nano_lex_g2p.c
    sanoTTS/esphome/g2p/lex/nano_lex_tables.c
    coralmicro/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/drivers/fsl_pit.c
    LINKER_SCRIPT
    ${PROJECT_SOURCE_DIR}/MIMXRT1176xxxxx_cm7_ram.ld
    DATA
    ${PROJECT_SOURCE_DIR}/models/llama2/stories15M_q80.bin
    ${PROJECT_SOURCE_DIR}/models/llama2/tokenizer.bin
    ${PROJECT_SOURCE_DIR}/models/yolov5/yolov5n-int8_edgetpu.tflite
    ${PROJECT_SOURCE_DIR}/models/yolov5/coco_labels.txt
    ${PROJECT_SOURCE_DIR}/models/sanotts/front_f32.bin
    ${PROJECT_SOURCE_DIR}/models/sanotts/amy_prefix_edgetpu.tflite
    ${PROJECT_SOURCE_DIR}/models/sanotts/amy_tail_edgetpu.tflite
)

target_link_libraries(llama4micro
    libs_base-m7_freertos
)

target_include_directories(llama4micro PRIVATE
    sanoTTS/mcu/include
    sanoTTS/esphome/g2p/lex
)

target_compile_options(llama4micro PRIVATE -O3)
