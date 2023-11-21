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

set(ADD_EXECUTABLE_M7_LINKER_SCRIPT
    ${PROJECT_SOURCE_DIR}/MIMXRT1176xxxxx_cm7_ram.ld
    CACHE INTERNAL ""
)

add_executable_m7(llama4micro
    main.cc
    DATA
    ${PROJECT_SOURCE_DIR}/models/llama2/stories15M_q80.bin
    ${PROJECT_SOURCE_DIR}/models/llama2/tokenizer.bin
    ${PROJECT_SOURCE_DIR}/models/yolov5/yolov5n-int8_edgetpu.tflite
    ${PROJECT_SOURCE_DIR}/models/yolov5/coco_labels.txt
)

target_link_libraries(llama4micro
    libs_base-m7_freertos
)
