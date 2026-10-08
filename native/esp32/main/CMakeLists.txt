idf_component_register(SRCS "main.cpp" "../../ruvi.cpp" "../../parallel.cpp" "../../tokenizer.cpp" "../../tokenizer_data.cpp" "../../kernels/kernel.cpp" "../../kernels/kernel_s3.cpp" "../../kernels/kernel_s3.S"
                       INCLUDE_DIRS "../.." "../../kernels"
                       REQUIRES esp_partition esp_timer esp_psram)
target_compile_features(${COMPONENT_LIB} PUBLIC cxx_std_17)
target_compile_options(${COMPONENT_LIB} PRIVATE -O3 -ffp-contract=off)
target_compile_definitions(${COMPONENT_LIB} PRIVATE RUVI_KERNEL_ESP32_S3=1)
