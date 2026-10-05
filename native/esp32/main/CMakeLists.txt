idf_component_register(SRCS "main.cpp" "../../ruvi.cpp"
                       INCLUDE_DIRS "../.."
                       REQUIRES esp_partition esp_timer esp_psram)
target_compile_features(${COMPONENT_LIB} PUBLIC cxx_std_17)
target_compile_options(${COMPONENT_LIB} PRIVATE -O3 -ffp-contract=off)
