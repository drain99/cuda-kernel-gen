cmake_minimum_required (VERSION 3.8)

add_executable (CKGSample main.cpp)

target_compile_features (CKGSample PUBLIC cxx_std_17)

target_include_directories (CKGSample 
PUBLIC "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.1\\include")

target_link_directories (CKGSample 
PUBLIC "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.1\\lib\\x64"
PUBLIC "..\\ckg\\lib")

target_link_libraries (CKGSample PUBLIC cudart)