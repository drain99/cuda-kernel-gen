cmake_minimum_required (VERSION 3.8)

set (CMAKE_EXPORT_COMPILE_COMMANDS ON)

project("CKGSample")

include_directories("include")
include_directories("ckg/include")

add_subdirectory("src")