cmake_minimum_required (VERSION 3.8)

set(CURR_SOURCES 
	"KernelWriter.cpp"
)

add_library(CKGKernelWriter STATIC ${CURR_SOURCES})

target_link_libraries(CKGKernelWriter CKGKernelGen)