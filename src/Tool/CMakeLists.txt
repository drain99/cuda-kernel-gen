cmake_minimum_required (VERSION 3.8)

set(CURR_SOURCES 
	"main.cpp"
)

add_executable(ckg-tool ${CURR_SOURCES})

target_link_libraries(ckg-tool CKGSourceParser CKGKernelGen)