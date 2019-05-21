cmake_minimum_required (VERSION 3.8)

set(CURR_SOURCES 
	"KernelExprGen.cpp"
	"InputParamsGen.cpp"
	"InputArgsGen.cpp"
	"KernelManager.cpp"
	"InputTypesGen.cpp"
	"KernelContext.cpp"
)

add_library(CKGKernelGen STATIC ${CURR_SOURCES})

target_link_libraries(CKGKernelGen CKGExprAST)