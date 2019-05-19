cmake_minimum_required (VERSION 3.8)

set(CURR_SOURCES 
	"ASTContext.cpp" 
	"Expr.cpp"
)

add_library(CKGExprAST STATIC ${CURR_SOURCES})

target_compile_features(CKGExprAST PUBLIC cxx_std_17)

target_compile_definitions(CKGExprAST PUBLIC ${LLVM_DEFINITIONS})

target_include_directories(CKGExprAST 
						PUBLIC ${LLVM_INCLUDE_DIRS} 
						PUBLIC "../../include")

target_link_libraries(CKGExprAST "LLVMSupport")