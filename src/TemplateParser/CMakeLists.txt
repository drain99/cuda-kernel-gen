cmake_minimum_required (VERSION 3.8)

set(CURR_SOURCES 
	"TemplateParser.cpp"
)

add_library(CKGTemplateParser STATIC ${CURR_SOURCES})

target_link_libraries(CKGTemplateParser CKGExprAST)