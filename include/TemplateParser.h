#ifndef _TEMPLATE_PARSER_H_
#define _TEMPLATE_PARSER_H_

#include <llvm/ADT/StringRef.h>
#include <string>

#include "ASTContext.h"

namespace Cuda {

class TemplateParser {
private:
  const std::string mET;
  const std::string mOT;

public:
  TemplateParser(const std::string &exprTemplate,
                 const std::string &outputTemplate);

  ASTContext createAST();

  TensorType createOutputTensorType();

private:
  void parseCurrExpr(ASTContext &C, Expr *P, llvm::StringRef E);

  TensorType parseTensorExpr(llvm::StringRef T);
};

} // namespace Cuda

#endif // !_TEMPLATE_PARSER_H_
