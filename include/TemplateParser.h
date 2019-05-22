#ifndef _TEMPLATE_PARSER_H_
#define _TEMPLATE_PARSER_H_

#include <llvm/ADT/StringRef.h>
#include <string>

#include "ASTContext.h"

namespace Cuda {

class TemplateParser {
private:
  const std::string mET;

public:
  TemplateParser(const std::string &exprTemplate);

  ASTContext createAST();

private:
  void parseCurrExpr(ASTContext &C, Expr *P, llvm::StringRef E);

  TensorType parseTensorExpr(llvm::StringRef T);
};

} // namespace Cuda

#endif // !_TEMPLATE_PARSER_H_
