#ifndef _INPUT_ARG_GEN_H_
#define _INPUT_ARG_GEN_H_

#include <vector>
#include <string>

#include "ASTVisitor.h"

namespace ckg {

class InputArgsGen : public ASTVisitor {
private:
  std::vector<std::string> mInputArgs;
  std::string mTerminalIdentifier;

public:
  InputArgsGen();

  std::vector<std::string> getInputArgs() const;

  virtual void visit(OperationExpr &E) override;

  virtual void visit(TensorExpr &E) override;

private:
  void processASMDExpr(const Expr &expr);

  void appendToTerminalIdentifier(uint16_t X);

  void popFromTerminalIdentifier();
};

} // namespace ckg

#endif // !_INPUT_ARG_GEN_H_
