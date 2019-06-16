#ifndef _INPUT_PARAMS_GEN_H_
#define _INPUT_PARAMS_GEN_H_

#include <vector>
#include <string>

#include "ASTVisitor.h"

namespace ckg {

class InputParamsGen : public ASTVisitor {
private:
  std::vector<std::string> mInputParams;
  std::string mTerminalIdentifier;

public:
  InputParamsGen();

  std::vector<std::string> getInputParams() const;

  virtual void visit(OperationExpr &E) override;

  virtual void visit(TensorExpr &E) override;

private:
  void appendToTerminalIdentifier(uint16_t X);

  void popFromTerminalIdentifier();
};

} // namespace ckg

#endif // !_INPUT_PARAMS_GEN_H_
