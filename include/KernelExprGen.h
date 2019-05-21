#ifndef _KERNEL_EXPR_GEN_H_
#define _KERNEL_EXPR_GEN_H_

#include <string>
#include <sstream>

#include "ASTVisitor.h"

namespace Cuda {

class KernelExprGen : public ASTVisitor {
private:
  std::stringstream mKernelExpr;
  std::string mTerminalIdentifier;

public:
  KernelExprGen();

  std::string getKernelExpr() const;

  virtual void visit(AddExpr &E) override;

  virtual void visit(SubtractExpr &E) override;

  virtual void visit(MultiplyExpr &E) override;

  virtual void visit(DivideExpr &E) override;

  virtual void visit(TensorExpr &E) override;

private:
  void processASMDExpr(Expr &E, char S);

  void appendToIdentifier(uint16_t X);

  void popFromIdentifier();
};

}


#endif // !_KERNEL_EXPR_GEN_H_
