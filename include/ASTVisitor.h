#ifndef _AST_VISITOR_H_
#define _AST_VISITOR_H_

#include "Expr.h"

namespace ckg {

class ASTVisitor {
public:
  virtual ~ASTVisitor() = 0;

  virtual void visit(AddExpr &E);

  virtual void visit(SubtractExpr &E);

  virtual void visit(MultiplyExpr &E);

  virtual void visit(DivideExpr &E);

  virtual void visit(TensorExpr &E);

  virtual void visit(OperationExpr &E);

  virtual void visit(Expr &E);
};

} // namespace ckg

#endif // !_AST_VISITOR_H_