#ifndef _INPUT_TYPES_GEN_H_
#define _INPUT_TYPES_GEN_H_

#include "ASTVisitor.h"
#include "KernelContext.h"

namespace ckg {

class InputTypesGen : public ASTVisitor {
private:
  std::vector<TensorType> mInputTypes;

public:
  std::vector<TensorType> getInputTypes() const;

  virtual void visit(OperationExpr &E) override;

  virtual void visit(TensorExpr &E) override;
};

} // namespace ckg

#endif // !_INPUT_TYPES_GEN_H_
