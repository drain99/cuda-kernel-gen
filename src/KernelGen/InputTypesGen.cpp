#include "InputTypesGen.h"
#include <iostream>
namespace Cuda {

std::vector<TensorType> InputTypesGen::getInputTypes() const {
  return mInputTypes;
}

void InputTypesGen::visit(Expr &E) {
  for (auto &&C : E.getChilds()) {
    C->accept(*this);
  }
}

void InputTypesGen::visit(TensorExpr &E) {
  mInputTypes.push_back(E.getTensorType());
}

}