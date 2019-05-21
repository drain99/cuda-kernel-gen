#include "InputParamsGen.h"

namespace Cuda {

InputParamsGen::InputParamsGen() : mTerminalIdentifier("in") {}

std::vector<std::string> InputParamsGen::getInputParams() const {
  return mInputParams;
}

void InputParamsGen::visit(Expr &E) {
  uint16_t I = 0;

  for (auto &&C : E.getChilds()) {
    appendToTerminalIdentifier(++I);
    C->accept(*this);
    popFromTerminalIdentifier();
  }
}

void InputParamsGen::visit(TensorExpr &E) {
  mInputParams.emplace_back(mTerminalIdentifier);
}

void InputParamsGen::appendToTerminalIdentifier(uint16_t X) {
  mTerminalIdentifier += "_" + std::to_string(X);
}

void InputParamsGen::popFromTerminalIdentifier() {
  while (mTerminalIdentifier.back() != '_') {
    mTerminalIdentifier.pop_back();
  }
  mTerminalIdentifier.pop_back();
}

} // namespace Cuda