#include "InputArgsGen.h"

namespace Cuda {

Cuda::InputArgsGen::InputArgsGen() : mTerminalIdentifier("(*this)") {}

std::vector<std::string> InputArgsGen::getInputArgs() const {
  return mInputArgs;
}

void InputArgsGen::visit(OperationExpr &E) {
  uint16_t I = 0;
  for (auto &&C : E.getChilds()) {
    appendToTerminalIdentifier(++I);
    C->accept(*this);
    popFromTerminalIdentifier();
  }
}

void InputArgsGen::visit(TensorExpr &E) {
  mInputArgs.emplace_back(mTerminalIdentifier + ".data()");
}

void InputArgsGen::appendToTerminalIdentifier(uint16_t X) {
  mTerminalIdentifier += ".mExpr" + std::to_string(X);
}

void InputArgsGen::popFromTerminalIdentifier() {
  while (mTerminalIdentifier.back() != '.') {
    mTerminalIdentifier.pop_back();
  }
  mTerminalIdentifier.pop_back();
}

} // namespace Cuda