#include "KernelExprGen.h"

namespace Cuda {

KernelExprGen::KernelExprGen() : mTerminalIdentifier("in") {}

std::string KernelExprGen::getKernelExpr() const { return mKernelExpr.str(); }

void KernelExprGen::visit(AddExpr &E) { processASMDExpr(E, '+'); }

void KernelExprGen::visit(SubtractExpr &E) { processASMDExpr(E, '-'); }

void KernelExprGen::visit(MultiplyExpr &E) { processASMDExpr(E, '*'); }

void KernelExprGen::visit(DivideExpr &E) { processASMDExpr(E, '/'); }

void KernelExprGen::visit(TensorExpr &E) {
  mKernelExpr << mTerminalIdentifier << "[idx]";
}

void KernelExprGen::processASMDExpr(Expr &E, char S) {
  mKernelExpr << '(';
  uint16_t I = 0;

  for (auto &&C : E.getChilds()) {
    appendToIdentifier(++I);
    C->accept(*this);
    popFromIdentifier();
    mKernelExpr << S;
  }

  mKernelExpr.seekp(-1, mKernelExpr.cur);
  mKernelExpr << ')';
}

void KernelExprGen::appendToIdentifier(uint16_t X) {
  mTerminalIdentifier += '_' + std::to_string(X);
}

void KernelExprGen::popFromIdentifier() {
  while (mTerminalIdentifier.back() != '_') {
    mTerminalIdentifier.pop_back();
  }
  mTerminalIdentifier.pop_back();
}

} // namespace Cuda