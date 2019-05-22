#include "Expr.h"
#include "ASTVisitor.h"

namespace Cuda {

Expr::Expr(const TensorType &tensorType, Expr *parent)
    : mParent(parent), mType(tensorType) {}

Expr::Expr(Expr *parent) : mParent(parent) {}

Expr::~Expr() {}

ChildVector<Expr *> &Expr::getChilds() { return mChilds; }

Expr *Expr::getParent() { return mParent; }

TensorType &Expr::getType() { return mType; }

uint16_t Expr::getChildCount() const { return mChilds.size(); }

void Expr::addChild(Expr *C) { mChilds.push_back(C); }

OperationExpr::OperationExpr(const TensorType &tensorType, Expr *parent)
    : Expr(tensorType, parent) {}

OperationExpr::OperationExpr(Expr *parent) : Expr(parent) {}

TensorType OperationExpr::getInputType(int32_t index) {
  return mChilds[index]->getType();
}

AddExpr::AddExpr(const TensorType &tensorType, Expr *parent)
    : OperationExpr(tensorType, parent) {}

AddExpr::AddExpr(Expr *parent) : OperationExpr(parent) {}

void AddExpr::accept(ASTVisitor &V) { V.visit(*this); }

SubtractExpr::SubtractExpr(const TensorType &tensorType, Expr *parent)
    : OperationExpr(tensorType, parent) {}

SubtractExpr::SubtractExpr(Expr *parent) : OperationExpr(parent) {}

void SubtractExpr::accept(ASTVisitor &V) { V.visit(*this); }

MultiplyExpr::MultiplyExpr(const TensorType &tensorType, Expr *parent)
    : OperationExpr(tensorType, parent) {}

MultiplyExpr::MultiplyExpr(Expr *parent) : OperationExpr(parent) {}

void MultiplyExpr::accept(ASTVisitor &V) { V.visit(*this); }

DivideExpr::DivideExpr(const TensorType &tensorType, Expr *parent)
    : OperationExpr(tensorType, parent) {}

DivideExpr::DivideExpr(Expr *parent) : OperationExpr(parent) {}

void DivideExpr::accept(ASTVisitor &V) { V.visit(*this); }

TensorExpr::TensorExpr(const TensorType &tensorType, Expr *parent)
    : Expr(tensorType, parent) {}

TensorExpr::TensorExpr(Expr *parent) : Expr(parent) {}

void TensorExpr::accept(ASTVisitor &V) { V.visit(*this); }

} // namespace Cuda