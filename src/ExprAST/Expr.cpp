#include "Expr.h"
#include "ASTVisitor.h"

namespace Cuda {

Expr::Expr(Expr *parent) : mParent(parent) {}

Expr::~Expr() {}

ChildVector<Expr *> &Expr::getChilds() { return mChilds; }

Expr *Expr::getParent() { return mParent; }

uint16_t Expr::getChildCount() const { return mChilds.size(); }

void Expr::addChild(Expr *C) { mChilds.push_back(C); }

AddExpr::AddExpr(Expr *parent) : Expr(parent) {}

void AddExpr::accept(ASTVisitor &V) { V.visit(*this); }

SubtractExpr::SubtractExpr(Expr *parent) : Expr(parent) {}

void SubtractExpr::accept(ASTVisitor &V) { V.visit(*this); }

MultiplyExpr::MultiplyExpr(Expr *parent) : Expr(parent) {}

void MultiplyExpr::accept(ASTVisitor &V) { V.visit(*this); }

DivideExpr::DivideExpr(Expr *parent) : Expr(parent) {}

void DivideExpr::accept(ASTVisitor &V) { V.visit(*this); }

TensorExpr::TensorExpr(Expr *parent, const TensorType &T)
    : Expr(parent), mTensorType(T) {}

void TensorExpr::accept(ASTVisitor &V) { V.visit(*this); }

TensorType &TensorExpr::getTensorType() { return mTensorType; }

} // namespace Cuda