#include <ASTVisitor.h>

namespace Cuda {

ASTVisitor::~ASTVisitor() {}

void ASTVisitor::visit(AddExpr &E) { visit(static_cast<OperationExpr &>(E)); }

void ASTVisitor::visit(SubtractExpr &E) { visit(static_cast<OperationExpr &>(E)); }

void ASTVisitor::visit(MultiplyExpr &E) { visit(static_cast<OperationExpr &>(E)); }

void ASTVisitor::visit(DivideExpr &E) { visit(static_cast<OperationExpr &>(E)); }

void ASTVisitor::visit(TensorExpr &E) { visit(static_cast<Expr &>(E)); }

void ASTVisitor::visit(OperationExpr &E) { visit(static_cast<Expr &>(E)); }

void ASTVisitor::visit(Expr &E) {}

} // namespace Cuda