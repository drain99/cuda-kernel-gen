#include "ASTContext.h"
#include "ASTVisitor.h"

namespace ckg {

Expr *ASTContext::getRootExpr() { return mRootExpr.get(); }

void ASTContext::visitRoot(ASTVisitor &V) { mRootExpr->accept(V); }

} // namespace ckg