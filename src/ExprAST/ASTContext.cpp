#include "ASTContext.h"
#include "ASTVisitor.h"

namespace Cuda
{

	Expr& ASTContext::getRootExpr() const
	{
		return *mRootExpr;
	}

	llvm::StringRef ASTContext::getFullTemplate() const
	{
		return { mFullTemplate };
	}

	void ASTContext::visitRoot(ASTVisitor& visitor) const
	{
		mRootExpr->accept(visitor);
	}

}
