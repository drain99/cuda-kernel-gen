#ifndef _AST_CONTEXT_H_
#define _AST_CONTEXT_H_

#include <vector>
#include <memory>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>

#include "Expr.h"

namespace Cuda
{

	class ASTVisitor;

	template <typename T>
	struct ExprOf {};

	class ASTContext
	{
	private:

		const std::string mFullTemplate;
		std::unique_ptr<Expr> mRootExpr;
		std::vector<std::unique_ptr<Expr>> mExprs;

	public:

		template <typename RootExpr>
		ASTContext(const ExprOf<RootExpr>&,
				   const std::string& fullTemplate);

		template <typename CurrentExpr>
		Expr& addNewExpr(Expr& parentExpr,
						 llvm::StringRef templateStr);

		template <typename CurrentExpr>
		Expr& addNewExpr(const ExprOf<CurrentExpr>&,
						 Expr& parentExpr,
						 llvm::StringRef templateStr);

		Expr& getRootExpr() const;

		llvm::StringRef getFullTemplate() const;

		void visitRoot(ASTVisitor& visitor) const;
	};

	template<typename RootExpr>
	inline ASTContext::ASTContext(const ExprOf<RootExpr>&,
								  const std::string& fullTemplate)
		: mFullTemplate(fullTemplate),
		mRootExpr(std::make_unique<RootExpr>(nullptr,
											 mFullTemplate))
	{
	}

	template<typename CurrentExpr>
	inline Expr& ASTContext::addNewExpr(Expr& parentExpr,
										llvm::StringRef templateStr)
	{
		mExprs.push_back(
			std::make_unique<CurrentExpr>(&parentExpr,
										  templateStr));
		parentExpr.addChild(mExprs.back().get());
		return *mExprs.back();
	}

	template<typename CurrentExpr>
	inline Expr& ASTContext::addNewExpr(const ExprOf<CurrentExpr>&,
										Expr& parentExpr,
										llvm::StringRef templateStr)
	{
		return addNewExpr<CurrentExpr>(parentExpr,
									   templateStr);
	}

}

#endif // !_AST_CONTEXT_H_