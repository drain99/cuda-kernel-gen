#include "Expr.h"
#include "ASTVisitor.h"

namespace Cuda
{

	Expr::Expr(Expr* parent,
			   llvm::StringRef templateStr)
		: mParent(parent), mTemplateStr(templateStr)
	{
	}

	Expr::~Expr() {}

	const SmallVector<Expr*>& Expr::getChilds() const
	{
		return mChilds;
	}

	OptionalRef<const Expr> Expr::getParent() const
	{
		if (mParent)
		{
			return *mParent;
		}
		else
		{
			return std::nullopt;
		}
	}

	int Expr::getChildCount() const
	{
		return mChilds.size();
	}

	llvm::StringRef Expr::getTemplateStr() const
	{
		return mTemplateStr;
	}

	void Expr::addChild(Expr* child)
	{
		mChilds.push_back(child);
	}

	AddExpr::AddExpr(Expr* parent,
					 llvm::StringRef templateStr)
		: Expr(parent, templateStr)
	{
	}

	void AddExpr::accept(ASTVisitor& visitor) const
	{
		visitor.visit({ *this });
	}

	SubtractExpr::SubtractExpr(Expr* parent,
							   llvm::StringRef templateStr)
		: Expr(parent, templateStr)
	{
	}

	void SubtractExpr::accept(ASTVisitor& visitor) const
	{
		visitor.visit({ *this });
	}

	MultiplyExpr::MultiplyExpr(Expr* parent,
							   llvm::StringRef templateStr)
		: Expr(parent, templateStr)
	{
	}

	void MultiplyExpr::accept(ASTVisitor& visitor) const
	{
		visitor.visit({ *this });
	}

	DivideExpr::DivideExpr(Expr* parent,
						   llvm::StringRef templateStr)
		: Expr(parent, templateStr)
	{
	}

	void DivideExpr::accept(ASTVisitor& visitor) const
	{
		visitor.visit({ *this });
	}

	TerminalExpr::TerminalExpr(Expr* parent,
							   llvm::StringRef templateStr)
		: Expr(parent, templateStr)
	{
	}

	void TerminalExpr::accept(ASTVisitor& visitor) const
	{
		visitor.visit({ *this });
	}

}