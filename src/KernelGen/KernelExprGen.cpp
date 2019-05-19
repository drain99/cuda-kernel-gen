#include "KernelExprGen.h"

namespace Cuda
{

	KernelExprGen::KernelExprGen() : mTerminalIdentifier("i")
	{
	}

	std::string KernelExprGen::getKernelExpr() const
	{
		return mKernelExpr.str();
	}

	void KernelExprGen::visit(const AddExpr& expr)
	{
		processASMDExpr(expr, '+');
	}

	void KernelExprGen::visit(const SubtractExpr& expr)
	{
		processASMDExpr(expr, '-');
	}

	void KernelExprGen::visit(const MultiplyExpr& expr)
	{
		processASMDExpr(expr, '*');
	}

	void KernelExprGen::visit(const DivideExpr& expr)
	{
		processASMDExpr(expr, '/');
	}

	void KernelExprGen::visit(const TerminalExpr& expr)
	{
		processTerminalExpr();
	}

	void KernelExprGen::processASMDExpr(const Expr& expr,
										char symbol)
	{
		mKernelExpr << '(';

		for (int i = 0; i < expr.getChildCount() - 1; ++i)
		{
			appendToIdentifier(i + 1);
			expr.getChilds()[i]->accept(*this);
			popFromIdentifier();
			mKernelExpr << symbol;
		}

		appendToIdentifier(expr.getChildCount());
		expr.getChilds().back()->accept(*this);
		popFromIdentifier();
		mKernelExpr << ')';
	}

	void KernelExprGen::processTerminalExpr()
	{
		mKernelExpr
			<< mTerminalIdentifier << "[indexX]";
	}

	void KernelExprGen::appendToIdentifier(int x)
	{
		mTerminalIdentifier += '_' + std::to_string(x);
	}

	void KernelExprGen::popFromIdentifier()
	{
		while (mTerminalIdentifier.back() != '_')
		{
			mTerminalIdentifier.pop_back();
		}
		mTerminalIdentifier.pop_back();
	}
}