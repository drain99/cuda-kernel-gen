#include "InputParamsGen.h"

namespace Cuda
{

	InputParamsGen::InputParamsGen() : mTerminalIdentifier("i")
	{
	}

	std::vector<std::string> InputParamsGen::getInputParams() const
	{
		return mInputParams;
	}

	void InputParamsGen::visit(const AddExpr& expr)
	{
		processASMDExpr(expr);
	}

	void InputParamsGen::visit(const SubtractExpr& expr)
	{
		processASMDExpr(expr);
	}

	void InputParamsGen::visit(const MultiplyExpr& expr)
	{
		processASMDExpr(expr);
	}

	void InputParamsGen::visit(const DivideExpr& expr)
	{
		processASMDExpr(expr);
	}

	void InputParamsGen::visit(const TerminalExpr& expr)
	{
		mInputParams.emplace_back(mTerminalIdentifier);
	}

	void InputParamsGen::processASMDExpr(const Expr& expr)
	{
		for (int i = 0; i < expr.getChildCount(); ++i)
		{
			appendToTerminalIdentifier(i + 1);
			expr.getChilds()[i]->accept(*this);
			popFromTerminalIdentifier();
		}
	}

	void InputParamsGen::appendToTerminalIdentifier(int x)
	{
		mTerminalIdentifier += "_" + std::to_string(x);
	}

	void InputParamsGen::popFromTerminalIdentifier()
	{
		while (mTerminalIdentifier.back() != '_')
		{
			mTerminalIdentifier.pop_back();
		}
		mTerminalIdentifier.pop_back();
	}

}