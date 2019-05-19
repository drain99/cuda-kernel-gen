#include "InputArgsGen.h"

namespace Cuda
{

	Cuda::InputArgsGen::InputArgsGen() : mTerminalIdentifier("(*this)")
	{
	}

	std::vector<std::string> InputArgsGen::getInputArgs() const
	{
		return mInputArgs;
	}

	void InputArgsGen::visit(const AddExpr& expr)
	{
		processASMDExpr(expr);
	}

	void InputArgsGen::visit(const SubtractExpr& expr)
	{
		processASMDExpr(expr);
	}

	void InputArgsGen::visit(const MultiplyExpr& expr)
	{
		processASMDExpr(expr);
	}

	void InputArgsGen::visit(const DivideExpr& expr)
	{
		processASMDExpr(expr);
	}

	void InputArgsGen::visit(const TerminalExpr& expr)
	{
		mInputArgs.emplace_back(mTerminalIdentifier + ".data()");
	}

	void InputArgsGen::processASMDExpr(const Expr& expr)
	{
		for (int i = 0; i < expr.getChildCount(); ++i)
		{
			appendToTerminalIdentifier(i + 1);
			expr.getChilds()[i]->accept(*this);
			popFromTerminalIdentifier();
		}
	}

	void InputArgsGen::appendToTerminalIdentifier(int x)
	{
		mTerminalIdentifier += ".expr" + std::to_string(x);
	}

	void InputArgsGen::popFromTerminalIdentifier()
	{
		while (mTerminalIdentifier.back() != '.')
		{
			mTerminalIdentifier.pop_back();
		}
		mTerminalIdentifier.pop_back();
	}

}