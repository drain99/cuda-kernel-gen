#include "InputTypesGen.h"
#include <iostream>
namespace Cuda
{

	std::vector<TensorHolder> InputTypesGen::getInputTypes() const
	{
		return mInputTypes;
	}

	void InputTypesGen::visit(const AddExpr& expr)
	{
		processASMDExpr(expr);
	}

	void InputTypesGen::visit(const SubtractExpr& expr)
	{
		processASMDExpr(expr);
	}

	void InputTypesGen::visit(const MultiplyExpr& expr)
	{
		processASMDExpr(expr);
	}

	void InputTypesGen::visit(const DivideExpr& expr)
	{
		processASMDExpr(expr);
	}

	void InputTypesGen::visit(const TerminalExpr& expr)
	{
		auto exprStr = expr.getTemplateStr();
		std::cout << exprStr.str() << std::endl;
		auto& result = mInputTypes.emplace_back();
		int balancedChevrons = 0;
		int leftIdx = exprStr.find('<');
		leftIdx = exprStr.find('<', leftIdx + 1) + 1;
		bool isFirst = true;

		for (int i = leftIdx; i < exprStr.size() - 2; ++i)
		{
			if (exprStr[i] == '<')
			{
				++balancedChevrons;
			}
			else if (exprStr[i] == '>')
			{
				--balancedChevrons;
			}
			else if (exprStr[i] == ',' && balancedChevrons == 0)
			{
				if (isFirst)
				{
					isFirst = false;
					result.DataType = exprStr.substr(leftIdx, 
													 i - leftIdx);
				}
				else
				{
					result.Dimensions.push_back(std::stoi(exprStr.substr(leftIdx,
																		 i - leftIdx)));
				}
				leftIdx = i + 1;
			}
		}
		if (isFirst)
		{
			result.DataType = exprStr.substr(leftIdx, 
											 exprStr.size() - 2 - leftIdx);
		}
		else
		{
			result.Dimensions.push_back(
				std::stoi(exprStr.substr(leftIdx,
										 exprStr.size() - 2 - leftIdx))
			);
		}
	}

	void InputTypesGen::processASMDExpr(const Expr& expr)
	{
		for (auto& C : expr.getChilds())
		{
			C->accept(*this);
		}
	}

}