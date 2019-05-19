#ifndef _INPUT_PARAMS_GEN_H_
#define _INPUT_PARAMS_GEN_H_

#include <vector>
#include <string>

#include "ASTVisitor.h"

namespace Cuda
{

	class InputParamsGen : public ASTVisitor
	{
	private:

		std::vector<std::string> mInputParams;
		std::string mTerminalIdentifier;

	public:

		InputParamsGen();

		std::vector<std::string> getInputParams() const;

		virtual void visit(const AddExpr& expr) override;

		virtual void visit(const SubtractExpr& expr) override;

		virtual void visit(const MultiplyExpr& expr) override;

		virtual void visit(const DivideExpr& expr) override;

		virtual void visit(const TerminalExpr& expr) override;

	private:

		void processASMDExpr(const Expr& expr);

		void appendToTerminalIdentifier(int x);

		void popFromTerminalIdentifier();
	};

}

#endif // !_INPUT_PARAMS_GEN_H_
