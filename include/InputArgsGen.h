#ifndef _INPUT_ARG_GEN_H_
#define _INPUT_ARG_GEN_H_

#include <vector>
#include <string>

#include "ASTVisitor.h"

namespace Cuda
{

	class InputArgsGen : public ASTVisitor
	{
	private:

		std::vector<std::string> mInputArgs;
		std::string mTerminalIdentifier;

	public:

		InputArgsGen();

		std::vector<std::string> getInputArgs() const;

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

#endif // !_INPUT_ARG_GEN_H_
