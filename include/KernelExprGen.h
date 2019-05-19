#ifndef _KERNEL_EXPR_GEN_H_
#define _KERNEL_EXPR_GEN_H_

#include <string>
#include <sstream>

#include "ASTVisitor.h"

namespace Cuda
{
	
	class KernelExprGen : public ASTVisitor
	{
	private:
		
		std::stringstream mKernelExpr;
		std::string mTerminalIdentifier;

	public:
		
		KernelExprGen();

		std::string getKernelExpr() const;

		virtual void visit(const AddExpr& expr) override;

		virtual void visit(const SubtractExpr& expr) override;

		virtual void visit(const MultiplyExpr& expr) override;

		virtual void visit(const DivideExpr& expr) override;

		virtual void visit(const TerminalExpr& expr) override;

	private:

		void processASMDExpr(const Expr& expr,
							 char symbol);

		void processTerminalExpr();

		void appendToIdentifier(int x);

		void popFromIdentifier();
	};

}


#endif // !_KERNEL_EXPR_GEN_H_
