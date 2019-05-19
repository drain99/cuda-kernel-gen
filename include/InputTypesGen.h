#ifndef _INPUT_TYPES_GEN_H_
#define _INPUT_TYPES_GEN_H_

#include "ASTVisitor.h"
#include "KernelHolder.h"

namespace Cuda
{

	class InputTypesGen : public ASTVisitor
	{
	private:

		std::vector<TensorHolder> mInputTypes;

	public:

		std::vector<TensorHolder> getInputTypes() const;

		virtual void visit(const AddExpr& expr) override;
		
		virtual void visit(const SubtractExpr& expr) override;

		virtual void visit(const MultiplyExpr& expr) override;

		virtual void visit(const DivideExpr& expr) override;

		virtual void visit(const TerminalExpr& expr) override;

	private:

		void processASMDExpr(const Expr& expr);
	};
}

#endif // !_INPUT_TYPES_GEN_H_
