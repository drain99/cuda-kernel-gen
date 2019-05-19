#ifndef _AST_VISITOR_H_
#define _AST_VISITOR_H_

#include "Expr.h"

namespace Cuda
{

	class ASTVisitor
	{
	public:

		virtual void visit(const AddExpr& expr) = 0;
		virtual void visit(const SubtractExpr& expr) = 0;
		virtual void visit(const MultiplyExpr& expr) = 0;
		virtual void visit(const DivideExpr& expr) = 0;
		virtual void visit(const TerminalExpr& expr) = 0;
	};

}

#endif // !_AST_VISITOR_H_