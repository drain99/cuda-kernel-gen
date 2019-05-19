#ifndef _EXPR_H_
#define _EXPR_H_

#include <optional>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>

namespace Cuda
{

	class ASTVisitor;

	template <typename V>
	using OptionalRef = std::optional<
		std::reference_wrapper<V>
	>;

	template <typename V>
	using SmallVector = llvm::SmallVector<V, 2>;

	class Expr
	{
	protected:

		Expr* mParent;
		SmallVector<Expr*> mChilds;
		llvm::StringRef mTemplateStr;

	public:

		Expr(Expr* parent,
			 llvm::StringRef templateStr);

		virtual ~Expr();

		virtual void accept(ASTVisitor& visitor) const = 0;

		const SmallVector<Expr*>& getChilds() const;

		OptionalRef<const Expr> getParent() const;

		int getChildCount() const;

		llvm::StringRef getTemplateStr() const;

		void addChild(Expr* child);
	};

	class AddExpr : public Expr
	{
	public:

		AddExpr(Expr* parent,
				llvm::StringRef templateStr);

		virtual void accept(ASTVisitor& visitor) const override;
	};

	class SubtractExpr : public Expr
	{
	public:

		SubtractExpr(Expr* parent,
					 llvm::StringRef templateStr);

		virtual void accept(ASTVisitor& visitor) const override;
	};

	class MultiplyExpr : public Expr
	{
	public:

		MultiplyExpr(Expr* parent,
					 llvm::StringRef templateStr);

		virtual void accept(ASTVisitor& visitor) const override;
	};

	class DivideExpr : public Expr
	{
	public:

		DivideExpr(Expr* parent,
				   llvm::StringRef templateStr);

		virtual void accept(ASTVisitor& visitor) const override;
	};

	class TerminalExpr : public Expr
	{
	public:

		TerminalExpr(Expr* parent,
					 llvm::StringRef templateStr);

		virtual void accept(ASTVisitor& visitor) const override;
	};

}

#endif // !_EXPR_H_
