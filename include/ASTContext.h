#ifndef _AST_CONTEXT_H_
#define _AST_CONTEXT_H_

#include <vector>
#include <memory>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>

#include "Expr.h"

namespace Cuda {

class ASTVisitor;

template <typename T> struct ExprOf {};

class ASTContext {
private:
  std::unique_ptr<Expr> mRootExpr;
  std::vector<std::unique_ptr<Expr>> mExprs;

public:
  template <typename RootExpr, typename... Args,
            typename = std::enable_if_t<IsOperationExpr_v<RootExpr>>>
  ASTContext(const ExprOf<RootExpr> &, Args &&... args);

  template <typename CurrExpr, typename... Args>
  Expr *addNewExpr(Args &&... args);

  Expr *getRootExpr();

  void visitRoot(ASTVisitor &V);
};

template <typename RootExpr, typename... Args, typename>
inline ASTContext::ASTContext(const ExprOf<RootExpr> &, Args &&... args)
    : mRootExpr(std::make_unique<RootExpr>(args...)) {}

template <typename CurrExpr, typename... Args>
inline Expr *ASTContext::addNewExpr(Args &&... args) {
  mExprs.push_back(std::make_unique<CurrExpr>(args...));
  mExprs.back()->getParent()->addChild(mExprs.back().get());
  return mExprs.back().get();
}

} // namespace Cuda

#endif // !_AST_CONTEXT_H_