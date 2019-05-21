#ifndef _EXPR_H_
#define _EXPR_H_

#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <type_traits>

#include "TensorType.h"

namespace Cuda {

class ASTVisitor;

struct ExprImpl {};
struct OperationExprImpl {};
struct TensorExprImpl {};

template <typename V> using ChildVector = llvm::SmallVector<V, 3>;

class Expr : private ExprImpl {
protected:
  Expr *mParent = nullptr;
  ChildVector<Expr *> mChilds;

public:
  Expr(Expr *parent = nullptr);

  virtual ~Expr();

  virtual void accept(ASTVisitor &V) = 0;

  ChildVector<Expr *> &getChilds();

  Expr *getParent();

  uint16_t getChildCount() const;

  void addChild(Expr *C);
};

class AddExpr : public Expr, private OperationExprImpl {
public:
  AddExpr(Expr *parent = nullptr);

  virtual void accept(ASTVisitor &V) override;
};

class SubtractExpr : public Expr, private OperationExprImpl {
public:
  SubtractExpr(Expr *parent = nullptr);

  virtual void accept(ASTVisitor &V) override;
};

class MultiplyExpr : public Expr, private OperationExprImpl {
public:
  MultiplyExpr(Expr *parent = nullptr);

  virtual void accept(ASTVisitor &V) override;
};

class DivideExpr : public Expr, private OperationExprImpl {
public:
  DivideExpr(Expr *parent = nullptr);

  virtual void accept(ASTVisitor &V) override;
};

class TensorExpr : public Expr, private TensorExprImpl {
protected:
  TensorType mTensorType;

public:
  TensorExpr(Expr *parent, const TensorType &T);

  virtual void accept(ASTVisitor &V) override;

  TensorType &getTensorType();
};

template <typename... Ts>
constexpr bool IsExpr_v = std::conjunction_v<std::is_base_of<ExprImpl, Ts>...>;

template <typename... Ts>
constexpr bool IsOperationExpr_v =
    std::conjunction_v<std::is_base_of<OperationExprImpl, Ts>...>;

template <typename... Ts>
constexpr bool IsTensorExpr_v =
    std::conjunction_v<std::is_base_of<TensorExprImpl, Ts>...>;

} // namespace Cuda

#endif // !_EXPR_H_
