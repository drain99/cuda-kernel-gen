#ifndef _EXPR_H_
#define _EXPR_H_

#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <type_traits>

#include "TensorType.h"

namespace ckg {

class ASTVisitor;

struct ExprImpl {};
struct OperationExprImpl {};
struct TensorExprImpl {};

template <typename V> using ChildVector = llvm::SmallVector<V, 3>;

class Expr : private ExprImpl {
protected:
  Expr *mParent = nullptr;
  TensorType mType = TensorType();
  ChildVector<Expr *> mChilds;

public:
  Expr(const TensorType &tensorType, Expr *parent = nullptr);
  
  Expr(Expr *parent = nullptr);

  virtual ~Expr();

  virtual void accept(ASTVisitor &V) = 0;

  ChildVector<Expr *> &getChilds();

  Expr *getParent();

  TensorType &getType();

  uint16_t getChildCount() const;

  void addChild(Expr *C);
};

class OperationExpr : public Expr, private OperationExprImpl {
public:
  OperationExpr(const TensorType &tensorType, Expr *parent = nullptr);

  OperationExpr(Expr *parent = nullptr);

  TensorType getInputType(int32_t index);
};

class AddExpr : public OperationExpr {
public:
  AddExpr(const TensorType &tensorType, Expr *parent = nullptr);

  AddExpr(Expr *parent = nullptr);

  virtual void accept(ASTVisitor &V) override;
};

class SubtractExpr : public OperationExpr {
public:
  SubtractExpr(const TensorType &tensorType, Expr *parent = nullptr);

  SubtractExpr(Expr *parent = nullptr);

  virtual void accept(ASTVisitor &V) override;
};

class MultiplyExpr : public OperationExpr {
public:
  MultiplyExpr(const TensorType &tensorType, Expr *parent = nullptr);

  MultiplyExpr(Expr *parent = nullptr);

  virtual void accept(ASTVisitor &V) override;
};

class DivideExpr : public OperationExpr {
public:
  DivideExpr(const TensorType &tensorType, Expr *parent = nullptr);

  DivideExpr(Expr *parent = nullptr);

  virtual void accept(ASTVisitor &V) override;
};

class TensorExpr : public Expr, private TensorExprImpl {
public:
  TensorExpr(const TensorType &tensorType, Expr *parent = nullptr);

  TensorExpr(Expr *parent = nullptr);

  virtual void accept(ASTVisitor &V) override;
};

template <typename... Ts>
constexpr bool IsExpr_v = std::conjunction_v<std::is_base_of<ExprImpl, Ts>...>;

template <typename... Ts>
constexpr bool IsOperationExpr_v =
    std::conjunction_v<std::is_base_of<OperationExprImpl, Ts>...>;

template <typename... Ts>
constexpr bool IsTensorExpr_v =
    std::conjunction_v<std::is_base_of<TensorExprImpl, Ts>...>;

} // namespace ckg

#endif // !_EXPR_H_
