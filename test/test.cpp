#include <utility>

#define CUDAKERNELGEN [[clang::cuda_kernel_gen]]

template <typename V, size_t... Dimensions> class Tensor {
private:
  V *mData = nullptr; // points to heap memory. (or cuda unified memory maybe)

public:
  Tensor() { mData = new V[100]; }

  using OutputType = Tensor<V, Dimensions...>;

  V *data() { return mData; }
};

template <typename T1, typename T2> struct AddType {};

template <typename V1, typename V2, size_t... Dimensions>
struct AddType<Tensor<V1, Dimensions...>, Tensor<V2, Dimensions...>> {
  using type =
      Tensor<decltype(std::declval<V1>() + std::declval<V2>()), Dimensions...>;
};

template <typename T1, typename T2>
using AddType_t = typename AddType<T1, T2>::type;

template <typename E1, typename E2> class AddExpr {
public:
  E1 mExpr1;
  E2 mExpr2;

public:
  AddExpr(const E1 &expr1, const E2 &expr2) : mExpr1(expr1), mExpr2(expr2) {}

  using ThisType = AddExpr<E1, E2>;

  using OutputType =
      AddType_t<typename E1::OutputType, typename E2::OutputType>;

  CUDAKERNELGEN OutputType eval() {
    OutputType result;
    if (std::is_same_v<
                      ThisType,
                      AddExpr<AddExpr<Tensor<int, 1000>, Tensor<float, 1000>>,
                              Tensor<long, 1000>>>) {
      kernel__0(mExpr1.mExpr1.data(), mExpr1.mExpr2.data(),
                mExpr2.data(), result.data());
      std::cout << "called." << std::endl;
    }
    return result;
  }
};

template <typename E1, typename E2,
          typename = std::void_t<
              AddType_t<typename E1::OutputType, typename E2::OutputType>>>
auto operator+(const E1 &A, const E2 &B) {
  return AddExpr<E1, E2>(A, B);
}

int main() {
  Tensor<int, 1000> A;
  Tensor<float, 1000> B;
  Tensor<long, 1000> C;
  auto X = A + B + C;
  auto y = X.eval();
  return 0;
}