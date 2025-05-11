#include <algorithm>
#include <memory>
#include <type_traits>
#include <utility>

#include <cuda_runtime.h>

#if __has_include("kernel_wrappers.h")
#include "kernel_wrappers.h"
#endif

#define CUDA_KERNEL_GEN_ATTR [[clang::annotate("cuda_kernel_gen")]]

template <typename V> V *allocateUnifiedMemory(size_t numOfItems) {
  V *unifiedPtr = nullptr;
  cudaMallocManaged(&unifiedPtr, numOfItems * sizeof(V));
  return unifiedPtr;
}

template <typename V> struct UnifiedMemoryDeleter {
  void operator()(V *dataPtr) const { cudaFree(dataPtr); }
};

template <typename V, size_t Dimensions> class Tensor {
private:
  std::shared_ptr<V> mData;

public:
  Tensor()
      : mData(allocateUnifiedMemory<V>(Dimensions), UnifiedMemoryDeleter<V>{}) {
  }

  Tensor(const V &value) : Tensor() {
    std::fill_n(mData.get(), Dimensions, value);
  }

  Tensor(const Tensor<V, Dimensions> &other) : mData(other.mData) {}

  Tensor(Tensor<V, Dimensions> &&other) : mData(other.mData) {}

  Tensor<V, Dimensions> &operator=(const Tensor<V, Dimensions> &other) {
    this->mData = other.mData;
    return *this;
  }

  Tensor<V, Dimensions> &operator=(Tensor<V, Dimensions> &&other) {
    this->mData = std::move(other.mData);
    return *this;
  }

  using OutputType = Tensor<V, Dimensions>;

  V *data() { return mData.get(); }
};

template <typename T1, typename T2> struct AddType {};

template <typename V1, typename V2, size_t Dimensions>
struct AddType<Tensor<V1, Dimensions>, Tensor<V2, Dimensions>> {
  using type =
      Tensor<decltype(std::declval<V1>() + std::declval<V2>()), Dimensions>;
};

template <typename T1, typename T2>
using AddType_t = typename AddType<T1, T2>::type;

template <typename E1, typename E2,
          typename OT =
              AddType_t<typename E1::OutputType, typename E2::OutputType>>
class AddExpr;

template <typename T1, typename T2> struct SubtractType {};

template <typename V1, typename V2, size_t Dimensions>
struct SubtractType<Tensor<V1, Dimensions>, Tensor<V2, Dimensions>> {
  using type =
      Tensor<decltype(std::declval<V1>() - std::declval<V2>()), Dimensions>;
};

template <typename T1, typename T2>
using SubtractType_t = typename SubtractType<T1, T2>::type;

template <typename E1, typename E2,
          typename OT =
              SubtractType_t<typename E1::OutputType, typename E2::OutputType>>
class SubtractExpr;

template <typename T1, typename T2> struct MultiplyType {};

template <typename V1, typename V2, size_t Dimensions>
struct MultiplyType<Tensor<V1, Dimensions>, Tensor<V2, Dimensions>> {
  using type =
      Tensor<decltype(std::declval<V1>() * std::declval<V2>()), Dimensions>;
};

template <typename T1, typename T2>
using MultiplyType_t = typename MultiplyType<T1, T2>::type;

template <typename E1, typename E2,
          typename OT =
              MultiplyType_t<typename E1::OutputType, typename E2::OutputType>>
class MultiplyExpr;

template <typename T1, typename T2> struct DivideType {};

template <typename V1, typename V2, size_t Dimensions>
struct DivideType<Tensor<V1, Dimensions>, Tensor<V2, Dimensions>> {
  using type =
      Tensor<decltype(std::declval<V1>() + std::declval<V2>()), Dimensions>;
};

template <typename T1, typename T2>
using DivideType_t = typename DivideType<T1, T2>::type;

template <typename E1, typename E2,
          typename OT =
              DivideType_t<typename E1::OutputType, typename E2::OutputType>>
class DivideExpr;

template <typename E1, typename E2, typename OT> class AddExpr {
public:
  E1 mExpr1;
  E2 mExpr2;

public:
  template <typename _E1, typename _E2>
  AddExpr(_E1 &&expr1, _E2 &&expr2)
      : mExpr1(std::forward<_E1>(expr1)), mExpr2(std::forward<_E2>(expr2)) {}

  using OutputType = OT;

  CUDA_KERNEL_GEN_ATTR OT eval() {
    OT result(0);
    /*AddExpr Call Space*/
    cudaDeviceSynchronize();
    return result;
  }
};

template <typename E1, typename E2>
AddExpr(E1 &&, E2 &&) -> AddExpr<std::decay_t<E1>, std::decay_t<E2>>;

template <
    typename E1, typename E2,
    typename = std::void_t<AddType_t<typename std::decay_t<E1>::OutputType,
                                     typename std::decay_t<E2>::OutputType>>>
auto operator+(E1 &&A, E2 &&B) {
  return AddExpr(std::forward<E1>(A), std::forward<E2>(B));
}

template <typename E1, typename E2, typename OT> class SubtractExpr {
public:
  E1 mExpr1;
  E2 mExpr2;

public:
  template <typename _E1, typename _E2>
  SubtractExpr(_E1 &&expr1, _E2 &&expr2)
      : mExpr1(std::forward<_E1>(expr1)), mExpr2(std::forward<_E2>(expr2)) {}

  using OutputType = OT;

  CUDA_KERNEL_GEN_ATTR OT eval() {
    OT result(0);
    /*SubtractExpr Call Space*/
    if constexpr (std::is_same_v<std::decay_t<decltype(*this)>,SubtractExpr<Tensor<int,25>,Tensor<float,25>,Tensor<float,25>>>) {
      kernel_wrapper__0((*this).mExpr1.data(), (*this).mExpr2.data(), result.data());
    }
    if constexpr (std::is_same_v<std::decay_t<decltype(*this)>,SubtractExpr<AddExpr<MultiplyExpr<AddExpr<Tensor<int,25>,Tensor<int,25>,Tensor<int,25>>,SubtractExpr<AddExpr<Tensor<int,25>,Tensor<int,25>,Tensor<int,25>>,Tensor<float,25>,Tensor<float,25>>,Tensor<float,25>>,Tensor<int,25>,Tensor<float,25>>,Tensor<int,25>,Tensor<float,25>>>) {
      kernel_wrapper__2((*this).mExpr1.mExpr1.mExpr1.mExpr1.data(), (*this).mExpr1.mExpr1.mExpr1.mExpr2.data(), (*this).mExpr1.mExpr1.mExpr2.mExpr1.mExpr1.data(), (*this).mExpr1.mExpr1.mExpr2.mExpr1.mExpr2.data(), (*this).mExpr1.mExpr1.mExpr2.mExpr2.data(), (*this).mExpr1.mExpr2.data(), (*this).mExpr2.data(), result.data());
    }
    cudaDeviceSynchronize();
    return result;
  }
};

template <typename E1, typename E2>
SubtractExpr(E1 &&, E2 &&) -> SubtractExpr<std::decay_t<E1>, std::decay_t<E2>>;

template <typename E1, typename E2,
          typename = std::void_t<
              SubtractType_t<typename std::decay_t<E1>::OutputType,
                             typename std::decay_t<E2>::OutputType>>>
auto operator-(E1 &&A, E2 &&B) {
  return SubtractExpr(std::forward<E1>(A), std::forward<E2>(B));
}

template <typename E1, typename E2, typename OT> class MultiplyExpr {
public:
  E1 mExpr1;
  E2 mExpr2;

public:
  template <typename _E1, typename _E2>
  MultiplyExpr(_E1 &&expr1, _E2 &&expr2)
      : mExpr1(std::forward<_E1>(expr1)), mExpr2(std::forward<_E2>(expr2)) {}

  using OutputType = OT;

  CUDA_KERNEL_GEN_ATTR OT eval() {
    OT result(0);
    /*MultiplyExpr Call Space*/
    if constexpr (std::is_same_v<std::decay_t<decltype(*this)>,MultiplyExpr<AddExpr<Tensor<int,25>,Tensor<int,25>,Tensor<int,25>>,SubtractExpr<AddExpr<Tensor<int,25>,Tensor<int,25>,Tensor<int,25>>,Tensor<float,25>,Tensor<float,25>>,Tensor<float,25>>>) {
      kernel_wrapper__1((*this).mExpr1.mExpr1.data(), (*this).mExpr1.mExpr2.data(), (*this).mExpr2.mExpr1.mExpr1.data(), (*this).mExpr2.mExpr1.mExpr2.data(), (*this).mExpr2.mExpr2.data(), result.data());
    }
    cudaDeviceSynchronize();
    return result;
  }
};

template <typename E1, typename E2>
MultiplyExpr(E1 &&, E2 &&) -> MultiplyExpr<std::decay_t<E1>, std::decay_t<E2>>;

template <typename E1, typename E2,
          typename = std::void_t<
              MultiplyType_t<typename std::decay_t<E1>::OutputType,
                             typename std::decay_t<E2>::OutputType>>>
auto operator*(E1 &&A, E2 &&B) {
  return MultiplyExpr(std::forward<E1>(A), std::forward<E2>(B));
}

template <typename E1, typename E2, typename OT> class DivideExpr {
public:
  E1 mExpr1;
  E2 mExpr2;

public:
  template <typename _E1, typename _E2>
  DivideExpr(_E1 &&expr1, _E2 &&expr2)
      : mExpr1(std::forward<_E1>(expr1)), mExpr2(std::forward<_E2>(expr2)) {}

  using OutputType = OT;

  CUDA_KERNEL_GEN_ATTR OT eval() {
    OT result(0);
    /*DivideExpr Call Space*/
    cudaDeviceSynchronize();
    return result;
  }
};

template <typename E1, typename E2>
DivideExpr(E1 &&, E2 &&) -> DivideExpr<std::decay_t<E1>, std::decay_t<E2>>;

template <
    typename E1, typename E2,
    typename = std::void_t<DivideType_t<typename std::decay_t<E1>::OutputType,
                                        typename std::decay_t<E2>::OutputType>>>
auto operator/(E1 &&A, E2 &&B) {
  return DivideExpr(std::forward<E1>(A), std::forward<E2>(B));
}
