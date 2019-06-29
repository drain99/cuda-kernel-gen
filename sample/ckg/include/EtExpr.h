#include <algorithm>
#include <cuda_runtime.h>
#include <memory>
#include <type_traits>
#include <utility>

#include "MyKernelWrappers.h"

#define CUDAKERNELGEN [[clang::cuda_kernel_gen]]

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
    this->mData = other.data;
    return *this;
  }

  Tensor<V, Dimensions> &operator=(Tensor<V, Dimensions> &&other) {
    this->mData = other.data;
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
class AddExpr {
public:
  E1 mExpr1;
  E2 mExpr2;

public:
  template <typename _E1, typename _E2>
  AddExpr(_E1 &&expr1, _E2 &&expr2)
      : mExpr1(std::forward<_E1>(expr1)), mExpr2(std::forward<_E2>(expr2)) {}

  using OutputType = OT;

  CUDAKERNELGEN OT eval() {
    OT result(0);
	/*AddExpr Call Space*/
	cudaDeviceSynchronize();
    return result;
  }
};

template <typename E1, typename E2>
AddExpr(E1 &&, E2 &&)->AddExpr<std::decay_t<E1>, std::decay_t<E2>>;

template <
    typename E1, typename E2,
    typename = std::void_t<AddType_t<typename std::decay_t<E1>::OutputType,
                                     typename std::decay_t<E2>::OutputType>>>
auto operator+(E1 &&A, E2 &&B) {
  return AddExpr(std::forward<E1>(A), std::forward<E2>(B));
}

template <typename T1, typename T2> struct SubtractType {};

template <typename V1, typename V2, size_t Dimensions>
struct SubtractType<Tensor<V1, Dimensions>, Tensor<V2, Dimensions>> {
  using type =
      Tensor<decltype(std::declval<V1>() + std::declval<V2>()), Dimensions>;
};

template <typename T1, typename T2>
using SubtractType_t = typename SubtractType<T1, T2>::type;

template <typename E1, typename E2,
          typename OT =
              SubtractType_t<typename E1::OutputType, typename E2::OutputType>>
class SubtractExpr {
public:
  E1 mExpr1;
  E2 mExpr2;

public:
  template <typename _E1, typename _E2>
  SubtractExpr(_E1 &&expr1, _E2 &&expr2)
      : mExpr1(std::forward<_E1>(expr1)), mExpr2(std::forward<_E2>(expr2)) {}

  using OutputType = OT;

  CUDAKERNELGEN OT eval() {
    OT result(0);
	/*SubtractExpr Call Space*/
	cudaDeviceSynchronize();
    return result;
  }
};

template <typename E1, typename E2>
SubtractExpr(E1 &&, E2 &&)->SubtractExpr<std::decay_t<E1>, std::decay_t<E2>>;

template <typename E1, typename E2,
          typename = std::void_t<
              SubtractType_t<typename std::decay_t<E1>::OutputType,
                             typename std::decay_t<E2>::OutputType>>>
auto operator-(E1 &&A, E2 &&B) {
  return SubtractExpr(std::forward<E1>(A), std::forward<E2>(B));
}

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
class MultiplyExpr {
public:
  E1 mExpr1;
  E2 mExpr2;

public:
  template <typename _E1, typename _E2>
  MultiplyExpr(_E1 &&expr1, _E2 &&expr2)
      : mExpr1(std::forward<_E1>(expr1)), mExpr2(std::forward<_E2>(expr2)) {}

  using OutputType = OT;

  CUDAKERNELGEN OT eval() {
    OT result(0);
	/*MultiplyExpr Call Space*/
	cudaDeviceSynchronize();
    return result;
  }
};

template <typename E1, typename E2>
MultiplyExpr(E1 &&, E2 &&)->MultiplyExpr<std::decay_t<E1>, std::decay_t<E2>>;

template <typename E1, typename E2,
          typename = std::void_t<
              MultiplyType_t<typename std::decay_t<E1>::OutputType,
                             typename std::decay_t<E2>::OutputType>>>
auto operator*(E1 &&A, E2 &&B) {
  return MultiplyExpr(std::forward<E1>(A), std::forward<E2>(B));
}

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
class DivideExpr {
public:
  E1 mExpr1;
  E2 mExpr2;

public:
  template <typename _E1, typename _E2>
  DivideExpr(_E1 &&expr1, _E2 &&expr2)
      : mExpr1(std::forward<_E1>(expr1)), mExpr2(std::forward<_E2>(expr2)) {}

  using OutputType = OT;

  CUDAKERNELGEN OT eval() {
    OT result(0);
	/*DivideExpr Call Space*/
	cudaDeviceSynchronize();
    return result;
  }
};

template <typename E1, typename E2>
DivideExpr(E1 &&, E2 &&)->DivideExpr<std::decay_t<E1>, std::decay_t<E2>>;

template <
    typename E1, typename E2,
    typename = std::void_t<DivideType_t<typename std::decay_t<E1>::OutputType,
                                        typename std::decay_t<E2>::OutputType>>>
auto operator/(E1 &&A, E2 &&B) {
  return DivideExpr(std::forward<E1>(A), std::forward<E2>(B));
}
