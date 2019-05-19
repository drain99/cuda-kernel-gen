#include <utility>

#define CUDAKERNELGEN [[clang::cuda_kernel_gen]] 

template <typename V, size_t... Dimensions>
class Tensor
{
public:
	Tensor() {}
};

template <typename V1, typename V2, size_t... Dimensions>
auto operator+(Tensor<V1, Dimensions...> const&,
			   Tensor<V2, Dimensions...> const&)
{
	using V = decltype(std::declval<V1>() + std::declval<V2>());
	return Tensor<V, Dimensions...>();
}

template <typename V>
class ConstExpr
{
public:
	using OutputType = V;
};

template <typename E1, typename E2>
class AddExpr
{
public:
	using OutputType = decltype(std::declval<typename E1::OutputType>() + std::declval<typename E2::OutputType>());
public:
	AddExpr(E1 const&, E2 const&) {}
	CUDAKERNELGEN OutputType eval() { return OutputType(); }
};


int main()
{
	ConstExpr<Tensor<long double, 5, 5>> x;
	ConstExpr<Tensor<double, 5, 5>> y;

	AddExpr<decltype(x), decltype(y)> z(x, y);
	z.eval();

	return 0;
}