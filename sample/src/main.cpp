#include <iostream>

#include "EtExpr.h"

template <typename V, size_t Dimensions>
void print(std::string const &Name, Tensor<V, Dimensions> &T) {
  std::cout << Name << ": [";
  for (size_t i = 0; i < Dimensions; ++i) {
    std::cout << T.data()[i] << " ";
  }
  std::cout << "]\n";
}

constexpr size_t dim = 25;

int main() {
  Tensor<int, dim> A(5), B(2), C(9), D(9);
  Tensor<float, dim> E(3);

  auto J = A - E;
  auto Q = (A + B) * (C + D - E);
  auto Y = (Q + C) - B;

  auto K = J.eval();
  auto P = Q.eval();
  auto Z = Y.eval();

  print("K", K);
  print("P", P);
  print("Z", Z);

  return 0;
}
