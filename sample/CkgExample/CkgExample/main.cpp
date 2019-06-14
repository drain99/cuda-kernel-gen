#include "EtExpr.h"
#include <iostream>

int main() {
  Tensor<int, 100> A(5), B(2), C(9), D(9);
  Tensor<float, 100> E(3);
  auto Q = (A + B) * (C + D - E);
  auto P = Q.eval();
  for (int i = 0; i < 100; i++) {
    std::cout << P.data()[i] << " ";
  }
  std::cout << std::endl;
  auto Y = (Q + C) - B;
  auto Z = Y.eval();
  for (int i = 0; i < 100; i++) {
    std::cout << Z.data()[i] << " ";
  }
  std::cout << std::endl;
  return 0;
}
