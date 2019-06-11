#include "include/EtExpr.h"
#include <iostream>

int main() {
  Tensor<int, 100> A(5), B(2), C(9), D(9);
  auto X = (A + B) * (C + D);
  auto Y = X.eval();
  for (int i = 0; i < 100; i++) {
    std::cout << Y.data()[i] << " ";
  }
  return 0;
}
