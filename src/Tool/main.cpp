#include <iostream>
#include <sstream>

#include "ASTContext.h"
#include "KernelManager.h"

using namespace Cuda;

int main() {
  std::string fullTemplate = "AddExpr<AddExpr<Tensor<int,1000>,Tensor<float,1000>>,Tensor<long,1000>>";

  ASTContext C = ASTContext(ExprOf<AddExpr>());

  auto Add1 = C.addNewExpr<AddExpr>(C.getRootExpr());

  C.addNewExpr<TensorExpr>(Add1, TensorType{"int", {1000}});
  C.addNewExpr<TensorExpr>(Add1, TensorType{"float", {1000}});
  C.addNewExpr<TensorExpr>(C.getRootExpr(), TensorType{"long", {1000}});

  KernelManager KM;
  auto &K = KM.createNewKernel(C, TensorType{"int", {100}});

  std::cout << KM.getKernelCallStr(K) << std::endl
            << KM.getKernelDeclStr(K) << std::endl
            << KM.getKernelDefStr(K) << std::endl;

  return 0;
}