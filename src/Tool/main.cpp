#include <iostream>
#include <sstream>

#include "ASTContext.h"
#include "KernelManager.h"
#include "TemplateParser.h"

using namespace Cuda;

int main() {
  TemplateParser TP("SubtractExpr<AddExpr<Tensor<int,101,10>,Tensor<int,101,10>>,Tensor<int,101,10>>",
                    "Tensor<int,101,10>");

  ASTContext C = TP.createAST();
  TensorType OT = TP.createOutputTensorType();

  KernelManager KM;
  auto &K = KM.createNewKernel(C, OT);

  std::cout << KM.getKernelCallStr(K) << std::endl
            << KM.getKernelDeclStr(K) << std::endl
            << KM.getKernelDefStr(K) << std::endl;

  return 0;
}