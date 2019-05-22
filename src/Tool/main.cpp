#include <iostream>
#include <sstream>

#include "ASTContext.h"
#include "KernelManager.h"
#include "TemplateParser.h"

using namespace Cuda;

int main() {
  TemplateParser TP("MultiplyExpr<AddExpr<Tensor<int,10,10>,Tensor<float,10,10>"
                    ",Tensor<float,10,10>>,AddExpr<Tensor<int,10,10>,Tensor<"
                    "float,10,10>,Tensor<float,10,10>>,Tensor<float,10,10>>");

  ASTContext C = TP.createAST();

  KernelManager KM;
  auto &K = KM.createNewKernel(C);

  std::cout << KM.getKernelCallStr(K) << std::endl
            << KM.getKernelDeclStr(K) << std::endl
            << KM.getKernelDefStr(K) << std::endl;

  return 0;
}