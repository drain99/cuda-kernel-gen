#include <iostream>
#include <sstream>

#include "ASTContext.h"
#include "KernelManager.h"
#include "TemplateParser.h"

using namespace Cuda;

int main() {
  TemplateParser TP("MultiplyExpr<AddExpr<Tensor<int,100>,Tensor<int,100>"
                    ",Tensor<int,100>>,AddExpr<Tensor<int,100>,Tensor<"
                    "int,100>,Tensor<int,100>>,Tensor<int,100>>");

  ASTContext C = TP.createAST();

  KernelManager KM;
  auto I = KM.createNewKernel(C);

  std::stringstream SS;
  KM.getKernelDeclStr(I, SS);
  SS << std::endl;
  KM.getKernelDefStr(I, SS);
  SS << std::endl;
  KM.getKernelWrapperCallStr(I, SS);
  SS << std::endl;
  KM.getKernelWrapperDeclStr(I, SS);
  SS << std::endl;
  KM.getKernelWrapperDefStr(I, SS);
  SS << std::endl;

  std::cout << SS.str() << std::endl;
  
  return 0;
}