#include <iostream>
#include <sstream>

#include <llvm/Support/CommandLine.h>
#include <clang/Tooling/CommonOptionsParser.h>

#include "ASTContext.h"
#include "KernelManager.h"
#include "TemplateParser.h"
#include "SourceParser.h"

using namespace Cuda;

static llvm::cl::OptionCategory MyCustomToolCatagory("my-custom-tool options");

int main(int argc, const char **argv) {

  clang::tooling::CommonOptionsParser OptionsParser(argc, argv,
                                                    MyCustomToolCatagory);

  std::vector<std::string> templateList;

  SourceParser::parseSources(OptionsParser.getSourcePathList(),
                             OptionsParser.getCompilations(), templateList);

  for (auto &&S : templateList) {
    int32_t J = 0;
    for (int32_t I = 0; I < S.size(); ++I) {
      if (S[I] != ' ') {
        S[J++] = S[I];
      }
    }
    S.erase(J);
    TemplateParser TP(S);
    ASTContext C = TP.createAST();
    KernelManager KM;
    auto I = KM.createNewKernel(C, S);
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
  }

  return 0;
}