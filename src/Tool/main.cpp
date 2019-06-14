#include <iostream>
#include <sstream>

#include <clang/Tooling/CommonOptionsParser.h>
#include <llvm/Support/CommandLine.h>

#include "ASTContext.h"
#include "KernelManager.h"
#include "KernelWriter.h"
#include "SourceParser.h"
#include "TemplateParser.h"

using namespace Cuda;

static llvm::cl::OptionCategory MyToolCatagory("CudaKernelGen options");
static llvm::cl::opt<std::string> CkgFolderPath(
    "ckg-path", llvm::cl::Required,
    llvm::cl::desc("Specify top-level ckg folder"),
    llvm::cl::value_desc("ckg-folder-path"), llvm::cl::cat(MyToolCatagory));

int main(int argc, const char **argv) {

  std::vector<std::string> MyArgsVec;
  MyArgsVec.emplace_back(argv[0]);
  MyArgsVec.emplace_back("-extra-arg=-IC:\\Program Files\\NVIDIA GPU Computing "
                      "Toolkit\\CUDA\\v10.1\\include");
  MyArgsVec.emplace_back("-extra-arg=-std=c++17");
  MyArgsVec.emplace_back("-cuda-path=\"C:\\Program Files\\NVIDIA GPU Computing "
                         "Toolkit\\CUDA\\v10.1\\include\"");
  MyArgsVec.emplace_back("-source-path=\"E:\\Desktop\\CudaKernelGen\\sample\"");
  MyArgsVec.emplace_back("main.cpp");

  int MyArgsC = MyArgsVec.size();

  const char **MyArgsv = new const char *[MyArgsC];
  for (int32_t I = 0; I < MyArgsC; ++I) {
    MyArgsv[I] = MyArgsVec[I].c_str();
  }

  clang::tooling::CommonOptionsParser OptionsParser(MyArgsC, MyArgsv,
                                                    MyToolCatagory);

  std::vector<std::string> templateList;
  KernelManager KM;

  std::cout << "Parsing sources..." << std::endl;
  SourceParser::parseSources(OptionsParser.getSourcePathList(),
                             OptionsParser.getCompilations(), templateList);

  std::cout << "Generating kernels..." << std::endl;
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
    KM.createNewKernel(C, S);
  }

  std::cout << "Writing kernels..." << std::endl;
  KernelWriter KW(OptionsParser.getSourcePathList(),
                  CkgFolderPath.getValue() + "\\include", KM);

  KW.writeKernelCalls();
  KW.writeKernelsHeader();
  KW.writeKernelsSource();
  KW.writeKernelWrappersHeader();
  KW.writeKernelWrappersSource();

  return 0;
}