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

  clang::tooling::CommonOptionsParser OptionsParser(argc, argv, MyToolCatagory);

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

  std::cout << "Compiling kernels..." << std::endl;
  std::string NvccCommand = "nvcc -c -O3 -odir=" + CkgFolderPath + "\\obj\\ " +
                            CkgFolderPath + "\\include\\MyKernels.cu " +
                            CkgFolderPath + "\\include\\MyKernelWrappers.cu";
  system(NvccCommand.c_str());

  // successfully created the obj's, now create executable by taking the objs into source.

  // the source files are already known, also the 2 extras are known. all thats required
  // are the compiler flags. get them from the compiler_database

  return 0;
}