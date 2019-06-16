#include <filesystem>
#include <iostream>
#include <sstream>

#include <clang/Tooling/CommonOptionsParser.h>
#include <llvm/Support/CommandLine.h>

#include "ASTContext.h"
#include "FileHelper.h"
#include "KernelManager.h"
#include "KernelWriter.h"
#include "SourceParser.h"
#include "TemplateParser.h"

using namespace ckg;
namespace cl = llvm::cl;
namespace tooling = clang::tooling;

std::vector<std::string> parseSourceFiles(tooling::CommonOptionsParser &OP) {
  std::cout << "Parsing sources..." << std::endl;
  std::vector<std::string> templateList;
  SourceParser::parseSources(OP.getSourcePathList(), OP.getCompilations(),
                             templateList);
  return templateList;
}

void removeSpaces(std::string &S) {
  int32_t J = 0;
  for (int32_t I = 0; I < S.size(); ++I) {
    if (S[I] != ' ') {
      S[J++] = S[I];
    }
  }
  S.erase(J);
}

KernelManager generateKernels(std::vector<std::string> &templateList) {
  std::cout << "Generating kernels..." << std::endl;
  KernelManager KM;
  for (auto &&S : templateList) {
    removeSpaces(S);
    TemplateParser TP(S);
    ASTContext C = TP.createAST();
    KM.createNewKernel(C, S);
  }
  return KM;
}

void writeKernels(KernelManager &KM, const fs::path &CkgFolder) {
  std::cout << "Writing kernels..." << std::endl;
  KernelWriter KW(CkgFolder, KM);
  KW.writeKernelCalls();
  KW.writeKernelsHeader();
  KW.writeKernelsSource();
  KW.writeKernelWrappersHeader();
  KW.writeKernelWrappersSource();
}

void compileKernels(const fs::path &ckgFolder) {
  std::cout << "Compiling kernels..." << std::endl;
  std::string NvccCommand =
      "nvcc -c -O2 -odir=" + (ckgFolder / "obj").string() + " " +
      (ckgFolder / "include" / "MyKernels.cu").string() + " " +
      (ckgFolder / "include" / "MyKernelWrappers.cu").string();
  system(NvccCommand.c_str());
}

static cl::OptionCategory MyToolCatagory("CudaKernelGen options");
static cl::opt<std::string>
    CkgFolderStr("ckg-path", cl::Required,
                 cl::desc("Specify top-level ckg folder"),
                 cl::value_desc("ckg-folder-path"), cl::cat(MyToolCatagory));

int main(int argc, const char **argv) {
  tooling::CommonOptionsParser OP(argc, argv, MyToolCatagory);
  fs::path CkgFolder = fs::canonical(CkgFolderStr.getValue());
  
  FileHelper::initiateCkgFolder(CkgFolder);
  auto templateList = parseSourceFiles(OP);
  auto KM = generateKernels(templateList);
  writeKernels(KM, CkgFolder);
  compileKernels(CkgFolder);
  FileHelper::revertOriginals(CkgFolder);
  FileHelper::deleteTempFolder(CkgFolder);

  return 0;
}