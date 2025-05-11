#include <cstdlib>
#include <filesystem>

#include <clang/Tooling/CommonOptionsParser.h>
#include <llvm-14/llvm/Support/raw_ostream.h>
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

static cl::OptionCategory CkgToolCategory("ckg-tool options");
static cl::opt<std::string>
    RootDirStr("root-dir", cl::Required,
               cl::desc("Specify project root directory"),
               cl::cat(CkgToolCategory));
static cl::opt<std::string> PluginDirStr("plugin-dir", cl::Required,
                                         cl::desc("Specify plugin directory"),
                                         cl::cat(CkgToolCategory));
static cl::opt<bool> QuietFlag("q", cl::init(false),
                               cl::desc("Hide progress messages"),
                               cl::cat(CkgToolCategory));

std::vector<std::string> parseSourceFiles(tooling::CommonOptionsParser &OP) {
  if (!QuietFlag)
    llvm::outs() << "Parsing sources...\n";
  std::vector<std::string> templateList;
  SourceParser::parseSources(OP.getSourcePathList(), OP.getCompilations(),
                             templateList);
  return templateList;
}

void removeSpaces(std::string &S) {
  int32_t J = 0;
  for (size_t I = 0; I < S.size(); ++I) {
    if (S[I] != ' ') {
      S[J++] = S[I];
    }
  }
  S.erase(J);
}

KernelManager generateKernels(std::vector<std::string> &templateList) {
  if (!QuietFlag)
    llvm::outs() << "Generating kernels...\n";
  KernelManager KM;
  for (auto &&S : templateList) {
    removeSpaces(S);
    TemplateParser TP(S);
    ASTContext C = TP.createAST();
    KM.createNewKernel(C, S);
  }
  return KM;
}

void writeKernels(KernelManager &KM, const fs::path &rootDir, const fs::path &pluginDir) {
  if (!QuietFlag)
    llvm::outs() << "Writing kernels...\n";
  KernelWriter KW(rootDir, pluginDir, KM);
  KW.writeKernelCalls();
  KW.writeKernelsHeader();
  KW.writeKernelsSource();
  KW.writeKernelWrappersHeader();
  KW.writeKernelWrappersSource();
}

int main(int argc, const char **argv) {
  auto OP = tooling::CommonOptionsParser::create(argc, argv, CkgToolCategory);
  if (!OP) {
    llvm::errs() << OP.takeError() << "\n";
    return 1;
  }

  fs::path const rootDir = fs::canonical(RootDirStr.getValue());
  fs::path const pluginDir = fs::canonical(PluginDirStr.getValue());

  FileHelper::initializeRoot(rootDir);
  auto templateList = parseSourceFiles(*OP);
  auto KM = generateKernels(templateList);
  writeKernels(KM, rootDir, pluginDir);
  // compileKernels(CkgFolder);
  // if (!CompileOnlyFlag) {
  //   compileAndLink(CkgFolder, OP.getSourcePathList());
  // }
  // if (RevertFlag) {
  //   FileHelper::revertOriginals(CkgFolder);
  // }
  // FileHelper::deleteTempFolder(CkgFolder);

  return 0;
}