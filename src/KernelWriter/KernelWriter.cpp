#include <fstream>

#include "KernelWriter.h"

namespace Cuda {

KernelWriter::KernelWriter(const std::vector<std::string> &userSources,
                           const std::string &pathTOIncludeDir,
                           const KernelManager &kernelManager)
    : mUserSources(userSources), mPathToIncludeDir(pathTOIncludeDir),
      mKernelManager(kernelManager) {}

void KernelWriter::writeKernelsHeader() {
  std::ofstream Ofs(mPathToIncludeDir + "\\MyKernels.h", std::ios::out);
  for (int32_t I = 0; I < mKernelManager.size(); ++I) {
    mKernelManager.getKernelDeclStr(I, Ofs);
    Ofs << std::endl;
  }
  Ofs.close();
}

void KernelWriter::writeKernelWrappersHeader() {
  std::ofstream Ofs(mPathToIncludeDir + "\\MyKernelWrappers.h", std::ios::out);
  for (int32_t I = 0; I < mKernelManager.size(); ++I) {
    mKernelManager.getKernelWrapperDeclStr(I, Ofs);
    Ofs << std::endl;
  }
  Ofs.close();
}

void KernelWriter::writeKernelsSource() {
  std::ofstream Ofs(mPathToIncludeDir + "\\MyKernels.cu", std::ios::out);
  Ofs << "#include \"MyKernels.h\"\n";
  for (int32_t I = 0; I < mKernelManager.size(); ++I) {
    mKernelManager.getKernelDefStr(I, Ofs);
    Ofs << std::endl;
  }
  Ofs.close();
}

void KernelWriter::writeKernelWrappersSource() {
  std::ofstream Ofs(mPathToIncludeDir + "\\MyKernelWrappers.cu", std::ios::out);
  Ofs << "#include \"MyKernels.h\"\n"
      << "#include \"MyKernelWrappers.h\"\n";
  for (int32_t I = 0; I < mKernelManager.size(); ++I) {
    mKernelManager.getKernelWrapperDefStr(I, Ofs);
    Ofs << std::endl;
  }
  Ofs.close();
}

void KernelWriter::writeKernelCalls() {
  // write to a temprary file until required piece of is found.
  // use a if constexpr block for each call which checks the full type of the
  // object using std::decay_t<decltype(*this)>. so need to store the
  // exprTemplateStr with each kernel to uniquely identify it.
  std::stringstream SS;
  std::fstream Ofs(mPathToIncludeDir + "\\EtExpr.h",
                   std::ios::out | std::ios::in);
  // clang ast gives the line and column, so that can be used to reach the position.
}

} // namespace Cuda