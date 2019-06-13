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
}

} // namespace Cuda