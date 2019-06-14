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
  for (int32_t I = 0; I < mKernelManager.size(); ++I) {
    std::ifstream Ifs(mPathToIncludeDir + "\\EtExpr.h", std::ios::in);
    std::ofstream Ofs(mPathToIncludeDir + "\\temp_et_expr.txt", std::ios::out);
    auto &exprTemplate = mKernelManager.get(I).FullExprTemplate;
    auto Pos = exprTemplate.find("Expr");
    std::string CallSpaceStr =
        "\t/*" + exprTemplate.substr(0, Pos + 4) + " Call Space*/";
    std::string Line;
    bool FoundCallSpace = false, FoundDeviceSync = false;
    while (std::getline(Ifs, Line)) {
      if (Line == "\tcudaDeviceSynchronize();" && FoundCallSpace) {
        FoundDeviceSync = true;
      }
      if (FoundCallSpace == FoundDeviceSync) {
        Ofs << Line << std::endl;
	  }
      if (Line == CallSpaceStr) {
        FoundCallSpace = true;
        Ofs << "\tif constexpr (std::is_same_v<std::decay_t<decltype(*this)>,"
            << exprTemplate << ">) {\n\t";
        mKernelManager.getKernelWrapperCallStr(I, Ofs);
        Ofs << "\n\t}\n";
      }
    }
    Ifs.close();
    Ofs.close();
    Ifs.open(mPathToIncludeDir + "\\temp_et_expr.txt", std::ios::in);
    Ofs.open(mPathToIncludeDir + "\\EtExpr.h", std::ios::out);
    while (std::getline(Ifs, Line)) {
      Ofs << Line << std::endl;
    }
    Ifs.close();
    Ofs.close();
  }
}

} // namespace Cuda