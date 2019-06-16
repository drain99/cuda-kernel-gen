#include <fstream>

#include "KernelWriter.h"
#include "FileHelper.h"

namespace ckg {

KernelWriter::KernelWriter(const fs::path &ckgFolder,
                           const KernelManager &kernelManager)
    : mCkgFolder(ckgFolder), mKernelManager(kernelManager) {}

void KernelWriter::writeKernelsHeader() const {
  std::ofstream Ofs(mCkgFolder / "include" / "MyKernels.h", std::ios::out);
  for (int32_t I = 0; I < mKernelManager.size(); ++I) {
    mKernelManager.getKernelDeclStr(I, Ofs);
    Ofs << std::endl;
  }
  Ofs.close();
}

void KernelWriter::writeKernelWrappersHeader() const {
  std::ofstream Ofs(mCkgFolder / "include" / "MyKernelWrappers.h",
                    std::ios::out);
  for (int32_t I = 0; I < mKernelManager.size(); ++I) {
    mKernelManager.getKernelWrapperDeclStr(I, Ofs);
    Ofs << std::endl;
  }
  Ofs.close();
}

void KernelWriter::writeKernelsSource() const {
  std::ofstream Ofs(mCkgFolder / "include" / "MyKernels.cu", std::ios::out);
  Ofs << "#include \"MyKernels.h\"\n";
  for (int32_t I = 0; I < mKernelManager.size(); ++I) {
    mKernelManager.getKernelDefStr(I, Ofs);
    Ofs << std::endl;
  }
  Ofs.close();
}

void KernelWriter::writeKernelWrappersSource() const {
  std::ofstream Ofs(mCkgFolder / "include" / "MyKernelWrappers.cu",
                    std::ios::out);
  Ofs << "#include \"MyKernels.h\"\n"
      << "#include \"MyKernelWrappers.h\"\n";
  for (int32_t I = 0; I < mKernelManager.size(); ++I) {
    mKernelManager.getKernelWrapperDefStr(I, Ofs);
    Ofs << std::endl;
  }
  Ofs.close();
}

void KernelWriter::writeKernelCalls() const {
  FileHelper::copyFile(mCkgFolder / "temp" / "OriginalEtExpr.h",
                       mCkgFolder / "include" / "EtExpr.h");
  for (int32_t I = 0; I < mKernelManager.size(); ++I) {
    std::ifstream Ifs(mCkgFolder / "include" / "EtExpr.h", std::ios::in);
    std::ofstream Ofs(mCkgFolder / "temp" / "BackupEtExpr.h", std::ios::out);
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
    Ifs.open(mCkgFolder / "temp" / "BackupEtExpr.h", std::ios::in);
    Ofs.open(mCkgFolder / "include" / "EtExpr.h", std::ios::out);
    while (std::getline(Ifs, Line)) {
      Ofs << Line << std::endl;
    }
    Ifs.close();
    Ofs.close();
  }
}

} // namespace ckg