#include <cstdint>
#include <fstream>
#include <llvm-14/llvm/Support/raw_ostream.h>

#include "KernelWriter.h"

namespace ckg {

KernelWriter::KernelWriter(const fs::path &rootDir, const fs::path &pluginDir,
                           KernelManager &kernelManager)
    : mRootDir(rootDir), mPluginDir(pluginDir),
      mKernelManager(std::move(kernelManager)) {}

void KernelWriter::writeKernelsHeader() const {
  std::ofstream Ofs(mRootDir / "generated" / "kernels.cuh", std::ios::out);
  for (uint32_t I = 0; I < mKernelManager.size(); ++I) {
    mKernelManager.getKernelDeclStr(I, Ofs);
    Ofs << '\n';
  }
  Ofs.close();
}

void KernelWriter::writeKernelWrappersHeader() const {
  std::ofstream Ofs(mRootDir / "generated" / "kernel_wrappers.h",
                    std::ios::out);
  for (uint32_t I = 0; I < mKernelManager.size(); ++I) {
    mKernelManager.getKernelWrapperDeclStr(I, Ofs);
    Ofs << '\n';
  }
  Ofs.close();
}

void KernelWriter::writeKernelsSource() const {
  std::ofstream Ofs(mRootDir / "generated" / "kernels.cu", std::ios::out);
  Ofs << "#include \"kernels.cuh\"\n";
  for (uint32_t I = 0; I < mKernelManager.size(); ++I) {
    mKernelManager.getKernelDefStr(I, Ofs);
    Ofs << '\n';
  }
  Ofs.close();
}

void KernelWriter::writeKernelWrappersSource() const {
  std::ofstream Ofs(mRootDir / "generated" / "kernel_wrappers.cu",
                    std::ios::out);
  Ofs << "#include \"kernels.cuh\"\n"
      << "#include \"kernel_wrappers.h\"\n";
  for (uint32_t I = 0; I < mKernelManager.size(); ++I) {
    mKernelManager.getKernelWrapperDefStr(I, Ofs);
    Ofs << '\n';
  }
  Ofs.close();
}

void KernelWriter::writeKernelCalls() const {
  std::unordered_map<std::string, std::vector<uint32_t>> CallSpaceToIndexMap;
  for (uint32_t I = 0; I < mKernelManager.size(); ++I) {
    auto const& exprTemplate = mKernelManager.get(I).FullExprTemplate;
    auto Pos = exprTemplate.find("Expr");
    std::string CallSpaceStr = "    /*" + exprTemplate.substr(0, Pos + 4) + " Call Space*/";
    CallSpaceToIndexMap[CallSpaceStr].emplace_back(I);
  }

  std::ifstream Ifs(mPluginDir / "EtExpr.h", std::ios::in);
  std::ofstream Ofs(mRootDir / "generated" / "EtExpr.h", std::ios::out);
  std::string Line;
  bool InsideCallSpace = false;
  while (std::getline(Ifs, Line)) {
    if (Line == "    cudaDeviceSynchronize();") {
      Ofs << Line << '\n';
      InsideCallSpace = false;
    } else if (auto const It = CallSpaceToIndexMap.find(Line); It != CallSpaceToIndexMap.end()) {
      Ofs << Line << '\n';
      InsideCallSpace = true;
      for (auto const I : It->second) {
        Ofs << "    if constexpr (std::is_same_v<std::decay_t<decltype(*this)>," << mKernelManager.get(I).FullExprTemplate << ">) {\n      ";
        mKernelManager.getKernelWrapperCallStr(I, Ofs);
        Ofs << "\n    }\n";
      }
    } else if (!InsideCallSpace) {
      Ofs << Line << '\n';
    }
  }
  Ifs.close();
  Ofs.close();
}

} // namespace ckg