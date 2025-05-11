#include <filesystem>
#include <fstream>

#include "FileHelper.h"

namespace ckg {

void FileHelper::initializeRoot(const fs::path &rootDir) {
  auto const genDir = rootDir / "generated";
  fs::remove_all(genDir);
  fs::create_directories(genDir);
  std::ofstream Ofs;
  Ofs.open(genDir / "EtExpr.h", std::ios::out);
  Ofs.close();
  Ofs.open(genDir / "kernels.cuh", std::ios::out);
  Ofs.close();
  Ofs.open(genDir / "kernels.cu", std::ios::out);
  Ofs.close();
  Ofs.open(genDir / "kernel_wrappers.h", std::ios::out);
  Ofs.close();
  Ofs.open(genDir / "kernel_wrappers.cu", std::ios::out);
  Ofs.close();
}

} // namespace ckg