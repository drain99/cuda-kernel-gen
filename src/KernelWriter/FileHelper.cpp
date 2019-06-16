#include <fstream>
#include <string>

#include "FileHelper.h"

namespace ckg {

void FileHelper::copyFile(const fs::path &destination, const fs::path &source) {
  std::ifstream Ifs(source, std::ios::in);
  std::ofstream Ofs(destination, std::ios::out);
  std::string S;
  while (std::getline(Ifs, S)) {
    Ofs << S << std::endl;
  }
  Ifs.close();
  Ofs.close();
}

void FileHelper::deleteTempFolder(const fs::path &ckgFolder) {
  fs::remove_all(ckgFolder / "temp");
}

void FileHelper::initiateCkgFolder(const fs::path &ckgFolder) {
  std::ofstream Ofs;
  Ofs.open(ckgFolder / "include" / "MyKernels.h", std::ios::out);
  Ofs.close();
  Ofs.open(ckgFolder / "include" / "MyKernels.cu", std::ios::out);
  Ofs.close();
  Ofs.open(ckgFolder / "include" / "MyKernelWrappers.h", std::ios::out);
  Ofs.close();
  Ofs.open(ckgFolder / "include" / "MyKernelWrappers.cu", std::ios::out);
  Ofs.close();
  fs::create_directory(ckgFolder / "temp");
  fs::create_directory(ckgFolder / "obj");
}

void FileHelper::revertOriginals(const fs::path &ckgFolder) {
  FileHelper::copyFile(ckgFolder / "include" / "EtExpr.h",
                       ckgFolder / "temp" / "OriginalEtExpr.h");
}

} // namespace ckg