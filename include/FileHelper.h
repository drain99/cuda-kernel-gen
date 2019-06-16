#ifndef _FILE_HELPER_H_
#define _FILE_HELPER_H_

#include <filesystem>

namespace fs = std::filesystem;

namespace ckg {

class FileHelper {
public:
  static void copyFile(const fs::path &destination, const fs::path &source);

  static void deleteTempFolder(const fs::path &ckgFolder);

  static void initiateCkgFolder(const fs::path &ckgFolder);

  static void revertOriginals(const fs::path &ckgFolder);
};

}

#endif // !_FILE_HELPER_H_
