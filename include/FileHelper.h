#ifndef _FILE_HELPER_H_
#define _FILE_HELPER_H_

#include <filesystem>

namespace fs = std::filesystem;

namespace ckg {

class FileHelper {
public:
  static void initializeRoot(const fs::path &rootDir);
};

}

#endif // !_FILE_HELPER_H_
