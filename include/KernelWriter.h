#ifndef _KERNEL_WRITER_H_
#define _KERNEL_WRITER_H_

#include <filesystem>
#include <string>
#include <vector>

#include "KernelManager.h"

namespace fs = std::filesystem;

namespace ckg {

class KernelWriter {
private:
  fs::path const mRootDir;
  fs::path const mPluginDir;
  KernelManager const mKernelManager;

public:
  KernelWriter(const fs::path &rootDir, const fs::path &pluginDir,
               KernelManager &kernelManager);

  void writeKernelsHeader() const;

  void writeKernelWrappersHeader() const;

  void writeKernelsSource() const;

  void writeKernelWrappersSource() const;

  void writeKernelCalls() const;
};

} // namespace ckg

#endif // !_KERNEL_WRITER_H_
