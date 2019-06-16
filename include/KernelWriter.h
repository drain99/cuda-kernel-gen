#ifndef _KERNEL_WRITER_H_
#define _KERNEL_WRITER_H_

#include <string>
#include <vector>
#include <filesystem>

#include "KernelManager.h"

namespace fs = std::filesystem;

namespace ckg {

class KernelWriter {
private:
  fs::path mCkgFolder;
  KernelManager mKernelManager;

public:
  KernelWriter(const fs::path& ckgFolder,
               const KernelManager &kernelManager);

  void writeKernelsHeader() const;

  void writeKernelWrappersHeader() const;

  void writeKernelsSource() const;

  void writeKernelWrappersSource() const;

  void writeKernelCalls() const;
};

} // namespace ckg

#endif // !_KERNEL_WRITER_H_
