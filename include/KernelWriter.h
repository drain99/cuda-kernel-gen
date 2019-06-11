#ifndef _KERNEL_WRITER_H_
#define _KERNEL_WRITER_H_

#include <string>
#include <vector>

#include "KernelManager.h"

namespace Cuda {

class KernelWriter {
private:
  std::vector<std::string> mUserSources;
  std::string mPathToIncludeDir;
  KernelManager mKernelManager;

public:
  KernelWriter(const std::vector<std::string> &userSources,
               const std::string &pathTOIncludeDir,
               const KernelManager &kernelManager);

  void writeKernelsHeader();

  void writeKernelWrappersHeader();

  void writeKernelsSource();

  void writeKernelWrappersSource();

  void writeKernelCalls();
};

} // namespace Cuda

#endif // !_KERNEL_WRITER_H_
