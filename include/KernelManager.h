#ifndef _KERNEL_MANAGER_H_
#define _KERNEL_MANAGER_H_

#include <vector>
#include <string>

#include "KernelContext.h"
#include "ASTContext.h"

namespace Cuda {

class KernelManager {
private:
  std::vector<KernelContext> mKernels;

public:
  KernelContext &createNewKernel(ASTContext &C);

  KernelContext &get(uint32_t I);

  uint32_t size() const;

  std::string getKernelCallStr(const KernelContext &K) const;

  std::string getKernelDeclStr(const KernelContext &K) const;

  std::string getKernelDefStr(const KernelContext &K) const;

private:
  uint32_t getNumOfElems(const TensorType &T);
};

} // namespace Cuda

#endif // !_KERNEL_MANAGER_H_
