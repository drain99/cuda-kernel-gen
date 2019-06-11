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
  uint32_t createNewKernel(ASTContext &C, const std::string &exprTemplate);

  KernelContext &get(uint32_t I);

  const KernelContext &get(uint32_t I) const;

  uint32_t size() const;

  void getKernelCallStr(uint32_t I, std::ostream& OS) const;

  void getKernelDeclStr(uint32_t I, std::ostream &OS) const;

  void getKernelDefStr(uint32_t I, std::ostream &OS) const;

  void getKernelNameStr(uint32_t I, std::ostream &OS) const;

  void getKernelWrapperNameStr(uint32_t I, std::ostream &OS) const;

  void getKernelWrapperCallStr(uint32_t I, std::ostream &OS) const;

  void getKernelWrapperDeclStr(uint32_t I, std::ostream &OS) const;

  void getKernelWrapperDefStr(uint32_t I, std::ostream &OS) const;

private:
  uint32_t getNumOfElems(const TensorType &T) const;
};

} // namespace Cuda

#endif // !_KERNEL_MANAGER_H_
