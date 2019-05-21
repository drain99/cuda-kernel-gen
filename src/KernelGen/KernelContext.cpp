#include "KernelContext.h"

namespace Cuda {

uint32_t KernelContext::ID = 0;

KernelContext::KernelContext() : FuncID(ID++) {}

} // namespace Cuda