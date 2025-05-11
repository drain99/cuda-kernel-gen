#include "KernelContext.h"

namespace ckg {

uint32_t KernelContext::ID = 0;

KernelContext::KernelContext() : FuncID(ID++) {}

} // namespace ckg