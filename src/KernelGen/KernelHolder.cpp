#include "KernelHolder.h"

namespace Cuda
{

	int KernelHolder::ID = 0;

	KernelHolder::KernelHolder() : FuncID(ID++) {}

}