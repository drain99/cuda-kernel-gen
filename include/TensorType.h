#ifndef _TENSOR_TYPE_H_
#define _TENSOR_TYPE_H_

#include <string>
#include <llvm/ADT/SmallVector.h>

namespace Cuda {

struct TensorType {
  std::string DataType;
  llvm::SmallVector<uint32_t, 2> Dimensions;
};

} // namespace Cuda

#endif // !_TENSOR_TYPE_H_
