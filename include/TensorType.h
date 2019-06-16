#ifndef _TENSOR_TYPE_H_
#define _TENSOR_TYPE_H_

#include <string>
#include <llvm/ADT/SmallVector.h>

namespace ckg {

struct TensorType {
  std::string DataType;
  llvm::SmallVector<uint32_t, 2> Dimensions;
};

} // namespace ckg

#endif // !_TENSOR_TYPE_H_
