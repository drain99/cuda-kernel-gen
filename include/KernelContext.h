#ifndef _KERNEL_CONTEXT_H_
#define _KERNEL_CONTEXT_H_

#include <vector>

#include "TensorType.h"

namespace ckg {

struct KernelContext {
public:
  std::vector<std::string> Attributes;
  uint16_t FuncID;
  std::vector<TensorType> InputTensorTypes;
  std::vector<std::string> InputParams;
  std::vector<std::string> InputArgs;
  TensorType OutputTensorType;
  std::string OutputParam;
  std::string OutputArg;
  std::string IndexCalc;
  std::string IndexCond;
  std::string KernelExpr;
  std::string FullExprTemplate;

  KernelContext();

private:
  static uint32_t ID;
};

} // namespace ckg

#endif // !_KERNEL_CONTEXT_H_
