#include "KernelManager.h"
#include "InputArgsGen.h"
#include "InputParamsGen.h"
#include "InputTypesGen.h"
#include "KernelExprGen.h"

namespace Cuda {

KernelContext &
KernelManager::createNewKernel(ASTContext &C) {
  auto &K = mKernels.emplace_back();

  K.IndexCalc = "int idx = blockDim.x * blockIdx.x + threadIdx.x;";
  K.Attributes.emplace_back("__global__");
  K.OutputTensorType = C.getRootExpr()->getType();
  K.OutputArg = "result";
  K.OutputParam = "out";
  K.IndexCond =
      "if (idx < " + std::to_string(getNumOfElems(K.OutputTensorType)) + ')';

  KernelExprGen KEG;
  C.visitRoot(KEG);
  K.KernelExpr = "out[idx] = " + KEG.getKernelExpr() + ';';

  InputArgsGen IAG;
  C.visitRoot(IAG);
  K.InputArgs = IAG.getInputArgs();

  InputParamsGen IPG;
  C.visitRoot(IPG);
  K.InputParams = IPG.getInputParams();

  InputTypesGen ITG;
  C.visitRoot(ITG);
  K.InputTensorTypes = ITG.getInputTypes();

  return K;
}

KernelContext &KernelManager::get(uint32_t I) { return mKernels[I]; }

uint32_t KernelManager::size() const { return mKernels.size(); }

std::string KernelManager::getKernelCallStr(const KernelContext &K) const {
  std::stringstream SS;

  SS << "kernel__" << K.FuncID << "(";

  for (auto &&Arg : K.InputArgs) {
    SS << Arg << ", ";
  }

  SS << "result.data());";

  return SS.str();
}

std::string KernelManager::getKernelDeclStr(const KernelContext &K) const {
  std::stringstream SS;

  for (auto &&ATTR : K.Attributes) {
    SS << ATTR << ' ';
  }

  SS << "\nvoid kernel__" << K.FuncID << "(";

  for (auto &&DT : K.InputTensorTypes) {
    SS << DT.DataType << "*, ";
  }

  SS << K.OutputTensorType.DataType << "*);";

  return SS.str();
}

std::string KernelManager::getKernelDefStr(const KernelContext &K) const {
  std::stringstream SS;

  for (auto &&ATTR : K.Attributes) {
    SS << ATTR << ' ';
  }

  SS << "\nvoid kernel__" << K.FuncID << "(";

  for (int i = 0; i < K.InputParams.size(); ++i) {
    SS << K.InputTensorTypes[i].DataType << "* " << K.InputParams[i]
       << ", ";
  }

  SS << K.OutputTensorType.DataType << "* " << K.OutputParam << ") {\n"
     << "\t" << K.IndexCalc << '\n'
     << "\t" << K.IndexCond << '\n'
     << "\t\t" << K.KernelExpr << "\n}";

  return SS.str();
}

uint32_t KernelManager::getNumOfElems(const TensorType &T) {
  uint32_t R = 1;
  
  for (auto &&X : T.Dimensions) {
    R *= X;
  }
  return R;
}

} // namespace Cuda