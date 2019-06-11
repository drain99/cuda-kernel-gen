#include "KernelManager.h"
#include "InputArgsGen.h"
#include "InputParamsGen.h"
#include "InputTypesGen.h"
#include "KernelExprGen.h"

namespace Cuda {

uint32_t KernelManager::createNewKernel(ASTContext &C, const std::string& exprTemplate) {
  auto &K = mKernels.emplace_back();

  K.FullExprTemplate = exprTemplate;
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

  return mKernels.size() - 1;
}

KernelContext &KernelManager::get(uint32_t I) { return mKernels[I]; }

const KernelContext &KernelManager::get(uint32_t I) const {
  return mKernels[I];
}

uint32_t KernelManager::size() const { return mKernels.size(); }

void KernelManager::getKernelCallStr(uint32_t I, std::ostream &OS) const {
  auto &K = mKernels[I];

  getKernelNameStr(I, OS);
  OS << "<<<1," << getNumOfElems(K.OutputTensorType) << ">>>(";
  for (auto &&Param : K.InputParams) {
    OS << Param << ", ";
  }
  OS << "out);";
}

void KernelManager::getKernelDeclStr(uint32_t I, std::ostream &OS) const {
  auto &K = mKernels[I];

  for (auto &&Attr : K.Attributes) {
    OS << Attr << ' ';
  }
  OS << "\nvoid ";
  getKernelNameStr(I, OS);
  OS << "(";
  for (auto &&DT : K.InputTensorTypes) {
    OS << DT.DataType << "*, ";
  }
  OS << K.OutputTensorType.DataType << "*);";
}

void KernelManager::getKernelDefStr(uint32_t I, std::ostream &OS) const {
  auto &K = mKernels[I];

  for (auto &&Attr : K.Attributes) {
    OS << Attr << ' ';
  }
  OS << "\nvoid ";
  getKernelNameStr(I, OS);
  OS << "(";
  for (int i = 0; i < K.InputParams.size(); ++i) {
    OS << K.InputTensorTypes[i].DataType << "* " << K.InputParams[i] << ", ";
  }
  OS << K.OutputTensorType.DataType << "* " << K.OutputParam << ") {\n"
     << "\t" << K.IndexCalc << '\n'
     << "\t" << K.IndexCond << '\n'
     << "\t\t" << K.KernelExpr << "\n}";
}

void KernelManager::getKernelNameStr(uint32_t I, std::ostream &OS) const {
  OS << "kernel__" << mKernels[I].FuncID;
}

void KernelManager::getKernelWrapperNameStr(uint32_t I,
                                            std::ostream &OS) const {
  OS << "kernel_wrapper__" << mKernels[I].FuncID;
}

void KernelManager::getKernelWrapperCallStr(uint32_t I,
                                            std::ostream &OS) const {
  auto &K = mKernels[I];

  getKernelWrapperNameStr(I, OS);
  OS << "(";
  for (auto &&Arg : K.InputArgs) {
    OS << Arg << ", ";
  }
  OS << "result.data());";
}

void KernelManager::getKernelWrapperDeclStr(uint32_t I,
                                            std::ostream &OS) const {
  auto &K = mKernels[I];

  OS << "void ";
  getKernelWrapperNameStr(I, OS);
  OS << "(";
  for (auto &&DT : K.InputTensorTypes) {
    OS << DT.DataType << "*, ";
  }
  OS << K.OutputTensorType.DataType << "*);";
}

void KernelManager::getKernelWrapperDefStr(uint32_t I,
                                           std::ostream &OS) const {
  auto &K = mKernels[I];

  OS << "void ";
  getKernelWrapperNameStr(I, OS);
  OS << "(";
  for (int i = 0; i < K.InputParams.size(); ++i) {
    OS << K.InputTensorTypes[i].DataType << "* " << K.InputParams[i] << ", ";
  }
  OS << K.OutputTensorType.DataType << "* " << K.OutputParam << ") {\n"
     << "\t";
  getKernelCallStr(I, OS);
  OS << "\n\tcudaDeviceSynchronize();\n}";
}

uint32_t KernelManager::getNumOfElems(const TensorType &T) const {
  uint32_t R = 1;

  for (auto &&X : T.Dimensions) {
    R *= X;
  }
  return R;
}

} // namespace Cuda