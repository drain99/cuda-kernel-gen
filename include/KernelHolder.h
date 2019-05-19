#ifndef _KERNEL_HOLDER_H_
#define _KERNEL_HOLDER_H_

#include <llvm/ADT/SmallVector.h>
#include <vector>

namespace Cuda
{

	struct TensorHolder
	{
		std::string DataType;
		llvm::SmallVector<int, 2> Dimensions;
	};

	struct KernelHolder
	{
	public:

		std::vector<std::string> Attributes;
		int FuncID;
		std::vector<TensorHolder> InputTensorTypes;
		std::vector<std::string> InputParams;
		std::vector<std::string> InputArgs;
		TensorHolder OutputTensorType;
		std::string OutputParam;
		std::string OutputArg;
		std::string IndexCalc;
		std::string IndexCond;
		std::string KernelExpr;

		KernelHolder();

	private:

		static int ID;
	};

}

#endif // !_KERNEL_HOLDER_H_
