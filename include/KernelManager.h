#ifndef _KERNEL_MANAGER_H_
#define _KERNEL_MANAGER_H_

#include <vector>

#include "KernelHolder.h"
#include "ASTContext.h"

namespace Cuda
{

	class KernelManager
	{
	private:

		std::vector<KernelHolder> mKernels;

	public:

		const KernelHolder& createNewKernel(ASTContext& context,
											const std::string& outputTensorType);

		std::string getKernelCallStr(const KernelHolder& K) const;

		std::string getKernelDeclStr(const KernelHolder& K) const;

		std::string getKernelDefStr(const KernelHolder& K) const;

	private:

		TensorHolder parseTensorHolder(const std::string& tensorStr) const;

		int getDimProduct(const TensorHolder& holder) const;
	};

}

#endif // !_KERNEL_MANAGER_H_
