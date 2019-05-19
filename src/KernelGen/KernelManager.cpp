#include "KernelManager.h"
#include "KernelHolder.h"
#include "KernelExprGen.h"
#include "InputArgsGen.h"
#include "InputParamsGen.h"
#include "InputTypesGen.h"

namespace Cuda
{

	const KernelHolder& KernelManager::createNewKernel(ASTContext& context, 
													   const std::string& outputTensorType)
	{
		auto& K = mKernels.emplace_back();

		K.IndexCalc = "int indexX=blockDim.x*blockIdx.x+threadIdx.x;";

		K.Attributes.emplace_back("__global__");

		K.OutputTensorType = parseTensorHolder(outputTensorType);

		K.OutputArg = "result";

		K.OutputParam = "o_1";

		K.IndexCond = 
			"if(indexX<" + std::to_string(getDimProduct(K.OutputTensorType)) + ')';
		
		KernelExprGen KEG;
		context.visitRoot(KEG);
		K.KernelExpr = "o_1[indexX]=" + KEG.getKernelExpr() + ';';

		InputArgsGen IAG;
		context.visitRoot(IAG);
		K.InputArgs = IAG.getInputArgs();

		InputParamsGen IPG;
		context.visitRoot(IPG);
		K.InputParams = IPG.getInputParams();

		InputTypesGen ITG;
		context.visitRoot(ITG);
		K.InputTensorTypes = ITG.getInputTypes();

		return K;
	}

	std::string KernelManager::getKernelCallStr(const KernelHolder& K) const
	{
		std::stringstream ss;

		ss << "kernel__"
			<< K.FuncID
			<< '(';

		for (auto&& Arg : K.InputArgs)
		{
			ss << Arg << ',';
		}
		
		ss << "result);";

		return ss.str();
	}

	std::string KernelManager::getKernelDeclStr(const KernelHolder& K) const
	{
		std::stringstream ss;

		for (auto&& ATTR : K.Attributes)
		{
			ss << ATTR << ' ';
		}

		ss << "kernel__"
			<< K.FuncID
			<< '(';

		for (auto&& DT : K.InputTensorTypes)
		{
			ss << DT.DataType << "*,";
		}

		ss << K.OutputTensorType.DataType
			<< "*);";

		return ss.str();
	}

	std::string KernelManager::getKernelDefStr(const KernelHolder& K) const
	{
		std::stringstream ss;

		for (auto&& ATTR : K.Attributes)
		{
			ss << ATTR << ' ';
		}

		ss << "kernel__"
			<< K.FuncID
			<< '(';

		for (int i = 0; i < K.InputParams.size(); ++i)
		{
			ss << K.InputTensorTypes[i].DataType
				<< "* "
				<< K.InputParams[i]
				<< ',';
		}

		ss << K.OutputTensorType.DataType
			<< "* "
			<< K.OutputParam
			<< "){\n"
			<< K.IndexCalc
			<< '\n'
			<< K.IndexCond
			<< '\n'
			<< K.KernelExpr
			<< "\n}";

		return ss.str();
	}

	TensorHolder KernelManager::parseTensorHolder(const std::string& tensorStr) const
	{
		TensorHolder result;
		int balancedChevrons = 0;
		int leftIdx = tensorStr.find('<') + 1;
		bool isFirst = true;

		for (int i = leftIdx; i < tensorStr.size() - 1; ++i)
		{
			if (tensorStr[i] == '<')
			{
				++balancedChevrons;
			}
			else if (tensorStr[i] == '>')
			{
				--balancedChevrons;
			}
			else if (tensorStr[i] == ',' && balancedChevrons == 0)
			{
				if (isFirst)
				{
					isFirst = false;
					result.DataType = tensorStr.substr(leftIdx, 
													   i - leftIdx);
				}
				else
				{
					result.Dimensions.push_back(
						std::stoi(tensorStr.substr(leftIdx,
												   i - leftIdx))
					);
				}
				leftIdx = i + 1;
			}
		}
		if (isFirst)
		{
			result.DataType = tensorStr.substr(leftIdx, 
											   tensorStr.size() - 1 - leftIdx);
		}
		else
		{
			result.Dimensions.push_back(
				std::stoi(tensorStr.substr(leftIdx,
										   tensorStr.size() - 1 - leftIdx))
			);
		}
		return result;
	}

	int KernelManager::getDimProduct(const TensorHolder& holder) const
	{
		int result = 1;
		for (auto&& D : holder.Dimensions)
		{
			result *= D;
		}
		return result;
	}

}