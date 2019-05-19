#include <iostream>
#include <sstream>

#include "ASTContext.h"
#include "KernelManager.h"

using namespace Cuda;

int main()
{
	std::string fullTemplate =
		"AddExpr<ConstantExpr<Tensor<int,100>>,ConstantExpr<Tensor<int,100>>>";

	ASTContext context = ASTContext(ExprOf<AddExpr>(),
									fullTemplate);

	context.addNewExpr(ExprOf<TerminalExpr>(),
					   context.getRootExpr(),
					   context.getFullTemplate().substr(8, 36 - 8 + 1));

	context.addNewExpr(ExprOf<TerminalExpr>(),
					   context.getRootExpr(),
					   context.getFullTemplate().substr(38, 66 - 38 + 1));

	KernelManager KM;
	auto& K = KM.createNewKernel(context,
								 "Tensor<int,100>");

	std::cout << KM.getKernelCallStr(K) << std::endl
		<< KM.getKernelDeclStr(K) << std::endl
		<< KM.getKernelDefStr(K) << std::endl;

	return 0;
}