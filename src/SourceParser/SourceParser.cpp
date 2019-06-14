#include "SourceParser.h"

namespace Cuda {

std::vector<std::string> *MyASTVisitor::mTemplateList = nullptr;

bool MyASTVisitor::VisitCXXMemberCallExpr(clang::CXXMemberCallExpr *E) {
  if (clang::CXXMethodDecl *D = E->getMethodDecl();
      isValidCudaKernelGenMethod(D)) {
    processKernelMethod(D);
  }
  return true;
}

bool MyASTVisitor::isValidCudaKernelGenMethod(clang::CXXMethodDecl *D) const {
  if (D == nullptr) {
    return false;
  }
  if (D->getNameAsString() != "eval") {
    return false;
  }
  for (auto it = D->attr_begin(); it != D->attr_end(); ++it) {
    if ((*it)->getSpelling() == "cuda_kernel_gen") {
      return true;
    }
  }
  return false;
}

void MyASTVisitor::processKernelMethod(clang::CXXMethodDecl *D) {
  clang::PrintingPolicy P{clang::LangOptions()};
  P.adjustForCPlusPlus();
  mTemplateList->push_back(D->getThisType().getAsString(P));
  auto &Expr = mTemplateList->back();
  Expr.erase(Expr.end() - 2, Expr.end());
}

void MyASTVisitor::setTemplatesList(std::vector<std::string> &templateList) {
  mTemplateList = &templateList;
}

void MyASTConsumer::HandleTranslationUnit(clang::ASTContext &C) {
  mVisitor.TraverseAST(C);
}

std::unique_ptr<clang::ASTConsumer>
MyFrontendAction::CreateASTConsumer(clang::CompilerInstance &Compiler,
                                    llvm::StringRef InFile) {
  return std::make_unique<MyASTConsumer>();
}

void SourceParser::parseSources(std::vector<std::string> sourcePaths,
                                const clang::tooling::CompilationDatabase &CD,
                                std::vector<std::string> &templateList) {
  MyASTVisitor::setTemplatesList(templateList);
  clang::tooling::ClangTool Tool(CD, sourcePaths);
  Tool.run(clang::tooling::newFrontendActionFactory<MyFrontendAction>().get());
}

} // namespace Cuda