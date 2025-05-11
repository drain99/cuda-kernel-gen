#include "SourceParser.h"
#include "clang/AST/PrettyPrinter.h"
#include "clang/Basic/AttrKinds.h"
#include "clang/Basic/LangOptions.h"
#include <llvm-14/llvm/Support/raw_ostream.h>

namespace ckg {

std::vector<std::string> *MyASTVisitor::mTemplateList = nullptr;

static const std::string KernelGenAttr = "[[clang::annotate(\"cuda_kernel_gen\")]]";

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
  for (auto const attr : D->attrs()) {
    if (attr->getKind() == clang::attr::Annotate) {
      std::string S;
      llvm::raw_string_ostream OS{S};
      attr->printPretty(OS, clang::PrintingPolicy(clang::LangOptions()));
      if (S.find(KernelGenAttr) != std::string::npos) {
        return true;
      }
    }
  }
  return false;
}

void MyASTVisitor::processKernelMethod(clang::CXXMethodDecl *D) {
  clang::PrintingPolicy P{clang::LangOptions()};
  P.adjustForCPlusPlus();
  mTemplateList->push_back(D->getThisType()->getPointeeType().getAsString(P));
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

} // namespace ckg