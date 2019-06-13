#ifndef _SOURCE_PARSER_H_
#define _SOURCE_PARSER_H_

#include <clang/Frontend/FrontendAction.h>
#include <clang/Tooling/Tooling.h>
#include <clang/AST/RecursiveASTVisitor.h>

namespace Cuda {

class MyASTVisitor : public clang::RecursiveASTVisitor<MyASTVisitor> {
private:
  static std::vector<std::string> *mTemplateList;

public:
  bool VisitCXXMemberCallExpr(clang::CXXMemberCallExpr *E);

  bool isValidCudaKernelGenMethod(clang::CXXMethodDecl *D) const;

  void processKernelMethod(clang::CXXMethodDecl *D);

  static void setTemplatesList(std::vector<std::string> &templateList);
};

class MyASTConsumer : public clang::ASTConsumer {
private:
  MyASTVisitor mVisitor;

public:
  void HandleTranslationUnit(clang::ASTContext &C) override;
};

class MyFrontendAction : public clang::ASTFrontendAction {
public:
  std::unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(clang::CompilerInstance &Compiler,
                    llvm::StringRef InFile) override;
};

class SourceParser {
public:
  static void parseSources(std::vector<std::string> sourcePaths,
                           const clang::tooling::CompilationDatabase &CD,
                           std::vector<std::string> &templateList);
};

} // namespace Cuda

#endif // !_SOURCE_PARSER_H_
