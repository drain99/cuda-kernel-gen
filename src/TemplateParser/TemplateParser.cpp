#include "TemplateParser.h"
#include<iostream>
namespace Cuda {

TemplateParser::TemplateParser(const std::string &exprTemplate,
                               const std::string &outputTemplate)
    : mET(exprTemplate), mOT(outputTemplate) {}

ASTContext TemplateParser::createAST() {
  llvm::StringRef ET = mET;
  ASTContext C = ASTContext(ExprOf<AddExpr>());
  uint32_t R = ET.find('<') - 1;
  llvm::StringRef OpName = ET.substr(0, R + 1);
  if (OpName == "AddExpr") {
    C = ASTContext(ExprOf<AddExpr>());
  } else if (OpName == "SubtractExpr") {
    C = ASTContext(ExprOf<SubtractExpr>());
  } else if (OpName == "MultiplyExpr") {
    C = ASTContext(ExprOf<MultiplyExpr>());
  } else if (OpName == "DivideExpr") {
    C = ASTContext(ExprOf<DivideExpr>());
  }
  int16_t chevronNum = 0;
  uint32_t L = R + 2;
  for (size_t I = R + 2; I + 1 < ET.size(); ++I) {
    char ch = ET[I];
    if (ch == '<') {
      ++chevronNum;
    } else if (ch == '>') {
      --chevronNum;
    }
    if (!chevronNum && (ch == ',' || (I + 2 == ET.size()))) {
      parseCurrExpr(C, C.getRootExpr(),
                    ET.substr(L, I - L + (I + 2 == ET.size())));
      L = I + 1;
    }
  }
  return C;
}

TensorType TemplateParser::createOutputTensorType() {
  return parseTensorExpr(mOT);
}

void TemplateParser::parseCurrExpr(ASTContext &C, Expr *P, llvm::StringRef E) {
  uint32_t R = E.find('<') - 1;
  llvm::StringRef OpName = E.substr(0, R + 1);
  Expr *curr = nullptr;
  if (OpName == "AddExpr") {
    curr = C.addNewExpr<AddExpr>(P);
  } else if (OpName == "SubtractExpr") {
    curr = C.addNewExpr<SubtractExpr>(P);
  } else if (OpName == "MultiplyExpr") {
    curr = C.addNewExpr<MultiplyExpr>(P);
  } else if (OpName == "DivideExpr") {
    curr = C.addNewExpr<DivideExpr>(P);
  } else if (OpName == "Tensor") {
    C.addNewExpr<TensorExpr>(P, parseTensorExpr(E));
  }
  if (!curr) {
    return;
  }
  int16_t chevronNum = 0;
  uint32_t L = R + 2;
  for (size_t I = R + 2; I + 1 < E.size(); ++I) {
    char ch = E[I];
    if (ch == '<') {
      ++chevronNum;
    } else if (ch == '>') {
      --chevronNum;
    }
    if (!chevronNum && (ch == ',' || (I + 2 == E.size()))) {
      parseCurrExpr(C, curr, E.substr(L, I - L + (I + 2 == E.size())));
      L = I + 1;
    }
  }
}

TensorType TemplateParser::parseTensorExpr(llvm::StringRef T) {
  TensorType TT;
  llvm::StringRef tensorName = T.substr(7, T.size() - 8);
  int16_t chevronNum = 0;
  uint32_t L = 0;
  bool isFirst = true;
  for (size_t I = 0; I < tensorName.size(); ++I) {
    char C = tensorName[I];
    if (C == '<') {
      ++chevronNum;
    } else if (C == '>') {
      --chevronNum;
    }
    if (!chevronNum && (C == ',' || (I + 1 == tensorName.size()))) {
      if (isFirst) {
        TT.DataType =
            tensorName.substr(L, I - L + (I + 1 == tensorName.size()));
        isFirst = false;
      } else {
        TT.Dimensions.push_back(std::stoi(
            tensorName.substr(L, I - L + (I + 1 == tensorName.size()))));
      }
      L = I + 1;
    }
  }
  return TT;
}

} // namespace Cuda