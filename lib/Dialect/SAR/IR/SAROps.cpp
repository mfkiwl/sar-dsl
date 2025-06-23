// lib/Dialect/SAR/IR/SAROps.cpp

#include "Dialect/SAR/IR/SARDialect.h"
#include "Dialect/SAR/IR/SARTypes.h"
#include "Dialect/SAR/IR/SAROps.h"

// Include generated op implementations
#define GET_OP_CLASSES
#include "Dialect/SAR/IR/SAROps.cpp.inc"

namespace mlir::sar {

void SARDialect::registerOps() {
    llvm::outs() << "Register " << getDialectNamespace() << " op\n";
    addOperations<
#define GET_OP_LIST
#include "Dialect/SAR/IR/SAROps.cpp.inc"
    >();
}

}
