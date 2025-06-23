// lib/Dialect/SAR/IR/SARDialect.cpp

#include "Dialect/SAR/IR/SARDialect.h"

#include "Dialect/SAR/IR/SARDialect.cpp.inc"

namespace mlir::sar {

void SARDialect::initialize() {
    llvm::outs() << "Initializing " << getDialectNamespace() << "\n";
    registerTypes();
    registerOps();
}

SARDialect::~SARDialect() {
    llvm::outs() << "Destroying " << getDialectNamespace() << "\n";
}

void SARDialect::sayHello() {
    llvm::outs() << "Hello from " << getDialectNamespace() << "\n";
}

} // namespace mlir::sar
