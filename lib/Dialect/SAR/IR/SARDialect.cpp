// lib/Dialect/SAR/IR/SARDialect.cpp

#include "llvm/Support/Casting.h"

#include "Dialect/SAR/IR/SARDialect.h"

#include "Dialect/SAR/IR/SARDialect.cpp.inc"

namespace mlir::sar {

// Initialize the SAR dialect
void SARDialect::initialize() {
    llvm::outs() << "Initializing " << getDialectNamespace() << "\n";
    registerTypes();
    registerOps();
}

// Destructor for SAR dialect
SARDialect::~SARDialect() {
    llvm::outs() << "Destroying " << getDialectNamespace() << "\n";
}

// Example method to demonstrate functionality
void SARDialect::sayHello() {
    llvm::outs() << "Hello from " << getDialectNamespace() << "\n";
}

} // namespace mlir::sar
