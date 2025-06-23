// include/Conversion/Passes.h

#ifndef CONVERSION_PASSES_H
#define CONVERSION_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir::sar {

#define GEN_PASS_DECL_CONVERTSARTOLINALGPASS
#define GEN_PASS_REGISTRATION
#include "Conversion/Passes.h.inc"

} // namespace mlir::sar

#endif
