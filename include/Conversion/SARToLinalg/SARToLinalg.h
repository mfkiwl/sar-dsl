// include/Conversion/SARToLinalg/SARToLinalg.h

#ifndef CONVERSION_SARTOLINALG_H
#define CONVERSION_SARTOLINALG_H

#include "mlir/Pass/Pass.h"

namespace mlir {
    class TypeConverter;
}  // namespace mlir

namespace mlir::sar {

void initSARToLinalgTypeConvert(TypeConverter &typeConverter);

void populateSARToLinalgPatterns(TypeConverter &typeConverter,
                                 RewritePatternSet &patterns);

#define GEN_PASS_DECL_CONVERTSARTOLINALGPASS
#include "Conversion/Passes.h.inc"

}  // namespace mlir::sar

#endif
