#ifndef CONVERSION_SARTOLINALG_H
#define CONVERSION_SARTOLINALG_H

#include "mlir/Pass/Pass.h"

namespace mlir {
    class TypeConverter;
    class ModuleOp;
}  // namespace mlir

namespace mlir::sar {

void initSARToLinalgTypeConvert(TypeConverter &typeConverter);

void populateSARToLinalgPatterns(TypeConverter &typeConverter,
                                 RewritePatternSet &patterns);

#define GEN_PASS_DECL_CONVERTSARTOLINALGPASS
#include "Conversion/Passes.h.inc"

void registerSARPassPipelines();

mlir::LogicalResult runSARToLinalgPipeline(mlir::ModuleOp module);

}  // namespace mlir::sar

#endif
