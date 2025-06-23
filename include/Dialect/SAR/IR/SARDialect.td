// include/Dialect/SAR/IR/SARDialect.td

#ifndef DIALECT_SAR_TD
#define DIALECT_SAR_TD

include "mlir/IR/DialectBase.td"

include "Dialect/SAR/IR/SARConstraints.td"

def SAR_Dialect : Dialect {
	let name = "sar";
	let summary = "SAR dialect";
	let description = "The SAR dialect is used for representing and manipulating SAR (Synthetic Aperture Radar) data.";

	let cppNamespace = "::mlir::sar";

	// Corrected dialect dependency
	let dependentDialects = ["::mlir::tensor::TensorDialect"];

	let discardableAttrs = (ins);

	// Declare dialect initialization methods
	let extraClassDeclaration = [{
		static void sayHello();
        void registerTypes();
        void registerOps();
	}];

	// Enable properties-based attribute storage
	let usePropertiesForAttributes = 1;

	// Default settings
    let hasConstantMaterializer = 0;
    let hasNonDefaultDestructor = 1;
    let hasOperationAttrVerify = 0;
    let hasRegionArgAttrVerify = 0;
    let hasRegionResultAttrVerify = 0;
    let hasOperationInterfaceFallback = 0;
    let useDefaultAttributePrinterParser = 0;
    let useDefaultTypePrinterParser = 1;
    let hasCanonicalizer = 0;
    let isExtensible = 0;
}

#endif
