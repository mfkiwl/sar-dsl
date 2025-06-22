#ifndef DIALECT_SAR_TYPES_TD
#define DIALECT_SAR_TYPES_TD

include "mlir/IR/Traits.td"
include "mlir/IR/AttrTypeBase.td"

include "Dialect/SAR/IR/SARDialect.td"

class SAR_Type<string typeMnemonic, list<Trait> traits = [], string baseCppClass = "::mlir::Type">
		: TypeDef<SAR_Dialect, typeMnemonic, traits, baseCppClass> {
	let mnemonic = typeMnemonic;
	let typeName = dialect.name # "." # typeMnemonic;
}

// SAR-specific tensor type with shape and element type
def TensorType : SAR_Type<"tensor", []> {
	let summary = "SAR tensor type";
	let description = "A specialized tensor type in SAR dialect";

	let parameters = (ins
		ArrayRefParameter<"int64_t">:$shape, // Tensor shape dimensions
		"Type":$elementType                  // Element data type
	);

	let genStorageClass = 1;
	let hasStorageCustomConstructor = 0;

	let builders = [
		TypeBuilder<(ins 
				"::mlir::ArrayRef<int64_t>":$shape,
                "::mlir::Type":$elementType), [{
			return $_get(elementType.getContext(), shape, elementType);
		}]>
	];

	let hasCustomAssemblyFormat = 1;

	let skipDefaultBuilders = 1;

	let genVerifyDecl = 1;

	let extraClassDeclaration = [{

	}];
}

#endif
