// include/Dialect/SAR/IR/SARTypes.td

#ifndef DIALECT_SAR_TYPES_TD
#define DIALECT_SAR_TYPES_TD

include "mlir/IR/Traits.td"
include "mlir/IR/AttrTypeBase.td"

include "Dialect/SAR/IR/SARDialect.td"

// Base class for SAR-specific types
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
        ArrayRefParameter<"int64_t">:$shape,
        "Type":$elementType
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

    let skipDefaultBuilders = 0;

    let genVerifyDecl = 1;

    let extraClassDeclaration = [{
        using ::mlir::ShapedType::Trait<tensorType>::getElementTypeBitWidth;
        using ::mlir::ShapedType::Trait<tensorType>::getRank;
        using ::mlir::ShapedType::Trait<tensorType>::getNumElements;
        using ::mlir::ShapedType::Trait<tensorType>::isDynamicDim;
        using ::mlir::ShapedType::Trait<tensorType>::hasStaticShape;
        using ::mlir::ShapedType::Trait<tensorType>::getNumDynamicDims;
        using ::mlir::ShapedType::Trait<tensorType>::getDimSize;
        using ::mlir::ShapedType::Trait<tensorType>::getDynamicDimIndex;
        
        ::mlir::ShapedType cloneWith(::std::optional<::llvm::ArrayRef<int64_t>> shape, 
                                     ::mlir::Type type) {
            if (shape)
                return mlir::sar::tensorType::get(getContext(), *shape, type);
            return mlir::sar::tensorType::get(getContext(), getShape(), type);
        }
    }];
}

#endif
