// include/Dialect/SAR/IR/SARTypes.td

#ifndef DIALECT_SAR_TYPES_TD
#define DIALECT_SAR_TYPES_TD

include "mlir/IR/DialectBase.td"
include "mlir/IR/Traits.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

include "Dialect/SAR/IR/SARDialect.td"

class SAR_Type<string name, string typeMnemonic, list<Trait> traits = [], string baseCppClass = "::mlir::Type">
        : TypeDef<SAR_Dialect, name, traits, baseCppClass> {
    let mnemonic = typeMnemonic;
    let typeName = dialect.name # "." # typeMnemonic;
}

def SAR_TensorType : SAR_Type<"Tensor", "tensor", [ShapedTypeInterface],"::mlir::TensorType"> {
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

    let skipDefaultBuilders = 1;

    let genVerifyDecl = 1;

    let extraClassDeclaration = [{
        using ::mlir::ShapedType::Trait<TensorType>::getElementTypeBitWidth;
        using ::mlir::ShapedType::Trait<TensorType>::getRank;
        using ::mlir::ShapedType::Trait<TensorType>::getNumElements;
        using ::mlir::ShapedType::Trait<TensorType>::isDynamicDim;
        using ::mlir::ShapedType::Trait<TensorType>::hasStaticShape;
        using ::mlir::ShapedType::Trait<TensorType>::getNumDynamicDims;
        using ::mlir::ShapedType::Trait<TensorType>::getDimSize;
        using ::mlir::ShapedType::Trait<TensorType>::getDynamicDimIndex;

        ::mlir::ShapedType cloneWith(::std::optional<::llvm::ArrayRef<int64_t>> shape, 
                                     ::mlir::Type type) {
            if (shape)
                return mlir::sar::TensorType::get(getContext(), *shape, type);
            return mlir::sar::TensorType::get(getContext(), getShape(), type);
        }
    }];
}

#endif
