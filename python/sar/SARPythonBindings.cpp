#include "SAR-c/Dialects.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "Dialect/SAR/IR/SARTypes.h"
#include "Dialect/SAR/IR/SARDialect.h"
#include "Conversion/Passes.h"
#include "Conversion/SARToLinalg/SARToLinalg.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Parser/Parser.h"
#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <vector>

using namespace mlir::python::adaptors;
namespace py = pybind11;

static std::string printModule(mlir::ModuleOp module) {
    std::string out;
    llvm::raw_string_ostream os(out);
    module.print(os);
    return out;
}

PYBIND11_MODULE(_sarDialects, m) {
    auto sarM = m.def_submodule("sar");

    mlir::sar::registerSARConversionPasses();
    mlir::sar::registerSARPassPipelines();

    sarM.def(
        "register_dialect",
        [](py::object contextObj, bool load) {
            MlirContext context;
            if (contextObj.is_none()) {
                context = mlirContextCreate();
            } else {
                context = py::cast<MlirContext>(contextObj);
            }
            MlirDialectHandle handle = mlirGetDialectHandle__sar__();
            mlirDialectHandleRegisterDialect(handle, context);
            if (load) {
                mlirDialectHandleLoadDialect(handle, context);
            }
        },
        py::arg("context") = py::none(), py::arg("load") = true);

    sarM.def(
        "lower_to_linalg",
        [](const std::string &ir) {
            mlir::MLIRContext ctx;
            mlir::DialectRegistry registry;
            registry.insert<mlir::BuiltinDialect, mlir::tensor::TensorDialect,
                            mlir::linalg::LinalgDialect, mlir::arith::ArithDialect,
                            mlir::sar::SARDialect, mlir::func::FuncDialect>();
            ctx.appendDialectRegistry(registry);
            ctx.loadAllAvailableDialects();

            auto module = mlir::parseSourceString<mlir::ModuleOp>(ir, &ctx);
            if (!module)
                throw std::runtime_error("Failed to parse module text");

            mlir::PassManager pm(&ctx);
            pm.addPass(mlir::sar::createConvertSARToLinalgPass());
            if (mlir::failed(pm.run(module.get())))
                throw std::runtime_error("PassManager failed to run SAR->Linalg");

            return printModule(module.get());
        },
        py::arg("ir"));

    mlir::python::adaptors::mlir_type_subclass(sarM, "TensorType",
                                               mlirTypeIsASARTensorType,
                                               mlirSARTensorTypeGetTypeID)
        .def_classmethod(
            "get",
            [](py::object cls, std::vector<int64_t> shape, MlirType elementType,
               MlirContext context) {
                if (context.ptr == nullptr) {
                    context = mlirContextCreate();
                }
                return cls(mlirSARTensorTypeGet(context, shape.size(), shape.data(),
                                                elementType));
            },
            py::arg("cls"), py::arg("shape"), py::arg("element_type"),
            py::arg("context") = py::none(), py::doc("Gets a TensorType"));

    mlir::python::adaptors::mlir_type_subclass(sarM, "MatrixType",
                                               mlirTypeIsASARMatrixType,
                                               mlirSARMatrixTypeGetTypeID)
        .def_classmethod(
            "get",
            [](py::object cls, std::vector<int64_t> shape, MlirType elementType,
               MlirContext context) {
                if (context.ptr == nullptr) {
                    context = mlirContextCreate();
                }
                return cls(mlirSARMatrixTypeGet(context, shape.size(), shape.data(),
                                                elementType));
            },
            py::arg("cls"), py::arg("shape"), py::arg("element_type"),
            py::arg("context") = py::none(), py::doc("Gets a MatrixType"));

    mlir::python::adaptors::mlir_type_subclass(sarM, "VectorType",
                                               mlirTypeIsASARVectorType,
                                               mlirSARVectorTypeGetTypeID)
        .def_classmethod(
            "get",
            [](py::object cls, std::vector<int64_t> shape, MlirType elementType,
               MlirContext context) {
                if (context.ptr == nullptr) {
                    context = mlirContextCreate();
                }
                return cls(mlirSARVectorTypeGet(context, shape.size(), shape.data(),
                                                elementType));
            },
            py::arg("cls"), py::arg("shape"), py::arg("element_type"),
            py::arg("context") = py::none(), py::doc("Gets a VectorType"));
}
