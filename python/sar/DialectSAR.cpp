// python/sar/DialectSAR.cpp

#include "SAR-c/Dialects.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "Dialect/SAR/IR/SARTypes.h"
#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <vector>

using namespace mlir::python::adaptors;
namespace py = pybind11;

PYBIND11_MODULE(_sarDialects, m) {
  auto sarM = m.def_submodule("sar");
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

  mlir::python::adaptors::mlir_type_subclass(sarM, "TensorType", mlirTypeIsASARTensorType, mlirSARTensorTypeGetTypeID)
      .def_classmethod(
          "get",
          [](py::object cls, std::vector<int64_t> shape, MlirType elementType, MlirContext context) {
            if (context.ptr == nullptr) {
              context = mlirContextCreate();
            }
            return cls(mlirSARTensorTypeGet(context, shape.size(), shape.data(), elementType));
          },
          py::arg("cls"),
          py::arg("shape"), py::arg("element_type"), py::arg("context") = py::none(), py::doc("Gets a TensorType"));

  mlir::python::adaptors::mlir_type_subclass(sarM, "MatrixType", mlirTypeIsASARMatrixType, mlirSARMatrixTypeGetTypeID)
      .def_classmethod(
          "get",
          [](py::object cls, std::vector<int64_t> shape, MlirType elementType, MlirContext context) {
            if (context.ptr == nullptr) {
              context = mlirContextCreate();
            }
            return cls(mlirSARMatrixTypeGet(context, shape.size(), shape.data(), elementType));
          },
          py::arg("cls"),
          py::arg("shape"), py::arg("element_type"), py::arg("context") = py::none(), py::doc("Gets a MatrixType"));

  mlir::python::adaptors::mlir_type_subclass(sarM, "VectorType", mlirTypeIsASARVectorType, mlirSARVectorTypeGetTypeID)
      .def_classmethod(
          "get",
          [](py::object cls, std::vector<int64_t> shape, MlirType elementType, MlirContext context) {
            if (context.ptr == nullptr) {
              context = mlirContextCreate();
            }
            return cls(mlirSARVectorTypeGet(context, shape.size(), shape.data(), elementType));
          },
          py::arg("cls"),
          py::arg("shape"), py::arg("element_type"), py::arg("context") = py::none(), py::doc("Gets a VectorType"));
}
