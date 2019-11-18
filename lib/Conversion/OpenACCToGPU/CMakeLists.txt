set(LLVM_TARGET_DEFINITIONS OpenACCToGPU.td)
mlir_tablegen(OpenACCToGPU.cpp.inc -gen-rewriters)
add_public_tablegen_target(MLIROpenACCToGPUIncGen)

set(LIBS
    MLIROpenACCOps
    MLIRAffineOps
    MLIRGPU
    MLIRIR
    MLIRLinalg
    MLIRPass
    MLIRStandardOps
    MLIRSupport
    MLIRTransforms
    LLVMSupport
    )

add_llvm_library(MLIROpenACCToGPU
    OpenACCToGPU.cpp
    OpenACCToGPUPass.cpp

    ADDITIONAL_HEADER_DIRS
    ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/OpenACCOps/
    )

add_dependencies(MLIROpenACCToGPU MLIROpenACCToGPUIncGen)
add_dependencies(MLIROpenACCToGPU ${LIBS})
target_link_libraries(MLIROpenACCToGPU ${LIBS})
