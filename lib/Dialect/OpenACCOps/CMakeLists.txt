add_llvm_library(MLIROpenACCOps
        OpenACCOps.cpp
        DialectRegistration.cpp

        ADDITIONAL_HEADER_DIRS
        ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/OpenACCOps
        )
add_dependencies(MLIRAffineOps
        MLIROpenACCOpsIncGen
        MLIRIR
        MLIRStandardOps)
target_link_libraries(MLIROpenACCOps MLIRIR MLIRStandardOps)