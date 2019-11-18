set(LLVM_TARGET_DEFINITIONS OpenACCOps.td)
mlir_tablegen(OpenACCOps.h.inc -gen-op-decls)
mlir_tablegen(OpenACCOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIROpenACCOpsIncGen)


