//===- Passes.h - OpenACC pass entry points ---------------------*- C++ -*-===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
// This header file defines prototypes that expose pass constructors.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_OPENACC_PASSES_H
#define MLIR_DIALECT_OPENACC_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
    std::unique_ptr<OpPassBase<mlir::ModuleOp>> createConvertOpenACCToGPUPass();
} // namespace mlir

#endif // MLIR_DIALECT_OPENACC_PASSES_H