//===- ConvertOpenACCToSeq.h - Convert Stencil to Affine ops -*- C++ -*-===//
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
// Provides patterns to convert from Stencil structure ops to affine ops.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_OPENACC_CONVERTOPENACCTOSEQ_H
#define MLIR_CONVERSION_OPENACC_CONVERTOPENACCTOSEQ_H

#include "mlir/Dialect/OpenACCOps/OpenACCOps.h"
#include "mlir/Support/StringExtras.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
    template <typename T> class OpPassBase;

    std::unique_ptr<OpPassBase<FuncOp>> createOpenACCToSeqPass();

} // namespace mlir

#endif // MLIR_CONVERSION_OPENACC_CONVERTOPENACCTOSEQ_H
