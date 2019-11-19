//===- ConvertStencilToAffine.h - Convert Stencil to Affine ops -*- C++ -*-===//
//
// Copyright 2019 Fabian Wolff
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

#ifndef MLIR_CONVERSION_OPENACCTOGPU_CONVERTOPENACCTOGPU_H
#define MLIR_CONVERSION_OPENACCTOGPU_CONVERTOPENACCTOGPU_H

#include "mlir/Dialect/OpenACCOps/OpenACCOps.h"
#include "mlir/Support/StringExtras.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
    template <typename T> class OpPassBase;
/// Collect a set of patterns to lower from Stencil structure
/// operations (stencil.stage, stencil.do_method etc.) to loop
/// operations within the Affine dialect; in particular, convert
/// abstract stencil descriptions into affine loop nests.
    void populateOpenACCToGPUConversionPatterns(OwningRewritePatternList &patterns,
                                                MLIRContext *ctx);

    std::unique_ptr<OpPassBase<FuncOp>> createOpenACCToGPUPass();

} // namespace mlir

#endif // MLIR_CONVERSION_OPENACCTOGPU_CONVERTOPENACCTOGPU_H
