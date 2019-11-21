//===- OpenACC.cpp - OpenACC to GPU dialect conversion ---------------===//
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
// This file implements a pass to convert the MLIR IIR dialect into the Stencil
// dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/OpenACC/ConvertOpenACCToGPU.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "llvm/ADT/StringMap.h"

#include <llvm/Support/raw_ostream.h>
#include <stack>
#include <string>
#include <unordered_map>
#include <utility>
#include <iostream>

using namespace mlir;

struct OpenACCLoopEmptyConstructFolder : public OpRewritePattern<acc::LoopOp> {
    using OpRewritePattern<acc::LoopOp>::OpRewritePattern;

    PatternMatchResult matchAndRewrite(acc::LoopOp loopOp,
                                       PatternRewriter &rewriter) const override {
        // Check that the body only contains a terminator.
        if (!has_single_element(loopOp.getBody()))
            return matchFailure();
        rewriter.eraseOp(loopOp);
        return matchSuccess();
    }
};

void
acc::LoopOp::getCanonicalizationPatterns(OwningRewritePatternList &patterns,
                                         MLIRContext *context) {
    patterns.insert<OpenACCLoopEmptyConstructFolder>(context);
}



