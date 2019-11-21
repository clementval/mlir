//===- OpenACCToSeqPass.cpp - Convert IIR Ops to Stencil Ops -------===//
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
// This file implements a pass to convert MLIR IIR ops into the Stencil
// ops.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/OpenACC/ConvertOpenACCToSeq.h"
#include "mlir/Dialect/OpenACCOps/Passes.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/Dialect/AffineOps/AffineOps.h"
#include "mlir/Dialect/OpenACCOps/OpenACCOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Module.h"

using namespace mlir;


struct OpenACCToSeqConversionPass
        : public ModulePass<OpenACCToSeqConversionPass> {
    void runOnModule() override;
};


template<typename TerminatorOp>
struct TerminatorOpLowering final : public OpRewritePattern<TerminatorOp> {
    using OpRewritePattern<TerminatorOp>::OpRewritePattern;

    PatternMatchResult matchAndRewrite(TerminatorOp terminatorOp,
                                       PatternRewriter &rewriter) const override {
        rewriter.eraseOp(terminatorOp);
        return Pattern::matchSuccess();
    }
};

template<typename StructureOp>
static void extractOperationsForSequential(StructureOp baseOp) {
    SmallVector < Operation * , 8 > toHoist;
    for (Operation &op : baseOp.getOperation()->getRegion(
            0).getBlocks().front().getOperations()) {
        if (&op == baseOp.getOperation()) {
            continue;
        } else {
            toHoist.push_back(&op);
        }
    }

    for (auto *op : toHoist) {
        op->moveBefore(baseOp.getOperation());
    }
}

/// Convert the OpenACC construct to run program in a sequential manner.
static void convertToSequential(ModuleOp m) {
    m.walk([&](acc::ParallelOp parallelOp) {
        parallelOp.walk([&](acc::LoopOp loopOp) {
            extractOperationsForSequential(loopOp);
            loopOp.erase();
        });
        extractOperationsForSequential(parallelOp);
        parallelOp.erase();
    });
}

void OpenACCToSeqConversionPass::runOnModule() {

    ConversionTarget target(getContext());
    target.addIllegalDialect<acc::OpenACCOpsDialect>();
    target.addLegalDialect<gpu::GPUDialect>();

    // If operation is considered legal the rewrite pattern in not called.
    OwningRewritePatternList patterns;
    patterns.insert<TerminatorOpLowering<acc::ParallelEndOp>>(&getContext());
    patterns.insert<TerminatorOpLowering<acc::LoopEndOp>>(&getContext());

    auto m = getModule();
    convertToSequential(m);

    if (failed(applyPartialConversion(m, target, patterns)))
        signalPassFailure();
}

static PassRegistration <OpenACCToSeqConversionPass> pass(
        "convert-openacc-to-seq",
        "Convert OpenACC to sequential execution");
