//===- OpenACCToGPUPass.cpp - Convert IIR Ops to Stencil Ops -------===//
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

#include "mlir/Conversion/OpenACC/ConvertOpenACCToGPU.h"
#include "mlir/Dialect/OpenACCOps/Passes.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/Dialect/AffineOps/AffineOps.h"
#include "mlir/Dialect/OpenACCOps/OpenACCOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Module.h"

using namespace mlir;


struct OpenACCToGPULoweringPass
        : public ModulePass<OpenACCToGPULoweringPass> {
    void runOnModule() override;
};

struct ParallelOpLowering final : public OpRewritePattern<acc::ParallelOp> {
    using OpRewritePattern<acc::ParallelOp>::OpRewritePattern;

    PatternMatchResult matchAndRewrite(acc::ParallelOp parallelOp,
                                       PatternRewriter &rewriter) const override;
};

struct LoopOpLowering final : public OpRewritePattern<acc::LoopOp> {
    using OpRewritePattern<acc::LoopOp>::OpRewritePattern;

    PatternMatchResult matchAndRewrite(acc::LoopOp loopOp,
                                       PatternRewriter &rewriter) const override;
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

PatternMatchResult
ParallelOpLowering::matchAndRewrite(acc::ParallelOp parallelOp,
                                    PatternRewriter &rewriter) const {
    // Not used now
    return matchSuccess();
}


PatternMatchResult
LoopOpLowering::matchAndRewrite(acc::LoopOp loopOp,
                                PatternRewriter &rewriter) const {
    // Not used now
    return matchSuccess();
}

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

static FuncOp outlineParallelKernel(acc::ParallelOp parallelOp) {
    auto loc = parallelOp.getLoc();
    std::string parallelKernelName =
            Twine(parallelOp.getParentOfType<FuncOp>().getName(),
                  "_kernel").str();
    Builder builder(parallelOp.getContext());
    FuncOp outlinedFunc = FuncOp::create(loc, parallelKernelName,
                                         builder.getFunctionType(llvm::None,
                                                                 llvm::None));
    outlinedFunc.getBody().takeBody(parallelOp.getOperation()->getRegion(0));

    OpBuilder opBuilder(parallelOp.getOperation());
    outlinedFunc.walk([&](acc::LoopOp loopOp) {
    });

    // Add a terminator at then end of the new func
    opBuilder.setInsertionPointToEnd(&outlinedFunc.getBody().back());
    opBuilder.create<mlir::ReturnOp>(loc);

    return outlinedFunc;
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

void OpenACCToGPULoweringPass::runOnModule() {

    ConversionTarget target(getContext());
    target.addIllegalDialect<acc::OpenACCOpsDialect>();
    target.addLegalDialect<gpu::GPUDialect>();

    // If operation is considered legal the rewrite pattern in not called.
    OwningRewritePatternList patterns;
//    patterns.insert<ParallelOpLowering>(&getContext());
    patterns.insert<TerminatorOpLowering<acc::ParallelEndOp>>(&getContext());
//    patterns.insert<LoopOpLowering>(&getContext());
    patterns.insert<TerminatorOpLowering<acc::LoopEndOp>>(&getContext());

    auto m = getModule();
    convertToSequential(m);

    if (failed(applyPartialConversion(m, target, patterns)))
        signalPassFailure();
}

static PassRegistration <OpenACCToGPULoweringPass> pass(
        "convert-openacc-to-gpu",
        "Convert OpenACC Ops to GPU dialect");
