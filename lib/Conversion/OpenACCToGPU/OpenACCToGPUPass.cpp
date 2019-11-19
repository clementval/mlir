//===- ConvertIIRToStencilPass.cpp - Convert IIR Ops to Stencil Ops -------===//
//
// Copyright 2019 Jean-Michel Gorius
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

#include "mlir/Conversion/OpenACCToGPU/ConvertOpenACCToGPU.h"
#include "mlir/Dialect/OpenACCOps/Passes.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/AffineOps/AffineOps.h"
#include "mlir/Dialect/OpenACCOps/OpenACCOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Module.h"

using namespace mlir;



    struct OpenACCToGPULoweringPass
            : public FunctionPass<OpenACCToGPULoweringPass> {
        void runOnFunction() override;
    };

    struct ParallelOpLowering final : public OpRewritePattern<acc::ParallelOp> {
        using OpRewritePattern<acc::ParallelOp>::OpRewritePattern;

        PatternMatchResult matchAndRewrite(acc::ParallelOp parallelOp,
                                           PatternRewriter &rewriter) const override;
    };





PatternMatchResult
ParallelOpLowering::matchAndRewrite(acc::ParallelOp parallelOp,
                                    PatternRewriter &rewriter) const {

    //auto parentRegion = parallelOp.getParentRegion();



    rewriter.eraseOp(parallelOp);
    return matchSuccess();
}


//void mlir::populateOpenACCToGPUConversionPatterns(
//        OwningRewritePatternList &patterns, MLIRContext *ctx) {
//
//}

void OpenACCToGPULoweringPass::runOnFunction() {

    ConversionTarget target(getContext());
    target.addLegalDialect<acc::OpenACCOpsDialect>();
    target.addLegalDialect<gpu::GPUDialect>();

    // If operation is considered legal the rewrite pattern in not called.
    target.addIllegalOp<acc::ParallelOp>();

    OwningRewritePatternList patterns;
    patterns.insert<ParallelOpLowering>(&getContext());

    auto f = getFunction();
    f.walk([](Operation *inst) {
        if(auto parallelOp = llvm::dyn_cast_or_null<acc::ParallelOp>(inst)) {

        }
    });

    if (failed(applyPartialConversion(f, target, patterns)))
        signalPassFailure();
}

//std::unique_ptr <OpPassBase<ModuleOp>> createOpenACCToGPUPass() {
//    return std::make_unique<OpenACCToGPULoweringPass>();
//}

/*class ConvertOpenACCToGPUPass : public ModulePass<ConvertOpenACCToGPUPass> {
    void runOnModule() override;
};

void ConvertOpenACCToGPUPass::runOnModule() {

    OwningRewritePatternList patterns;
    auto module = getModule();


//    getCanonicalizationPatterns(patterns, module.getContext());
    ConversionTarget target(getContext());
    target.addLegalDialect<gpu::GPUDialect, AffineOpsDialect, acc::OpenACCOpsDialect>();
    //target.addIllegalDialect<OpenACCOpsDialect>();
    if (failed(applyPartialConversion(module, target, patterns))) {
        return signalPassFailure();
    }
}*/

//std::unique_ptr <OpPassBase<ModuleOp>>
//mlir::createConvertOpenACCToGPUPass() {
//    return std::make_unique<ConvertOpenACCToGPUPass>();
//}

static PassRegistration <OpenACCToGPULoweringPass> pass(
        "convert-openacc-to-gpu",
        "Convert OpenACC Ops to GPU dialect");
