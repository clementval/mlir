//===- Ops.cpp - Loop MLIR Operations -------------------------------------===//
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

#include "mlir/Dialect/OpenACCOps/OpenACCOps.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/MathExtras.h"
#include "mlir/Support/STLExtras.h"
#include "mlir/Transforms/SideEffectsInterface.h"

using namespace mlir;
using namespace acc;

//===----------------------------------------------------------------------===//
// OpenACCOpsDialect
//===----------------------------------------------------------------------===//

OpenACCOpsDialect::OpenACCOpsDialect(MLIRContext *context)
        : Dialect(getDialectNamespace(), context) {
    addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/OpenACCOps/OpenACCOps.cpp.inc"
    >();
    allowsUnknownOperations();
}

// Parses an op that has no inputs and no outputs.
static ParseResult parseNoIOOp(OpAsmParser &parser, OperationState &state) {
    if (parser.parseOptionalAttrDict(state.attributes))
        return failure();
    return success();
}

/*
template <typename StructureOp>
static ParseResult parseRegionOp(OpAsmParser &parser, OperationState &state,
                                 unsigned int nRegions = 1) {
    llvm::SmallVector<Region *, 2> regions;
    for (unsigned int i = 0; i < nRegions; ++i)
        regions.push_back(state.addRegion());

    for (auto &region : regions) {
        if (parser.parseRegion(*region, */
/*arguments=*//*
{}, */
/*argTypes=*//*
{}))
            return failure();
        StructureOp::ensureTerminator(*region, parser.getBuilder(), state.location);
    }

    if (succeeded(parser.parseOptionalKeyword("attributes"))) {
        if (parser.parseOptionalAttrDict(state.attributes))
            return failure();
    }

    return success();
}*/

static void printNoIOOp(Operation *op, OpAsmPrinter &printer) {
    printer << op->getName();
    printer.printOptionalAttrDict(op->getAttrs());
}

template <typename StructureOp>
static ParseResult parseRegionOp(OpAsmParser &parser, OperationState &state,
                                 unsigned int nRegions = 1) {

    llvm::SmallVector<Region *, 2> regions;
    for (unsigned int i = 0; i < nRegions; ++i) {
        regions.push_back(state.addRegion());
    }

    for (auto &region : regions) {
        if (parser.parseRegion(*region, /*arguments=*/{}, /*argTypes=*/{})) {
            return failure();
        }
        StructureOp::ensureTerminator(*region, parser.getBuilder(), state.location);
    }

    if (succeeded(parser.parseOptionalKeyword("attributes"))) {
        if (parser.parseOptionalAttrDict(state.attributes))
            return failure();
    }

    return success();
}

//===----------------------------------------------------------------------===//
// ParallelOp
//===----------------------------------------------------------------------===//

//static void print(OpAsmPrinter &p, ParallelOp op) {
//    p << op.getOperationName();
//   // p.printRegion(op.region(),
//            ///*printEntryBlockArgs=*/false,
//            ///*printBlockTerminators=*/false);
//    p.printOptionalAttrDict(op.getAttrs());
//}

static void printRegionOp(Operation *op, OpAsmPrinter &printer) {
    printer << op->getName();

    for (auto &region : op->getRegions()) {
        printer.printRegion(region, false, false);
    }

    if (!op->getAttrs().empty()) {
        printer << " attributes";
        printer.printOptionalAttrDict(op->getAttrs());
    }
}

void ParallelOp::build(Builder *builder, OperationState &state) {
    ensureTerminator(*state.addRegion(), *builder, state.location);
}


//===----------------------------------------------------------------------===//
// LoopOp
//===----------------------------------------------------------------------===//
/*
void LoopOp::build(Builder *builder, OperationState &result) {
    Region *bodyRegion = result.addRegion();
}*/

//static void print(OpAsmPrinter &p, LoopOp op) {
//    p << LoopOp::getOperationName();
//}

// TODO use the Region parser for now
/*
static ParseResult parseLoopOp(OpAsmParser &parser, OperationState &result) {


    //auto &builder = parser.getBuilder();

    Region *bodyRegion = result.addRegion();
    if (parser.parseRegion(*bodyRegion, {}, {}))
        return failure();
//    IfOp::ensureTerminator(*thenRegion, parser.getBuilder(), result.location);

    // Parse the optional attribute list.
    if (parser.parseOptionalAttrDict(result.attributes))
        return failure();

    return success();
}*/


/*
static void print(OpAsmPrinter &p, IfOp op) {
    p << LoopOp::getOperationName();
    //p.printRegion(op.bodyRegion(),
            *//*printEntryBlockArgs=*//*false,
            *//*printBlockTerminators=*//*false);

    p.printOptionalAttrDict(op.getAttrs());
}*/

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/OpenACCOps/OpenACCOps.cpp.inc"
