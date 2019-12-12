//===- mlir_runner_utils.cpp - Utils for MLIR CPU execution ---------------===//
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
// Utilities for interfacing MLIR types with C code as well as printing,
// debugging etc.
//
//===----------------------------------------------------------------------===//

#include "include/mlir_runner_utils.h"

extern "C" void
print_memref_vector_4x4xf32(StridedMemRefType<Vector2D<4, 4, float>, 2> *M) {
  impl::printMemRef(*M);
}

extern "C" void print_memref_f32(UnrankedMemRefType<float> *M) {
  printUnrankedMemRefMetaData(std::cout, *M);
  int rank = M->rank;
  void *ptr = M->descriptor;

#define MEMREF_CASE(RANK)                                                      \
  case RANK:                                                                   \
    impl::printMemRef(*(static_cast<StridedMemRefType<float, RANK> *>(ptr)));  \
    break

  switch (rank) {
    MEMREF_CASE(0);
    MEMREF_CASE(1);
    MEMREF_CASE(2);
    MEMREF_CASE(3);
    MEMREF_CASE(4);
  default:
    assert(0 && "Unsupported rank to print");
  }
}

extern "C" void print_memref_0d_f32(StridedMemRefType<float, 0> *M) {
  impl::printMemRef(*M);
}
extern "C" void print_memref_1d_f32(StridedMemRefType<float, 1> *M) {
  impl::printMemRef(*M);
}
extern "C" void print_memref_2d_f32(StridedMemRefType<float, 2> *M) {
  impl::printMemRef(*M);
}
extern "C" void print_memref_3d_f32(StridedMemRefType<float, 3> *M) {
  impl::printMemRef(*M);
}
extern "C" void print_memref_4d_f32(StridedMemRefType<float, 4> *M) {
  impl::printMemRef(*M);
}
