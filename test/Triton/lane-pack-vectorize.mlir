// RUN: triton-opt -triton-vectorize-lane-pack -split-input-file %s | FileCheck %s

module {
  func.func @packed_loop(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>, %arg2: tensor<4xf32>, %arg3: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %0, %1, %2, %3 = scf.for %i = %c0 to %c4 step %c1 iter_args(%r0 = %arg0, %r1 = %arg1, %r2 = %arg2, %r3 = %arg3) -> (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) {
      %s0 = tt.addf %r0, %r0 : tensor<4xf32>
      %s1 = tt.addf %r1, %r1 : tensor<4xf32>
      %s2 = tt.addf %r2, %r2 : tensor<4xf32>
      %s3 = tt.addf %r3, %r3 : tensor<4xf32>
      scf.yield %s0, %s1, %s2, %s3 : tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>
    }
    return %0, %1, %2, %3 : tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>
  }

  // CHECK-LABEL: func.func @packed_loop
  // CHECK-NOT: scf.for
  // CHECK: tt.join
  // CHECK: tt.join
  // CHECK: tt.join
}

// Negative case: mixed loop-carried values must be left alone.
module {
  func.func @mixed_loop(%arg0: tensor<4xf32>, %arg1: i32) -> (tensor<4xf32>, i32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %0, %1 = scf.for %i = %c0 to %c4 step %c1 iter_args(%r0 = %arg0, %r1 = %arg1) -> (tensor<4xf32>, i32) {
      %s0 = tt.addf %r0, %r0 : tensor<4xf32>
      %v1 = arith.addi %r1, %r1 : i32
      scf.yield %s0, %v1 : tensor<4xf32>, i32
    }
    return %0, %1 : tensor<4xf32>, i32
  }

  // CHECK-LABEL: func.func @mixed_loop
  // CHECK: scf.for
}
