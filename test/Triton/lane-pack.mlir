// RUN: triton-opt --triton-lane-pack %s | FileCheck %s

module {
  tt.func @row_col_row(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>, %eps: f32, %lb: index, %ub: index, %step: index) -> (tensor<4xf32>, tensor<4xf32>) {
    %0:2 = scf.for %iv = %lb to %ub step %step iter_args(%lane0 = %arg0, %lane1 = %arg1) -> (tensor<4xf32>, tensor<4xf32>) {
      %r0 = "tt.reduce"(%lane0) <{axis = 0 : i32}> ({
      ^bb0(%a: f32, %b: f32):
        %sum0 = arith.addf %a, %b : f32
        tt.reduce.return %sum0 : f32
      }) : (tensor<4xf32>) -> f32
      %den0s = arith.addf %r0, %eps : f32
      %den0 = tt.splat %den0s : f32 -> tensor<4xf32>
      %row0 = arith.divf %lane0, %den0 : tensor<4xf32>

      %r1 = "tt.reduce"(%lane1) <{axis = 0 : i32}> ({
      ^bb0(%a0: f32, %b0: f32):
        %sum1 = arith.addf %a0, %b0 : f32
        tt.reduce.return %sum1 : f32
      }) : (tensor<4xf32>) -> f32
      %den1s = arith.addf %r1, %eps : f32
      %den1 = tt.splat %den1s : f32 -> tensor<4xf32>
      %row1 = arith.divf %lane1, %den1 : tensor<4xf32>

      %colsum = arith.addf %row0, %row1 : tensor<4xf32>
      %epst = tt.splat %eps : f32 -> tensor<4xf32>
      %colden = arith.addf %colsum, %epst : tensor<4xf32>
      %col0 = arith.divf %row0, %colden : tensor<4xf32>
      %col1 = arith.divf %row1, %colden : tensor<4xf32>

      %r2 = "tt.reduce"(%col0) <{axis = 0 : i32}> ({
      ^bb0(%a1: f32, %b1: f32):
        %sum2 = arith.addf %a1, %b1 : f32
        tt.reduce.return %sum2 : f32
      }) : (tensor<4xf32>) -> f32
      %den2s = arith.addf %r2, %eps : f32
      %den2 = tt.splat %den2s : f32 -> tensor<4xf32>
      %row2 = arith.divf %col0, %den2 : tensor<4xf32>

      %r3 = "tt.reduce"(%col1) <{axis = 0 : i32}> ({
      ^bb0(%a2: f32, %b2: f32):
        %sum3 = arith.addf %a2, %b2 : f32
        tt.reduce.return %sum3 : f32
      }) : (tensor<4xf32>) -> f32
      %den3s = arith.addf %r3, %eps : f32
      %den3 = tt.splat %den3s : f32 -> tensor<4xf32>
      %row3 = arith.divf %col1, %den3 : tensor<4xf32>
      scf.yield %row2, %row3 : tensor<4xf32>, tensor<4xf32>
    }
    return %0#0, %0#1 : tensor<4xf32>, tensor<4xf32>
  }

  // CHECK-LABEL: tt.func @row_col_row(
  // CHECK: %[[PACK:.*]] = tensor.concat
  // CHECK: %[[FOR:.*]] = scf.for
  // CHECK: %[[ROWSUM0:.*]] = "tt.reduce"(%{{.*}}) <{axis = 1 : i32}>
  // CHECK: %[[COLSUM:.*]] = "tt.reduce"(%{{.*}}) <{axis = 0 : i32}>
  // CHECK: %[[ROWSUM1:.*]] = "tt.reduce"(%{{.*}}) <{axis = 1 : i32}>

  tt.func @col_row(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>, %eps: f32, %lb: index, %ub: index, %step: index) -> (tensor<4xf32>, tensor<4xf32>) {
    %0:2 = scf.for %iv = %lb to %ub step %step iter_args(%lane0 = %arg0, %lane1 = %arg1) -> (tensor<4xf32>, tensor<4xf32>) {
      %sum = arith.addf %lane0, %lane1 : tensor<4xf32>
      %epsden = tt.splat %eps : f32 -> tensor<4xf32>
      %den = arith.addf %sum, %epsden : tensor<4xf32>
      %col0 = arith.divf %lane0, %den : tensor<4xf32>
      %col1 = arith.divf %lane1, %den : tensor<4xf32>

      %r0 = "tt.reduce"(%col0) <{axis = 0 : i32}> ({
      ^bb0(%a: f32, %b: f32):
        %sum0 = arith.addf %a, %b : f32
        tt.reduce.return %sum0 : f32
      }) : (tensor<4xf32>) -> f32
      %den0s = arith.addf %r0, %eps : f32
      %rowden0 = tt.splat %den0s : f32 -> tensor<4xf32>
      %row0 = arith.divf %col0, %rowden0 : tensor<4xf32>

      %r1 = "tt.reduce"(%col1) <{axis = 0 : i32}> ({
      ^bb0(%a0: f32, %b0: f32):
        %sum1 = arith.addf %a0, %b0 : f32
        tt.reduce.return %sum1 : f32
      }) : (tensor<4xf32>) -> f32
      %den1s = arith.addf %r1, %eps : f32
      %rowden1 = tt.splat %den1s : f32 -> tensor<4xf32>
      %row1 = arith.divf %col1, %rowden1 : tensor<4xf32>
      scf.yield %row0, %row1 : tensor<4xf32>, tensor<4xf32>
    }
    return %0#0, %0#1 : tensor<4xf32>, tensor<4xf32>
  }

  // CHECK-LABEL: tt.func @col_row(
  // CHECK: scf.for
  // CHECK: "tt.reduce"(%{{.*}}) <{axis = 0 : i32}>
  // CHECK: "tt.reduce"(%{{.*}}) <{axis = 1 : i32}>
}
