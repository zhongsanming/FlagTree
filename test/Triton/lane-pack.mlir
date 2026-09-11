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
    tt.return %0#0, %0#1 : tensor<4xf32>, tensor<4xf32>
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
    tt.return %0#0, %0#1 : tensor<4xf32>, tensor<4xf32>
  }

  // CHECK-LABEL: tt.func @col_row(
  // CHECK: scf.for
  // CHECK: "tt.reduce"(%{{.*}}) <{axis = 0 : i32}>
  // CHECK: "tt.reduce"(%{{.*}}) <{axis = 1 : i32}>

  tt.func @row_row(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>, %eps: f32, %lb: index, %ub: index, %step: index) -> (tensor<4xf32>, tensor<4xf32>) {
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

      %r2 = "tt.reduce"(%row0) <{axis = 0 : i32}> ({
      ^bb0(%a1: f32, %b1: f32):
        %sum2 = arith.addf %a1, %b1 : f32
        tt.reduce.return %sum2 : f32
      }) : (tensor<4xf32>) -> f32
      %den2s = arith.addf %r2, %eps : f32
      %den2 = tt.splat %den2s : f32 -> tensor<4xf32>
      %row2 = arith.divf %row0, %den2 : tensor<4xf32>

      %r3 = "tt.reduce"(%row1) <{axis = 0 : i32}> ({
      ^bb0(%a2: f32, %b2: f32):
        %sum3 = arith.addf %a2, %b2 : f32
        tt.reduce.return %sum3 : f32
      }) : (tensor<4xf32>) -> f32
      %den3s = arith.addf %r3, %eps : f32
      %den3 = tt.splat %den3s : f32 -> tensor<4xf32>
      %row3 = arith.divf %row1, %den3 : tensor<4xf32>
      scf.yield %row2, %row3 : tensor<4xf32>, tensor<4xf32>
    }
    tt.return %0#0, %0#1 : tensor<4xf32>, tensor<4xf32>
  }

  // CHECK-LABEL: tt.func @row_row(
  // CHECK: "tt.reduce"(%{{.*}}) <{axis = 1 : i32}>
  // CHECK: "tt.reduce"(%{{.*}}) <{axis = 1 : i32}>

  tt.func @col_col(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>, %eps: f32, %lb: index, %ub: index, %step: index) -> (tensor<4xf32>, tensor<4xf32>) {
    %0:2 = scf.for %iv = %lb to %ub step %step iter_args(%lane0 = %arg0, %lane1 = %arg1) -> (tensor<4xf32>, tensor<4xf32>) {
      %sum0 = arith.addf %lane0, %lane1 : tensor<4xf32>
      %eps0 = tt.splat %eps : f32 -> tensor<4xf32>
      %den0 = arith.addf %sum0, %eps0 : tensor<4xf32>
      %col0 = arith.divf %lane0, %den0 : tensor<4xf32>
      %col1 = arith.divf %lane1, %den0 : tensor<4xf32>

      %sum1 = arith.addf %col0, %col1 : tensor<4xf32>
      %eps1 = tt.splat %eps : f32 -> tensor<4xf32>
      %den1 = arith.addf %sum1, %eps1 : tensor<4xf32>
      %col2 = arith.divf %col0, %den1 : tensor<4xf32>
      %col3 = arith.divf %col1, %den1 : tensor<4xf32>
      scf.yield %col2, %col3 : tensor<4xf32>, tensor<4xf32>
    }
    tt.return %0#0, %0#1 : tensor<4xf32>, tensor<4xf32>
  }

  // CHECK-LABEL: tt.func @col_col(
  // CHECK: "tt.reduce"(%{{.*}}) <{axis = 0 : i32}>
  // CHECK: "tt.reduce"(%{{.*}}) <{axis = 0 : i32}>

  tt.func @row_row_no_eps(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>, %lb: index, %ub: index, %step: index) -> (tensor<4xf32>, tensor<4xf32>) {
    %0:2 = scf.for %iv = %lb to %ub step %step iter_args(%lane0 = %arg0, %lane1 = %arg1) -> (tensor<4xf32>, tensor<4xf32>) {
      %r0 = "tt.reduce"(%lane0) <{axis = 0 : i32}> ({
      ^bb0(%a: f32, %b: f32):
        %sum0 = arith.addf %a, %b : f32
        tt.reduce.return %sum0 : f32
      }) : (tensor<4xf32>) -> f32
      %den0 = tt.splat %r0 : f32 -> tensor<4xf32>
      %row0 = arith.divf %lane0, %den0 : tensor<4xf32>

      %r1 = "tt.reduce"(%lane1) <{axis = 0 : i32}> ({
      ^bb0(%a0: f32, %b0: f32):
        %sum1 = arith.addf %a0, %b0 : f32
        tt.reduce.return %sum1 : f32
      }) : (tensor<4xf32>) -> f32
      %den1 = tt.splat %r1 : f32 -> tensor<4xf32>
      %row1 = arith.divf %lane1, %den1 : tensor<4xf32>

      %r2 = "tt.reduce"(%row0) <{axis = 0 : i32}> ({
      ^bb0(%a1: f32, %b1: f32):
        %sum2 = arith.addf %a1, %b1 : f32
        tt.reduce.return %sum2 : f32
      }) : (tensor<4xf32>) -> f32
      %den2 = tt.splat %r2 : f32 -> tensor<4xf32>
      %row2 = arith.divf %row0, %den2 : tensor<4xf32>

      %r3 = "tt.reduce"(%row1) <{axis = 0 : i32}> ({
      ^bb0(%a2: f32, %b2: f32):
        %sum3 = arith.addf %a2, %b2 : f32
        tt.reduce.return %sum3 : f32
      }) : (tensor<4xf32>) -> f32
      %den3 = tt.splat %r3 : f32 -> tensor<4xf32>
      %row3 = arith.divf %row1, %den3 : tensor<4xf32>
      scf.yield %row2, %row3 : tensor<4xf32>, tensor<4xf32>
    }
    tt.return %0#0, %0#1 : tensor<4xf32>, tensor<4xf32>
  }

  // CHECK-LABEL: tt.func @row_row_no_eps(
  // CHECK: "tt.reduce"(%{{.*}}) <{axis = 1 : i32}>
  // CHECK: "tt.reduce"(%{{.*}}) <{axis = 1 : i32}>

  tt.func @col_col_no_eps(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>, %lb: index, %ub: index, %step: index) -> (tensor<4xf32>, tensor<4xf32>) {
    %0:2 = scf.for %iv = %lb to %ub step %step iter_args(%lane0 = %arg0, %lane1 = %arg1) -> (tensor<4xf32>, tensor<4xf32>) {
      %sum0 = arith.addf %lane0, %lane1 : tensor<4xf32>
      %col0 = arith.divf %lane0, %sum0 : tensor<4xf32>
      %col1 = arith.divf %lane1, %sum0 : tensor<4xf32>

      %sum1 = arith.addf %col0, %col1 : tensor<4xf32>
      %col2 = arith.divf %col0, %sum1 : tensor<4xf32>
      %col3 = arith.divf %col1, %sum1 : tensor<4xf32>
      scf.yield %col2, %col3 : tensor<4xf32>, tensor<4xf32>
    }
    tt.return %0#0, %0#1 : tensor<4xf32>, tensor<4xf32>
  }

  // CHECK-LABEL: tt.func @col_col_no_eps(
  // CHECK: "tt.reduce"(%{{.*}}) <{axis = 0 : i32}>
  // CHECK: "tt.reduce"(%{{.*}}) <{axis = 0 : i32}>

  tt.func @col_row_no_eps(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>, %lb: index, %ub: index, %step: index) -> (tensor<4xf32>, tensor<4xf32>) {
    %0:2 = scf.for %iv = %lb to %ub step %step iter_args(%lane0 = %arg0, %lane1 = %arg1) -> (tensor<4xf32>, tensor<4xf32>) {
      %sum = arith.addf %lane0, %lane1 : tensor<4xf32>
      %col0 = arith.divf %lane0, %sum : tensor<4xf32>
      %col1 = arith.divf %lane1, %sum : tensor<4xf32>

      %r0 = "tt.reduce"(%col0) <{axis = 0 : i32}> ({
      ^bb0(%a: f32, %b: f32):
        %sum0 = arith.addf %a, %b : f32
        tt.reduce.return %sum0 : f32
      }) : (tensor<4xf32>) -> f32
      %rowden0 = tt.splat %r0 : f32 -> tensor<4xf32>
      %row0 = arith.divf %col0, %rowden0 : tensor<4xf32>

      %r1 = "tt.reduce"(%col1) <{axis = 0 : i32}> ({
      ^bb0(%a0: f32, %b0: f32):
        %sum1 = arith.addf %a0, %b0 : f32
        tt.reduce.return %sum1 : f32
      }) : (tensor<4xf32>) -> f32
      %rowden1 = tt.splat %r1 : f32 -> tensor<4xf32>
      %row1 = arith.divf %col1, %rowden1 : tensor<4xf32>
      scf.yield %row0, %row1 : tensor<4xf32>, tensor<4xf32>
    }
    tt.return %0#0, %0#1 : tensor<4xf32>, tensor<4xf32>
  }

  // CHECK-LABEL: tt.func @col_row_no_eps(
  // CHECK: scf.for
  // CHECK: "tt.reduce"(%{{.*}}) <{axis = 0 : i32}>
  // CHECK: "tt.reduce"(%{{.*}}) <{axis = 1 : i32}>

  tt.func @col_row_four_lane(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>, %arg2: tensor<4xf32>, %arg3: tensor<4xf32>, %eps_t: tensor<4xf32>, %eps: f32, %lb: index, %ub: index, %step: index) -> (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) {
    %0:4 = scf.for %iv = %lb to %ub step %step iter_args(%lane0 = %arg0, %lane1 = %arg1, %lane2 = %arg2, %lane3 = %arg3) -> (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) {
      %sum0 = arith.addf %lane0, %lane1 : tensor<4xf32>
      %sum1 = arith.addf %sum0, %lane2 : tensor<4xf32>
      %sum2 = arith.addf %sum1, %lane3 : tensor<4xf32>
      %den = arith.addf %sum2, %eps_t : tensor<4xf32>
      %col0 = arith.divf %lane0, %den : tensor<4xf32>
      %col1 = arith.divf %lane1, %den : tensor<4xf32>
      %col2 = arith.divf %lane2, %den : tensor<4xf32>
      %col3 = arith.divf %lane3, %den : tensor<4xf32>

      %r0 = "tt.reduce"(%col0) <{axis = 0 : i32}> ({
      ^bb0(%a: f32, %b: f32):
        %sumr0 = arith.addf %a, %b : f32
        tt.reduce.return %sumr0 : f32
      }) : (tensor<4xf32>) -> f32
      %d0s = arith.addf %r0, %eps : f32
      %d0 = tt.splat %d0s : f32 -> tensor<4xf32>
      %row0 = arith.divf %col0, %d0 : tensor<4xf32>

      %r1 = "tt.reduce"(%col1) <{axis = 0 : i32}> ({
      ^bb0(%a1: f32, %b1: f32):
        %sumr1 = arith.addf %a1, %b1 : f32
        tt.reduce.return %sumr1 : f32
      }) : (tensor<4xf32>) -> f32
      %d1s = arith.addf %r1, %eps : f32
      %d1 = tt.splat %d1s : f32 -> tensor<4xf32>
      %row1 = arith.divf %col1, %d1 : tensor<4xf32>

      %r2 = "tt.reduce"(%col2) <{axis = 0 : i32}> ({
      ^bb0(%a2: f32, %b2: f32):
        %sumr2 = arith.addf %a2, %b2 : f32
        tt.reduce.return %sumr2 : f32
      }) : (tensor<4xf32>) -> f32
      %d2s = arith.addf %r2, %eps : f32
      %d2 = tt.splat %d2s : f32 -> tensor<4xf32>
      %row2 = arith.divf %col2, %d2 : tensor<4xf32>

      %r3 = "tt.reduce"(%col3) <{axis = 0 : i32}> ({
      ^bb0(%a3: f32, %b3: f32):
        %sumr3 = arith.addf %a3, %b3 : f32
        tt.reduce.return %sumr3 : f32
      }) : (tensor<4xf32>) -> f32
      %d3s = arith.addf %r3, %eps : f32
      %d3 = tt.splat %d3s : f32 -> tensor<4xf32>
      %row3 = arith.divf %col3, %d3 : tensor<4xf32>
      scf.yield %row0, %row1, %row2, %row3 : tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>
    }
    tt.return %0#0, %0#1, %0#2, %0#3 : tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>
  }

  // CHECK-LABEL: tt.func @col_row_four_lane(
  // CHECK: tensor.concat
  // CHECK: "tt.reduce"(%{{.*}}) <{axis = 0 : i32}>
  // CHECK: "tt.reduce"(%{{.*}}) <{axis = 1 : i32}>

  tt.func @reject_unmatched_extra_op(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>, %eps: f32, %lb: index, %ub: index, %step: index) -> (tensor<4xf32>, tensor<4xf32>) {
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

      %extra = arith.addf %row0, %row0 : tensor<4xf32>
      scf.yield %extra, %row1 : tensor<4xf32>, tensor<4xf32>
    }
    tt.return %0#0, %0#1 : tensor<4xf32>, tensor<4xf32>
  }

  // CHECK-LABEL: tt.func @reject_unmatched_extra_op(
  // CHECK-NOT: tensor.concat

  tt.func @row_softmax(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>, %lb: index, %ub: index, %step: index) -> (tensor<4xf32>, tensor<4xf32>) {
    %0:2 = scf.for %iv = %lb to %ub step %step iter_args(%lane0 = %arg0, %lane1 = %arg1) -> (tensor<4xf32>, tensor<4xf32>) {
      %m0 = "tt.reduce"(%lane0) <{axis = 0 : i32}> ({
      ^bb0(%a: f32, %b: f32):
        %mx = arith.maximumf %a, %b : f32
        tt.reduce.return %mx : f32
      }) : (tensor<4xf32>) -> f32
      %mb0 = tt.splat %m0 : f32 -> tensor<4xf32>
      %e0 = arith.subf %lane0, %mb0 : tensor<4xf32>
      %ex0 = math.exp %e0 : tensor<4xf32>
      %s0 = "tt.reduce"(%ex0) <{axis = 0 : i32}> ({
      ^bb0(%c: f32, %d: f32):
        %sm = arith.addf %c, %d : f32
        tt.reduce.return %sm : f32
      }) : (tensor<4xf32>) -> f32
      %sb0 = tt.splat %s0 : f32 -> tensor<4xf32>
      %out0 = arith.divf %ex0, %sb0 : tensor<4xf32>

      %m1 = "tt.reduce"(%lane1) <{axis = 0 : i32}> ({
      ^bb0(%a0: f32, %b0: f32):
        %mx1 = arith.maximumf %a0, %b0 : f32
        tt.reduce.return %mx1 : f32
      }) : (tensor<4xf32>) -> f32
      %mb1 = tt.splat %m1 : f32 -> tensor<4xf32>
      %e1 = arith.subf %lane1, %mb1 : tensor<4xf32>
      %ex1 = math.exp %e1 : tensor<4xf32>
      %s1 = "tt.reduce"(%ex1) <{axis = 0 : i32}> ({
      ^bb0(%c0: f32, %d0: f32):
        %sm1 = arith.addf %c0, %d0 : f32
        tt.reduce.return %sm1 : f32
      }) : (tensor<4xf32>) -> f32
      %sb1 = tt.splat %s1 : f32 -> tensor<4xf32>
      %out1 = arith.divf %ex1, %sb1 : tensor<4xf32>
      scf.yield %out0, %out1 : tensor<4xf32>, tensor<4xf32>
    }
    tt.return %0#0, %0#1 : tensor<4xf32>, tensor<4xf32>
  }

  // CHECK-LABEL: tt.func @row_softmax(
  // CHECK: tensor.concat
  // CHECK: "tt.reduce"(%{{.*}}) <{axis = 1 : i32}>
  // CHECK: math.exp
  // CHECK: "tt.reduce"(%{{.*}}) <{axis = 1 : i32}>

  tt.func @clamp_lanes(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>, %lo: tensor<4xf32>, %hi: tensor<4xf32>, %lb: index, %ub: index, %step: index) -> (tensor<4xf32>, tensor<4xf32>) {
    %0:2 = scf.for %iv = %lb to %ub step %step iter_args(%lane0 = %arg0, %lane1 = %arg1) -> (tensor<4xf32>, tensor<4xf32>) {
      %c0 = arith.maximumf %lane0, %lo : tensor<4xf32>
      %d0 = arith.minimumf %c0, %hi : tensor<4xf32>
      %c1 = arith.maximumf %lane1, %lo : tensor<4xf32>
      %d1 = arith.minimumf %c1, %hi : tensor<4xf32>
      scf.yield %d0, %d1 : tensor<4xf32>, tensor<4xf32>
    }
    tt.return %0#0, %0#1 : tensor<4xf32>, tensor<4xf32>
  }

  // CHECK-LABEL: tt.func @clamp_lanes(
  // CHECK: tensor.concat
  // CHECK: arith.maximumf
  // CHECK: arith.minimumf

  tt.func @reject_divergent_lanes(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>, %lb: index, %ub: index, %step: index) -> (tensor<4xf32>, tensor<4xf32>) {
    %0:2 = scf.for %iv = %lb to %ub step %step iter_args(%lane0 = %arg0, %lane1 = %arg1) -> (tensor<4xf32>, tensor<4xf32>) {
      %a = arith.addf %lane0, %lane0 : tensor<4xf32>
      %b = arith.mulf %lane1, %lane1 : tensor<4xf32>
      scf.yield %a, %b : tensor<4xf32>, tensor<4xf32>
    }
    tt.return %0#0, %0#1 : tensor<4xf32>, tensor<4xf32>
  }

  // CHECK-LABEL: tt.func @reject_divergent_lanes(
  // CHECK-NOT: tensor.concat

  tt.func @mixed_iter_args(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>, %acc0: f32, %lb: index, %ub: index, %step: index) -> (tensor<4xf32>, tensor<4xf32>, f32) {
    %0:3 = scf.for %iv = %lb to %ub step %step iter_args(%lane0 = %arg0, %lane1 = %arg1, %acc = %acc0) -> (tensor<4xf32>, tensor<4xf32>, f32) {
      %r0 = "tt.reduce"(%lane0) <{axis = 0 : i32}> ({
      ^bb0(%a: f32, %b: f32):
        %sum0 = arith.addf %a, %b : f32
        tt.reduce.return %sum0 : f32
      }) : (tensor<4xf32>) -> f32
      %den0 = tt.splat %r0 : f32 -> tensor<4xf32>
      %row0 = arith.divf %lane0, %den0 : tensor<4xf32>

      %r1 = "tt.reduce"(%lane1) <{axis = 0 : i32}> ({
      ^bb0(%a0: f32, %b0: f32):
        %sum1 = arith.addf %a0, %b0 : f32
        tt.reduce.return %sum1 : f32
      }) : (tensor<4xf32>) -> f32
      %den1 = tt.splat %r1 : f32 -> tensor<4xf32>
      %row1 = arith.divf %lane1, %den1 : tensor<4xf32>

      %acc2 = arith.addf %acc, %acc : f32
      scf.yield %row0, %row1, %acc2 : tensor<4xf32>, tensor<4xf32>, f32
    }
    tt.return %0#0, %0#1, %0#2 : tensor<4xf32>, tensor<4xf32>, f32
  }

  // CHECK-LABEL: tt.func @mixed_iter_args(
  // CHECK: tensor.concat
  // CHECK: "tt.reduce"(%{{.*}}) <{axis = 1 : i32}>

  tt.func @trans_lanes(%arg0: tensor<4x2xf32>, %arg1: tensor<4x2xf32>, %lb: index, %ub: index, %step: index) -> (tensor<4x2xf32>, tensor<4x2xf32>) {
    %0:2 = scf.for %iv = %lb to %ub step %step iter_args(%lane0 = %arg0, %lane1 = %arg1) -> (tensor<4x2xf32>, tensor<4x2xf32>) {
      %t0 = "tt.trans"(%lane0) <{order = array<i32: 1, 0>}> : (tensor<4x2xf32>) -> tensor<2x4xf32>
      %u0 = "tt.trans"(%t0) <{order = array<i32: 1, 0>}> : (tensor<2x4xf32>) -> tensor<4x2xf32>
      %t1 = "tt.trans"(%lane1) <{order = array<i32: 1, 0>}> : (tensor<4x2xf32>) -> tensor<2x4xf32>
      %u1 = "tt.trans"(%t1) <{order = array<i32: 1, 0>}> : (tensor<2x4xf32>) -> tensor<4x2xf32>
      scf.yield %u0, %u1 : tensor<4x2xf32>, tensor<4x2xf32>
    }
    tt.return %0#0, %0#1 : tensor<4x2xf32>, tensor<4x2xf32>
  }

  // CHECK-LABEL: tt.func @trans_lanes(
  // CHECK: tensor.concat
  // CHECK: tt.trans %{{.*}} {order = array<i32: 0, 2, 1>}

  tt.func @straight_line(%x0: tensor<4xf32>, %x1: tensor<4xf32>, %y0: tensor<4xf32>, %y1: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
    %r0 = arith.addf %x0, %y0 : tensor<4xf32>
    %r1 = arith.addf %x1, %y1 : tensor<4xf32>
    %s0 = arith.mulf %r0, %x0 : tensor<4xf32>
    %s1 = arith.mulf %r1, %x1 : tensor<4xf32>
    tt.return %s0, %s1 : tensor<4xf32>, tensor<4xf32>
  }

  // CHECK-LABEL: tt.func @straight_line(
  // CHECK: tensor.concat
  // CHECK: arith.addf
  // CHECK: arith.mulf

  tt.func @straight_line_shared(%x0: tensor<4xf32>, %x1: tensor<4xf32>, %c: f32) -> (tensor<4xf32>, tensor<4xf32>) {
    %cs = tt.splat %c : f32 -> tensor<4xf32>
    %r0 = arith.mulf %x0, %cs : tensor<4xf32>
    %r1 = arith.mulf %x1, %cs : tensor<4xf32>
    tt.return %r0, %r1 : tensor<4xf32>, tensor<4xf32>
  }

  // CHECK-LABEL: tt.func @straight_line_shared(
  // CHECK: tensor.concat
  // CHECK: tt.broadcast
  // CHECK: arith.mulf

  tt.func @straight_line_four(%x0: tensor<4xf32>, %x1: tensor<4xf32>, %x2: tensor<4xf32>, %x3: tensor<4xf32>, %y0: tensor<4xf32>, %y1: tensor<4xf32>, %y2: tensor<4xf32>, %y3: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) {
    %r0 = arith.addf %x0, %y0 : tensor<4xf32>
    %r1 = arith.addf %x1, %y1 : tensor<4xf32>
    %r2 = arith.addf %x2, %y2 : tensor<4xf32>
    %r3 = arith.addf %x3, %y3 : tensor<4xf32>
    tt.return %r0, %r1, %r2, %r3 : tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>
  }

  // CHECK-LABEL: tt.func @straight_line_four(
  // CHECK: %[[C:.*]] = tensor.concat
  // CHECK: arith.addf %[[C]]

  tt.func @prologue_in_outer_loop(%x0: tensor<4xf32>, %x1: tensor<4xf32>, %y0: tensor<4xf32>, %y1: tensor<4xf32>, %o0: tensor<4x!tt.ptr<f32>>, %o1: tensor<4x!tt.ptr<f32>>, %lb: index, %ub: index, %step: index) {
    scf.for %i = %lb to %ub step %step {
      %r0 = arith.addf %x0, %y0 : tensor<4xf32>
      %r1 = arith.addf %x1, %y1 : tensor<4xf32>
      tt.store %o0, %r0 : tensor<4x!tt.ptr<f32>>
      tt.store %o1, %r1 : tensor<4x!tt.ptr<f32>>
      scf.yield
    }
    tt.return
  }

  // CHECK-LABEL: tt.func @prologue_in_outer_loop(
  // CHECK: tensor.concat
  // CHECK: arith.addf

  tt.func @prologue_feeds_packed_loop(%x0: tensor<4xf32>, %x1: tensor<4xf32>, %y0: tensor<4xf32>, %y1: tensor<4xf32>, %lb: index, %ub: index, %step: index) -> (tensor<4xf32>, tensor<4xf32>) {
    %r0 = arith.addf %x0, %y0 : tensor<4xf32>
    %r1 = arith.addf %x1, %y1 : tensor<4xf32>
    %cst = arith.constant dense<[1, 4]> : tensor<2xi64>
    %reshape0 = tensor.reshape %r0(%cst) : (tensor<4xf32>, tensor<2xi64>) -> tensor<1x4xf32>
    %reshape1 = tensor.reshape %r1(%cst) : (tensor<4xf32>, tensor<2xi64>) -> tensor<1x4xf32>
    %concat = tensor.concat dim(0) %reshape0, %reshape1 : (tensor<1x4xf32>, tensor<1x4xf32>) -> tensor<2x4xf32>
    %0 = scf.for %i = %lb to %ub step %step iter_args(%acc = %concat) -> (tensor<2x4xf32>) {
      %rd = "tt.reduce"(%acc) <{axis = 1 : i32}> ({
      ^bb0(%a: f32, %b: f32):
        %s = arith.addf %a, %b : f32
        tt.reduce.return %s : f32
      }) : (tensor<2x4xf32>) -> tensor<2xf32>
      %ed = tt.expand_dims %rd {axis = 1 : i32} : tensor<2xf32> -> tensor<2x1xf32>
      %bc = tt.broadcast %ed : tensor<2x1xf32> -> tensor<2x4xf32>
      %d = arith.divf %acc, %bc : tensor<2x4xf32>
      scf.yield %d : tensor<2x4xf32>
    }
    %cst2 = arith.constant dense<4> : tensor<1xi64>
    %e0 = tensor.extract_slice %0[0, 0] [1, 4] [1, 1] : tensor<2x4xf32> to tensor<1x4xf32>
    %res0 = tensor.reshape %e0(%cst2) : (tensor<1x4xf32>, tensor<1xi64>) -> tensor<4xf32>
    %e1 = tensor.extract_slice %0[1, 0] [1, 4] [1, 1] : tensor<2x4xf32> to tensor<1x4xf32>
    %res1 = tensor.reshape %e1(%cst2) : (tensor<1x4xf32>, tensor<1xi64>) -> tensor<4xf32>
    tt.return %res0, %res1 : tensor<4xf32>, tensor<4xf32>
  }

  // CHECK-LABEL: tt.func @prologue_feeds_packed_loop(
  // CHECK: tensor.concat
  // CHECK: tensor.concat
  // CHECK: %[[PACKED:.*]] = arith.addf
  // CHECK: scf.for {{.*}} iter_args(%{{.*}} = %[[PACKED]])

  tt.func @softmax_prologue_feeds_packed_loop(%x0: tensor<4xf32>, %x1: tensor<4xf32>, %lb: index, %ub: index, %step: index) -> (tensor<4xf32>, tensor<4xf32>) {
    %m0 = "tt.reduce"(%x0) <{axis = 0 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %mx = arith.maxnumf %a, %b : f32
      tt.reduce.return %mx : f32
    }) : (tensor<4xf32>) -> f32
    %m1 = "tt.reduce"(%x1) <{axis = 0 : i32}> ({
    ^bb0(%c: f32, %d: f32):
      %mx1 = arith.maxnumf %c, %d : f32
      tt.reduce.return %mx1 : f32
    }) : (tensor<4xf32>) -> f32
    %mb0 = tt.splat %m0 : f32 -> tensor<4xf32>
    %mb1 = tt.splat %m1 : f32 -> tensor<4xf32>
    %e0 = arith.subf %x0, %mb0 : tensor<4xf32>
    %e1 = arith.subf %x1, %mb1 : tensor<4xf32>
    %ex0 = math.exp %e0 : tensor<4xf32>
    %ex1 = math.exp %e1 : tensor<4xf32>
    %s0 = "tt.reduce"(%ex0) <{axis = 0 : i32}> ({
    ^bb0(%e: f32, %f: f32):
      %sm = arith.addf %e, %f : f32
      tt.reduce.return %sm : f32
    }) : (tensor<4xf32>) -> f32
    %s1 = "tt.reduce"(%ex1) <{axis = 0 : i32}> ({
    ^bb0(%g: f32, %h: f32):
      %sm1 = arith.addf %g, %h : f32
      tt.reduce.return %sm1 : f32
    }) : (tensor<4xf32>) -> f32
    %sb0 = tt.splat %s0 : f32 -> tensor<4xf32>
    %sb1 = tt.splat %s1 : f32 -> tensor<4xf32>
    %o0 = arith.divf %ex0, %sb0 : tensor<4xf32>
    %o1 = arith.divf %ex1, %sb1 : tensor<4xf32>
    %cs = arith.addf %o0, %o1 : tensor<4xf32>
    %cst = arith.constant dense<1.000000e-06> : tensor<4xf32>
    %cs2 = arith.addf %cs, %cst : tensor<4xf32>
    %r0 = arith.divf %o0, %cs2 : tensor<4xf32>
    %r1 = arith.divf %o1, %cs2 : tensor<4xf32>
    %c = arith.constant dense<[1, 4]> : tensor<2xi64>
    %rs0 = tensor.reshape %r0(%c) : (tensor<4xf32>, tensor<2xi64>) -> tensor<1x4xf32>
    %rs1 = tensor.reshape %r1(%c) : (tensor<4xf32>, tensor<2xi64>) -> tensor<1x4xf32>
    %concat = tensor.concat dim(0) %rs0, %rs1 : (tensor<1x4xf32>, tensor<1x4xf32>) -> tensor<2x4xf32>
    %0 = scf.for %i = %lb to %ub step %step iter_args(%acc = %concat) -> (tensor<2x4xf32>) {
      %rd = "tt.reduce"(%acc) <{axis = 1 : i32}> ({
      ^bb0(%a: f32, %b: f32):
        %s = arith.addf %a, %b : f32
        tt.reduce.return %s : f32
      }) : (tensor<2x4xf32>) -> tensor<2xf32>
      %ed = tt.expand_dims %rd {axis = 1 : i32} : tensor<2xf32> -> tensor<2x1xf32>
      %bc = tt.broadcast %ed : tensor<2x1xf32> -> tensor<2x4xf32>
      %d = arith.divf %acc, %bc : tensor<2x4xf32>
      scf.yield %d : tensor<2x4xf32>
    }
    %c2 = arith.constant dense<4> : tensor<1xi64>
    %es0 = tensor.extract_slice %0[0, 0] [1, 4] [1, 1] : tensor<2x4xf32> to tensor<1x4xf32>
    %res0 = tensor.reshape %es0(%c2) : (tensor<1x4xf32>, tensor<1xi64>) -> tensor<4xf32>
    %es1 = tensor.extract_slice %0[1, 0] [1, 4] [1, 1] : tensor<2x4xf32> to tensor<1x4xf32>
    %res1 = tensor.reshape %es1(%c2) : (tensor<1x4xf32>, tensor<1xi64>) -> tensor<4xf32>
    tt.return %res0, %res1 : tensor<4xf32>, tensor<4xf32>
  }

  // CHECK-LABEL: tt.func @softmax_prologue_feeds_packed_loop(
  // CHECK: tensor.concat
  // CHECK: "tt.reduce"(%{{.*}}) <{axis = 1 : i32}>
  // CHECK: math.exp
  // CHECK: "tt.reduce"(%{{.*}}) <{axis = 1 : i32}>
  // CHECK: "tt.reduce"(%{{.*}}) <{axis = 0 : i32}>

  // A scalar (lane-invariant) select condition must be broadcast to the packed
  // value shape, not packed to its own lane shape, or arith.select fails.
  tt.func @select_scalar_condition(%c: i1, %a0: tensor<4xf32>, %a1: tensor<4xf32>, %b0: tensor<4xf32>, %b1: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
    %r0 = arith.select %c, %a0, %b0 : tensor<4xf32>
    %r1 = arith.select %c, %a1, %b1 : tensor<4xf32>
    tt.return %r0, %r1 : tensor<4xf32>, tensor<4xf32>
  }

  // CHECK-LABEL: tt.func @select_scalar_condition(
  // CHECK: %[[COND:.*]] = tt.splat %{{.*}} : i1 -> tensor<2x4xi1>
  // CHECK: arith.select %[[COND]],

  // Two independent lane families in the same block must both be packed.
  tt.func @two_families(%a0: tensor<4xf32>, %a1: tensor<4xf32>, %b0: tensor<4xf32>, %b1: tensor<4xf32>, %c0: tensor<4xf32>, %c1: tensor<4xf32>, %d0: tensor<4xf32>, %d1: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) {
    %r0 = arith.addf %a0, %a1 : tensor<4xf32>
    %r1 = arith.addf %b0, %b1 : tensor<4xf32>
    %s0 = arith.mulf %c0, %c1 : tensor<4xf32>
    %s1 = arith.mulf %d0, %d1 : tensor<4xf32>
    tt.return %r0, %r1, %s0, %s1 : tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>
  }

  // CHECK-LABEL: tt.func @two_families(
  // CHECK: arith.addf %{{.*}} : tensor<2x4xf32>
  // CHECK: arith.mulf %{{.*}} : tensor<2x4xf32>
}
