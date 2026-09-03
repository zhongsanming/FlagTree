#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"

namespace mlir::triton {

#define GEN_PASS_DEF_TRITONLANEPACK
#include "triton/Dialect/Triton/Transforms/Passes.h.inc"

namespace {

static RankedTensorType getRankedTensorType(Value v) {
  return dyn_cast<RankedTensorType>(v.getType());
}

static bool isSameLaneTensorGroup(ArrayRef<Value> vals) {
  if (vals.size() < 2)
    return false;
  auto ty = getRankedTensorType(vals.front());
  if (!ty)
    return false;
  for (Value v : vals.drop_front()) {
    auto otherTy = getRankedTensorType(v);
    if (!otherTy || otherTy != ty)
      return false;
  }
  return true;
}

static bool hasSingleAddCombiner(triton::ReduceOp reduceOp) {
  Region &combine = reduceOp.getCombineOp();
  if (!combine.hasOneBlock())
    return false;
  if (combine.front().getOperations().size() != 2)
    return false;
  return isa<arith::AddFOp>(combine.front().front());
}

static Value matchScalarWithAddEps(Value v, Value &eps) {
  auto add = v.getDefiningOp<arith::AddFOp>();
  if (!add)
    return {};
  if (!eps) {
    if (isa<OpResult>(add.getLhs()) && !isa<OpResult>(add.getRhs())) {
      eps = add.getRhs();
      return add.getLhs();
    }
    if (isa<OpResult>(add.getRhs()) && !isa<OpResult>(add.getLhs())) {
      eps = add.getLhs();
      return add.getRhs();
    }
    return {};
  }
  if (add.getLhs() == eps)
    return add.getRhs();
  if (add.getRhs() == eps)
    return add.getLhs();
  return {};
}

static bool matchPerLaneRowNorm(ArrayRef<BlockArgument> laneArgs,
                                SmallVectorImpl<Value> &rowNorms,
                                Value &eps) {
  rowNorms.clear();
  rowNorms.reserve(laneArgs.size());
  for (BlockArgument laneArg : laneArgs) {
    auto div = dyn_cast_or_null<arith::DivFOp>(*laneArg.getUsers().begin());
    (void)div;
    bool matchedLane = false;
    for (Operation *user : laneArg.getUsers()) {
      auto divOp = dyn_cast<arith::DivFOp>(user);
      if (!divOp || divOp.getLhs() != laneArg)
        continue;
      auto splat = divOp.getRhs().getDefiningOp<triton::SplatOp>();
      if (!splat)
        continue;
      Value reduceScalar = matchScalarWithAddEps(splat.getSrc(), eps);
      if (!reduceScalar)
        continue;
      auto reduce = reduceScalar.getDefiningOp<triton::ReduceOp>();
      if (!reduce || reduce.getAxis() != 0 || reduce.getSrcs().size() != 1 ||
          reduce.getSrcs().front() != laneArg || !hasSingleAddCombiner(reduce))
        continue;
      rowNorms.push_back(divOp.getResult());
      matchedLane = true;
      break;
    }
    if (!matchedLane)
      return false;
  }
  return rowNorms.size() == laneArgs.size();
}

static bool collectAddTreeLeaves(Value root, SmallVectorImpl<Value> &leaves,
                                 Value &eps) {
  SmallVector<Value> worklist{root};
  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    auto add = current.getDefiningOp<arith::AddFOp>();
    if (!add) {
      leaves.push_back(current);
      continue;
    }
    if (!eps && (!isa<OpResult>(add.getLhs()) || !isa<OpResult>(add.getRhs()))) {
      if (!isa<OpResult>(add.getLhs())) {
        eps = add.getLhs();
        worklist.push_back(add.getRhs());
      } else {
        eps = add.getRhs();
        worklist.push_back(add.getLhs());
      }
      continue;
    }
    if (eps && add.getLhs() == eps) {
      worklist.push_back(add.getRhs());
      continue;
    }
    if (eps && add.getRhs() == eps) {
      worklist.push_back(add.getLhs());
      continue;
    }
    worklist.push_back(add.getLhs());
    worklist.push_back(add.getRhs());
  }
  return true;
}

static bool matchSharedColumnNorm(ArrayRef<Value> rowNorms,
                                  ArrayRef<Value> yieldedLanes, Value &eps) {
  if (rowNorms.size() != yieldedLanes.size())
    return false;

  Value sharedDenom;
  for (auto [rowNorm, yielded] : llvm::zip(rowNorms, yieldedLanes)) {
    auto div = yielded.getDefiningOp<arith::DivFOp>();
    if (!div || div.getLhs() != rowNorm)
      return false;
    if (!sharedDenom)
      sharedDenom = div.getRhs();
    else if (sharedDenom != div.getRhs())
      return false;
  }

  SmallVector<Value> leaves;
  if (!collectAddTreeLeaves(sharedDenom, leaves, eps))
    return false;
  if (leaves.size() != rowNorms.size())
    return false;

  llvm::SmallPtrSet<void *, 8> leafSet;
  for (Value leaf : leaves)
    leafSet.insert(leaf.getAsOpaquePointer());
  for (Value rowNorm : rowNorms) {
    if (!leafSet.contains(rowNorm.getAsOpaquePointer()))
      return false;
  }
  return true;
}

static bool isLanePackCandidate(scf::ForOp forOp) {
  SmallVector<Value> iterArgs(forOp.getInitArgs().begin(),
                              forOp.getInitArgs().end());
  if (!isSameLaneTensorGroup(iterArgs))
    return false;

  auto yield = dyn_cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  if (!yield)
    return false;
  if (yield.getNumOperands() != iterArgs.size())
    return false;

  SmallVector<Value> yielded(yield.getOperands().begin(),
                             yield.getOperands().end());
  if (!isSameLaneTensorGroup(yielded))
    return false;

  SmallVector<BlockArgument> laneArgs(forOp.getRegionIterArgs().begin(),
                                      forOp.getRegionIterArgs().end());
  Value eps;
  SmallVector<Value> rowNorms;
  if (!matchPerLaneRowNorm(laneArgs, rowNorms, eps))
    return false;
  return matchSharedColumnNorm(rowNorms, yielded, eps);
}

struct LanePackPass : public impl::TritonLanePackBase<LanePackPass> {
  using TritonLanePackBase::TritonLanePackBase;

  void runOnOperation() override {
    bool foundCandidate = false;
    getOperation().walk([&](scf::ForOp forOp) {
      if (!isLanePackCandidate(forOp))
        return;
      foundCandidate = true;
      (void)forOp;
    });

    (void)foundCandidate;
  }
};

} // namespace

} // namespace mlir::triton
