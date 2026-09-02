#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"

using namespace mlir;

namespace mlir::triton {

#define GEN_PASS_DEF_TRITONVECTORIZELANEPACK
#include "triton/Dialect/Triton/Transforms/Passes.h.inc"

namespace {

using LaneMap = DenseMap<Value, Value>;

static bool isRankedTensorValue(Value value) {
  return isa<RankedTensorType>(value.getType());
}

static bool isPureTensorOp(Operation *op) {
  return op && op->getNumRegions() == 0 && isMemoryEffectFree(op) &&
         llvm::all_of(op->getResults(), isRankedTensorValue);
}

static bool areLaneIsomorphic(Value lhs, Value rhs, LaneMap &lhsToRhs,
                              LaneMap &rhsToLhs,
                              DenseMap<std::pair<Value, Value>, bool> &memo) {
  if (lhs == rhs)
    return true;
  if (lhs.getType() != rhs.getType())
    return false;

  auto key = (lhs.getAsOpaquePointer() < rhs.getAsOpaquePointer())
                 ? std::make_pair(lhs, rhs)
                 : std::make_pair(rhs, lhs);
  if (auto it = memo.find(key); it != memo.end())
    return it->second;

  memo[key] = true;
  if (!lhs.getDefiningOp() || !rhs.getDefiningOp()) {
    bool consistent = (!lhsToRhs.count(lhs) || lhsToRhs[lhs] == rhs) &&
                      (!rhsToLhs.count(rhs) || rhsToLhs[rhs] == lhs);
    if (consistent) {
      lhsToRhs[lhs] = rhs;
      rhsToLhs[rhs] = lhs;
    }
    memo[key] = consistent;
    return consistent;
  }

  auto *lhsDef = lhs.getDefiningOp();
  auto *rhsDef = rhs.getDefiningOp();
  if (!isPureTensorOp(lhsDef) || !isPureTensorOp(rhsDef) ||
      lhsDef->getName() != rhsDef->getName() ||
      cast<OpResult>(lhs).getResultNumber() != cast<OpResult>(rhs).getResultNumber() ||
      lhsDef->getAttrs() != rhsDef->getAttrs() ||
      lhsDef->getNumOperands() != rhsDef->getNumOperands()) {
    memo[key] = false;
    return false;
  }

  for (auto operands : llvm::zip(lhsDef->getOperands(), rhsDef->getOperands())) {
    if (!areLaneIsomorphic(std::get<0>(operands), std::get<1>(operands),
                           lhsToRhs, rhsToLhs, memo)) {
      memo[key] = false;
      return false;
    }
  }
  return true;
}

static bool isSupportedLoop(scf::ForOp loop) {
  if (!loop.getBody() || !loop.getBody()->hasOneBlock())
    return false;
  if (loop.getNumRegionIterArgs() == 0 ||
      loop.getNumRegionIterArgs() != loop.getNumResults())
    return false;
  if (!llvm::all_of(loop.getInitArgs(), isRankedTensorValue))
    return false;

  Block &body = loop.getBody()->front();
  auto yieldOp = dyn_cast<scf::YieldOp>(body.getTerminator());
  if (!yieldOp || yieldOp.getNumOperands() != loop.getNumResults())
    return false;
  if (!llvm::all_of(yieldOp.getOperands(), isRankedTensorValue))
    return false;

  for (Operation &op : body.without_terminator()) {
    if (!isPureTensorOp(&op))
      return false;
  }
  return true;
}

static bool canPackYieldedFamily(scf::ForOp loop, SmallVectorImpl<Value> &lanes) {
  lanes.clear();
  auto yieldOp = cast<scf::YieldOp>(loop.getBody()->front().getTerminator());
  DenseMap<std::pair<Value, Value>, bool> memo;
  LaneMap lhsToRhs;
  LaneMap rhsToLhs;

  for (Value yielded : yieldOp.getOperands()) {
    if (lanes.empty()) {
      lanes.push_back(yielded);
      continue;
    }
    if (yielded.getType() != lanes.front().getType())
      return false;
    if (!areLaneIsomorphic(lanes.front(), yielded, lhsToRhs, rhsToLhs, memo))
      return false;
    lanes.push_back(yielded);
  }
  return lanes.size() > 1;
}

static Value buildJoinTree(OpBuilder &builder, Location loc,
                           ArrayRef<Value> lanes) {
  if (lanes.empty())
    return Value();
  if (lanes.size() == 1)
    return lanes.front();

  SmallVector<Value> current(lanes.begin(), lanes.end());
  while (current.size() > 1) {
    SmallVector<Value> next;
    next.reserve((current.size() + 1) / 2);
    for (size_t i = 0; i < current.size(); i += 2) {
      if (i + 1 == current.size()) {
        next.push_back(current[i]);
        continue;
      }
      next.push_back(builder.create<JoinOp>(loc, current[i], current[i + 1]));
    }
    current.swap(next);
  }
  return current.front();
}

static LogicalResult rewritePackedLoop(scf::ForOp loop,
                                       ArrayRef<Value> yieldedLanes) {
  OpBuilder builder(loop);
  builder.setInsertionPoint(loop);

  Value packedInit = buildJoinTree(builder, loop.getLoc(), yieldedLanes);
  if (!packedInit)
    return failure();

  auto newLoop = builder.create<scf::ForOp>(loop.getLoc(), loop.getLowerBound(),
                                            loop.getUpperBound(), loop.getStep(),
                                            ValueRange{packedInit});
  newLoop->setAttrs(loop->getAttrs());

  IRMapping mapping;
  mapping.map(loop.getInductionVar(), newLoop.getInductionVar());
  mapping.map(loop.getRegionIterArg(0), newLoop.getRegionIterArg(0));

  Block &oldBody = loop.getBody()->front();
  Block &newBody = newLoop.getRegion().front();
  newBody.getOperations().splice(newBody.begin(), oldBody.getOperations(),
                                 oldBody.begin(), oldBody.end());

  for (Operation &op : llvm::make_early_inc_range(newBody.without_terminator())) {
    op.replaceUsesWithIf(mapping.lookupOrDefault(op.getResults()),
                         [&](OpOperand &use) {
                           return !newLoop->isAncestor(use.getOwner());
                         });
  }

  auto yieldOp = cast<scf::YieldOp>(newBody.getTerminator());
  builder.setInsertionPoint(yieldOp);
  Value packedYield = buildJoinTree(builder, yieldOp.getLoc(),
                                    yieldOp.getOperands());
  if (!packedYield)
    return failure();
  builder.create<scf::YieldOp>(yieldOp.getLoc(), packedYield);
  yieldOp.erase();

  loop.replaceAllUsesWith(newLoop.getResult(0));
  loop.erase();
  return success();
}

struct VectorizeLanePackPass
    : public impl::TritonVectorizeLanePackBase<VectorizeLanePackPass> {
  using TritonVectorizeLanePackBase::TritonVectorizeLanePackBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<scf::ForOp> loops;
    module.walk([&](scf::ForOp loop) { loops.push_back(loop); });

    for (scf::ForOp loop : loops) {
      if (!isSupportedLoop(loop))
        continue;
      SmallVector<Value> yieldedLanes;
      if (!canPackYieldedFamily(loop, yieldedLanes))
        continue;
      if (failed(rewritePackedLoop(loop, yieldedLanes)))
        signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::triton
