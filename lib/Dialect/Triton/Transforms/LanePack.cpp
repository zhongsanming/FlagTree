#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"
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
  if (vals.size() < 2) {
    llvm::errs() << "[lane-pack] reject: need >= 2 lane values, got "
                 << vals.size() << "\n";
    return false;
  }
  auto ty = getRankedTensorType(vals.front());
  if (!ty) {
    llvm::errs() << "[lane-pack] reject: first lane is not ranked tensor\n";
    return false;
  }
  for (Value v : vals.drop_front()) {
    auto otherTy = getRankedTensorType(v);
    if (!otherTy || otherTy != ty) {
      llvm::errs() << "[lane-pack] reject: lane tensor type mismatch\n";
      return false;
    }
  }
  return true;
}

static bool hasSingleAddCombiner(triton::ReduceOp reduceOp) {
  auto *combiner = reduceOp.getSingleCombiner();
  if (!combiner || !isa<arith::AddFOp>(combiner)) {
    llvm::errs() << "[lane-pack] reject: reduce combiner is not addf\n";
    return false;
  }
  return true;
}

template <typename OpTy>
static bool hasOneUseOfType(Value v, Operation *&user) {
  user = nullptr;
  for (Operation *candidate : v.getUsers()) {
    if (!isa<OpTy>(candidate))
      continue;
    if (user)
      return false;
    user = candidate;
  }
  return user != nullptr;
}

static bool matchLaneReduceToSplatDiv(BlockArgument laneArg, Value &eps,
                                      triton::ReduceOp &reduceOp,
                                      arith::DivFOp &divOp) {
  Operation *user = nullptr;
  if (!hasOneUseOfType<arith::DivFOp>(laneArg, user)) {
    llvm::errs() << "[lane-pack] reject: lane arg has no unique divf user\n";
    return false;
  }

  divOp = cast<arith::DivFOp>(user);
  if (divOp.getLhs() != laneArg)
    return false;

  auto splat = divOp.getRhs().getDefiningOp<triton::SplatOp>();
  if (!splat) {
    llvm::errs() << "[lane-pack] reject: div rhs is not triton.splat\n";
    return false;
  }
  auto add = splat.getSrc().getDefiningOp<arith::AddFOp>();
  if (!add) {
    llvm::errs() << "[lane-pack] reject: splat src is not addf\n";
    return false;
  }

  Value reduceScalar;
  Value candidateEps;
  if (isa<OpResult>(add.getLhs()) && !isa<OpResult>(add.getRhs())) {
    reduceScalar = add.getLhs();
    candidateEps = add.getRhs();
  } else if (isa<OpResult>(add.getRhs()) && !isa<OpResult>(add.getLhs())) {
    reduceScalar = add.getRhs();
    candidateEps = add.getLhs();
  } else {
    llvm::errs() << "[lane-pack] reject: add tree around row norm is not binary addf with scalar eps\n";
    return false;
  }

  if (!eps)
    eps = candidateEps;
  else if (eps != candidateEps) {
    llvm::errs() << "[lane-pack] reject: epsilon mismatch across lanes\n";
    return false;
  }

  reduceOp = reduceScalar.getDefiningOp<triton::ReduceOp>();
  if (!reduceOp || reduceOp.getAxis() != 0 || reduceOp.getNumOperands() != 1 ||
      reduceOp.getOperand(0) != laneArg || !hasSingleAddCombiner(reduceOp)) {
    llvm::errs() << "[lane-pack] reject: reduce op shape does not match lane sum\n";
    return false;
  }
  return true;
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
    if (!isa<OpResult>(add.getLhs())) {
      if (!eps)
        eps = add.getLhs();
      if (eps == add.getLhs()) {
        worklist.push_back(add.getRhs());
        continue;
      }
    }
    if (!isa<OpResult>(add.getRhs())) {
      if (!eps)
        eps = add.getRhs();
      if (eps == add.getRhs()) {
        worklist.push_back(add.getLhs());
        continue;
      }
    }
    worklist.push_back(add.getLhs());
    worklist.push_back(add.getRhs());
  }
  return true;
}

struct LanePackMatch {
  SmallVector<Value> initLanes;
  SmallVector<BlockArgument> laneArgs;
  SmallVector<arith::DivFOp> rowDivs;
  SmallVector<Value> rowNorms;
  SmallVector<arith::DivFOp> yieldDivs;
  Value eps;
  scf::YieldOp yieldOp;
};

static FailureOr<LanePackMatch> matchLanePackLoop(scf::ForOp forOp) {
  llvm::errs() << "[lane-pack] inspect loop: ";
  forOp->print(llvm::errs());
  llvm::errs() << "\n";
  LanePackMatch match;
  match.initLanes.assign(forOp.getInitArgs().begin(), forOp.getInitArgs().end());
  if (!isSameLaneTensorGroup(match.initLanes)) {
    llvm::errs() << "[lane-pack] reject: init args not same lane tensor group\n";
    return failure();
  }

  match.yieldOp = dyn_cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  if (!match.yieldOp) {
    llvm::errs() << "[lane-pack] reject: loop terminator is not scf.yield\n";
    return failure();
  }
  if (match.yieldOp.getNumOperands() != match.initLanes.size()) {
    llvm::errs() << "[lane-pack] reject: yield arity mismatch\n";
    return failure();
  }

  SmallVector<Value> yielded(match.yieldOp.getOperands().begin(),
                             match.yieldOp.getOperands().end());
  if (!isSameLaneTensorGroup(yielded)) {
    llvm::errs() << "[lane-pack] reject: yielded values not same lane tensor group\n";
    return failure();
  }

  match.laneArgs.assign(forOp.getRegionIterArgs().begin(),
                        forOp.getRegionIterArgs().end());
  match.rowDivs.reserve(match.laneArgs.size());
  match.rowNorms.reserve(match.laneArgs.size());
  for (BlockArgument laneArg : match.laneArgs) {
    triton::ReduceOp reduceOp;
    arith::DivFOp divOp;
    if (!matchLaneReduceToSplatDiv(laneArg, match.eps, reduceOp, divOp)) {
      llvm::errs() << "[lane-pack] reject: lane arg failed reduce/div match\n";
      return failure();
    }
    match.rowDivs.push_back(divOp);
    match.rowNorms.push_back(divOp.getResult());
  }

  Value sharedDenom;
  match.yieldDivs.reserve(yielded.size());
  for (auto [rowNorm, yieldedLane] : llvm::zip(match.rowNorms, yielded)) {
    auto divOp = yieldedLane.getDefiningOp<arith::DivFOp>();
    if (!divOp || divOp.getLhs() != rowNorm) {
      llvm::errs() << "[lane-pack] reject: yield div does not consume row norm\n";
      return failure();
    }
    match.yieldDivs.push_back(divOp);
    if (!sharedDenom)
      sharedDenom = divOp.getRhs();
    else if (sharedDenom != divOp.getRhs())
      return failure();
  }

  SmallVector<Value> addLeaves;
  Value treeEps = match.eps;
  if (!collectAddTreeLeaves(sharedDenom, addLeaves, treeEps)) {
    llvm::errs() << "[lane-pack] reject: failed to collect add tree leaves\n";
    return failure();
  }
  if (treeEps != match.eps || addLeaves.size() != match.rowNorms.size()) {
    llvm::errs() << "[lane-pack] reject: add tree leaf count or epsilon mismatch\n";
    return failure();
  }

  llvm::SmallPtrSet<void *, 8> leafSet;
  for (Value leaf : addLeaves)
    leafSet.insert(leaf.getAsOpaquePointer());
  for (Value rowNorm : match.rowNorms) {
    if (!leafSet.contains(rowNorm.getAsOpaquePointer()))
      return failure();
  }

  llvm::errs() << "[lane-pack] match success: lanes=" << match.initLanes.size()
               << "\n";
  return match;
}

static Value buildPackedLanes(OpBuilder &builder, Location loc,
                              ArrayRef<Value> lanes) {
  assert(lanes.size() >= 2 && "expected at least two lanes");
  Value packed = builder.create<triton::JoinOp>(loc, lanes[0], lanes[1]);
  for (Value lane : lanes.drop_front(2)) {
    auto laneTy = cast<RankedTensorType>(lane.getType());
    auto packedTy = cast<RankedTensorType>(packed.getType());
    SmallVector<int64_t> expandedShape(laneTy.getShape().begin(),
                                       laneTy.getShape().end());
    expandedShape.push_back(1);
    auto expandedTy = RankedTensorType::get(expandedShape,
                                            laneTy.getElementType(),
                                            packedTy.getEncoding());
    Value expandedLane =
        builder.create<triton::ExpandDimsOp>(loc, expandedTy, lane, -1);
    packed = builder.create<triton::CatOp>(loc, packedTy, packed, expandedLane);
  }
  return packed;
}

static SmallVector<Value> unpackPackedLanes(OpBuilder &builder, Location loc,
                                            Value packed,
                                            unsigned laneCount) {
  SmallVector<Value> lanes;
  lanes.reserve(laneCount);

  std::function<void(Value)> splitRec = [&](Value v) {
    if (lanes.size() + 1 == laneCount) {
      lanes.push_back(v);
      return;
    }
    auto shape = cast<RankedTensorType>(v.getType()).getShape();
    if (shape.empty() || shape.back() != 2)
      return;
    auto split = builder.create<triton::SplitOp>(loc, v);
    splitRec(split.getOutLHS());
    splitRec(split.getOutRHS());
  };

  splitRec(packed);
  return lanes;
}

static Value buildSumReduce(OpBuilder &builder, Location loc, Value src,
                            int axis) {
  auto reduce = builder.create<triton::ReduceOp>(loc, src, axis);
  {
    OpBuilder::InsertionGuard guard(builder);
    Region &region = reduce.getCombineOp();
    Block *block = builder.createBlock(&region);
    auto elemTy = cast<RankedTensorType>(src.getType()).getElementType();
    block->addArgument(elemTy, loc);
    block->addArgument(elemTy, loc);
    builder.setInsertionPointToStart(block);
    Value sum = builder.create<arith::AddFOp>(loc, block->getArgument(0),
                                              block->getArgument(1));
    builder.create<triton::ReduceReturnOp>(loc, sum);
  }
  return *reduce.getResult().begin();
}

static Value broadcastScalarTo(OpBuilder &builder, Location loc, Value scalar,
                               RankedTensorType dstTy) {
  SmallVector<int64_t> splatShape(dstTy.getRank(), 1);
  auto splatTy = RankedTensorType::get(splatShape, dstTy.getElementType(),
                                       dstTy.getEncoding());
  Value splat = builder.create<triton::SplatOp>(loc, splatTy, scalar);
  return builder.create<triton::BroadcastOp>(loc, dstTy, splat);
}

static LogicalResult rewriteLanePackLoop(scf::ForOp forOp,
                                         const LanePackMatch &match) {
  Location loc = forOp.getLoc();
  OpBuilder builder(forOp);

  Value packedInit = buildPackedLanes(builder, loc, match.initLanes);
  auto packedInitTy = cast<RankedTensorType>(packedInit.getType());
  SmallVector<int64_t> transposedShape(packedInitTy.getShape().begin(),
                                       packedInitTy.getShape().end());
  std::rotate(transposedShape.rbegin(), transposedShape.rbegin() + 1,
              transposedShape.rend());
  auto packedLaneMajorTy = RankedTensorType::get(
      transposedShape, packedInitTy.getElementType(), packedInitTy.getEncoding());
  Value laneMajorInit = builder.create<triton::TransOp>(
      loc, packedLaneMajorTy, packedInit,
      DenseI32ArrayAttr::get(builder.getContext(),
                             SmallVector<int32_t>{1, 0}));

  auto newFor = builder.create<scf::ForOp>(loc, forOp.getLowerBound(),
                                           forOp.getUpperBound(),
                                           forOp.getStep(), ValueRange{laneMajorInit});

  Block *oldBody = forOp.getBody();
  Block *newBody = newFor.getBody();
  builder.setInsertionPointToStart(newBody);
  Value packedArg = newBody->getArgument(newBody->getNumArguments() - 1);
  auto packedArgTy = cast<RankedTensorType>(packedArg.getType());

  Value rowSums = buildSumReduce(builder, loc, packedArg, 1);
  auto rowSumsTy = cast<RankedTensorType>(rowSums.getType());
  Value epsVec = broadcastScalarTo(builder, loc, match.eps, rowSumsTy);
  Value rowDenom = builder.create<arith::AddFOp>(loc, rowSums, epsVec);
  auto expandRowTy = RankedTensorType::get(packedArgTy.getShape(),
                                           packedArgTy.getElementType(),
                                           packedArgTy.getEncoding());
  Value rowDenomExpanded =
      builder.create<triton::ExpandDimsOp>(loc, expandRowTy, rowDenom, 1);
  Value rowDenomBroadcast =
      builder.create<triton::BroadcastOp>(loc, packedArgTy, rowDenomExpanded);
  Value rowNormalized =
      builder.create<arith::DivFOp>(loc, packedArg, rowDenomBroadcast);

  Value colSums = buildSumReduce(builder, loc, rowNormalized, 0);
  auto colSumsTy = cast<RankedTensorType>(colSums.getType());
  Value epsCols = broadcastScalarTo(builder, loc, match.eps, colSumsTy);
  Value colDenom = builder.create<arith::AddFOp>(loc, colSums, epsCols);
  auto expandColTy = RankedTensorType::get(packedArgTy.getShape(),
                                           packedArgTy.getElementType(),
                                           packedArgTy.getEncoding());
  Value colDenomExpanded =
      builder.create<triton::ExpandDimsOp>(loc, expandColTy, colDenom, 0);
  Value colDenomBroadcast =
      builder.create<triton::BroadcastOp>(loc, packedArgTy, colDenomExpanded);
  Value packedYield =
      builder.create<arith::DivFOp>(loc, rowNormalized, colDenomBroadcast);
  builder.create<scf::YieldOp>(loc, packedYield);

  builder.setInsertionPointAfter(newFor);
  Value packedResult = newFor.getResult(0);
  auto packedResultTy = cast<RankedTensorType>(packedResult.getType());
  SmallVector<int64_t> unpackShape(packedResultTy.getShape().begin(),
                                   packedResultTy.getShape().end());
  std::rotate(unpackShape.begin(), unpackShape.begin() + 1, unpackShape.end());
  auto unpackTy = RankedTensorType::get(unpackShape, packedResultTy.getElementType(),
                                        packedResultTy.getEncoding());
  Value unpackInput = builder.create<triton::TransOp>(
      loc, unpackTy, packedResult,
      DenseI32ArrayAttr::get(builder.getContext(), SmallVector<int32_t>{1, 0}));

  SmallVector<Value> unpacked = unpackPackedLanes(builder, loc, unpackInput,
                                                  match.initLanes.size());
  if (unpacked.size() != match.initLanes.size())
    return failure();

  for (auto [oldResult, newResult] : llvm::zip(forOp.getResults(), unpacked))
    oldResult.replaceAllUsesWith(newResult);
  forOp.erase();
  return success();
}

struct LanePackPass : public impl::TritonLanePackBase<LanePackPass> {
  using TritonLanePackBase::TritonLanePackBase;

  void runOnOperation() override {
    SmallVector<scf::ForOp> candidates;
    getOperation().walk([&](scf::ForOp forOp) {
      if (succeeded(matchLanePackLoop(forOp)))
        candidates.push_back(forOp);
    });

    for (scf::ForOp forOp : candidates) {
      FailureOr<LanePackMatch> match = matchLanePackLoop(forOp);
      if (failed(match))
        continue;
      if (failed(rewriteLanePackLoop(forOp, *match))) {
        llvm::errs() << "[lane-pack] rewrite failed\n";
        signalPassFailure();
        return;
      }
      llvm::errs() << "[lane-pack] rewrite applied\n";
    }
  }
};

} // namespace

} // namespace mlir::triton
