#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir-ext/Dialect/CommonIR/IR/CommonIRDialect.h"
#include "tle/dialect/include/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir::triton::tle {

#define GEN_PASS_DEF_TRITONTLELANEPACK
#include "tle/dialect/include/Transforms/Passes.h.inc"

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

static bool isScalar(Value v) {
  return v.getType() && !isa<ShapedType>(v.getType());
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
  auto lhsReduce = add.getLhs().getDefiningOp<triton::ReduceOp>();
  auto rhsReduce = add.getRhs().getDefiningOp<triton::ReduceOp>();
  if (lhsReduce && !rhsReduce) {
    reduceScalar = add.getLhs();
    candidateEps = add.getRhs();
  } else if (rhsReduce && !lhsReduce) {
    reduceScalar = add.getRhs();
    candidateEps = add.getLhs();
  } else {
    llvm::errs()
        << "[lane-pack] reject: row denominator is not reduce + scalar eps\n";
    return false;
  }

  if (!eps)
    eps = candidateEps;
  // Epsilon may be represented as a scalar in one lane and as a promoted
  // tensor in another lane. Keep the first value for reconstruction and do
  // not require SSA identity or type equality across the original lanes.

  reduceOp = reduceScalar.getDefiningOp<triton::ReduceOp>();
  if (!reduceOp || reduceOp.getAxis() != 0 || reduceOp.getNumOperands() != 1 ||
      reduceOp.getOperand(0) != laneArg || !hasSingleAddCombiner(reduceOp)) {
    llvm::errs() << "[lane-pack] reject: reduce op shape does not match lane sum\n";
    return false;
  }
  return true;
}

static bool collectAddTreeLeaves(Value root, SmallVectorImpl<Value> &leaves) {
  SmallVector<Value> worklist{root};
  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    if (auto add = current.getDefiningOp<arith::AddFOp>()) {
      worklist.push_back(add.getLhs());
      worklist.push_back(add.getRhs());
      continue;
    }
    leaves.push_back(current);
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
  int64_t laneReduceAxis = 0;
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
    if (match.rowDivs.empty())
      match.laneReduceAxis = reduceOp.getAxis();
    else if (match.laneReduceAxis != reduceOp.getAxis())
      return failure();
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
  if (!collectAddTreeLeaves(sharedDenom, addLeaves)) {
    llvm::errs() << "[lane-pack] reject: failed to collect add tree leaves\n";
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

static DenseIntElementsAttr buildReassociationAttr(OpBuilder &builder,
                                                    ArrayRef<int64_t> shape) {
  SmallVector<int64_t> reassociation(shape.begin(), shape.end());
  auto reassociationTy =
      RankedTensorType::get({static_cast<int64_t>(reassociation.size())},
                            builder.getI64Type());
  return DenseIntElementsAttr::get(reassociationTy, reassociation);
}

static Value buildPackedLanes(OpBuilder &builder, Location loc,
                              ArrayRef<Value> lanes) {
  assert(lanes.size() >= 2 && "expected at least two lanes");
  auto laneTy = cast<RankedTensorType>(lanes.front().getType());

  SmallVector<int64_t> singletonLaneShape;
  singletonLaneShape.push_back(1);
  singletonLaneShape.append(laneTy.getShape().begin(), laneTy.getShape().end());
  auto singletonLaneTy = RankedTensorType::get(singletonLaneShape,
                                               laneTy.getElementType(),
                                               laneTy.getEncoding());

  auto reshapeLane = [&](Value lane) -> Value {
    return builder.create<tensor::ReshapeOp>(
        loc, singletonLaneTy, lane,
        buildReassociationAttr(builder, singletonLaneShape));
  };

  SmallVector<Value> concatOperands;
  concatOperands.reserve(lanes.size());
  for (Value lane : lanes)
    concatOperands.push_back(reshapeLane(lane));

  SmallVector<int64_t> packedShape;
  packedShape.push_back(lanes.size());
  packedShape.append(laneTy.getShape().begin(), laneTy.getShape().end());
  auto packedTy = RankedTensorType::get(packedShape, laneTy.getElementType(),
                                        laneTy.getEncoding());
  return builder.create<tensor::ConcatOp>(loc, packedTy, 0, concatOperands);
}

static SmallVector<Value> unpackPackedLanes(OpBuilder &builder, Location loc,
                                            Value packed,
                                            unsigned laneCount) {
  SmallVector<Value> lanes;
  lanes.reserve(laneCount);
  auto packedTy = dyn_cast<RankedTensorType>(packed.getType());
  if (!packedTy || packedTy.getRank() < 1 ||
      packedTy.getShape().front() != static_cast<int64_t>(laneCount)) {
    llvm::errs() << "[lane-pack] reject: packed tensor shape mismatch: "
                 << packed.getType() << "\n";
    return lanes;
  }

  SmallVector<int64_t> laneShape(packedTy.getShape().drop_front().begin(),
                                 packedTy.getShape().drop_front().end());
  auto laneTy = RankedTensorType::get(laneShape, packedTy.getElementType(),
                                      packedTy.getEncoding());

  SmallVector<int64_t> singletonLaneShape;
  singletonLaneShape.push_back(1);
  singletonLaneShape.append(laneShape.begin(), laneShape.end());
  auto singletonLaneTy = RankedTensorType::get(
      singletonLaneShape, packedTy.getElementType(), packedTy.getEncoding());

  SmallVector<OpFoldResult> sizes;
  sizes.reserve(packedTy.getRank());
  sizes.push_back(builder.getIndexAttr(1));
  for (int64_t dim : laneShape)
    sizes.push_back(builder.getIndexAttr(dim));
  SmallVector<OpFoldResult> strides(packedTy.getRank(), builder.getIndexAttr(1));

  for (unsigned laneIdx = 0; laneIdx < laneCount; ++laneIdx) {
    SmallVector<OpFoldResult> offsets(packedTy.getRank(), builder.getIndexAttr(0));
    offsets[0] = builder.getIndexAttr(laneIdx);
    Value slice = builder.create<tensor::ExtractSliceOp>(
        loc, singletonLaneTy, packed, offsets, sizes, strides);
    Value lane = builder.create<tensor::ReshapeOp>(
        loc, laneTy, slice,
        buildReassociationAttr(builder, singletonLaneShape));
    lanes.push_back(lane);
  }
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

static Value broadcastEpsilonTo(OpBuilder &builder, Location loc,
                                Value epsilon, RankedTensorType dstTy) {
  if (auto epsilonTy = dyn_cast<RankedTensorType>(epsilon.getType())) {
    if (epsilonTy == dstTy)
      return epsilon;
    return builder.create<triton::BroadcastOp>(loc, dstTy, epsilon);
  }

  SmallVector<int64_t> splatShape(dstTy.getRank(), 1);
  auto splatTy = RankedTensorType::get(splatShape, dstTy.getElementType(),
                                       dstTy.getEncoding());
  Value splat = builder.create<triton::SplatOp>(loc, splatTy, epsilon);
  return builder.create<triton::BroadcastOp>(loc, dstTy, splat);
}

static LogicalResult rewriteLanePackLoop(scf::ForOp forOp,
                                         const LanePackMatch &match) {
  Location loc = forOp.getLoc();
  OpBuilder builder(forOp);
  llvm::errs() << "[lane-pack] rewrite begin: lanes="
               << match.initLanes.size() << ", epsilon type="
               << match.eps.getType() << "\n";

  Value packedInit = buildPackedLanes(builder, loc, match.initLanes);
  llvm::errs() << "[lane-pack] packed init type: " << packedInit.getType()
               << "\n";

  auto newFor = builder.create<scf::ForOp>(
      loc, forOp.getLowerBound(), forOp.getUpperBound(), forOp.getStep(),
      ValueRange{packedInit});
  llvm::errs() << "[lane-pack] created packed loop\n";

  Block *oldBody = forOp.getBody();
  Block *newBody = newFor.getBody();
  builder.setInsertionPointToStart(newBody);
  Value packedArg = newBody->getArgument(newBody->getNumArguments() - 1);
  auto packedArgTy = cast<RankedTensorType>(packedArg.getType());
  llvm::errs() << "[lane-pack] packed loop argument type: " << packedArgTy
               << "\n";

  constexpr int packedLaneAxis = 0;
  const int64_t packedOriginalReduceAxis = match.laneReduceAxis + 1;

  Value rowSums =
      buildSumReduce(builder, loc, packedArg, packedOriginalReduceAxis);
  auto rowSumsTy = cast<RankedTensorType>(rowSums.getType());
  Value epsVec = broadcastEpsilonTo(builder, loc, match.eps, rowSumsTy);
  llvm::errs() << "[lane-pack] row sums/epsilon types: " << rowSumsTy << " / "
               << epsVec.getType() << "\n";
  Value rowDenom = builder.create<arith::AddFOp>(loc, rowSums, epsVec);
  SmallVector<int64_t> rowExpandedShape(rowSumsTy.getShape().begin(),
                                        rowSumsTy.getShape().end());
  rowExpandedShape.insert(rowExpandedShape.begin() + packedOriginalReduceAxis,
                          1);
  auto expandRowTy = RankedTensorType::get(
      rowExpandedShape, packedArgTy.getElementType(), packedArgTy.getEncoding());
  Value rowDenomExpanded = builder.create<triton::ExpandDimsOp>(
      loc, expandRowTy, rowDenom, packedOriginalReduceAxis);
  Value rowDenomBroadcast =
      builder.create<triton::BroadcastOp>(loc, packedArgTy, rowDenomExpanded);
  Value rowNormalized =
      builder.create<arith::DivFOp>(loc, packedArg, rowDenomBroadcast);
  llvm::errs() << "[lane-pack] created row normalization\n";

  Value colSums = buildSumReduce(builder, loc, rowNormalized, packedLaneAxis);
  auto colSumsTy = cast<RankedTensorType>(colSums.getType());
  Value epsCols = broadcastEpsilonTo(builder, loc, match.eps, colSumsTy);
  llvm::errs() << "[lane-pack] column sums/epsilon types: " << colSumsTy
               << " / " << epsCols.getType() << "\n";
  Value colDenom = builder.create<arith::AddFOp>(loc, colSums, epsCols);
  SmallVector<int64_t> colExpandedShape(colSumsTy.getShape().begin(),
                                        colSumsTy.getShape().end());
  colExpandedShape.insert(colExpandedShape.begin() + packedLaneAxis, 1);
  auto expandColTy = RankedTensorType::get(
      colExpandedShape, packedArgTy.getElementType(), packedArgTy.getEncoding());
  Value colDenomExpanded = builder.create<triton::ExpandDimsOp>(
      loc, expandColTy, colDenom, packedLaneAxis);
  Value colDenomBroadcast =
      builder.create<triton::BroadcastOp>(loc, packedArgTy, colDenomExpanded);
  Value packedYield =
      builder.create<arith::DivFOp>(loc, rowNormalized, colDenomBroadcast);
  builder.create<scf::YieldOp>(loc, packedYield);
  llvm::errs() << "[lane-pack] created column normalization and loop yield\n";

  builder.setInsertionPointAfter(newFor);
  Value packedResult = newFor.getResult(0);
  llvm::errs() << "[lane-pack] unpack input type: " << packedResult.getType()
               << "\n";

  SmallVector<Value> unpacked = unpackPackedLanes(builder, loc, packedResult,
                                                  match.initLanes.size());
  if (unpacked.size() != match.initLanes.size()) {
    llvm::errs() << "[lane-pack] rewrite reject: unpack produced "
                 << unpacked.size() << " lanes; expected "
                 << match.initLanes.size() << "\n";
    return failure();
  }
  llvm::errs() << "[lane-pack] unpacked " << unpacked.size() << " lanes\n";

  for (auto [oldResult, newResult] : llvm::zip(forOp.getResults(), unpacked))
    oldResult.replaceAllUsesWith(newResult);
  forOp.erase();
  llvm::errs() << "[lane-pack] rewrite complete\n";
  return success();
}

struct TritonTleLanePackPass
    : public impl::TritonTleLanePackBase<TritonTleLanePackPass> {
  using TritonTleLanePackBase::TritonTleLanePackBase;

  void runOnOperation() override {
    getOperation().walk([&](scf::ForOp forOp) {
      FailureOr<LanePackMatch> match = matchLanePackLoop(forOp);
      if (failed(match))
        return;
      if (failed(rewriteLanePackLoop(forOp, *match))) {
        llvm::errs() << "[lane-pack] rewrite failed\n";
        signalPassFailure();
        return;
      }
      llvm::errs() << "[lane-pack] rewrite applied\n";
    });
  }
};

} // namespace

} // namespace mlir::triton::tle
