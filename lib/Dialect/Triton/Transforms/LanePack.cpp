//===----------------------------------------------------------------------===//
// LanePack.cpp - Lane-parallel operation packing
//
// Recognizes groups of loop-carried Triton tensor "lanes" (for example four
// independent tensor<Mxf32> values updated by the same computation) and
// rewrites the loop so the lanes are packed into a single tensor with a new
// leading dimension.
//
// The transform is purely structural. An operation is lifted to the packed
// form when each of its per-lane instances computes the same math and its
// operands are either already packed or lane-invariant. An associative combine
// across the lanes (e.g. an add tree over all lanes) is recognized as a
// reduction over the synthesized lane axis. No semantic pattern (row/column
// normalization, softmax, ...) is hard-coded; anything whose packed form is
// mathematically equivalent is allowed.
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"

#define DEBUG_TYPE "lane-pack"

namespace mlir::triton {

#define GEN_PASS_DEF_TRITONLANEPACK
#include "triton/Dialect/Triton/Transforms/Passes.h.inc"

namespace {

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

static Value buildShapeConst(OpBuilder &builder, Location loc,
                             ArrayRef<int64_t> shape) {
  return builder.create<arith::ConstantOp>(loc, builder.getI64TensorAttr(shape));
}

// Packs `lanes` (all of the same ranked tensor type) into one tensor with a new
// leading dimension of size lanes.size().
static Value buildPackedLanes(OpBuilder &builder, Location loc,
                              ArrayRef<Value> lanes) {
  assert(lanes.size() >= 2 && "expected at least two lanes");
  auto laneTy = cast<RankedTensorType>(lanes.front().getType());

  SmallVector<int64_t> singletonLaneShape;
  singletonLaneShape.push_back(1);
  singletonLaneShape.append(laneTy.getShape().begin(), laneTy.getShape().end());
  auto singletonLaneTy =
      RankedTensorType::get(singletonLaneShape, laneTy.getElementType(),
                            laneTy.getEncoding());

  auto reshapeLane = [&](Value lane) -> Value {
    return builder.create<tensor::ReshapeOp>(
        loc, singletonLaneTy, lane,
        buildShapeConst(builder, loc, singletonLaneShape));
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
                                            Value packed, unsigned laneCount) {
  SmallVector<Value> lanes;
  lanes.reserve(laneCount);
  auto packedTy = dyn_cast<RankedTensorType>(packed.getType());
  if (!packedTy || packedTy.getRank() < 1 ||
      packedTy.getShape().front() != static_cast<int64_t>(laneCount))
    return lanes;

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
    SmallVector<OpFoldResult> offsets(packedTy.getRank(),
                                      builder.getIndexAttr(0));
    offsets[0] = builder.getIndexAttr(laneIdx);
    Value slice = builder.create<tensor::ExtractSliceOp>(
        loc, singletonLaneTy, packed, offsets, sizes, strides);
    Value lane = builder.create<tensor::ReshapeOp>(
        loc, laneTy, slice, buildShapeConst(builder, loc, laneShape));
    lanes.push_back(lane);
  }
  return lanes;
}

static void eraseDeadTree(Value v) {
  Operation *op = v.getDefiningOp();
  if (!op || !op->use_empty())
    return;
  SmallVector<Value> operands(op->getOperands().begin(),
                              op->getOperands().end());
  op->erase();
  for (Value o : operands)
    eraseDeadTree(o);
}

// ---------------------------------------------------------------------------
// Math predicates
// ---------------------------------------------------------------------------

// Associative/commutative ops that can be realized as a tt.reduce over the
// synthesized lane axis.
static bool isAssociativeCombine(Operation *op) {
  return isa<arith::AddFOp, arith::AddIOp, arith::MulFOp, arith::MulIOp,
             arith::MaximumFOp, arith::MinimumFOp, arith::MaxSIOp,
             arith::MaxUIOp, arith::MinSIOp, arith::MinUIOp, arith::AndIOp,
             arith::OrIOp, arith::XOrIOp>(op);
}

static void collectAssociativeLeaves(Value v, StringRef opName,
                                     SmallVectorImpl<Value> &leaves) {
  if (Operation *def = v.getDefiningOp()) {
    if (def->getName().getStringRef() == opName) {
      for (Value operand : def->getOperands())
        collectAssociativeLeaves(operand, opName, leaves);
      return;
    }
  }
  leaves.push_back(v);
}

// ---------------------------------------------------------------------------
// Packed value bookkeeping
// ---------------------------------------------------------------------------

struct PackedValue {
  Value value;
  // A shared value is lane-invariant: it has the shape of a single lane (or is
  // a scalar) and must be broadcast along the synthesized lane axis at use.
  bool shared = false;
};

// The packed type of a per-lane value: an extra leading dim of size laneCount.
static RankedTensorType packedTypeOf(Value v, int64_t laneCount) {
  SmallVector<int64_t> shape;
  shape.push_back(laneCount);
  Type elemTy;
  Attribute enc;
  if (auto rt = dyn_cast<RankedTensorType>(v.getType())) {
    llvm::append_range(shape, rt.getShape());
    elemTy = rt.getElementType();
    enc = rt.getEncoding();
  } else {
    elemTy = v.getType();
  }
  return RankedTensorType::get(shape, elemTy, enc);
}

// Grows an already packed value to dstTy by appending size-1 dims and
// broadcasting. Used when a scalar-lane packed value meets a tensor-lane one.
static Value widenPacked(OpBuilder &builder, Location loc, Value v,
                         RankedTensorType dstTy) {
  auto srcTy = dyn_cast<RankedTensorType>(v.getType());
  if (!srcTy)
    return Value();
  if (srcTy == dstTy)
    return v;

  Value cur = v;
  auto curTy = srcTy;
  while (curTy.getRank() < dstTy.getRank()) {
    SmallVector<int64_t> shape(curTy.getShape().begin(),
                               curTy.getShape().end());
    shape.push_back(1);
    auto nextTy = RankedTensorType::get(shape, curTy.getElementType(),
                                        curTy.getEncoding());
    cur = builder.create<triton::ExpandDimsOp>(loc, nextTy, cur,
                                               curTy.getRank());
    curTy = nextTy;
  }
  if (curTy == dstTy)
    return cur;
  return builder.create<triton::BroadcastOp>(loc, dstTy, cur);
}

// Brings a packed/shared value to the packed shape `dstTy`.
static Value materialize(OpBuilder &builder, Location loc, PackedValue pv,
                         RankedTensorType dstTy) {
  if (!pv.shared)
    return widenPacked(builder, loc, pv.value, dstTy);

  if (auto srcTy = dyn_cast<RankedTensorType>(pv.value.getType())) {
    if (srcTy == dstTy)
      return pv.value;
    // Lane-invariant tensor: add the lane axis and broadcast.
    SmallVector<int64_t> shape;
    shape.push_back(1);
    llvm::append_range(shape, srcTy.getShape());
    auto expandedTy = RankedTensorType::get(shape, srcTy.getElementType(),
                                            srcTy.getEncoding());
    Value expanded =
        builder.create<triton::ExpandDimsOp>(loc, expandedTy, pv.value, 0);
    return builder.create<triton::BroadcastOp>(loc, dstTy, expanded);
  }

  // Scalar value: broadcast to the packed shape.
  return builder.create<triton::SplatOp>(loc, dstTy, pv.value);
}

// Builds a tt.reduce over `src` whose combine region is cloned from `proto`.
// Used for per-lane reduces, where `proto` is the original reduce op.
static Value buildReduceFromRegion(OpBuilder &builder, Location loc,
                                   Value src, int64_t axis,
                                   triton::ReduceOp proto) {
  auto reduce = builder.create<triton::ReduceOp>(loc, src, axis);
  Region &region = reduce.getCombineOp();
  IRMapping mapping;
  proto.getCombineOp().cloneInto(&region, mapping);
  return reduce->getResult(0);
}

// Builds a tt.reduce over `src` whose combine region computes a single instance
// of `kind`. Used for cross-lane tree reductions, where `kind` is a top-level
// associative op rather than an op nested in a reduce region.
static Value buildReduceFromKind(OpBuilder &builder, Location loc, Value src,
                                 int64_t axis, Operation *kind) {
  auto reduce = builder.create<triton::ReduceOp>(loc, src, axis);
  Region &region = reduce.getCombineOp();
  Block *block = builder.createBlock(&region);
  auto elemTy = cast<RankedTensorType>(src.getType()).getElementType();
  block->addArgument(elemTy, loc);
  block->addArgument(elemTy, loc);
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(block);
  OperationState state(loc, kind->getName());
  state.addOperands({block->getArgument(0), block->getArgument(1)});
  state.addTypes(elemTy);
  state.addAttributes(kind->getAttrs());
  Operation *c = builder.create(state);
  builder.create<triton::ReduceReturnOp>(loc, c->getResult(0));
  return reduce->getResult(0);
}

// ---------------------------------------------------------------------------
// Structural lifter
// ---------------------------------------------------------------------------

struct Lifter {
  scf::ForOp forOp;
  OpBuilder &builder;
  Location loc;
  unsigned n;
  SmallVector<Value> initLanes;
  DenseMap<Value, PackedValue> packedOf;
  SmallVector<Value> referenceOrder;
  // laneImages[i][ref] is the lane-i value corresponding to reference value
  // `ref` (lane 0). laneImages[0] is unused (identity).
  SmallVector<DenseMap<Value, Value>> laneImages;

  Lifter(scf::ForOp forOp, OpBuilder &builder, unsigned n)
      : forOp(forOp), builder(builder), loc(forOp.getLoc()), n(n),
        laneImages(n) {}

  void seed(Value packedInit) {
    initLanes.assign(forOp.getInitArgs().begin(), forOp.getInitArgs().end());
    packedOf[initLanes[0]] = {packedInit, false};
    referenceOrder.push_back(initLanes[0]);
    for (unsigned i = 1; i < n; ++i)
      laneImages[i][initLanes[0]] = initLanes[i];
  }

  // A loop body block argument that is one of the lane iter args.
  bool isLaneArg(Value v) {
    auto barg = dyn_cast<BlockArgument>(v);
    if (!barg || barg.getOwner() != forOp.getBody())
      return false;
    return barg.getArgNumber() >= 1; // arg 0 is the induction variable
  }

  bool isShared(Value v) {
    auto it = packedOf.find(v);
    if (it != packedOf.end())
      return it->second.shared;
    if (isLaneArg(v))
      return false;
    Operation *def = v.getDefiningOp();
    if (!def)
      return true; // an outer block argument
    return !forOp->isAncestor(def);
  }

  std::optional<PackedValue> getPacked(Value v) {
    auto it = packedOf.find(v);
    if (it != packedOf.end())
      return it->second;
    if (isLaneArg(v))
      return std::nullopt;
    Operation *def = v.getDefiningOp();
    if (!def || !forOp->isAncestor(def))
      return PackedValue{v, true};
    return std::nullopt;
  }

  bool allOperandsShared(Operation *op) {
    return llvm::all_of(op->getOperands(), [&](Value a) { return isShared(a); });
  }

  bool allOperandsResolved(Operation *op) {
    return llvm::all_of(op->getOperands(),
                        [&](Value a) { return getPacked(a).has_value(); });
  }

  Operation *findLaneOp(Operation *refOp, unsigned lane) {
    SmallVector<Value> expected;
    for (Value a : refOp->getOperands()) {
      if (isShared(a)) {
        expected.push_back(a);
        continue;
      }
      Value img = laneImages[lane].lookup(a);
      if (!img)
        return nullptr;
      expected.push_back(img);
    }
    if (expected.empty())
      return nullptr;

    for (Operation *u : expected.front().getUsers()) {
      if (u->getName() != refOp->getName())
        continue;
      if (u->getNumOperands() != refOp->getNumOperands())
        continue;
      if (u->getNumResults() != refOp->getNumResults())
        continue;
      if (!OperationEquivalence::isEquivalentTo(
              refOp, u, OperationEquivalence::ignoreValueEquivalence, nullptr,
              OperationEquivalence::Flags::IgnoreLocations))
        continue;
      if (!llvm::equal(u->getOperands(), expected))
        continue;
      return u;
    }
    return nullptr;
  }

  Value emitClonedOp(Operation *op, ValueRange operands, TypeRange resultTypes) {
    IRMapping mapping;
    for (auto [from, to] : llvm::zip(op->getOperands(), operands))
      mapping.map(from, to);
    Operation *newOp = builder.clone(*op, mapping);
    for (auto [result, type] : llvm::zip(newOp->getResults(), resultTypes))
      result.setType(type);
    return newOp->getResult(0);
  }

  // Lane-invariant op inside the loop: clone it (with already packed operands)
  // and treat the result as shared.
  LogicalResult liftSharedOp(Operation *op) {
    if (op->getNumResults() != 1)
      return failure();
    SmallVector<Value> operands;
    for (Value a : op->getOperands()) {
      auto pv = getPacked(a);
      if (!pv)
        return failure();
      operands.push_back(pv->value);
    }
    Value result = emitClonedOp(op, operands, op->getResult(0).getType());
    packedOf[op->getResult(0)] = {result, true};
    return success();
  }

  // A tt.reduce applied independently to every lane.
  LogicalResult liftLaneReduce(triton::ReduceOp reduce) {
    if (reduce->getNumOperands() != 1 || reduce->getNumResults() != 1)
      return failure();
    Operation *combiner = reduce.getSingleCombiner();
    if (!combiner || !isAssociativeCombine(combiner))
      return failure();
    Value src = reduce.getOperand(0);
    auto pv = getPacked(src);
    if (!pv || pv->shared)
      return failure();

    SmallVector<Operation *> laneOps;
    for (unsigned i = 1; i < n; ++i) {
      Operation *oi = findLaneOp(reduce, i);
      if (!oi)
        return failure();
      auto other = dyn_cast<triton::ReduceOp>(oi);
      if (!other || !other.getSingleCombiner() ||
          other.getSingleCombiner()->getName() != combiner->getName())
        return failure();
      laneOps.push_back(oi);
    }

    Value packed = buildReduceFromRegion(builder, loc, pv->value,
                                         reduce.getAxis() + 1, reduce);
    packedOf[reduce->getResult(0)] = {packed, false};
    referenceOrder.push_back(reduce->getResult(0));
    for (unsigned i = 1; i < n; ++i)
      laneImages[i][reduce->getResult(0)] = laneOps[i - 1]->getResult(0);
    return success();
  }

  // A pointwise/shape op applied independently to every lane.
  LogicalResult liftPerLaneOp(Operation *op) {
    if (op->getNumResults() != 1)
      return failure();
    if (isa<triton::ReduceOp>(op))
      return liftLaneReduce(cast<triton::ReduceOp>(op));
    // Shape-changing / region ops need explicit lane-axis remapping; defer.
    if (isa<triton::TransOp, tensor::ReshapeOp>(op) || op->getNumRegions() != 0)
      return failure();

    SmallVector<PackedValue> pvs;
    for (Value a : op->getOperands()) {
      auto pv = getPacked(a);
      if (!pv)
        return failure();
      pvs.push_back(*pv);
    }

    SmallVector<Operation *> laneOps;
    for (unsigned i = 1; i < n; ++i) {
      Operation *oi = findLaneOp(op, i);
      if (!oi)
        return failure();
      laneOps.push_back(oi);
    }

    auto resultTy = packedTypeOf(op->getResult(0), n);
    Value packed;
    if (auto splat = dyn_cast<triton::SplatOp>(op)) {
      // Scalar-lane -> tensor-lane: widen the packed source.
      packed = widenPacked(builder, loc, pvs[0].value, resultTy);
    } else if (auto expand = dyn_cast<triton::ExpandDimsOp>(op)) {
      packed = builder.create<triton::ExpandDimsOp>(loc, resultTy,
                                                    pvs[0].value,
                                                    expand.getAxis() + 1);
    } else {
      SmallVector<Value> operands;
      for (auto [a, pv] : llvm::zip(op->getOperands(), pvs)) {
        Value v = materialize(builder, loc, pv, packedTypeOf(a, n));
        if (!v)
          return failure();
        operands.push_back(v);
      }
      packed = emitClonedOp(op, operands, resultTy);
    }
    if (!packed)
      return failure();

    packedOf[op->getResult(0)] = {packed, false};
    referenceOrder.push_back(op->getResult(0));
    for (unsigned i = 1; i < n; ++i)
      laneImages[i][op->getResult(0)] = laneOps[i - 1]->getResult(0);
    return success();
  }

  // An associative combine over the lane values (e.g. an add tree). Emitted as
  // a reduction over the lane axis, plus any lane-invariant leaves.
  LogicalResult liftCrossLane(Operation *op) {
    if (op->getNumResults() != 1 || !isAssociativeCombine(op))
      return failure();

    SmallVector<Value> leaves;
    collectAssociativeLeaves(op->getResult(0), op->getName().getStringRef(),
                             leaves);

    for (Value r : referenceOrder) {
      auto pvIt = packedOf.find(r);
      if (pvIt == packedOf.end() || pvIt->second.shared)
        continue;
      if (!isa<RankedTensorType>(r.getType()))
        continue;

      SmallVector<Value> expected;
      bool complete = true;
      for (unsigned i = 0; i < n; ++i) {
        Value e = (i == 0) ? r : laneImages[i].lookup(r);
        if (!e) {
          complete = false;
          break;
        }
        expected.push_back(e);
      }
      if (!complete)
        continue;

      // Every expected lane leaf must appear exactly once.
      llvm::SmallDenseMap<Value, unsigned> counts;
      for (Value l : leaves)
        counts[l]++;
      bool matched = true;
      for (Value e : expected) {
        auto it = counts.find(e);
        if (it == counts.end() || it->second == 0) {
          matched = false;
          break;
        }
        it->second--;
      }
      if (!matched)
        continue;

      SmallPtrSet<void *, 8> expectedSet;
      for (Value e : expected)
        expectedSet.insert(e.getAsOpaquePointer());
      bool restShared = true;
      for (Value l : leaves) {
        if (!expectedSet.contains(l.getAsOpaquePointer()) && !isShared(l)) {
          restShared = false;
          break;
        }
      }
      if (!restShared)
        continue;

      Value reduced =
          buildReduceFromKind(builder, loc, pvIt->second.value, 0, op);
      for (Value l : leaves) {
        if (expectedSet.contains(l.getAsOpaquePointer()))
          continue;
        auto lp = getPacked(l);
        if (!lp)
          return failure();
        Value operand = lp->value;
        if (operand.getType() != reduced.getType()) {
          auto dstTy = dyn_cast<RankedTensorType>(reduced.getType());
          if (!dstTy)
            return failure();
          operand = materialize(builder, loc, *lp, dstTy);
          if (!operand)
            return failure();
        }
        IRMapping foldMapping;
        foldMapping.map(op->getOperand(0), reduced);
        foldMapping.map(op->getOperand(1), operand);
        reduced = builder.clone(*op, foldMapping)->getResult(0);
      }
      packedOf[op->getResult(0)] = {reduced, true};
      return success();
    }
    return failure();
  }

  LogicalResult liftOp(Operation *op) {
    if (allOperandsShared(op))
      return liftSharedOp(op);
    if (allOperandsResolved(op))
      return liftPerLaneOp(op);
    if (succeeded(liftCrossLane(op)))
      return success();
    // Unmatched: drop it from the packed loop. Only safe for side-effect-free
    // ops; anything else must abort the rewrite.
    if (!isMemoryEffectFree(op)) {
      LLVM_DEBUG(llvm::dbgs() << "[lane-pack] reject: effectful unmatched op "
                              << *op << "\n");
      return failure();
    }
    LLVM_DEBUG(llvm::dbgs() << "[lane-pack] dropping unmatched pure op " << *op
                            << "\n");
    return success();
  }

  FailureOr<Value> run() {
    for (Operation &op : forOp.getBody()->without_terminator()) {
      if (failed(liftOp(&op)))
        return failure();
    }

    auto yield = dyn_cast<scf::YieldOp>(forOp.getBody()->getTerminator());
    if (!yield || yield.getNumOperands() != n)
      return failure();
    Value y0 = yield.getOperand(0);
    auto it = packedOf.find(y0);
    if (it == packedOf.end() || it->second.shared)
      return failure();
    for (unsigned i = 1; i < n; ++i) {
      auto img = laneImages[i].find(y0);
      if (img == laneImages[i].end() || img->second != yield.getOperand(i))
        return failure();
    }
    return it->second.value;
  }
};

// ---------------------------------------------------------------------------
// Rewrite
// ---------------------------------------------------------------------------

static LogicalResult rewriteLanePackLoop(scf::ForOp forOp) {
  OpBuilder builder(forOp);
  Location loc = forOp.getLoc();

  auto initArgs = forOp.getInitArgs();
  if (initArgs.size() < 2)
    return failure();
  if (!isa<RankedTensorType>(initArgs[0].getType()))
    return failure();
  for (Value a : initArgs)
    if (a.getType() != initArgs[0].getType())
      return failure();
  unsigned n = initArgs.size();

  SmallVector<Value> initLanesVec(initArgs.begin(), initArgs.end());
  Value packedInit = buildPackedLanes(builder, loc, initLanesVec);
  auto newFor = builder.create<scf::ForOp>(loc, forOp.getLowerBound(),
                                           forOp.getUpperBound(),
                                           forOp.getStep(),
                                           ValueRange{packedInit});
  newFor->setAttrs(forOp->getAttrs());

  auto fail = [&]() -> LogicalResult {
    newFor.erase();
    eraseDeadTree(packedInit);
    return failure();
  };

  Lifter lifter(forOp, builder, n);
  lifter.seed(packedInit);
  builder.setInsertionPointToStart(newFor.getBody());
  FailureOr<Value> packedYield = lifter.run();
  if (failed(packedYield))
    return fail();

  builder.setInsertionPointToEnd(newFor.getBody());
  builder.create<scf::YieldOp>(loc, *packedYield);

  builder.setInsertionPointAfter(newFor);
  SmallVector<Value> unpacked =
      unpackPackedLanes(builder, loc, newFor.getResult(0), n);
  if (unpacked.size() != n)
    return fail();

  for (auto [oldResult, newResult] : llvm::zip(forOp.getResults(), unpacked))
    oldResult.replaceAllUsesWith(newResult);
  forOp.erase();
  return success();
}

// ---------------------------------------------------------------------------
// Pass
// ---------------------------------------------------------------------------

struct LanePackPass : public impl::TritonLanePackBase<LanePackPass> {
  using TritonLanePackBase::TritonLanePackBase;

  void runOnOperation() override {
    getOperation().walk([&](scf::ForOp forOp) {
      if (succeeded(rewriteLanePackLoop(forOp)))
        LLVM_DEBUG(llvm::dbgs() << "[lane-pack] packed loop\n");
    });
  }
};

} // namespace

} // namespace mlir::triton
