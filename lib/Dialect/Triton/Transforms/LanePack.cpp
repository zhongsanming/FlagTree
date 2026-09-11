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
//
// Scope: only side-effect-free "math" ops are lifted. Ops with side effects
// (loads/stores, copies, barriers, ...) are treated as boundaries and left for
// other passes to vectorize. Leaf lane groups are packed either with a concat
// or, when they are a contiguous run of tensor.extract_slice of one source,
// with a single wider slice + reshape.
//
// Two modes share one lifter:
//   * loop mode packs the iter args of an scf.for and replays its body once,
//   * block mode (SLP) discovers a lane cone in straight-line code, seeded by a
//     tensor.concat packing boundary or by shallow structural signatures.
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <map>
#include <vector>

#define DEBUG_TYPE "lane-pack"

namespace mlir::triton {

#define GEN_PASS_DEF_TRITONLANEPACK
#include "triton/Dialect/Triton/Transforms/Passes.h.inc"

namespace {

// Debug logging, off unless LANE_PACK_DEBUG is set in the environment. This is
// intentionally independent of LLVM_DEBUG so release builds can produce
// diagnostics on request.
static bool lanePackDebug() {
  static const bool enabled = ::getenv("LANE_PACK_DEBUG") != nullptr;
  return enabled;
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

static Value buildShapeConst(OpBuilder &builder, Location loc,
                             ArrayRef<int64_t> shape) {
  return builder.create<arith::ConstantOp>(loc,
                                           builder.getI64TensorAttr(shape));
}

// If the lanes are a contiguous run of tensor.extract_slice of one source
// (e.g. mix_l0..3 = src[8,12,16,20]), pack them with a single wider slice +
// reshape instead of a concat. Returns null when the pattern does not apply.
static Value buildPackedContiguousSlices(OpBuilder &builder, Location loc,
                                         ArrayRef<Value> lanes) {
  auto toStatic =
      [](ArrayRef<OpFoldResult> ofrs) -> std::optional<SmallVector<int64_t>> {
    SmallVector<int64_t> out;
    for (OpFoldResult ofr : ofrs) {
      std::optional<int64_t> value = getConstantIntValue(ofr);
      if (!value)
        return std::nullopt;
      out.push_back(*value);
    }
    return out;
  };

  auto first = lanes.front().getDefiningOp<tensor::ExtractSliceOp>();
  if (!first)
    return Value();
  auto firstView = cast<OffsetSizeAndStrideOpInterface>(first.getOperation());
  Value source = first.getSource();
  auto sizes = toStatic(firstView.getMixedSizes());
  auto strides = toStatic(firstView.getMixedStrides());
  auto base = toStatic(firstView.getMixedOffsets());
  if (!sizes || !strides || !base || sizes->empty() || (*strides)[0] != 1)
    return Value();
  int64_t laneExtent = (*sizes)[0];
  if (laneExtent <= 0)
    return Value();

  for (auto [i, lane] : llvm::enumerate(lanes.drop_front())) {
    auto slice = lane.getDefiningOp<tensor::ExtractSliceOp>();
    if (!slice || slice.getSource() != source)
      return Value();
    auto view = cast<OffsetSizeAndStrideOpInterface>(slice.getOperation());
    auto s = toStatic(view.getMixedSizes());
    auto st = toStatic(view.getMixedStrides());
    auto off = toStatic(view.getMixedOffsets());
    if (!s || !st || !off || *s != *sizes || *st != *strides)
      return Value();
    if ((*off)[0] != (*base)[0] + (int64_t)(i + 1) * laneExtent)
      return Value();
    for (unsigned d = 1; d < off->size(); ++d)
      if ((*off)[d] != (*base)[d])
        return Value();
  }

  auto laneTy = cast<RankedTensorType>(lanes.front().getType());
  int64_t n = (int64_t)lanes.size();

  SmallVector<OpFoldResult> newOffsets;
  for (int64_t o : *base)
    newOffsets.push_back(builder.getIndexAttr(o));
  SmallVector<OpFoldResult> newSizes;
  newSizes.push_back(builder.getIndexAttr(n * laneExtent));
  for (unsigned d = 1; d < sizes->size(); ++d)
    newSizes.push_back(builder.getIndexAttr((*sizes)[d]));
  SmallVector<OpFoldResult> newStrides;
  for (int64_t s : *strides)
    newStrides.push_back(builder.getIndexAttr(s));

  SmallVector<int64_t> wideShape;
  wideShape.push_back(n * laneExtent);
  wideShape.append(laneTy.getShape().begin() + 1, laneTy.getShape().end());
  auto wideTy = RankedTensorType::get(wideShape, laneTy.getElementType(),
                                      laneTy.getEncoding());
  Value wide = builder.create<tensor::ExtractSliceOp>(
      loc, wideTy, source, newOffsets, newSizes, newStrides);

  SmallVector<int64_t> packedShape;
  packedShape.push_back(n);
  packedShape.append(laneTy.getShape().begin(), laneTy.getShape().end());
  auto packedTy = RankedTensorType::get(packedShape, laneTy.getElementType(),
                                        laneTy.getEncoding());
  return builder.create<tensor::ReshapeOp>(
      loc, packedTy, wide, buildShapeConst(builder, loc, packedShape));
}

// Packs `lanes` (all of the same ranked tensor type) into one tensor with a new
// leading dimension of size lanes.size().
static Value buildPackedLanes(OpBuilder &builder, Location loc,
                              ArrayRef<Value> lanes) {
  assert(lanes.size() >= 2 && "expected at least two lanes");
  if (Value coalesced = buildPackedContiguousSlices(builder, loc, lanes))
    return coalesced;
  auto laneTy = cast<RankedTensorType>(lanes.front().getType());

  SmallVector<int64_t> singletonLaneShape;
  singletonLaneShape.push_back(1);
  singletonLaneShape.append(laneTy.getShape().begin(), laneTy.getShape().end());
  auto singletonLaneTy = RankedTensorType::get(
      singletonLaneShape, laneTy.getElementType(), laneTy.getEncoding());

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
  SmallVector<OpFoldResult> strides(packedTy.getRank(),
                                    builder.getIndexAttr(1));

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

// Ops that map to a single packed op when applied lane-wise. Shape ops are
// allowed here but handled explicitly by the emitter.
static bool isPackableElementwise(Operation *op) {
  if (op->hasTrait<OpTrait::Elementwise>())
    return true;
  return isa<arith::SelectOp, arith::BitcastOp, triton::SplatOp,
             triton::BroadcastOp, triton::ExpandDimsOp, triton::TransOp,
             triton::ReshapeOp>(op);
}

// Associative/commutative ops that can be realized as a tt.reduce over the
// synthesized lane axis.
static bool isAssociativeCombine(Operation *op) {
  return isa<arith::AddFOp, arith::AddIOp, arith::MulFOp, arith::MulIOp,
             arith::MaximumFOp, arith::MinimumFOp, arith::MaxNumFOp,
             arith::MinNumFOp, arith::MaxSIOp, arith::MaxUIOp, arith::MinSIOp,
             arith::MinUIOp, arith::AndIOp, arith::OrIOp, arith::XOrIOp>(op);
}

// Ops the lifter can emit for a lane group: elementwise/shape ops and
// tt.reduce with an associative combiner.
static bool isLiftableOp(Operation *op) {
  if (auto reduce = dyn_cast<triton::ReduceOp>(op)) {
    Operation *combiner = reduce.getSingleCombiner();
    return combiner && isAssociativeCombine(combiner);
  }
  return isPackableElementwise(op) && op->getNumRegions() == 0;
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

static void collectAssociativeTreeOps(Value v, StringRef opName,
                                      SmallVectorImpl<Operation *> &ops) {
  Operation *def = v.getDefiningOp();
  if (!def || def->getName().getStringRef() != opName)
    return;
  ops.push_back(def);
  for (Value operand : def->getOperands())
    collectAssociativeTreeOps(operand, opName, ops);
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

// The packed type an operand of an elementwise op must take: the op broadcasts
// its operands to the result shape, so use the result's packed shape but keep
// the operand's element type (e.g. an i1 select condition becomes
// tensor<[N]x...xi1>).
static RankedTensorType packedOperandType(Value operand,
                                          RankedTensorType resultPackedTy) {
  Type elemTy = isa<RankedTensorType>(operand.getType())
                    ? cast<RankedTensorType>(operand.getType()).getElementType()
                    : operand.getType();
  if (elemTy == resultPackedTy.getElementType())
    return resultPackedTy;
  return RankedTensorType::get(resultPackedTy.getShape(), elemTy,
                               resultPackedTy.getEncoding());
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
    cur =
        builder.create<triton::ExpandDimsOp>(loc, nextTy, cur, curTy.getRank());
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
static Value buildReduceFromRegion(OpBuilder &builder, Location loc, Value src,
                                   int64_t axis, triton::ReduceOp proto) {
  auto reduce = builder.create<triton::ReduceOp>(loc, src, axis);
  OpBuilder::InsertionGuard guard(builder);
  Region &region = reduce.getCombineOp();
  Block *block = builder.createBlock(&region);
  auto elemTy = cast<RankedTensorType>(src.getType()).getElementType();
  block->addArgument(elemTy, loc);
  block->addArgument(elemTy, loc);
  builder.setInsertionPointToStart(block);
  Operation *combiner = proto.getSingleCombiner();
  Block &protoBlock = proto.getCombineOp().front();
  IRMapping mapping;
  mapping.map(protoBlock.getArgument(0), block->getArgument(0));
  mapping.map(protoBlock.getArgument(1), block->getArgument(1));
  Operation *c = builder.clone(*combiner, mapping);
  builder.create<triton::ReduceReturnOp>(loc, c->getResult(0));
  return reduce->getResult(0);
}

// Builds a tt.reduce over `src` whose combine region computes a single instance
// of `kind`. Used for cross-lane tree reductions, where `kind` is a top-level
// associative op rather than an op nested in a reduce region.
static Value buildReduceFromKind(OpBuilder &builder, Location loc, Value src,
                                 int64_t axis, Operation *kind) {
  auto reduce = builder.create<triton::ReduceOp>(loc, src, axis);
  OpBuilder::InsertionGuard guard(builder);
  Region &region = reduce.getCombineOp();
  Block *block = builder.createBlock(&region);
  auto elemTy = cast<RankedTensorType>(src.getType()).getElementType();
  block->addArgument(elemTy, loc);
  block->addArgument(elemTy, loc);
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
  // Loop mode: the loop being packed. Block mode: null.
  scf::ForOp forOp;
  // Operation whose body contains the packed computation. Values defined
  // outside it are lane-invariant.
  Operation *scope = nullptr;
  // Block whose arguments are the lanes (loop body in loop mode, null in block
  // mode).
  Block *laneBody = nullptr;
  // In block mode, values defined before this op are lane-invariant. Null in
  // loop mode.
  Operation *coneStart = nullptr;
  OpBuilder &builder;
  Location loc;
  unsigned n;
  DenseMap<Value, PackedValue> packedOf;
  // Old loop body block args (induction var and non-lane iter args) remapped to
  // the corresponding args of the new loop.
  DenseMap<Value, Value> argRemap;
  SmallVector<Value> referenceOrder;
  // Indices (into the loop iter args) that form the packed lane group.
  SmallVector<unsigned> laneIndices;
  // Block mode does not clone lane-invariant ops and does not abort on
  // effectful/unmatched ops.
  bool blockMode = false;
  // Results of cross-lane reductions (shared across lanes).
  SmallVector<Value> sharedRefs;
  // Original operations replaced by the packed computation.
  SmallPtrSet<Operation *, 32> liftedOps;
  // Values known to be lane-varying (members/images of a lane group).
  DenseSet<Value> laneVarying;
  // laneImages[i][ref] is the lane-i value corresponding to reference value
  // `ref` (lane 0). laneImages[0] is unused (identity).
  SmallVector<DenseMap<Value, Value>> laneImages;

  Lifter(scf::ForOp forOp, OpBuilder &builder, unsigned n)
      : forOp(forOp), scope(forOp.getOperation()), laneBody(forOp.getBody()),
        builder(builder), loc(forOp.getLoc()), n(n), laneImages(n) {}

  // Block (straight-line) mode constructor.
  Lifter(Operation *scope, Operation *coneStart, OpBuilder &builder, unsigned n)
      : scope(scope), coneStart(coneStart), builder(builder),
        loc(scope->getLoc()), n(n), laneImages(n) {
    blockMode = true;
  }

  void seed(Value packedCurrent, ArrayRef<unsigned> indices,
            scf::ForOp newFor) {
    laneIndices.assign(indices.begin(), indices.end());
    auto iterArgs = forOp.getRegionIterArgs();
    Value ref = iterArgs[laneIndices[0]];
    packedOf[ref] = {packedCurrent, false};
    referenceOrder.push_back(ref);
    for (unsigned i = 1; i < n; ++i)
      laneImages[i][ref] = iterArgs[laneIndices[i]];

    // Remap the induction variable and every non-lane iter arg to the new
    // loop's corresponding argument.
    argRemap[forOp.getInductionVar()] = newFor.getInductionVar();
    auto newIterArgs = newFor.getRegionIterArgs();
    unsigned nextNew = 1; // newIterArgs[0] is the packed lane group
    for (unsigned j = 0; j < iterArgs.size(); ++j) {
      if (llvm::is_contained(laneIndices, j))
        continue;
      argRemap[iterArgs[j]] = newIterArgs[nextNew++];
    }
  }

  // A loop body block argument that is one of the lane iter args.
  bool isLaneArg(Value v) {
    if (!laneBody)
      return false;
    auto barg = dyn_cast<BlockArgument>(v);
    if (!barg || barg.getOwner() != laneBody)
      return false;
    unsigned idx = barg.getArgNumber();
    if (idx == 0)
      return false; // induction variable
    return llvm::is_contained(laneIndices, idx - 1);
  }

  bool isShared(Value v) {
    auto it = packedOf.find(v);
    if (it != packedOf.end())
      return it->second.shared;
    // A lane-varying value is never lane-invariant, even if it is defined
    // before the cone start (e.g. a per-lane splat or a leaf lane).
    if (laneVarying.contains(v))
      return false;
    if (isLaneArg(v))
      return false;
    if (auto barg = dyn_cast<BlockArgument>(v))
      return true; // induction var, non-lane iter arg, or an outer arg
    Operation *def = v.getDefiningOp();
    if (!def)
      return true;
    if (!scope->isAncestor(def))
      return true;
    if (coneStart && def->getBlock() == coneStart->getBlock() &&
        def->isBeforeInBlock(coneStart))
      return true;
    return false;
  }

  std::optional<PackedValue> getPacked(Value v) {
    auto it = packedOf.find(v);
    if (it != packedOf.end())
      return it->second;
    if (laneVarying.contains(v))
      return std::nullopt;
    if (isLaneArg(v))
      return std::nullopt;
    if (auto barg = dyn_cast<BlockArgument>(v)) {
      if (laneBody && barg.getOwner() == laneBody && barg.getArgNumber() >= 1) {
        auto rit = argRemap.find(v);
        if (rit != argRemap.end())
          return PackedValue{rit->second, true};
      }
      return PackedValue{v, true};
    }
    Operation *def = v.getDefiningOp();
    if (!def)
      return PackedValue{v, true};
    if (!scope->isAncestor(def))
      return PackedValue{v, true};
    if (coneStart && def->getBlock() == coneStart->getBlock() &&
        def->isBeforeInBlock(coneStart))
      return PackedValue{v, true};
    return std::nullopt;
  }

  bool allOperandsShared(Operation *op) {
    return llvm::all_of(op->getOperands(),
                        [&](Value a) { return isShared(a); });
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

  Value emitClonedOp(Operation *op, ValueRange operands,
                     TypeRange resultTypes) {
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
      // Prefer the mapping computed by cone discovery: structurally identical
      // siblings (e.g. several tt.splat of the same scalar) cannot be told
      // apart by findLaneOp, so re-discovering here could pick the wrong one.
      Operation *oi = nullptr;
      if (Value img = laneImages[i].lookup(reduce->getResult(0)))
        oi = img.getDefiningOp();
      else
        oi = findLaneOp(reduce, i);
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
    liftedOps.insert(reduce.getOperation());
    for (unsigned i = 1; i < n; ++i) {
      laneImages[i][reduce->getResult(0)] = laneOps[i - 1]->getResult(0);
      laneVarying.insert(laneOps[i - 1]->getResult(0));
      liftedOps.insert(laneOps[i - 1]);
    }
    return success();
  }

  // A pointwise/shape op applied independently to every lane.
  LogicalResult liftPerLaneOp(Operation *op) {
    if (op->getNumResults() != 1)
      return failure();
    if (isa<triton::ReduceOp>(op))
      return liftLaneReduce(cast<triton::ReduceOp>(op));
    if (op->getNumRegions() != 0 || !isPackableElementwise(op))
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
      // Prefer the discovery mapping (see liftLaneReduce).
      Operation *oi = nullptr;
      if (Value img = laneImages[i].lookup(op->getResult(0)))
        oi = img.getDefiningOp();
      else
        oi = findLaneOp(op, i);
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
      packed = builder.create<triton::ExpandDimsOp>(loc, resultTy, pvs[0].value,
                                                    expand.getAxis() + 1);
    } else if (auto trans = dyn_cast<triton::TransOp>(op)) {
      // The lane axis is inserted at 0, so every transposed axis shifts by one.
      IRMapping mapping;
      mapping.map(trans.getSrc(), pvs[0].value);
      auto newTrans = cast<triton::TransOp>(builder.clone(*op, mapping));
      newTrans.getResult().setType(resultTy);
      SmallVector<int32_t> order{0};
      for (int32_t axis : trans.getOrder())
        order.push_back(axis + 1);
      newTrans.setOrder(order);
      packed = newTrans.getResult();
    } else if (auto reshape = dyn_cast<triton::ReshapeOp>(op)) {
      // Element reordering could move data across lanes.
      if (reshape.getAllowReorder())
        return failure();
      packed = emitClonedOp(op, {pvs[0].value}, resultTy);
    } else {
      SmallVector<Value> operands;
      for (auto [a, pv] : llvm::zip(op->getOperands(), pvs)) {
        Value v = materialize(builder, loc, pv, packedOperandType(a, resultTy));
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
    liftedOps.insert(op);
    for (unsigned i = 1; i < n; ++i) {
      laneImages[i][op->getResult(0)] = laneOps[i - 1]->getResult(0);
      laneVarying.insert(laneOps[i - 1]->getResult(0));
      liftedOps.insert(laneOps[i - 1]);
    }
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
      SmallVector<Operation *> treeOps;
      collectAssociativeTreeOps(op->getResult(0), op->getName().getStringRef(),
                                treeOps);
      for (Operation *treeOp : treeOps)
        liftedOps.insert(treeOp);
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
      sharedRefs.push_back(op->getResult(0));
      return success();
    }
    return failure();
  }

  LogicalResult liftOp(Operation *op) {
    // A value already seeded as a leaf (or otherwise handled) needs no lifting;
    // re-cloning it as a shared op would be wrong.
    if (op->getNumResults() == 1 && packedOf.contains(op->getResult(0)))
      return success();
    if (allOperandsShared(op))
      return liftSharedOp(op);
    if (allOperandsResolved(op))
      return liftPerLaneOp(op);
    if (succeeded(liftCrossLane(op)))
      return success();
    // Unmatched: drop it from the packed loop. Only safe for side-effect-free
    // ops; anything else must abort the rewrite.
    if (!isMemoryEffectFree(op)) {
      if (blockMode)
        return success(); // boundary: leave the op in place
      LLVM_DEBUG(llvm::dbgs() << "[lane-pack] reject: effectful unmatched op "
                              << *op << "\n");
      return failure();
    }
    LLVM_DEBUG(llvm::dbgs()
               << "[lane-pack] dropping unmatched pure op " << *op << "\n");
    return success();
  }

  FailureOr<Value> run() {
    for (Operation &op : forOp.getBody()->without_terminator()) {
      if (failed(liftOp(&op)))
        return failure();
    }

    auto yield = dyn_cast<scf::YieldOp>(forOp.getBody()->getTerminator());
    if (!yield)
      return failure();
    Value y0 = yield.getOperand(laneIndices[0]);
    auto it = packedOf.find(y0);
    if (it == packedOf.end() || it->second.shared)
      return failure();
    for (unsigned i = 1; i < n; ++i) {
      auto img = laneImages[i].find(y0);
      if (img == laneImages[i].end() ||
          img->second != yield.getOperand(laneIndices[i]))
        return failure();
    }
    return it->second.value;
  }
};

// ---------------------------------------------------------------------------
// Rewrite
// ---------------------------------------------------------------------------

static LogicalResult
rewriteLanePackLoop(scf::ForOp forOp, SmallPtrSetImpl<Block *> &packedBodies) {
  OpBuilder builder(forOp);
  Location loc = forOp.getLoc();

  auto initArgs = forOp.getInitArgs();
  unsigned m = initArgs.size();
  if (lanePackDebug())
    llvm::errs() << "[lane-pack] loop: iterArgs=" << m << "\n";
  if (m < 2)
    return failure();

  // Pick the largest group of same-typed tensor iter args to pack. Other iter
  // args are carried through unchanged.
  unsigned bestSize = 0;
  Type bestTy;
  for (unsigned i = 0; i < m; ++i) {
    if (!isa<RankedTensorType>(initArgs[i].getType()))
      continue;
    unsigned size = 0;
    for (unsigned j = 0; j < m; ++j)
      if (initArgs[j].getType() == initArgs[i].getType())
        ++size;
    if (size > bestSize) {
      bestSize = size;
      bestTy = initArgs[i].getType();
    }
  }
  if (bestSize < 2)
    return failure();

  SmallVector<unsigned> laneIndices;
  for (unsigned j = 0; j < m; ++j)
    if (initArgs[j].getType() == bestTy)
      laneIndices.push_back(j);
  unsigned n = laneIndices.size();

  SmallVector<Value> initLanes;
  for (unsigned idx : laneIndices)
    initLanes.push_back(initArgs[idx]);
  Value packedInit = buildPackedLanes(builder, loc, initLanes);

  SmallVector<unsigned> otherIndices;
  SmallVector<Value> newInitArgs{packedInit};
  for (unsigned j = 0; j < m; ++j) {
    if (llvm::is_contained(laneIndices, j))
      continue;
    otherIndices.push_back(j);
    newInitArgs.push_back(initArgs[j]);
  }

  auto newFor = builder.create<scf::ForOp>(loc, forOp.getLowerBound(),
                                           forOp.getUpperBound(),
                                           forOp.getStep(), newInitArgs);
  newFor->setAttrs(forOp->getAttrs());

  auto fail = [&]() -> LogicalResult {
    newFor.erase();
    eraseDeadTree(packedInit);
    return failure();
  };

  Lifter lifter(forOp, builder, n);
  // The body must start from the loop-carried packed value, not the pre-loop
  // packed init.
  lifter.seed(newFor.getRegionIterArg(0), laneIndices, newFor);
  builder.setInsertionPointToStart(newFor.getBody());
  FailureOr<Value> packedYield = lifter.run();
  if (failed(packedYield))
    return fail();

  auto oldYield = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  builder.setInsertionPointToEnd(newFor.getBody());
  SmallVector<Value> newYieldOperands{*packedYield};
  for (unsigned j : otherIndices) {
    auto pv = lifter.getPacked(oldYield.getOperand(j));
    if (!pv)
      return fail();
    newYieldOperands.push_back(pv->value);
  }
  builder.create<scf::YieldOp>(loc, newYieldOperands);

  builder.setInsertionPointAfter(newFor);
  SmallVector<Value> unpacked =
      unpackPackedLanes(builder, loc, newFor.getResult(0), n);
  if (unpacked.size() != n)
    return fail();

  SmallVector<Value> newResults(m);
  for (unsigned k = 0; k < n; ++k)
    newResults[laneIndices[k]] = unpacked[k];
  for (unsigned k = 0; k < otherIndices.size(); ++k)
    newResults[otherIndices[k]] = newFor.getResult(k + 1);

  for (unsigned j = 0; j < m; ++j)
    forOp.getResult(j).replaceAllUsesWith(newResults[j]);
  packedBodies.insert(newFor.getBody());
  if (lanePackDebug())
    llvm::errs() << "[lane-pack] loop packed: lanes=" << n << "\n";
  forOp.erase();
  return success();
}

// ---------------------------------------------------------------------------
// Straight-line (SLP) packing
// ---------------------------------------------------------------------------

// Two ops are structurally identical if they have the same op, operand types,
// result types and attributes/regions, ignoring operand identities.
static bool sameOpStructure(Operation *a, Operation *b) {
  if (a->getName() != b->getName())
    return false;
  if (a->getNumOperands() != b->getNumOperands() ||
      a->getNumResults() != b->getNumResults())
    return false;
  if (!llvm::equal(a->getOperandTypes(), b->getOperandTypes()))
    return false;
  if (!llvm::equal(a->getResultTypes(), b->getResultTypes()))
    return false;
  return OperationEquivalence::isEquivalentTo(
      a, b, OperationEquivalence::ignoreValueEquivalence, nullptr,
      OperationEquivalence::IgnoreLocations);
}

// Candidate lane groups in a block: single-result packable tensor ops grouped
// by shallow structure.
static SmallVector<SmallVector<Value>>
findSiblingGroups(Block *block, const DenseSet<Operation *> &skip) {
  // Largest lane group we are willing to synthesize. A real family is small;
  // anything larger is almost certainly a mistaken merge and would blow up the
  // cone/lifter.
  constexpr unsigned kMaxLanes = 32;

  SmallVector<Value> candidates;
  for (Operation &op : *block) {
    if (skip.contains(&op))
      continue;
    if (op.getNumResults() != 1)
      continue;
    // Reduces (and scalar results, e.g. a per-lane reduce) take part in the
    // lane structure but cannot seed a cone, so include them in the congruence
    // refinement even though the final groups are filtered to tensors below.
    if (!isLiftableOp(&op))
      continue;
    // These are the pack/unpack scaffolding used by this pass itself.
    if (isa<tensor::ReshapeOp, tensor::ExtractSliceOp>(&op))
      continue;
    candidates.push_back(op.getResult(0));
  }

  // Initial partition by shallow structure (op + operand/result types +
  // attributes).
  SmallVector<SmallVector<Value>> groups;
  for (Value v : candidates) {
    Operation *op = v.getDefiningOp();
    bool added = false;
    for (SmallVector<Value> &g : groups) {
      if (sameOpStructure(op, g.front().getDefiningOp())) {
        g.push_back(v);
        added = true;
        break;
      }
    }
    if (!added)
      groups.push_back({v});
  }

  // Refine by operand congruence: two values stay together only if their
  // corresponding operands live in the same group. A shallow signature merges
  // every independent instance of the same structure (e.g. all iterations of
  // an unrolled loop); congruence separates them because iteration N consumes
  // iteration N-1's results, which are a different group.
  bool changed = true;
  while (changed) {
    changed = false;
    DenseMap<Value, unsigned> groupOf;
    for (unsigned gi = 0; gi < groups.size(); ++gi)
      for (Value v : groups[gi])
        groupOf[v] = gi;
    SmallVector<SmallVector<Value>> refined;
    for (SmallVector<Value> &g : groups) {
      std::map<std::vector<unsigned>, SmallVector<Value>> buckets;
      for (Value v : g) {
        std::vector<unsigned> key;
        for (Value operand : v.getDefiningOp()->getOperands()) {
          auto it = groupOf.find(operand);
          key.push_back(it == groupOf.end() ? 0 : it->second + 1);
        }
        buckets[key].push_back(v);
      }
      if (buckets.size() > 1)
        changed = true;
      for (auto &entry : buckets)
        refined.push_back(entry.second);
    }
    groups = std::move(refined);
  }

  llvm::erase_if(groups, [&](const SmallVector<Value> &g) {
    return g.size() < 2 || g.size() > kMaxLanes ||
           !isa<RankedTensorType>(g.front().getType());
  });
  return groups;
}

struct Cone {
  unsigned n = 0;
  SmallVector<SmallVector<Value>> leafGroups;
  SmallVector<DenseMap<Value, Value>> laneImages;
  // Reference (lane 0) intermediate ops.
  SmallVector<Operation *> refOps;
};

// Groups the operands of `seed` recursively. Every group has the same size as
// the seed; groups whose members are not structurally identical become leaves.
static FailureOr<Cone> discoverCone(ArrayRef<Value> seed) {
  Cone cone;
  cone.n = seed.size();
  cone.laneImages.resize(cone.n);
  DenseSet<Value> seen;
  SmallVector<SmallVector<Value>> worklist;
  worklist.push_back(SmallVector<Value>(seed.begin(), seed.end()));
  while (!worklist.empty()) {
    SmallVector<Value> group = worklist.pop_back_val();
    Value ref = group.front();
    if (!seen.insert(ref).second)
      continue;

    for (unsigned i = 1; i < cone.n; ++i)
      cone.laneImages[i][ref] = group[i];

    Operation *op0 = ref.getDefiningOp();
    bool isLeaf = !op0 || !isLiftableOp(op0);
    bool mismatchedComputational = false;
    if (!isLeaf) {
      for (unsigned i = 1; i < cone.n; ++i) {
        Operation *oi = group[i].getDefiningOp();
        if (!oi || !sameOpStructure(op0, oi)) {
          isLeaf = true;
          mismatchedComputational = true;
          break;
        }
      }
    }

    if (isLeaf) {
      // A group of liftable ops that are not structurally identical is not a
      // lane family, just unrelated ops that happen to share a shallow
      // signature. Packing them would be correct but pointless and can blow up
      // the fixpoint, so reject the whole cone.
      if (mismatchedComputational) {
        if (lanePackDebug())
          llvm::errs() << "[lane-pack] cone fail: mixed computational leaf at "
                       << op0->getName() << "\n";
        return failure();
      }
      if (!isa<RankedTensorType>(ref.getType())) {
        if (lanePackDebug())
          llvm::errs() << "[lane-pack] cone fail: non-tensor leaf " << ref
                       << "\n";
        return failure();
      }
      cone.leafGroups.push_back(group);
      continue;
    }

    cone.refOps.push_back(op0);
    for (unsigned k = 0; k < op0->getNumOperands(); ++k) {
      SmallVector<Value> operands;
      operands.reserve(cone.n);
      for (unsigned i = 0; i < cone.n; ++i)
        operands.push_back(group[i].getDefiningOp()->getOperand(k));
      if (llvm::all_of(operands,
                       [&](Value v) { return v == operands.front(); }))
        continue; // lane-invariant
      Type ty = operands.front().getType();
      if (!llvm::all_of(operands, [&](Value v) { return v.getType() == ty; })) {
        if (lanePackDebug())
          llvm::errs() << "[lane-pack] cone fail: operand type mismatch at "
                       << op0->getName() << " operand " << k << "\n";
        return failure();
      }
      worklist.push_back(operands);
    }
  }
  return cone;
}

// A packing boundary (tensor.concat created by the loop rewrite, or by an
// explicit pack) exposes exactly the lane values of a cone: use the values
// feeding the concat (looking through the per-lane reshapes) as the seed, and
// remember the concat so it can be replaced by the packed result.
struct ConcatBoundary {
  Operation *concat;
  SmallVector<Value> sources;
};

static std::optional<ConcatBoundary>
findConcatBoundary(Block *block, const DenseSet<Operation *> &skip) {
  for (Operation &op : *block) {
    if (skip.contains(&op))
      continue;
    auto concat = dyn_cast<tensor::ConcatOp>(&op);
    if (!concat)
      continue;
    SmallVector<Value> sources;
    for (Value input : concat.getInputs()) {
      Value src = input;
      if (auto reshape = input.getDefiningOp<tensor::ReshapeOp>())
        src = reshape.getSource();
      sources.push_back(src);
    }
    if (sources.size() < 2)
      continue;
    Type ty = sources.front().getType();
    if (!isa<RankedTensorType>(ty))
      continue;
    if (!llvm::all_of(sources, [&](Value v) { return v.getType() == ty; }))
      continue;
    return ConcatBoundary{concat.getOperation(), std::move(sources)};
  }
  return std::nullopt;
}

// Packs one lane cone in a straight-line block. Returns true if anything was
// rewritten.
static bool rewriteBlock(Block *block, Operation *scope,
                         DenseSet<Operation *> &skip) {
  // Snapshot so that, on success, every op the rewrite emitted can be marked
  // and excluded from later fixpoint iterations in the same block.
  SmallPtrSet<Operation *, 32> before;
  for (Operation &op : *block)
    before.insert(&op);

  SmallVector<SmallVector<Value>> candidates;
  std::optional<ConcatBoundary> boundary = findConcatBoundary(block, skip);
  if (boundary)
    candidates.push_back(boundary->sources);
  SmallVector<SmallVector<Value>> siblingGroups =
      findSiblingGroups(block, skip);
  llvm::stable_sort(siblingGroups, [](const SmallVector<Value> &a,
                                      const SmallVector<Value> &b) {
    return a.size() > b.size();
  });
  for (SmallVector<Value> &g : siblingGroups)
    candidates.push_back(std::move(g));

  if (lanePackDebug())
    llvm::errs() << "[lane-pack] rewriteBlock("
                 << block->getParentOp()->getName()
                 << ") boundary=" << (boundary ? boundary->sources.size() : 0)
                 << " candidates=" << candidates.size() << "\n";

  for (unsigned ci = 0; ci < candidates.size(); ++ci) {
    SmallVector<Value> &seed = candidates[ci];
    bool isBoundarySeed = boundary && ci == 0;
    FailureOr<Cone> cone = discoverCone(seed);
    if (lanePackDebug())
      llvm::errs() << "[lane-pack]   cand " << ci << " n=" << seed.size()
                   << " boundarySeed=" << isBoundarySeed
                   << " cone=" << (failed(cone) ? "FAIL" : "ok") << " refOps="
                   << (failed(cone) ? 0 : (int)cone->refOps.size()) << "\n";
    if (failed(cone))
      continue;

    // Earliest cone ref op: the lifter scans from here so every cone op is
    // visited, regardless of where the packed ops are emitted.
    Operation *processStart = nullptr;
    for (Operation *op : cone->refOps)
      if (!processStart || op->isBeforeInBlock(processStart))
        processStart = op;
    if (!processStart) {
      if (lanePackDebug())
        llvm::errs() << "[lane-pack]   skip: no ref op\n";
      continue;
    }

    // Emit the packed ops after the last leaf defined in this block so every
    // leaf dominates them, even when leaves are interleaved with cone ops.
    Operation *emissionPoint = processStart;
    for (SmallVector<Value> &g : cone->leafGroups) {
      for (Value v : g) {
        Operation *def = v.getDefiningOp();
        if (!def || def->getBlock() != block)
          continue;
        if (def->isBeforeInBlock(emissionPoint))
          continue;
        if (Operation *next = def->getNextNode())
          emissionPoint = next;
      }
    }
    if (lanePackDebug()) {
      llvm::errs() << "[lane-pack]   processStart: " << *processStart
                   << "\n[lane-pack]   emissionPoint: " << *emissionPoint
                   << "\n[lane-pack]   leafGroups=" << cone->leafGroups.size()
                   << "\n";
      for (SmallVector<Value> &g : cone->leafGroups) {
        llvm::errs() << "[lane-pack]     leaf:";
        for (Value v : g)
          llvm::errs() << " " << v;
        llvm::errs() << "\n";
      }
    }

    OpBuilder builder(emissionPoint);
    Location loc = emissionPoint->getLoc();
    Lifter lifter(scope, processStart, builder, cone->n);
    lifter.laneImages = std::move(cone->laneImages);
    for (unsigned i = 1; i < cone->n; ++i)
      for (auto &entry : lifter.laneImages[i])
        lifter.laneVarying.insert(entry.second);

    // Snapshot the ops to visit before emitting anything: emitted ops are
    // inserted before emissionPoint, which lies inside [processStart, end), so
    // walking getNextNode() live would also visit (and re-lift) them.
    SmallVector<Operation *> worklist;
    for (Operation *op = processStart; op; op = op->getNextNode())
      worklist.push_back(op);

    DenseSet<Value> leafRefs;
    for (SmallVector<Value> &g : cone->leafGroups) {
      Value packed = buildPackedLanes(builder, loc, g);
      // Seed every lane of the leaf, not just the reference: otherwise the
      // sibling lanes (defined before the cone start) are misclassified as
      // lane-invariant by isShared and their whole chain is skipped.
      for (Value v : g)
        lifter.packedOf[v] = {packed, false};
      leafRefs.insert(g.front());
    }

    // Lift the cone plus any forward lane-parallel extension.
    for (Operation *op : worklist)
      (void)lifter.liftOp(op);
    if (lanePackDebug())
      llvm::errs() << "[lane-pack]   lifted=" << lifter.liftedOps.size()
                   << "\n";

    // If this cone is the producer of a pack boundary, hand the packed result
    // straight to the concat's users instead of unpacking and re-packing.
    if (isBoundarySeed) {
      // Never retry this boundary, consumed or not.
      skip.insert(boundary->concat);
      auto it = lifter.packedOf.find(boundary->sources.front());
      if (it != lifter.packedOf.end() && !it->second.shared &&
          it->second.value.getType() ==
              boundary->concat->getResult(0).getType()) {
        boundary->concat->getResult(0).replaceAllUsesWith(it->second.value);
        SmallVector<Operation *> reshapes;
        for (Value input : boundary->concat->getOperands())
          if (auto reshape = input.getDefiningOp<tensor::ReshapeOp>())
            reshapes.push_back(reshape.getOperation());
        boundary->concat->erase();
        for (Operation *reshape : reshapes)
          if (reshape->use_empty())
            reshape->erase();
      }
    }

    // The unpacked values are materialised at the emission point, so only uses
    // at or after it can be served; earlier uses keep the original (unpacked)
    // computation, which therefore is not erased.
    auto canServe = [&](Operation *user) {
      return user->getBlock() != block || !user->isBeforeInBlock(emissionPoint);
    };

    // Materialise escaping lane values, then erase the original computation.
    for (Value ref : lifter.referenceOrder) {
      auto it = lifter.packedOf.find(ref);
      if (it == lifter.packedOf.end() || it->second.shared ||
          leafRefs.contains(ref))
        continue;
      SmallVector<Value> lanes(cone->n);
      lanes[0] = ref;
      for (unsigned i = 1; i < cone->n; ++i)
        lanes[i] = lifter.laneImages[i].lookup(ref);
      bool external = false;
      for (Value lane : lanes) {
        if (!lane)
          continue;
        for (OpOperand &use : lane.getUses())
          if (!lifter.liftedOps.contains(use.getOwner()) &&
              canServe(use.getOwner())) {
            external = true;
            break;
          }
        if (external)
          break;
      }
      if (!external)
        continue;
      SmallVector<Value> unpacked =
          unpackPackedLanes(builder, loc, it->second.value, cone->n);
      if (unpacked.size() != cone->n)
        continue;
      for (unsigned i = 0; i < cone->n; ++i) {
        if (!lanes[i])
          continue;
        for (OpOperand &use : llvm::make_early_inc_range(lanes[i].getUses()))
          if (!lifter.liftedOps.contains(use.getOwner()) &&
              canServe(use.getOwner()))
            use.set(unpacked[i]);
      }
    }
    for (Value ref : lifter.sharedRefs) {
      auto it = lifter.packedOf.find(ref);
      if (it == lifter.packedOf.end())
        continue;
      for (OpOperand &use : llvm::make_early_inc_range(ref.getUses()))
        if (!lifter.liftedOps.contains(use.getOwner()) &&
            canServe(use.getOwner()))
          use.set(it->second.value);
    }

    SmallVector<Operation *> toErase;
    for (Operation *op : lifter.liftedOps)
      if (op->getBlock() == block)
        toErase.push_back(op);
    // Anything this rewrite consumed or emitted must not become a candidate
    // again, even if it survives (still used) instead of being erased.
    for (Operation *op : toErase)
      skip.insert(op);
    llvm::stable_sort(toErase, [](Operation *a, Operation *b) {
      return a->isBeforeInBlock(b);
    });
    for (Operation *op : llvm::reverse(toErase))
      if (op->use_empty())
        op->erase();

    // Sweep up any original ops that became dead (e.g. the sibling lanes of a
    // value that escaped only through an erased reshape).
    bool changed = true;
    while (changed) {
      changed = false;
      for (Operation &op : llvm::make_early_inc_range(*block)) {
        if (&op == block->getTerminator())
          continue;
        if (op.use_empty() && isMemoryEffectFree(&op)) {
          op.erase();
          changed = true;
        }
      }
    }
    // If nothing was actually lifted this candidate was a no-op (e.g. the
    // boundary seed failed); roll forward to the next candidate.
    if (lifter.liftedOps.empty()) {
      if (lanePackDebug())
        llvm::errs() << "[lane-pack]   no-op candidate, trying next\n";
      continue;
    }
    // Mark every op this rewrite emitted so later fixpoint iterations in the
    // same block do not re-pack them.
    for (Operation &op : *block)
      if (!before.contains(&op))
        skip.insert(&op);
    return true;
  }
  return false;
}

// ---------------------------------------------------------------------------
// Pass
// ---------------------------------------------------------------------------

struct LanePackPass : public impl::TritonLanePackBase<LanePackPass> {
  using TritonLanePackBase::TritonLanePackBase;

  void runOnOperation() override {
    SmallPtrSet<Block *, 16> packedBodies;
    getOperation().walk([&](scf::ForOp forOp) {
      if (succeeded(rewriteLanePackLoop(forOp, packedBodies)))
        LLVM_DEBUG(llvm::dbgs() << "[lane-pack] packed loop\n");
    });

    // Straight-line packing runs on every block (the lane-parallel prologue is
    // often inside an outer loop body), except the loop bodies already produced
    // by the loop rewrite. Each block is rewritten to a fixpoint so several
    // independent lane cones can be packed.
    SmallVector<Block *> blocks;
    getOperation().walk([&](Operation *op) {
      for (Region &region : op->getRegions())
        for (Block &block : region)
          blocks.push_back(&block);
    });
    if (lanePackDebug())
      llvm::errs() << "[lane-pack] run: blocks=" << blocks.size()
                   << " packedBodies=" << packedBodies.size() << "\n";
    DenseSet<Operation *> packedOps;
    for (Block *block : blocks) {
      Operation *parent = block->getParentOp();
      if (packedBodies.contains(block)) {
        if (lanePackDebug())
          llvm::errs() << "[lane-pack] block parent=" << parent->getName()
                       << " skip=packedBody\n";
        continue;
      }
      // Loop bodies with iter args are the loop rewrite's job; only pack
      // straight-line code (e.g. the body of the outer token loop, which has no
      // iter args).
      if (auto forOp = dyn_cast<scf::ForOp>(parent)) {
        if (forOp.getNumRegionIterArgs() > 0) {
          if (lanePackDebug())
            llvm::errs() << "[lane-pack] block parent=" << parent->getName()
                         << " skip=loopWithIterArgs\n";
          continue;
        }
      }
      if (lanePackDebug())
        llvm::errs() << "[lane-pack] block parent=" << parent->getName()
                     << " visiting\n";
      // Bound the fixpoint; a block with a great many families still gets
      // several of them packed without risking an unbounded rewrite loop.
      unsigned rewrites = 0;
      while (rewrites < 64 && rewriteBlock(block, parent, packedOps))
        ++rewrites;
    }
  }
};

} // namespace

} // namespace mlir::triton
