//===----------------------------------------------------------------------===//
// LaneVectorize.cpp - Lane-parallel operation vectorization
//
// Recognizes groups of loop-carried Triton tensor "lanes" (for example four
// independent tensor<Mxf32> values updated by the same computation) and
// rewrites them so the lanes are packed into a single tensor with a new
// leading dimension.
//
// The transform is purely structural: an operation is lifted to packed form
// when each of its per-lane instances computes the same math and its operands
// are either already packed or lane-invariant. An associative combine across
// the lanes (e.g. an add tree over all lanes) is recognized as a reduction
// over the synthesized lane axis. No semantic pattern (softmax, norms, ...) is
// hard-coded; anything whose packed form is mathematically equivalent is
// allowed.
//
// Two modes share one packer:
//   * loop mode packs the iter args of an scf.for and replays its body once,
//   * block mode (SLP) discovers a "lane cone" in straight-line code, seeded
//     by a tensor.concat packing boundary or by shallow structural signatures.
//
// Scope / hard boundaries (kept deliberately conservative):
//   * Only side-effect-free "math" ops are lifted. Loads/stores/copies/barriers
//     are boundaries, left for other passes to vectorize (memory vectorization
//     is out of scope).
//   * Only "compute" types are packed: scalar/vector/tensor of integer, float,
//     index or complex. Pointer/buffer/memref/token types are never packed, so
//     a cone can never materialize a pointer-typed tensor.
//   * arith.select (tl.where) is NOT packable even though it is elementwise.
//     It is used to select between the arms of a conditional ping-pong, and
//     packing those arms makes the (Ascend) emitter produce non-zero-offset
//     lane subviews -> dynamic-stride memrefs that the downstream stride-align,
//     PlanMemory and hivmc stages cannot handle. Everything else allowed is
//     offset-free, so its lane views keep statically unit striding.
//   * tt.reshape with allow_reorder is rejected: element reordering could move
//     data across lanes.
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
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/raw_ostream.h"

#include <map>
#include <vector>

namespace mlir::triton {

#define GEN_PASS_DEF_TRITONLANEVECTORIZE
#include "triton/Dialect/Triton/Transforms/Passes.h.inc"

namespace {

// Diagnostics go to stderr regardless of build type when LANE_VECTORIZE_DEBUG
// is set in the environment. This is intentionally independent of LLVM_DEBUG.
static bool debugEnabled() {
  static const bool enabled = ::getenv("LANE_VECTORIZE_DEBUG") != nullptr;
  return enabled;
}

// Prints a compact before/after view of a lane group and its packed result:
//
//   [lane-vectorize] lift: 2 lanes -> tensor<2x4xf32>
//   [lane-vectorize]   | lane0: %r0 = arith.addf %l0, %y : tensor<4xf32>
//   [lane-vectorize]   | lane1: %r1 = arith.addf %l1, %y : tensor<4xf32>
//   [lane-vectorize]   v
//   [lane-vectorize]   %rp = arith.addf %lp, %yb : tensor<2x4xf32>
//
// Only emits output when LANE_VECTORIZE_DEBUG is set.
static void dumpLaneGroup(StringRef label, ArrayRef<Value> lanes,
                          Value packed) {
  if (!debugEnabled())
    return;
  llvm::errs() << "[lane-vectorize] " << label << ": " << lanes.size()
               << " lanes -> " << packed.getType() << "\n";
  for (auto [i, lane] : llvm::enumerate(lanes)) {
    llvm::errs() << "  | lane" << i << ": ";
    if (Operation *def = lane.getDefiningOp())
      llvm::errs() << *def;
    else
      llvm::errs() << lane << " : " << lane.getType();
    llvm::errs() << "\n";
  }
  llvm::errs() << "  v\n  ";
  if (Operation *def = packed.getDefiningOp())
    llvm::errs() << *def;
  else
    llvm::errs() << packed << " : " << packed.getType();
  llvm::errs() << "\n";
}

// ---------------------------------------------------------------------------
// Packing / unpacking primitives
// ---------------------------------------------------------------------------

static Value buildShapeConst(OpBuilder &builder, Location loc,
                             ArrayRef<int64_t> shape) {
  return builder.create<arith::ConstantOp>(loc,
                                           builder.getI64TensorAttr(shape));
}

// If the lanes are a contiguous run of tensor.extract_slice of one source
// (e.g. src[8], src[12], src[16], src[20]), pack them with a single wider
// slice + reshape instead of a concat. Returns null when it does not apply.
static Value packContiguousSlices(OpBuilder &builder, Location loc,
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

  // Every lane must be the same slice shape/strides, offset only in dim 0 by
  // exactly one lane extent each step, and identical in all other dims.
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
static Value packLanes(OpBuilder &builder, Location loc, ArrayRef<Value> lanes) {
  assert(lanes.size() >= 2 && "expected at least two lanes");
  if (Value coalesced = packContiguousSlices(builder, loc, lanes))
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

// Inverse of packLanes: slices the leading lane dimension back into `laneCount`
// individual lane tensors.
static SmallVector<Value> unpackLanes(OpBuilder &builder, Location loc,
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
  if (debugEnabled()) {
    llvm::errs() << "[lane-vectorize] unpack: ";
    if (Operation *def = packed.getDefiningOp())
      llvm::errs() << *def;
    else
      llvm::errs() << packed;
    llvm::errs() << "\n";
  }
  return lanes;
}

// Recursively erases the defining op of `v` (and its dead operands) once it has
// no remaining uses.
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
// Packability predicates
// ---------------------------------------------------------------------------

// A "compute" value is a scalar, or a vector/tensor whose element type is a
// number. Pointers, buffers, memrefs and tokens are addressing/memory, not
// math, and must never be materialized in packed form.
static bool isComputeType(Type type) {
  if (isa<BaseMemRefType>(type))
    return false;
  Type elem = type;
  if (auto shaped = dyn_cast<ShapedType>(type))
    elem = shaped.getElementType();
  return isa<IntegerType, FloatType, IndexType, ComplexType>(elem);
}

// Packable ops may only produce and consume compute values.
static bool hasOnlyComputeTypes(Operation *op) {
  return llvm::all_of(op->getOperands(),
                      [](Value v) { return isComputeType(v.getType()); }) &&
         llvm::all_of(op->getResultTypes(),
                      [](Type t) { return isComputeType(t); });
}

static bool isPackableElementwise(Operation *op) {
  // Reject addressing/buffer ops (tt.addptr, tt.ptr_to_int, buffer
  // materializations) so a packed cone can never contain pointer/buffer types.
  if (!hasOnlyComputeTypes(op))
    return false;

  // See the file header: packing tl.where breaks the Ascend backend's
  // stride-align / PlanMemory / hivmc stages (dynamic-stride lane views).
  if (isa<arith::SelectOp>(op))
    return false;

  // An elementwise op applies pointwise and broadcasts operands to the result
  // shape, so adding a leading lane dimension preserves it.
  if (op->hasTrait<OpTrait::Elementwise>())
    return true;
  return isa<arith::BitcastOp, triton::SplatOp, triton::BroadcastOp,
             triton::ExpandDimsOp, triton::TransOp, triton::ReshapeOp>(op);
}

// Associative/commutative ops that can be realized as a tt.reduce over the
// synthesized lane axis.
static bool isAssociativeCombine(Operation *op) {
  return isa<arith::AddFOp, arith::AddIOp, arith::MulFOp, arith::MulIOp,
             arith::MaximumFOp, arith::MinimumFOp, arith::MaxNumFOp,
             arith::MinNumFOp, arith::MaxSIOp, arith::MaxUIOp, arith::MinSIOp,
             arith::MinUIOp, arith::AndIOp, arith::OrIOp, arith::XOrIOp>(op);
}

// Ops the packer can lift for a lane group: elementwise/shape ops and tt.reduce
// with an associative single combiner.
static bool isLiftableOp(Operation *op) {
  if (auto reduce = dyn_cast<triton::ReduceOp>(op)) {
    Operation *combiner = reduce.getSingleCombiner();
    return combiner && isAssociativeCombine(combiner);
  }
  return isPackableElementwise(op) && op->getNumRegions() == 0;
}

// Collects the leaves of an associative tree rooted at `v` (recursing through
// same-name ops only).
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

// Collects every op of the associative tree rooted at `v`.
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

// The packed form of a value.
struct Packing {
  Value value;
  // A shared value is lane-invariant: it has the shape of a single lane (or is
  // a scalar) and must be broadcast along the lane axis at each use.
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

// Grows an already packed value to `dstTy` by appending size-1 dims and
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
static Value materialize(OpBuilder &builder, Location loc, Packing pv,
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

  // Lane-invariant scalar: broadcast to the packed shape.
  return builder.create<triton::SplatOp>(loc, dstTy, pv.value);
}

// Builds a tt.reduce over `src` whose combine region is cloned from `proto`
// (a per-lane reduce).
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
// of `kind` (a cross-lane associative combine).
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
// Structural packer
// ---------------------------------------------------------------------------
//
// Lifts per-lane operations into packed form. State:
//   * lanesOf[ref] — the n per-lane values corresponding to the lane-0
//                    reference `ref` (lanesOf[ref][0] == ref). Built by cone
//                    discovery (block mode) or incrementally as ops are lifted.
//   * packed[ref]  — the packed form of `ref`, or the value itself with
//                    shared==true when `ref` is lane-invariant.
//   * laneVarying  — every value known to vary per lane, for a fast resolve().
//
// resolve(v) is the single classification entry point for operands of a lifted
// op: it returns the packing to use for `v`, or nullopt when `v` is
// lane-varying but not yet lifted.

struct Packer {
  // -- configuration --
  scf::ForOp forOp;               // loop mode: the loop; block mode: null
  Operation *scope = nullptr;     // values defined outside it are lane-invariant
  Block *laneBody = nullptr;      // loop body (loop mode); null in block mode
  Operation *coneStart = nullptr; // block mode: values before this are shared
  bool blockMode = false;
  OpBuilder &builder;
  Location loc;
  unsigned n;

  // -- state --
  DenseMap<Value, SmallVector<Value>> lanesOf; // ref -> {lane0..laneN-1}
  DenseMap<Value, Packing> packed;             // ref -> packed/shared form
  DenseSet<Value> laneVarying;                 // every lane-varying value
  DenseMap<Value, Value> argRemap;             // old loop args -> new loop args
  SmallVector<unsigned> laneIndices;           // iter-arg indices of the lanes
  SmallVector<Value> packedRefs;   // lane-varying refs in lift order
  SmallVector<Value> sharedRefs;   // cross-lane reduce results (shared)
  SmallPtrSet<Operation *, 32> liftedOps;      // original ops replaced

  Packer(scf::ForOp forOp, OpBuilder &builder, unsigned n)
      : forOp(forOp), scope(forOp.getOperation()), laneBody(forOp.getBody()),
        builder(builder), loc(forOp.getLoc()), n(n) {}

  // Block (straight-line) mode constructor.
  Packer(Operation *scope, Operation *coneStart, OpBuilder &builder, unsigned n)
      : scope(scope), coneStart(coneStart), builder(builder),
        loc(scope->getLoc()), n(n), blockMode(true) {}

  // Seeds the lane group from the loop's iter args, and remaps the induction
  // variable and every non-lane iter arg to the new loop's args.
  void seed(Value packedCurrent, ArrayRef<unsigned> indices,
            scf::ForOp newFor) {
    laneIndices.assign(indices.begin(), indices.end());
    auto iterArgs = forOp.getRegionIterArgs();

    SmallVector<Value> lanes;
    lanes.reserve(n);
    for (unsigned idx : laneIndices)
      lanes.push_back(iterArgs[idx]);
    Value ref = lanes[0];
    lanesOf[ref] = lanes;
    laneVarying.insert(lanes.begin(), lanes.end());
    packed[ref] = {packedCurrent, /*shared=*/false};
    packedRefs.push_back(ref);

    argRemap[forOp.getInductionVar()] = newFor.getInductionVar();
    auto newIterArgs = newFor.getRegionIterArgs();
    unsigned nextNew = 1; // newIterArgs[0] is the packed lane group
    for (unsigned j = 0; j < iterArgs.size(); ++j) {
      if (llvm::is_contained(laneIndices, j))
        continue;
      argRemap[iterArgs[j]] = newIterArgs[nextNew++];
    }
  }

  // Classifies `v` as an operand of a lifted op:
  //   - packed/seeded:      returns its Packing,
  //   - lane-invariant:     returns {remappedValue, shared=true},
  //   - lane-varying, unresolved: nullopt.
  std::optional<Packing> resolve(Value v) {
    if (auto it = packed.find(v); it != packed.end())
      return it->second;
    if (laneVarying.contains(v))
      return std::nullopt;

    // Block arguments are lane-invariant. Loop body args (induction var and
    // non-lane iter args) are remapped to the new loop so the packed ops they
    // feed dominate their uses.
    if (isa<BlockArgument>(v)) {
      if (auto it = argRemap.find(v); it != argRemap.end())
        return Packing{it->second, /*shared=*/true};
      return Packing{v, /*shared=*/true};
    }

    // Constants and values defined outside the packed scope are lane-invariant.
    Operation *def = v.getDefiningOp();
    if (!def || !scope->isAncestor(def))
      return Packing{v, /*shared=*/true};

    // Block mode: values defined before the cone start are lane-invariant.
    if (coneStart && def->getBlock() == coneStart->getBlock() &&
        def->isBeforeInBlock(coneStart))
      return Packing{v, /*shared=*/true};

    return std::nullopt;
  }

  bool isShared(Value v) {
    auto p = resolve(v);
    return p.has_value() && p->shared;
  }

  bool allOperandsShared(Operation *op) {
    return llvm::all_of(op->getOperands(),
                        [&](Value a) { return isShared(a); });
  }

  bool allOperandsResolved(Operation *op) {
    return llvm::all_of(op->getOperands(),
                        [&](Value a) { return resolve(a).has_value(); });
  }

  // Finds the lane-`lane` counterpart of `op`. Prefers the structural mapping
  // recorded by cone discovery: findLaneOp cannot tell apart structurally
  // identical siblings (e.g. two tt.splat of the same scalar).
  Operation *findSiblingOp(Operation *op, unsigned lane) {
    if (auto it = lanesOf.find(op->getResult(0)); it != lanesOf.end()) {
      Value img = it->second[lane];
      if (img)
        return img.getDefiningOp();
    }
    return findLaneOp(op, lane);
  }

  // Structural search for the lane-`lane` counterpart of `op`, matching by
  // operand list (with lane-varying operands mapped to their lane images).
  Operation *findLaneOp(Operation *op, unsigned lane) {
    SmallVector<Value> expected;
    for (Value a : op->getOperands()) {
      if (isShared(a)) {
        expected.push_back(a);
        continue;
      }
      auto it = lanesOf.find(a);
      if (it == lanesOf.end())
        return nullptr;
      expected.push_back(it->second[lane]);
    }
    if (expected.empty())
      return nullptr;

    for (Operation *u : expected.front().getUsers()) {
      if (u->getName() != op->getName() ||
          u->getNumOperands() != op->getNumOperands() ||
          u->getNumResults() != op->getNumResults())
        continue;
      if (!llvm::equal(u->getOperands(), expected))
        continue;
      if (!OperationEquivalence::isEquivalentTo(
              op, u, OperationEquivalence::ignoreValueEquivalence, nullptr,
              OperationEquivalence::Flags::IgnoreLocations))
        continue;
      return u;
    }
    return nullptr;
  }

  // Clones `op` with `operands` substituted and retypes the results.
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

  // Lane-invariant op inside the loop: clone it (with packed operands) and
  // treat the result as shared.
  LogicalResult liftSharedOp(Operation *op) {
    if (op->getNumResults() != 1)
      return failure();
    SmallVector<Value> operands;
    for (Value a : op->getOperands()) {
      auto pv = resolve(a);
      if (!pv)
        return failure();
      operands.push_back(pv->value);
    }
    Value result = emitClonedOp(op, operands, op->getResult(0).getType());
    packed[op->getResult(0)] = {result, /*shared=*/true};
    return success();
  }

  // Records the lifted op `op` (reference) with its per-lane sibling ops.
  void recordLifted(Operation *op, Value packedValue,
                    ArrayRef<Operation *> laneOps) {
    SmallVector<Value> lanes;
    lanes.push_back(op->getResult(0));
    for (Operation *oi : laneOps)
      lanes.push_back(oi->getResult(0));
    dumpLaneGroup("lift", lanes, packedValue);

    lanesOf[op->getResult(0)] = lanes;
    laneVarying.insert(lanes.begin() + 1, lanes.end());
    packed[op->getResult(0)] = {packedValue, /*shared=*/false};
    packedRefs.push_back(op->getResult(0));
    liftedOps.insert(op);
    for (Operation *oi : laneOps)
      liftedOps.insert(oi);
  }

  // A tt.reduce applied independently to every lane -> one tt.reduce over the
  // same axis + 1 (the new lane axis is 0).
  LogicalResult liftLaneReduce(triton::ReduceOp reduce) {
    if (reduce->getNumOperands() != 1 || reduce->getNumResults() != 1)
      return failure();
    Operation *combiner = reduce.getSingleCombiner();
    if (!combiner || !isAssociativeCombine(combiner))
      return failure();
    auto pv = resolve(reduce.getOperand(0));
    if (!pv || pv->shared)
      return failure();

    SmallVector<Operation *> laneOps;
    for (unsigned i = 1; i < n; ++i) {
      Operation *oi = findSiblingOp(reduce, i);
      if (!oi)
        return failure();
      auto other = dyn_cast<triton::ReduceOp>(oi);
      if (!other || !other.getSingleCombiner() ||
          other.getSingleCombiner()->getName() != combiner->getName())
        return failure();
      laneOps.push_back(oi);
    }

    Value packedValue = buildReduceFromRegion(builder, loc, pv->value,
                                              reduce.getAxis() + 1, reduce);
    recordLifted(reduce, packedValue, laneOps);
    return success();
  }

  // A pointwise/shape op applied independently to every lane -> one packed op.
  LogicalResult liftPerLaneOp(Operation *op) {
    if (op->getNumResults() != 1)
      return failure();
    if (isa<triton::ReduceOp>(op))
      return liftLaneReduce(cast<triton::ReduceOp>(op));
    if (op->getNumRegions() != 0 || !isPackableElementwise(op))
      return failure();

    SmallVector<Packing> pvs;
    for (Value a : op->getOperands()) {
      auto pv = resolve(a);
      if (!pv)
        return failure();
      pvs.push_back(*pv);
    }

    SmallVector<Operation *> laneOps;
    for (unsigned i = 1; i < n; ++i) {
      Operation *oi = findSiblingOp(op, i);
      if (!oi)
        return failure();
      laneOps.push_back(oi);
    }

    auto resultTy = packedTypeOf(op->getResult(0), n);
    Value packedValue;
    if (auto splat = dyn_cast<triton::SplatOp>(op)) {
      // Scalar-lane -> tensor-lane: widen the packed source.
      packedValue = widenPacked(builder, loc, pvs[0].value, resultTy);
    } else if (auto expand = dyn_cast<triton::ExpandDimsOp>(op)) {
      packedValue = builder.create<triton::ExpandDimsOp>(
          loc, resultTy, pvs[0].value, expand.getAxis() + 1);
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
      packedValue = newTrans.getResult();
    } else if (auto reshape = dyn_cast<triton::ReshapeOp>(op)) {
      // Element reordering could move data across lanes.
      if (reshape.getAllowReorder())
        return failure();
      packedValue = emitClonedOp(op, {pvs[0].value}, resultTy);
    } else {
      SmallVector<Value> operands;
      for (auto [a, pv] : llvm::zip(op->getOperands(), pvs)) {
        Value v = materialize(builder, loc, pv, packedOperandType(a, resultTy));
        if (!v)
          return failure();
        operands.push_back(v);
      }
      packedValue = emitClonedOp(op, operands, resultTy);
    }
    if (!packedValue)
      return failure();

    recordLifted(op, packedValue, laneOps);
    return success();
  }

  // An associative combine over the lanes (e.g. an add tree) -> a tt.reduce
  // over lane axis 0, plus any lane-invariant leaves folded back in.
  LogicalResult liftCrossLane(Operation *op) {
    if (op->getNumResults() != 1 || !isAssociativeCombine(op))
      return failure();

    SmallVector<Value> leaves;
    collectAssociativeLeaves(op->getResult(0), op->getName().getStringRef(),
                             leaves);

    for (Value r : packedRefs) {
      auto pvIt = packed.find(r);
      if (pvIt == packed.end() || pvIt->second.shared)
        continue;
      if (!isa<RankedTensorType>(r.getType()))
        continue;

      auto lanesIt = lanesOf.find(r);
      if (lanesIt == lanesOf.end() || lanesIt->second.size() != n)
        continue;
      const SmallVector<Value> &expected = lanesIt->second;

      // Every expected lane leaf must appear exactly once among the tree's
      // leaves.
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

      // All remaining leaves must be lane-invariant.
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

      // Reduce over the lane axis, then fold the lane-invariant leaves back in.
      Value reduced =
          buildReduceFromKind(builder, loc, pvIt->second.value, 0, op);
      if (debugEnabled())
        llvm::errs() << "[lane-vectorize] cross-lane reduce: " << *op << "\n"
                     << "  v\n  " << *reduced.getDefiningOp() << "\n";
      SmallVector<Operation *> treeOps;
      collectAssociativeTreeOps(op->getResult(0), op->getName().getStringRef(),
                                treeOps);
      for (Operation *treeOp : treeOps)
        liftedOps.insert(treeOp);
      for (Value l : leaves) {
        if (expectedSet.contains(l.getAsOpaquePointer()))
          continue;
        auto lp = resolve(l);
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
      packed[op->getResult(0)] = {reduced, /*shared=*/true};
      sharedRefs.push_back(op->getResult(0));
      return success();
    }
    return failure();
  }

  LogicalResult liftOp(Operation *op) {
    // Values already seeded as leaves need no lifting.
    if (op->getNumResults() == 1 && packed.contains(op->getResult(0)))
      return success();

    // Effectful ops are hard boundaries: in loop mode this aborts the whole
    // rewrite; in block mode the cone simply stops here.
    if (!isMemoryEffectFree(op)) {
      if (blockMode)
        return success();
      if (debugEnabled())
        llvm::errs() << "[lane-vectorize] reject: effectful op " << *op << "\n";
      return failure();
    }

    if (allOperandsShared(op))
      return liftSharedOp(op);
    if (allOperandsResolved(op))
      return liftPerLaneOp(op);
    if (succeeded(liftCrossLane(op)))
      return success();

    // An unmatched pure op is simply dropped from the packed loop.
    if (debugEnabled())
      llvm::errs() << "[lane-vectorize] drop: unmatched pure op " << *op
                   << "\n";
    return success();
  }

  // Loop mode: lifts the whole body and validates the yield.
  FailureOr<Value> run() {
    for (Operation &op : forOp.getBody()->without_terminator())
      if (failed(liftOp(&op)))
        return failure();

    auto yield = dyn_cast<scf::YieldOp>(forOp.getBody()->getTerminator());
    if (!yield)
      return failure();
    Value y0 = yield.getOperand(laneIndices[0]);
    auto it = packed.find(y0);
    if (it == packed.end() || it->second.shared)
      return failure();
    auto lanesIt = lanesOf.find(y0);
    if (lanesIt == lanesOf.end())
      return failure();
    for (unsigned i = 1; i < n; ++i)
      if (lanesIt->second[i] != yield.getOperand(laneIndices[i]))
        return failure();
    return it->second.value;
  }
};

// ---------------------------------------------------------------------------
// Loop mode
// ---------------------------------------------------------------------------

static LogicalResult
rewriteLaneVectorizeLoop(scf::ForOp forOp,
                         SmallPtrSetImpl<Block *> &packedBodies) {
  OpBuilder builder(forOp);
  Location loc = forOp.getLoc();

  auto initArgs = forOp.getInitArgs();
  unsigned m = initArgs.size();
  if (m < 2)
    return failure();

  // Pick the largest group of same-typed tensor iter args to pack; the rest
  // are carried through unchanged.
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
  Value packedInit = packLanes(builder, loc, initLanes);
  dumpLaneGroup("loop init", initLanes, packedInit);

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

  // The loop rewrite is all-or-nothing: any failure rolls back to the original
  // loop.
  auto fail = [&]() -> LogicalResult {
    newFor.erase();
    eraseDeadTree(packedInit);
    return failure();
  };

  Packer packer(forOp, builder, n);
  // The body must start from the loop-carried packed value, not the pre-loop
  // packed init.
  packer.seed(newFor.getRegionIterArg(0), laneIndices, newFor);
  builder.setInsertionPointToStart(newFor.getBody());
  FailureOr<Value> packedYield = packer.run();
  if (failed(packedYield))
    return fail();

  auto oldYield = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  builder.setInsertionPointToEnd(newFor.getBody());
  SmallVector<Value> newYieldOperands{*packedYield};
  for (unsigned j : otherIndices) {
    auto pv = packer.resolve(oldYield.getOperand(j));
    if (!pv)
      return fail();
    newYieldOperands.push_back(pv->value);
  }
  builder.create<scf::YieldOp>(loc, newYieldOperands);

  builder.setInsertionPointAfter(newFor);
  SmallVector<Value> unpacked =
      unpackLanes(builder, loc, newFor.getResult(0), n);
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
  if (debugEnabled())
    llvm::errs() << "[lane-vectorize] packed loop: lanes=" << n << "\n";
  forOp.erase();
  return success();
}

// ---------------------------------------------------------------------------
// Block mode (SLP)
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

// A cheap structural signature of an op: its name, operand/result types and
// attributes (all uniqued). Two ops can only be sameOpStructure if their
// signatures match, so bucketing by this keeps the partition linear instead of
// doing O(n^2) OperationEquivalence comparisons. Regions are not part of the
// signature, so sameOpStructure is still applied within each bucket.
static uint64_t structureHash(Operation *op) {
  llvm::hash_code h = llvm::hash_value(op->getName().getStringRef());
  for (Type t : op->getOperandTypes())
    h = llvm::hash_combine(h, t.getAsOpaquePointer());
  for (Type t : op->getResultTypes())
    h = llvm::hash_combine(h, t.getAsOpaquePointer());
  for (NamedAttribute a : op->getAttrs())
    h = llvm::hash_combine(h, a.getName().str(),
                           a.getValue().getAsOpaquePointer());
  return h;
}

// Candidate lane groups in a block: single-result liftable tensor ops grouped
// by shallow structure, then refined by operand congruence.
static SmallVector<SmallVector<Value>>
findSiblingGroups(Block *block, const DenseSet<Operation *> &skip) {
  // Largest lane group we are willing to synthesize. A real family is small;
  // anything larger is almost certainly a mistaken merge.
  constexpr unsigned kMaxLanes = 32;

  SmallVector<Value> candidates;
  for (Operation &op : *block) {
    if (skip.contains(&op))
      continue;
    if (op.getNumResults() != 1 || !isLiftableOp(&op))
      continue;
    // The pack/unpack scaffolding used by this pass itself.
    if (isa<tensor::ReshapeOp, tensor::ExtractSliceOp>(&op))
      continue;
    candidates.push_back(op.getResult(0));
  }

  // Initial partition by a cheap structural signature (op name + operand/result
  // types + attributes), then split each signature bucket by sameOpStructure
  // (which also compares regions).
  SmallVector<SmallVector<Value>> groups;
  DenseMap<uint64_t, SmallVector<unsigned>> bySignature;
  for (Value v : candidates) {
    Operation *op = v.getDefiningOp();
    uint64_t sig = structureHash(op);
    bool added = false;
    for (unsigned gi : bySignature[sig]) {
      if (sameOpStructure(op, groups[gi].front().getDefiningOp())) {
        groups[gi].push_back(v);
        added = true;
        break;
      }
    }
    if (!added) {
      groups.push_back({v});
      bySignature[sig].push_back(groups.size() - 1);
    }
  }

  // Refine by operand congruence: two values stay together only if their
  // corresponding operands live in the same group. This separates e.g.
  // different iterations of an unrolled loop that share a shallow signature
  // (iteration N consumes iteration N-1's results).
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
  DenseMap<Value, SmallVector<Value>> lanesOf; // ref -> {lane0..laneN-1}
  SmallVector<Operation *> refOps;             // reference (lane 0) ops
};

// Discovers a lane cone from a seed group, recursing into operands. Every group
// has the same size as the seed; groups whose members are not structurally
// identical become leaves.
static FailureOr<Cone> discoverCone(ArrayRef<Value> seed) {
  Cone cone;
  cone.n = seed.size();
  DenseSet<Value> seen;
  SmallVector<SmallVector<Value>> worklist;
  worklist.push_back(SmallVector<Value>(seed.begin(), seed.end()));
  while (!worklist.empty()) {
    SmallVector<Value> group = worklist.pop_back_val();
    Value ref = group.front();
    if (!seen.insert(ref).second)
      continue;

    cone.lanesOf[ref] = group;

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
      // Liftable but structurally different ops are not a lane family, just
      // unrelated ops sharing a shallow signature; packing them would be
      // pointless and can blow up the fixpoint.
      if (mismatchedComputational)
        return failure();
      if (!isa<RankedTensorType>(ref.getType()) ||
          !isComputeType(ref.getType()))
        return failure();
      // An effectful leaf is a hard boundary: packing it would concat memory
      // results (memory vectorization), which is left to other passes.
      if (Operation *def = ref.getDefiningOp();
          def && !isMemoryEffectFree(def))
        return failure();
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
      if (!llvm::all_of(operands, [&](Value v) { return v.getType() == ty; }))
        return failure();
      worklist.push_back(operands);
    }
  }
  return cone;
}

// A packing boundary (tensor.concat created by a previous pack) exposes the
// lane values of a cone: its inputs (through per-lane reshapes) seed the cone,
// and the concat itself is replaced by the packed result.
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

// A candidate lane group for block mode. The seed values are kept alongside
// their defining ops (captured while live), so a candidate invalidated by an
// earlier rewrite can be dropped by pointer comparison without dereferencing a
// dangling Value.
struct Candidate {
  SmallVector<Value> seed;
  SmallVector<Operation *> ops;
  bool boundary = false;
};

// Packs one lane cone in a straight-line block. `candidates`, `boundary` and
// `before` are computed once per block by the caller, so the O(n^2)-prone
// discovery is not repeated on every fixpoint iteration; `liveOps` is the set
// of ops currently in the block, refreshed after every successful rewrite.
// Returns true if anything was rewritten.
static bool rewriteBlock(Block *block, Operation *scope,
                         DenseSet<Operation *> &skip,
                         ArrayRef<Candidate> candidates,
                         const std::optional<ConcatBoundary> &boundary,
                         const llvm::SmallPtrSetImpl<Operation *> &liveOps,
                         const llvm::SmallPtrSetImpl<Operation *> &before) {
  for (unsigned ci = 0; ci < candidates.size(); ++ci) {
    const Candidate &cand = candidates[ci];
    if (cand.boundary && !boundary)
      continue;
    // Drop candidates whose ops an earlier rewrite in this block erased. A
    // block-argument seed (e.g. a concat boundary over function arguments) has
    // no defining op and is always live.
    bool live = true;
    for (Operation *op : cand.ops)
      if (op && !liveOps.contains(op)) {
        live = false;
        break;
      }
    if (!live)
      continue;

    const SmallVector<Value> &seed = cand.seed;
    bool isBoundarySeed = cand.boundary;
    FailureOr<Cone> cone = discoverCone(seed);
    if (failed(cone))
      continue;

    // Earliest cone ref op: the packer scans from here so every cone op is
    // visited, regardless of where the packed ops are emitted.
    Operation *processStart = nullptr;
    for (Operation *op : cone->refOps)
      if (!processStart || op->isBeforeInBlock(processStart))
        processStart = op;
    if (!processStart)
      continue;

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

    OpBuilder builder(emissionPoint);
    Location loc = emissionPoint->getLoc();
    Packer packer(scope, processStart, builder, cone->n);
    packer.lanesOf = std::move(cone->lanesOf);
    for (auto &entry : packer.lanesOf)
      packer.laneVarying.insert(entry.second.begin(), entry.second.end());

    // Snapshot the ops to visit before emitting anything: emitted ops are
    // inserted before emissionPoint, which lies inside [processStart, end), so
    // walking getNextNode() live would also visit (and re-lift) them.
    SmallVector<Operation *> worklist;
    for (Operation *op = processStart; op; op = op->getNextNode())
      worklist.push_back(op);

    DenseSet<Value> leafRefs;
    for (SmallVector<Value> &g : cone->leafGroups) {
      Value packedLeaf = packLanes(builder, loc, g);
      dumpLaneGroup("leaf", g, packedLeaf);
      // Seed every lane of the leaf, not just the reference: otherwise the
      // sibling lanes (defined before the cone start) are misclassified as
      // lane-invariant and their whole chain is skipped.
      for (Value v : g)
        packer.packed[v] = {packedLeaf, /*shared=*/false};
      leafRefs.insert(g.front());
    }

    for (Operation *op : worklist)
      (void)packer.liftOp(op);

    // If this cone is the producer of a pack boundary, hand the packed result
    // straight to the concat's users instead of unpacking and re-packing.
    if (isBoundarySeed) {
      skip.insert(boundary->concat); // never retry this boundary
      auto it = packer.packed.find(boundary->sources.front());
      if (it != packer.packed.end() && !it->second.shared &&
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

    // The unpacked values are materialized at the emission point, so only uses
    // at or after it can be served; earlier uses keep the original (unpacked)
    // computation, which therefore is not erased.
    auto canServe = [&](Operation *user) {
      return user->getBlock() != block || !user->isBeforeInBlock(emissionPoint);
    };

    // Materialize escaping lane values, then erase the original computation.
    for (Value ref : packer.packedRefs) {
      auto it = packer.packed.find(ref);
      if (it == packer.packed.end() || it->second.shared ||
          leafRefs.contains(ref))
        continue;
      auto lanesIt = packer.lanesOf.find(ref);
      if (lanesIt == packer.lanesOf.end())
        continue;
      const SmallVector<Value> &lanes = lanesIt->second;

      bool external = false;
      for (Value lane : lanes) {
        for (OpOperand &use : lane.getUses()) {
          if (!packer.liftedOps.contains(use.getOwner()) &&
              canServe(use.getOwner())) {
            external = true;
            break;
          }
        }
        if (external)
          break;
      }
      if (!external)
        continue;

      SmallVector<Value> unpacked =
          unpackLanes(builder, loc, it->second.value, cone->n);
      if (unpacked.size() != cone->n)
        continue;
      for (unsigned i = 0; i < cone->n; ++i) {
        if (!lanes[i])
          continue;
        for (OpOperand &use : llvm::make_early_inc_range(lanes[i].getUses()))
          if (!packer.liftedOps.contains(use.getOwner()) &&
              canServe(use.getOwner()))
            use.set(unpacked[i]);
      }
    }
    for (Value ref : packer.sharedRefs) {
      auto it = packer.packed.find(ref);
      if (it == packer.packed.end())
        continue;
      for (OpOperand &use : llvm::make_early_inc_range(ref.getUses()))
        if (!packer.liftedOps.contains(use.getOwner()) &&
            canServe(use.getOwner()))
          use.set(it->second.value);
    }

    SmallVector<Operation *> toErase;
    for (Operation *op : packer.liftedOps)
      if (op->getBlock() == block)
        toErase.push_back(op);
    for (Operation *op : toErase)
      skip.insert(op);
    llvm::stable_sort(toErase, [](Operation *a, Operation *b) {
      return a->isBeforeInBlock(b);
    });
    for (Operation *op : llvm::reverse(toErase))
      if (op->use_empty())
        op->erase();

    // Sweep up any original ops that became dead.
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

    // A no-op candidate (e.g. the boundary seed failed) rolls to the next one.
    if (packer.liftedOps.empty())
      continue;

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

struct LaneVectorizePass : public impl::TritonLaneVectorizeBase<LaneVectorizePass> {
  using TritonLaneVectorizeBase::TritonLaneVectorizeBase;

  void runOnOperation() override {
    SmallPtrSet<Block *, 16> packedBodies;
    getOperation().walk([&](scf::ForOp forOp) {
      if (succeeded(rewriteLaneVectorizeLoop(forOp, packedBodies)))
        if (debugEnabled())
          llvm::errs() << "[lane-vectorize] packed loop\n";
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
    DenseSet<Operation *> packedOps;
    for (Block *block : blocks) {
      Operation *parent = block->getParentOp();
      if (packedBodies.contains(block))
        continue;
      // Loop bodies with iter args are the loop rewrite's job; only pack
      // straight-line code (e.g. the body of the outer token loop, which has no
      // iter args).
      if (auto forOp = dyn_cast<scf::ForOp>(parent)) {
        if (forOp.getNumRegionIterArgs() > 0)
          continue;
      }
      // Discover the lane groups once; the fixpoint below then only drops the
      // groups a rewrite invalidated instead of recomputing the partition on
      // every iteration.
      SmallPtrSet<Operation *, 32> before;
      for (Operation &op : *block)
        before.insert(&op);

      std::optional<ConcatBoundary> boundary =
          findConcatBoundary(block, packedOps);
      SmallVector<Candidate> candidates;
      auto addCandidate = [&](ArrayRef<Value> seed, bool isBoundary) {
        Candidate c;
        c.seed.assign(seed.begin(), seed.end());
        for (Value v : seed)
          c.ops.push_back(v.getDefiningOp());
        c.boundary = isBoundary;
        candidates.push_back(std::move(c));
      };
      if (boundary)
        addCandidate(boundary->sources, /*isBoundary=*/true);
      SmallVector<SmallVector<Value>> siblingGroups =
          findSiblingGroups(block, packedOps);
      llvm::stable_sort(siblingGroups, [](const SmallVector<Value> &a,
                                          const SmallVector<Value> &b) {
        return a.size() > b.size();
      });
      for (SmallVector<Value> &g : siblingGroups)
        addCandidate(g, /*isBoundary=*/false);

      SmallPtrSet<Operation *, 32> liveOps;
      auto refreshLive = [&]() {
        liveOps.clear();
        for (Operation &op : *block)
          liveOps.insert(&op);
      };
      refreshLive();

      // Bound the fixpoint; a block with many families still gets several of
      // them packed without risking an unbounded rewrite loop.
      unsigned rewrites = 0;
      while (rewrites < 64 &&
             rewriteBlock(block, parent, packedOps, candidates, boundary,
                          liveOps, before)) {
        ++rewrites;
        refreshLive();
        // The concat boundary is consumed by its rewrite; drop it so the cached
        // boundary candidate is not retried.
        if (boundary && packedOps.contains(boundary->concat))
          boundary.reset();
      }
    }
  }
};

} // namespace

} // namespace mlir::triton
