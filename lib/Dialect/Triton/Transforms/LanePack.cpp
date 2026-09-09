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

struct ProbeLogger {
  bool emit = false;

  template <typename... Args>
  void reject(Args &&...args) const {
    if (!emit)
      return;
    llvm::errs() << "[lane-pack] reject: ";
    (llvm::errs() << ... << std::forward<Args>(args));
    llvm::errs() << "\n";
  }
};

static bool matchReducePlusEpsilon(Value value, Value &epsilon,
                                   triton::ReduceOp &reduceOp,
                                   ProbeLogger log = {}) {
  auto add = value.getDefiningOp<arith::AddFOp>();
  if (!add) {
    log.reject("denominator is not addf");
    return false;
  }

  auto lhsReduce = add.getLhs().getDefiningOp<triton::ReduceOp>();
  auto rhsReduce = add.getRhs().getDefiningOp<triton::ReduceOp>();
  Value candidateEps;
  if (lhsReduce && !rhsReduce) {
    reduceOp = lhsReduce;
    candidateEps = add.getRhs();
  } else if (rhsReduce && !lhsReduce) {
    reduceOp = rhsReduce;
    candidateEps = add.getLhs();
  } else {
    log.reject("denominator is not reduce + epsilon");
    return false;
  }

  if (!epsilon)
    epsilon = candidateEps;
  return true;
}

static bool matchBroadcastedDenominator(Value denominator, Value src,
                                        Value &epsilon,
                                        triton::ReduceOp &reduceOp,
                                        ProbeLogger log = {}) {
  auto splat = denominator.getDefiningOp<triton::SplatOp>();
  if (!splat) {
    log.reject("row denominator is not triton.splat");
    return false;
  }

  if (!matchReducePlusEpsilon(splat.getSrc(), epsilon, reduceOp, log))
    return false;

  if (reduceOp.getNumOperands() != 1 || reduceOp.getOperand(0) != src ||
      !hasSingleAddCombiner(reduceOp)) {
    log.reject("row reduce op shape does not match source");
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

enum class NormStepKind {
  Row,
  Col,
};

struct NormStepMatch {
  NormStepKind kind;
  SmallVector<Value> inputs;
  SmallVector<Operation *> supportOps;
  SmallVector<arith::DivFOp> divs;
  SmallVector<Value> outputs;
  Value epsilon;
  int64_t reduceAxis = -1;
};

static void addSupportOpIfPresent(Value value,
                                  SmallVectorImpl<Operation *> &supportOps) {
  if (Operation *def = value.getDefiningOp())
    supportOps.push_back(def);
}

struct LanePackMatch {
  SmallVector<Value> initLanes;
  SmallVector<BlockArgument> laneArgs;
  SmallVector<NormStepMatch, 4> steps;
  scf::YieldOp yieldOp;
};

static bool matchRowNormStep(ArrayRef<Value> inputs, ArrayRef<Value> outputs,
                             NormStepMatch &step, ProbeLogger log = {}) {
  if (inputs.size() != outputs.size() || inputs.empty())
    return false;

  step.kind = NormStepKind::Row;
  step.inputs.assign(inputs.begin(), inputs.end());
  step.outputs.assign(outputs.begin(), outputs.end());
  step.supportOps.clear();
  step.divs.clear();
  step.epsilon = Value();
  step.reduceAxis = -1;

  for (auto [input, output] : llvm::zip(inputs, outputs)) {
    auto divOp = output.getDefiningOp<arith::DivFOp>();
    if (!divOp || divOp.getLhs() != input) {
      log.reject("row step output is not input/div");
      return false;
    }

    triton::ReduceOp reduceOp;
    if (!matchBroadcastedDenominator(divOp.getRhs(), input, step.epsilon,
                                     reduceOp, log)) {
      log.reject("row step denominator mismatch");
      return false;
    }

    if (step.reduceAxis < 0)
      step.reduceAxis = reduceOp.getAxis();
    else if (step.reduceAxis != reduceOp.getAxis()) {
      log.reject("row step axis mismatch");
      return false;
    }

    step.divs.push_back(divOp);
    addSupportOpIfPresent(divOp.getRhs(), step.supportOps);
    if (auto splat = divOp.getRhs().getDefiningOp<triton::SplatOp>()) {
      addSupportOpIfPresent(splat.getSrc(), step.supportOps);
      if (auto add = splat.getSrc().getDefiningOp<arith::AddFOp>()) {
        addSupportOpIfPresent(add.getLhs(), step.supportOps);
        addSupportOpIfPresent(add.getRhs(), step.supportOps);
      }
    }
  }

  return true;
}

static bool matchColNormStep(ArrayRef<Value> inputs, ArrayRef<Value> outputs,
                             NormStepMatch &step, ProbeLogger log = {}) {
  if (inputs.size() != outputs.size() || inputs.empty())
    return false;

  step.kind = NormStepKind::Col;
  step.inputs.assign(inputs.begin(), inputs.end());
  step.outputs.assign(outputs.begin(), outputs.end());
  step.supportOps.clear();
  step.divs.clear();
  step.epsilon = Value();
  step.reduceAxis = 0;

  Value sharedDenom;
  for (auto [input, output] : llvm::zip(inputs, outputs)) {
    auto divOp = output.getDefiningOp<arith::DivFOp>();
    if (!divOp || divOp.getLhs() != input) {
      log.reject("col step output is not input/div");
      return false;
    }
    step.divs.push_back(divOp);
    if (!sharedDenom)
      sharedDenom = divOp.getRhs();
    else if (sharedDenom != divOp.getRhs()) {
      log.reject("col step denominators differ");
      return false;
    }
  }

  SmallVector<Value> addLeaves;
  if (!collectAddTreeLeaves(sharedDenom, addLeaves)) {
    log.reject("failed to collect col add tree");
    return false;
  }

  llvm::SmallPtrSet<void *, 8> inputSet;
  for (Value input : inputs)
    inputSet.insert(input.getAsOpaquePointer());

  unsigned matchedInputs = 0;
  for (Value leaf : addLeaves) {
    if (inputSet.contains(leaf.getAsOpaquePointer()))
      ++matchedInputs;
  }
  if (matchedInputs != inputs.size()) {
    log.reject("col denominator does not use all inputs");
    return false;
  }

  SmallVector<Value> nonInputLeaves;
  for (Value leaf : addLeaves) {
    if (!inputSet.contains(leaf.getAsOpaquePointer()))
      nonInputLeaves.push_back(leaf);
  }
  if (nonInputLeaves.size() != 1) {
    log.reject("col denominator must have one epsilon leaf");
    return false;
  }
  step.epsilon = nonInputLeaves.front();
  addSupportOpIfPresent(sharedDenom, step.supportOps);
  for (Value leaf : addLeaves)
    addSupportOpIfPresent(leaf, step.supportOps);
  return true;
}

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
    llvm::errs()
        << "[lane-pack] reject: yielded values not same lane tensor group\n";
    return failure();
  }

  match.laneArgs.assign(forOp.getRegionIterArgs().begin(),
                        forOp.getRegionIterArgs().end());

  SmallVector<Value> current(match.laneArgs.begin(), match.laneArgs.end());
  SmallPtrSet<Operation *, 32> matchedOps;
  while (current != yielded) {
    SmallVector<Value> candidateOutputs;
    candidateOutputs.reserve(current.size());
    for (Value input : current) {
      Operation *user = nullptr;
      if (!hasOneUseOfType<arith::DivFOp>(input, user)) {
        llvm::errs() << "[lane-pack] reject: expected unique div user for step input\n";
        return failure();
      }
      auto divOp = cast<arith::DivFOp>(user);
      if (divOp.getLhs() != input) {
        llvm::errs() << "[lane-pack] reject: div user does not consume input as lhs\n";
        return failure();
      }
      candidateOutputs.push_back(divOp.getResult());
    }

    NormStepMatch step;
    if (!matchRowNormStep(current, candidateOutputs, step) &&
        !matchColNormStep(current, candidateOutputs, step)) {
      ProbeLogger log{true};
      NormStepMatch debugStep;
      (void)matchRowNormStep(current, candidateOutputs, debugStep, log);
      (void)matchColNormStep(current, candidateOutputs, debugStep, log);
      llvm::errs() << "[lane-pack] reject: failed to extend step chain\n";
      return failure();
    }

    for (Operation *op : step.supportOps)
      if (op)
        matchedOps.insert(op);
    for (arith::DivFOp div : step.divs)
      matchedOps.insert(div);
    match.steps.push_back(step);

    if (!isSameLaneTensorGroup(step.outputs)) {
      llvm::errs() << "[lane-pack] reject: step outputs are not lane group\n";
      return failure();
    }
    current.assign(step.outputs.begin(), step.outputs.end());
  }

  if (match.steps.empty()) {
    llvm::errs() << "[lane-pack] reject: no normalization steps found\n";
    return failure();
  }

  for (Operation &op : forOp.getBody()->without_terminator()) {
    if (matchedOps.contains(&op))
      continue;
    llvm::errs() << "[lane-pack] reject: unsupported or unmatched op in loop body: ";
    op.print(llvm::errs());
    llvm::errs() << "\n";
    return failure();
  }

  llvm::errs() << "[lane-pack] match success: lanes=" << match.initLanes.size()
               << ", steps=" << match.steps.size() << "\n";
  return match;
}

static Value buildShapeConst(OpBuilder &builder, Location loc,
                             ArrayRef<int64_t> shape) {
  return builder.create<arith::ConstantOp>(loc, builder.getI64TensorAttr(shape));
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

static Value buildPackedRowNorm(OpBuilder &builder, Location loc, Value input,
                                Value epsilon, int64_t originalReduceAxis) {
  auto inputTy = cast<RankedTensorType>(input.getType());
  const int64_t packedReduceAxis = originalReduceAxis + 1;
  Value sums = buildSumReduce(builder, loc, input, packedReduceAxis);
  auto sumsTy = cast<RankedTensorType>(sums.getType());
  Value eps = broadcastEpsilonTo(builder, loc, epsilon, sumsTy);
  Value denom = builder.create<arith::AddFOp>(loc, sums, eps);

  SmallVector<int64_t> expandedShape(sumsTy.getShape().begin(),
                                     sumsTy.getShape().end());
  expandedShape.insert(expandedShape.begin() + packedReduceAxis, 1);
  auto expandedTy = RankedTensorType::get(expandedShape,
                                          inputTy.getElementType(),
                                          inputTy.getEncoding());
  Value expanded =
      builder.create<triton::ExpandDimsOp>(loc, expandedTy, denom, packedReduceAxis);
  Value broadcast = builder.create<triton::BroadcastOp>(loc, inputTy, expanded);
  return builder.create<arith::DivFOp>(loc, input, broadcast);
}

static Value buildPackedColNorm(OpBuilder &builder, Location loc, Value input,
                                Value epsilon) {
  auto inputTy = cast<RankedTensorType>(input.getType());
  constexpr int64_t packedLaneAxis = 0;
  Value sums = buildSumReduce(builder, loc, input, packedLaneAxis);
  auto sumsTy = cast<RankedTensorType>(sums.getType());
  Value eps = broadcastEpsilonTo(builder, loc, epsilon, sumsTy);
  Value denom = builder.create<arith::AddFOp>(loc, sums, eps);

  SmallVector<int64_t> expandedShape(sumsTy.getShape().begin(),
                                     sumsTy.getShape().end());
  expandedShape.insert(expandedShape.begin() + packedLaneAxis, 1);
  auto expandedTy = RankedTensorType::get(expandedShape,
                                          inputTy.getElementType(),
                                          inputTy.getEncoding());
  Value expanded =
      builder.create<triton::ExpandDimsOp>(loc, expandedTy, denom, packedLaneAxis);
  Value broadcast = builder.create<triton::BroadcastOp>(loc, inputTy, expanded);
  return builder.create<arith::DivFOp>(loc, input, broadcast);
}

static LogicalResult rewriteLanePackLoop(scf::ForOp forOp,
                                         const LanePackMatch &match) {
  Location loc = forOp.getLoc();
  OpBuilder builder(forOp);
  llvm::errs() << "[lane-pack] rewrite begin: lanes="
               << match.initLanes.size() << ", steps=" << match.steps.size()
               << "\n";

  Value packedInit = buildPackedLanes(builder, loc, match.initLanes);
  auto newFor = builder.create<scf::ForOp>(
      loc, forOp.getLowerBound(), forOp.getUpperBound(), forOp.getStep(),
      ValueRange{packedInit});

  Block *newBody = newFor.getBody();
  builder.setInsertionPointToStart(newBody);
  Value packedCurrent = newBody->getArgument(newBody->getNumArguments() - 1);

  for (const NormStepMatch &step : match.steps) {
    if (step.kind == NormStepKind::Row) {
      packedCurrent =
          buildPackedRowNorm(builder, loc, packedCurrent, step.epsilon,
                             step.reduceAxis);
    } else {
      packedCurrent = buildPackedColNorm(builder, loc, packedCurrent,
                                         step.epsilon);
    }
  }
  builder.create<scf::YieldOp>(loc, packedCurrent);

  builder.setInsertionPointAfter(newFor);
  Value packedResult = newFor.getResult(0);
  SmallVector<Value> unpacked = unpackPackedLanes(builder, loc, packedResult,
                                                  match.initLanes.size());
  if (unpacked.size() != match.initLanes.size()) {
    llvm::errs() << "[lane-pack] rewrite reject: unpack produced "
                 << unpacked.size() << " lanes; expected "
                 << match.initLanes.size() << "\n";
    return failure();
  }

  for (auto [oldResult, newResult] : llvm::zip(forOp.getResults(), unpacked))
    oldResult.replaceAllUsesWith(newResult);
  forOp.erase();
  llvm::errs() << "[lane-pack] rewrite complete\n";
  return success();
}

struct LanePackPass : public impl::TritonLanePackBase<LanePackPass> {
  using TritonLanePackBase::TritonLanePackBase;

  void runOnOperation() override {
    getOperation().walk([&](scf::ForOp forOp) {
      FailureOr<LanePackMatch> match = matchLanePackLoop(forOp);
      if (failed(match))
        return;
      if (failed(rewriteLanePackLoop(forOp, match.value()))) {
        llvm::errs() << "[lane-pack] rewrite failed\n";
        signalPassFailure();
        return;
      }
      llvm::errs() << "[lane-pack] rewrite applied\n";
    });
  }
};

} // namespace

} // namespace mlir::triton
