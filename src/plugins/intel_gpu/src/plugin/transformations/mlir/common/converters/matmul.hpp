// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "../convert_common.hpp"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "openvino/op/matmul.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace ov::intel_gpu::mlir {

struct ConvertMatMul {
    Operation* operator()(ConversionContext& context, const NodePtr& node) {
        auto loc = createLocation(context.context, node);
        auto& builder = context.builder();
        const auto inputs = context.getInputs(node);
        const auto elemType = node->get_output_element_type(0);
        const auto outPShape = node->get_output_partial_shape(0);
        auto origOutType = mlir::cast<RankedTensorType>(importTensor(context.context, outPShape, elemType));

        auto makeFill = [&](RankedTensorType type, const ov::PartialShape& pshape) -> Value {
            auto empty = tensor::EmptyOp::create(builder, loc, type, context.get_dynamic_dimension_values(pshape));
            auto zero = getConstant(builder, elemType, 0);
            return linalg::FillOp::create(builder, loc, mlir::ValueRange{zero}, mlir::ValueRange{empty}).getResult(0);
        };

        auto mmNode = std::dynamic_pointer_cast<ov::op::v0::MatMul>(node);
        assert(mmNode);
        bool trA = mmNode->get_transpose_a();
        bool trB = mmNode->get_transpose_b();
        assert(!(trA && trB));

        int64_t targetRank = outPShape.size();
        mlir::SmallVector<Value, 2> ins{inputs[0], inputs[1]};

        // 1. Pure 2D MatMul (static or dynamic)
        if (targetRank <= 2) {
            mlir::SmallVector<Value, 1> outs{makeFill(origOutType, outPShape)};
            if (trA) return linalg::MatmulTransposeAOp::create(builder, loc, ins, outs);
            if (trB) return linalg::MatmulTransposeBOp::create(builder, loc, ins, outs);
            return linalg::MatmulOp::create(builder, loc, ins, outs);
        }

        auto tA = mlir::cast<RankedTensorType>(ins[0].getType());
        auto tB = mlir::cast<RankedTensorType>(ins[1].getType());

        auto allLeadingUnit = [](RankedTensorType t) {
            auto lead = t.getShape().drop_back(2);
            return std::all_of(lead.begin(), lead.end(), [](int64_t d) { return d == 1; });
        };

        auto collapseLeadingTo2D = [&](Value v) -> Value {
            auto shape = mlir::cast<RankedTensorType>(v.getType()).getShape();
            int64_t r = shape.size();
            if (r <= 2) return v;
            SmallVector<ReassociationIndices> reassoc;
            ReassociationIndices leading;
            for (int64_t i = 0; i < r - 1; ++i) leading.push_back(i);
            reassoc.push_back(leading);
            reassoc.push_back(ReassociationIndices{r - 1});
            return tensor::CollapseShapeOp::create(builder, loc, v, reassoc).getResult();
        };

        // 2. Unit leading dims OR ND x 2D (!trA, static shape, any batch B >= 1) -> collapse to 2D MatMul
        bool unitBatch = allLeadingUnit(tA) && allLeadingUnit(tB) && allLeadingUnit(origOutType);
        bool ndBy2d = (!trA && tB.getRank() == 2 && origOutType.hasStaticShape());
        if (unitBatch || ndBy2d) {
            Value outFull = makeFill(origOutType, outPShape);
            ins[0] = collapseLeadingTo2D(ins[0]);
            ins[1] = collapseLeadingTo2D(ins[1]);
            mlir::SmallVector<Value, 1> outs{collapseLeadingTo2D(outFull)};

            Operation* mm = trA   ? linalg::MatmulTransposeAOp::create(builder, loc, ins, outs)
                            : trB ? linalg::MatmulTransposeBOp::create(builder, loc, ins, outs)
                                  : linalg::MatmulOp::create(builder, loc, ins, outs);

            SmallVector<ReassociationIndices> reassoc;
            ReassociationIndices leading;
            for (int64_t i = 0; i < targetRank - 1; ++i) leading.push_back(i);
            reassoc.push_back(leading);
            reassoc.push_back(ReassociationIndices{targetRank - 1});
            return tensor::ExpandShapeOp::create(builder, loc, origOutType, mm->getResult(0), reassoc).getOperation();
        }

        // 3. General 3D+ BatchMatMul (collapse batch axes [0 .. r-3] to 3D, broadcast batch 1 -> B if needed)
        auto to3D = [&](Value v) -> Value {
            auto t = mlir::cast<RankedTensorType>(v.getType());
            int64_t r = t.getRank();
            if (r == 2) {
                SmallVector<ReassociationIndices> reassoc{{0, 1}, {2}};
                auto newT = RankedTensorType::get({1, t.getDimSize(0), t.getDimSize(1)}, t.getElementType());
                return tensor::ExpandShapeOp::create(builder, loc, newT, v, reassoc).getResult();
            }
            if (r == 3) return v;
            SmallVector<ReassociationIndices> reassoc;
            ReassociationIndices bIdx;
            for (int64_t i = 0; i <= r - 3; ++i) bIdx.push_back(i);
            reassoc.push_back(bIdx);
            reassoc.push_back(ReassociationIndices{r - 2});
            reassoc.push_back(ReassociationIndices{r - 1});
            return tensor::CollapseShapeOp::create(builder, loc, v, reassoc).getResult();
        };

        Value out3D = to3D(makeFill(origOutType, outPShape));
        int64_t outBatch = mlir::cast<RankedTensorType>(out3D.getType()).getDimSize(0);

        auto broadcastBatchIfNeeded = [&](Value v3d) -> Value {
            auto t = mlir::cast<RankedTensorType>(v3d.getType());
            if (t.getDimSize(0) == 1 && outBatch > 1) {
                SmallVector<ReassociationIndices> drop0{{0, 1}, {2}};
                Value v2d = tensor::CollapseShapeOp::create(builder, loc, v3d, drop0).getResult();
                auto bType = RankedTensorType::get({outBatch, t.getDimSize(1), t.getDimSize(2)}, t.getElementType());
                auto emp = tensor::EmptyOp::create(builder, loc, bType, ValueRange{});
                return linalg::BroadcastOp::create(builder, loc, v2d, emp, SmallVector<int64_t>{0}).getResult()[0];
            }
            return v3d;
        };

        ins[0] = broadcastBatchIfNeeded(to3D(ins[0]));
        ins[1] = broadcastBatchIfNeeded(to3D(ins[1]));
        mlir::SmallVector<Value, 1> outs{out3D};

        Operation* mm = trA   ? linalg::BatchMatmulTransposeAOp::create(builder, loc, ins, outs)
                        : trB ? linalg::BatchMatmulTransposeBOp::create(builder, loc, ins, outs)
                              : linalg::BatchMatmulOp::create(builder, loc, ins, outs);

        if (targetRank > 3) {
            SmallVector<ReassociationIndices> reassoc;
            ReassociationIndices bIdx;
            for (int64_t i = 0; i <= targetRank - 3; ++i) bIdx.push_back(i);
            reassoc.push_back(bIdx);
            reassoc.push_back(ReassociationIndices{targetRank - 2});
            reassoc.push_back(ReassociationIndices{targetRank - 1});
            mm = tensor::ExpandShapeOp::create(builder, loc, origOutType, mm->getResult(0), reassoc).getOperation();
        }
        return mm;
    }
};

}  // namespace ov::intel_gpu::mlir
