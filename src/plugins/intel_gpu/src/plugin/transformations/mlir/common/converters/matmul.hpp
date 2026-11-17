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

        auto mmNode = std::dynamic_pointer_cast<ov::op::v0::MatMul>(node);
        assert(mmNode);
        bool trA = mmNode->get_transpose_a(), trB = mmNode->get_transpose_b();
        assert(!(trA && trB));

        int64_t targetRank = outPShape.size();
        mlir::SmallVector<Value, 2> ins{inputs[0], inputs[1]};

        auto makeFill = [&](RankedTensorType type, const ov::PartialShape& pshape) -> Value {
            auto empty = tensor::EmptyOp::create(builder, loc, type, context.get_dynamic_dimension_values(pshape));
            auto zero = getConstant(builder, elemType, 0);
            return linalg::FillOp::create(builder, loc, mlir::ValueRange{zero}, mlir::ValueRange{empty}).getResult(0);
        };

        if (targetRank <= 2) {
            mlir::SmallVector<Value, 1> outs{makeFill(origOutType, outPShape)};
            if (trA) return linalg::MatmulTransposeAOp::create(builder, loc, ins, outs);
            if (trB) return linalg::MatmulTransposeBOp::create(builder, loc, ins, outs);
            return linalg::MatmulOp::create(builder, loc, ins, outs);
        }

        auto to3D = [&](Value tensor) -> Value {
            auto t = mlir::cast<RankedTensorType>(tensor.getType());
            int64_t r = t.getRank();
            if (r == 2) {
                SmallVector<ReassociationIndices> reassoc{{0, 1}, {2}};
                auto newT = RankedTensorType::get({1, t.getDimSize(0), t.getDimSize(1)}, t.getElementType());
                return tensor::ExpandShapeOp::create(builder, loc, newT, tensor, reassoc).getResult();
            }
            if (r <= 3) return tensor;
            SmallVector<ReassociationIndices> reassoc;
            ReassociationIndices bIdx;
            for (int64_t i = 0; i <= r - 3; ++i) bIdx.push_back(i);
            reassoc.push_back(bIdx);
            reassoc.push_back(ReassociationIndices{r - 2});
            reassoc.push_back(ReassociationIndices{r - 1});
            return tensor::CollapseShapeOp::create(builder, loc, tensor, reassoc).getResult();
        };

        ins[0] = to3D(ins[0]);
        ins[1] = to3D(ins[1]);
        mlir::SmallVector<Value, 1> outs{to3D(makeFill(origOutType, outPShape))};

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
