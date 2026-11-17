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
        const auto ov_output_element_type = node->get_output_element_type(0);
        const auto ov_output_shape = node->get_output_partial_shape(0);
        auto outType = importTensor(context.context, ov_output_shape, ov_output_element_type);
        auto dynamic_dimensions = context.get_dynamic_dimension_values(ov_output_shape);
        auto empty = tensor::EmptyOp::create(builder, loc, outType, dynamic_dimensions);
        auto zero = getConstant(builder, ov_output_element_type, 0);
        auto fill = linalg::FillOp::create(builder, loc, mlir::ValueRange{zero}, mlir::ValueRange{empty});

        mlir::SmallVector<Value, 2> ins{inputs[0], inputs[1]};
        mlir::SmallVector<Value, 1> outs{fill.getResult(0)};

        auto matmul_node = std::dynamic_pointer_cast<ov::op::v0::MatMul>(node);
        assert(matmul_node);
        bool isTransposedA = matmul_node->get_transpose_a();
        bool isTransposedB = matmul_node->get_transpose_b();
        assert(!(isTransposedA && isTransposedB));

        int64_t targetRank = ov_output_shape.size();

        // 1. If rank is 2, run simple 2D matmul
        if (targetRank <= 2) {
            Operation* matmul = nullptr;
            if (isTransposedA) {
                matmul = linalg::MatmulTransposeAOp::create(builder, loc, ins, outs);
            } else if (isTransposedB) {
                matmul = linalg::MatmulTransposeBOp::create(builder, loc, ins, outs);
            } else {
                matmul = linalg::MatmulOp::create(builder, loc, ins, outs);
            }
            return matmul;
        }

        // 2. Check if all leading batch dimensions are 1 (can be collapsed to pure 2D)
        auto canCollapseTo2D = [](mlir::Value tensor) {
            auto shape = mlir::cast<RankedTensorType>(tensor.getType()).getShape().drop_back(2);
            return std::all_of(shape.begin(), shape.end(), [](int64_t d) {
                return d == 1;
            });
        };

        if (canCollapseTo2D(ins[0]) && canCollapseTo2D(ins[1]) && canCollapseTo2D(outs[0])) {
            auto collapseTo2D = [&](Value tensor) -> Value {
                auto shape = mlir::cast<RankedTensorType>(tensor.getType()).getShape();
                int64_t rank = shape.size();
                if (rank <= 2) return tensor;
                SmallVector<ReassociationIndices> reassoc;
                ReassociationIndices leading;
                for (int64_t i = 0; i < rank - 1; ++i) leading.push_back(i);
                reassoc.push_back(leading);
                reassoc.push_back(ReassociationIndices{rank - 1});
                return tensor::CollapseShapeOp::create(builder, loc, tensor, reassoc).getResult();
            };

            auto expandFrom2D = [&](Value tensor) -> Value {
                auto type = mlir::cast<RankedTensorType>(tensor.getType());
                auto shape = type.getShape();
                if (static_cast<int64_t>(shape.size()) == targetRank) return tensor;
                SmallVector<ReassociationIndices> reassoc;
                ReassociationIndices leading;
                for (int64_t i = 0; i < targetRank - 1; ++i) leading.push_back(i);
                reassoc.push_back(leading);
                reassoc.push_back(ReassociationIndices{targetRank - 1});
                SmallVector<int64_t> new_shape(targetRank, 1);
                std::copy(shape.begin(), shape.end(), new_shape.end() - shape.size());
                auto new_type = mlir::RankedTensorType::get(new_shape, type.getElementType());
                return tensor::ExpandShapeOp::create(builder, loc, new_type, tensor, reassoc).getResult();
            };

            ins[0] = collapseTo2D(ins[0]);
            ins[1] = collapseTo2D(ins[1]);
            outs[0] = collapseTo2D(outs[0]);

            Operation* matmul = nullptr;
            if (isTransposedA) {
                matmul = linalg::MatmulTransposeAOp::create(builder, loc, ins, outs);
            } else if (isTransposedB) {
                matmul = linalg::MatmulTransposeBOp::create(builder, loc, ins, outs);
            } else {
                matmul = linalg::MatmulOp::create(builder, loc, ins, outs);
            }
            return expandFrom2D(matmul->getResult(0)).getDefiningOp();
        }

        // 3. Batch dimensions present (targetRank >= 3).
        // If targetRank > 3, collapse all leading batch dims [0 .. rank-3] into a single 3D batch dimension.
        auto collapseBatchTo3D = [&](Value tensor) -> Value {
            auto shape = mlir::cast<RankedTensorType>(tensor.getType()).getShape();
            int64_t r = shape.size();
            if (r <= 3) return tensor;
            SmallVector<ReassociationIndices> reassoc;
            ReassociationIndices batchIndices;
            for (int64_t i = 0; i <= r - 3; ++i) batchIndices.push_back(i);
            reassoc.push_back(batchIndices);
            reassoc.push_back(ReassociationIndices{r - 2});
            reassoc.push_back(ReassociationIndices{r - 1});
            return tensor::CollapseShapeOp::create(builder, loc, tensor, reassoc).getResult();
        };

        auto expandBatchFrom3D = [&](Value tensor, RankedTensorType originalType) -> Value {
            auto shape = originalType.getShape();
            int64_t r = shape.size();
            if (r <= 3) return tensor;
            SmallVector<ReassociationIndices> reassoc;
            ReassociationIndices batchIndices;
            for (int64_t i = 0; i <= r - 3; ++i) batchIndices.push_back(i);
            reassoc.push_back(batchIndices);
            reassoc.push_back(ReassociationIndices{r - 2});
            reassoc.push_back(ReassociationIndices{r - 1});
            return tensor::ExpandShapeOp::create(builder, loc, originalType, tensor, reassoc).getResult();
        };

        auto origOutType = mlir::cast<RankedTensorType>(outs[0].getType());
        ins[0] = collapseBatchTo3D(ins[0]);
        ins[1] = collapseBatchTo3D(ins[1]);
        outs[0] = collapseBatchTo3D(outs[0]);

        Operation* matmul = nullptr;
        if (isTransposedA) {
            matmul = linalg::BatchMatmulTransposeAOp::create(builder, loc, ins, outs);
        } else if (isTransposedB) {
            matmul = linalg::BatchMatmulTransposeBOp::create(builder, loc, ins, outs);
        } else {
            matmul = linalg::BatchMatmulOp::create(builder, loc, ins, outs);
        }

        if (targetRank > 3) {
            matmul = expandBatchFrom3D(matmul->getResult(0), origOutType).getDefiningOp();
        }
        return matmul;
    }
};

}  // namespace ov::intel_gpu::mlir
