// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "../convert_common.hpp"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "ov_ops/rms.hpp"

namespace ov::intel_gpu::mlir {

// y = x / sqrt(mean(x^2, last_dim) + eps) [* gamma]
struct ConvertRMS {
    Operation* operator()(ConversionContext& context, const NodePtr& node) {
        auto rms = ov::as_type_ptr<ov::op::internal::RMS>(node);
        OPENVINO_ASSERT(rms, "Failed to cast to RMS");

        auto loc = createLocation(context.context, node);
        auto& builder = context.builder();
        const auto inputs = context.getInputs(node);
        const auto x = inputs[0];

        const auto el_type = node->get_input_element_type(0);
        const auto calc_el_type = (el_type == ov::element::f16) ? ov::element::f32 : el_type;
        const auto mlir_el_type = importPrecision(context.context, el_type);
        const auto mlir_calc_type = importPrecision(context.context, calc_el_type);
        const auto shape = node->get_output_partial_shape(0);
        const auto rank = shape.rank().get_length();
        const int64_t axis = rank - 1;
        const int64_t num_els = shape[axis].get_length();

        auto out_type = importTensor(context.context, shape, el_type);
        auto calc_type = importTensor(context.context, shape, calc_el_type);
        auto out_dims = context.get_dynamic_dimension_values(shape);

        auto castTensor = [&](Value v, ov::element::Type from_t, ov::element::Type to_t,
                              RankedTensorType to_mlir_tensor, ArrayRef<Value> dyn_dims) -> Value {
            if (from_t == to_t) return v;
            auto to_mlir_el = importPrecision(context.context, to_t);
            auto r = to_mlir_tensor.getRank();
            auto id = builder.getMultiDimIdentityMap(r);
            SmallVector<utils::IteratorType> par(r, utils::IteratorType::parallel);
            auto emp = tensor::EmptyOp::create(builder, loc, to_mlir_tensor, dyn_dims);
            return linalg::GenericOp::create(
                builder, loc, TypeRange{to_mlir_tensor}, ValueRange{v}, ValueRange{emp},
                SmallVector<AffineMap>{id, id}, par,
                [&](::mlir::OpBuilder& b, ::mlir::Location l, ValueRange args) {
                    Value res = (to_t == ov::element::f32)
                                    ? Value{arith::ExtFOp::create(b, l, to_mlir_el, args[0])}
                                    : Value{arith::TruncFOp::create(b, l, to_mlir_el, args[0])};
                    linalg::YieldOp::create(b, l, ValueRange{res});
                }).getResult(0);
        };

        Value x_calc = castTensor(x, el_type, calc_el_type, mlir::cast<RankedTensorType>(calc_type), out_dims);

        PartialShape reduced_shape(std::vector<Dimension>(shape.begin(), shape.begin() + axis));
        auto reduced_type = importTensor(context.context, reduced_shape, calc_el_type);
        auto reduced_dims = context.get_dynamic_dimension_values(reduced_shape);

        // x^2
        auto two = arith::ConstantOp::create(builder, loc, ::mlir::DenseElementsAttr::get(calc_type, builder.getFloatAttr(mlir_calc_type, 2.0)));
        auto sq_empty = tensor::EmptyOp::create(builder, loc, calc_type, out_dims);
        Value squared = linalg::ElementwiseOp::create(builder, loc, ValueRange{x_calc, two}, ValueRange{sq_empty}, linalg::ElementwiseKind::powf).getResult(0);

        // sum(x^2, axis)
        auto sum_empty = tensor::EmptyOp::create(builder, loc, reduced_type, reduced_dims);
        auto zero = getConstant(builder, mlir_calc_type, 0, loc);
        auto sum_init = linalg::FillOp::create(builder, loc, ValueRange{zero}, ValueRange{sum_empty});
        Value sum = linalg::ReduceOp::create(builder,
                                             loc,
                                             ValueRange{squared},
                                             ValueRange{sum_init.getResult(0)},
                                             SmallVector<int64_t>{axis},
                                             [&](::mlir::OpBuilder& b, ::mlir::Location l, ValueRange args) {
                                                 linalg::YieldOp::create(b, l, Value{arith::AddFOp::create(b, l, args[0], args[1])});
                                             })
                        .getResult(0);

        // sum / N
        auto n_const =
            arith::ConstantOp::create(builder,
                                      loc,
                                      ::mlir::DenseElementsAttr::get(reduced_type, builder.getFloatAttr(mlir_calc_type, static_cast<double>(num_els))));
        auto div_empty = tensor::EmptyOp::create(builder, loc, reduced_type, reduced_dims);
        Value mean = linalg::ElementwiseOp::create(builder, loc, ValueRange{sum, n_const}, ValueRange{div_empty}, linalg::ElementwiseKind::div).getResult(0);

        // 1 / sqrt(mean + eps)
        PartialShape scale_shape = reduced_shape;
        scale_shape.push_back(Dimension(1));
        auto scale_type = importTensor(context.context, scale_shape, calc_el_type);
        auto scale_dims = context.get_dynamic_dimension_values(scale_shape);
        auto scale_const = [&](double v) {
            return Value{arith::ConstantOp::create(builder, loc, ::mlir::DenseElementsAttr::get(scale_type, builder.getFloatAttr(mlir_calc_type, v)))};
        };
        auto scale_empty = [&] {
            return tensor::EmptyOp::create(builder, loc, scale_type, scale_dims);
        };
        Value inv_rms = linalg::BroadcastOp::create(builder, loc, mean, scale_empty(), SmallVector<int64_t>{axis}).getResult()[0];
        inv_rms = linalg::ElementwiseOp::create(builder,
                                                loc,
                                                ValueRange{inv_rms, scale_const(rms->get_epsilon())},
                                                ValueRange{scale_empty()},
                                                linalg::ElementwiseKind::add)
                      .getResult(0);
        inv_rms = linalg::ElementwiseOp::create(builder, loc, ValueRange{inv_rms}, ValueRange{scale_empty()}, linalg::ElementwiseKind::sqrt).getResult(0);
        inv_rms = linalg::ElementwiseOp::create(builder, loc, ValueRange{scale_const(1.0), inv_rms}, ValueRange{scale_empty()}, linalg::ElementwiseKind::div)
                      .getResult(0);

        // broadcast back over the reduced axis and scale x
        SmallVector<::mlir::ReassociationIndices> collapse(axis);
        for (int64_t i = 0; i < axis; ++i) {
            collapse[i].push_back(i);
        }
        collapse.back().push_back(axis);
        Value squeezed = tensor::CollapseShapeOp::create(builder, loc, inv_rms, collapse);
        auto bcast_empty = tensor::EmptyOp::create(builder, loc, calc_type, out_dims);
        Value bcast = linalg::BroadcastOp::create(builder, loc, squeezed, bcast_empty, SmallVector<int64_t>{axis}).getResult()[0];
        auto mul_empty = tensor::EmptyOp::create(builder, loc, calc_type, out_dims);
        Value res_calc = linalg::ElementwiseOp::create(builder, loc, ValueRange{bcast, x_calc}, ValueRange{mul_empty}, linalg::ElementwiseKind::mul).getResult(0);

        if (rms->get_elementwise_affine()) {
            Value gamma = inputs[1];
            auto [collapse_groups, dimensions] = broadcast_dimensions(node->get_input_partial_shape(1), shape);
            if (!dimensions.empty()) {
                auto squeezed_g = tensor::CollapseShapeOp::create(builder, loc, gamma, collapse_groups);
                auto empty = tensor::EmptyOp::create(builder, loc, out_type, out_dims);
                gamma = linalg::BroadcastOp::create(builder, loc, squeezed_g, empty, dimensions).getResult()[0];
            }
            Value gamma_calc = castTensor(gamma, el_type, calc_el_type, mlir::cast<RankedTensorType>(calc_type), out_dims);
            auto empty = tensor::EmptyOp::create(builder, loc, calc_type, out_dims);
            res_calc = linalg::ElementwiseOp::create(builder, loc, ValueRange{res_calc, gamma_calc}, ValueRange{empty}, linalg::ElementwiseKind::mul).getResult(0);
        }
        Value res_final = castTensor(res_calc, calc_el_type, el_type, mlir::cast<RankedTensorType>(out_type), out_dims);
        return res_final.getDefiningOp();
    }
};

}  // namespace ov::intel_gpu::mlir
