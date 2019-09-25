#pragma once

#include <torch/nn/options/padding.h>
#include <torch/expanding_array.h>


namespace torch {
namespace nn {
namespace functional {

inline Tensor reflection_pad1d(const Tensor& input, const ReflectionPad1dOptions& options) {

    //TODO: There must be a cleaner way to do this?
    auto paddingArray = options.padding();
    auto paddingValue = paddingArray->front();
    auto paddingAsInt = (int)paddingValue;

    return torch::reflection_pad1d(
        input,
        {paddingAsInt, paddingAsInt});
}

inline Tensor reflection_pad2d(const Tensor& input, const ReflectionPad2dOptions& options) {
    return torch::reflection_pad2d(
        input,
        options.padding());
}

} // namespace functional
} // namespace nn
} // namespace torch
