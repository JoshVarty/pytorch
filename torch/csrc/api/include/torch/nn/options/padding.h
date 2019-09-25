#pragma once

#include <torch/expanding_array.h>

namespace torch {
namespace nn {

/// Options for a `D`-dimensional avgpool functional and module.
template <size_t D>
struct ReflectionPadOptions {
  ReflectionPadOptions(ExpandingArray<D> padding)
      : padding_(padding) {}

  /// The padding to add to the input volumes.
  /// For a `D`-dim padding, must be a single number or a list of `D`
  /// numbers.
  /// This parameter __can__ be changed after construction.
  TORCH_ARG(ExpandingArray<D>, padding) = 0;
};

/// `ReflectionPadOptions` specialized for 1-D maxpool.
using ReflectionPad1dOptions = ReflectionPadOptions<1>;

/// `ReflectionPadOptions` specialized for 2-D maxpool.
using ReflectionPad2dOptions = ReflectionPadOptions<2>;

} // namespace nn
} // namespace torch
