/**
 * Copyright (c) 2020-2021, Hanseul Jun.
 */

#pragma once

#include "depth_decoder.hpp"
#include "rvl.hpp"

namespace rgbd
{
class TDC1Decoder : public DepthDecoderImpl
{
public:
    TDC1Decoder() noexcept;
    Int32Frame decode(gsl::span<const std::byte> bytes) noexcept;

private:
    // Using int32_t to be compatible with the differences that can have
    // negative values.
    vector<int32_t> previous_depth_values_;
};
} // namespace tg
