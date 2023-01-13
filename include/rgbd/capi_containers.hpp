#pragma once

#include "constants.hpp"

namespace rgbd
{
// NativeByteArray is for converting Bytes (i.e., vector<byte>)
// into something Swift can hold.
// Mainly, it is for moving a tg::Bytes in stack memory to heap memory by
// e.g., auto native_byte_array = new NativeByteArray(std::move(bytes));
class NativeByteArray
{
public:
    NativeByteArray(const Bytes& byte_values);
    NativeByteArray(Bytes&& byte_values);
    // Converting std::byte* into uint8_t* since
    // Swift does not know about std::byte.
    uint8_t* data()
    {
        return reinterpret_cast<uint8_t*>(byte_values_.data());
    }
    size_t size()
    {
        return byte_values_.size();
    }

private:
    Bytes byte_values_;
};

class NativeFloatArray
{
public:
    NativeFloatArray(vector<float>&& float_values);
    float* data()
    {
        return float_values_.data();
    }
    size_t size()
    {
        return float_values_.size();
    }

private:
    vector<float> float_values_;
};

class NativeInt32Array
{
public:
    NativeInt32Array(const vector<int32_t>& int32_values);
    int32_t* data()
    {
        return int32_values_.data();
    }
    size_t size()
    {
        return int32_values_.size();
    }

private:
    vector<int32_t> int32_values_;
};

class NativeUInt8Array
{
public:
    NativeUInt8Array(const vector<uint8_t>& uint8_values);
    uint8_t* data()
    {
        return uint8_values_.data();
    }
    size_t size()
    {
        return uint8_values_.size();
    }

private:
    vector<uint8_t> uint8_values_;
};

class NativeString
{
public:
    NativeString(const string& value);
    const char* c_str()
    {
        return value_.c_str();
    }

private:
    string value_;
};
}
