#pragma once
#include "laic/gguf.hpp"
#include <cstddef>
#include <cstdint>
namespace laic::quant_v2 { float value(GgmlType type,const uint8_t*data,size_t index); }
