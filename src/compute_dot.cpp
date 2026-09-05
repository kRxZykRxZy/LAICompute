#include "laic/compute_dot.hpp"
#include <cmath>
#include <cstring>
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

namespace laic::compute {
static inline float half(uint16_t h) noexcept {
    const uint32_t sign = uint32_t(h & 0x8000u) << 16;
    const uint32_t exp = (h >> 10) & 0x1fu;
    const uint32_t mant = h & 0x03ffu;
    if (exp == 0) {
        if (!mant) {
            uint32_t bits = sign; float out; std::memcpy(&out, &bits, 4); return out;
        }
        return std::ldexp((h & 0x8000u) ? -float(mant) : float(mant), -24);
    }
    if (exp == 31) {
        uint32_t bits = sign | 0x7f800000u | (mant << 13);
        float out; std::memcpy(&out, &bits, 4); return out;
    }
    const uint32_t bits = sign | ((exp + 112u) << 23) | (mant << 13);
    float out; std::memcpy(&out, &bits, 4); return out;
}

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
static inline float dot_half_neon(const uint16_t *w, const float *x, size_t n) noexcept {
    float32x4_t a0=vdupq_n_f32(0.f),a1=vdupq_n_f32(0.f),a2=vdupq_n_f32(0.f),a3=vdupq_n_f32(0.f);
    size_t i=0;
    for(;i+16<=n;i+=16){
        float32x4_t w0={half(w[i]),half(w[i+1]),half(w[i+2]),half(w[i+3])};
        float32x4_t w1={half(w[i+4]),half(w[i+5]),half(w[i+6]),half(w[i+7])};
        float32x4_t w2={half(w[i+8]),half(w[i+9]),half(w[i+10]),half(w[i+11])};
        float32x4_t w3={half(w[i+12]),half(w[i+13]),half(w[i+14]),half(w[i+15])};
        a0=vmlaq_f32(a0,vld1q_f32(x+i),w0);a1=vmlaq_f32(a1,vld1q_f32(x+i+4),w1);
        a2=vmlaq_f32(a2,vld1q_f32(x+i+8),w2);a3=vmlaq_f32(a3,vld1q_f32(x+i+12),w3);
    }
    float32x4_t a=vaddq_f32(vaddq_f32(a0,a1),vaddq_f32(a2,a3));
    float32x2_t p=vadd_f32(vget_low_f32(a),vget_high_f32(a));p=vpadd_f32(p,p);
    float out=vget_lane_f32(p,0);for(;i<n;++i)out+=half(w[i])*x[i];return out;
}
#endif

float dot_f16(const uint8_t *weights,const float *x,size_t n) noexcept {
    const uint16_t*w=reinterpret_cast<const uint16_t*>(weights);
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    return dot_half_neon(w,x,n);
#else
    float out=0.f;for(size_t i=0;i<n;++i)out+=half(w[i])*x[i];return out;
#endif
}

float dot_f32(const uint8_t *weights,const float *x,size_t n) noexcept {
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    const float*w=reinterpret_cast<const float*>(weights);float32x4_t a0=vdupq_n_f32(0.f),a1=vdupq_n_f32(0.f);size_t i=0;
    for(;i+8<=n;i+=8){a0=vmlaq_f32(a0,vld1q_f32(x+i),vld1q_f32(w+i));a1=vmlaq_f32(a1,vld1q_f32(x+i+4),vld1q_f32(w+i+4));}
    float32x4_t a=vaddq_f32(a0,a1);float32x2_t p=vadd_f32(vget_low_f32(a),vget_high_f32(a));p=vpadd_f32(p,p);float out=vget_lane_f32(p,0);for(;i<n;++i)out+=x[i]*w[i];return out;
#else
    const float*w=reinterpret_cast<const float*>(weights);float out=0.f;for(size_t i=0;i<n;++i)out+=x[i]*w[i];return out;
#endif
}

void axpy_f32(float*dst,const float*src,float scale,size_t n) noexcept {
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    const float32x4_t s=vdupq_n_f32(scale);size_t i=0;
    for(;i+16<=n;i+=16){vst1q_f32(dst+i,vmlaq_f32(vld1q_f32(dst+i),vld1q_f32(src+i),s));vst1q_f32(dst+i+4,vmlaq_f32(vld1q_f32(dst+i+4),vld1q_f32(src+i+4),s));vst1q_f32(dst+i+8,vmlaq_f32(vld1q_f32(dst+i+8),vld1q_f32(src+i+8),s));vst1q_f32(dst+i+12,vmlaq_f32(vld1q_f32(dst+i+12),vld1q_f32(src+i+12),s));}
    for(;i+4<=n;i+=4)vst1q_f32(dst+i,vmlaq_f32(vld1q_f32(dst+i),vld1q_f32(src+i),s));for(;i<n;++i)dst[i]+=scale*src[i];
#else
    for(size_t i=0;i<n;++i)dst[i]+=scale*src[i];
#endif
}
}
