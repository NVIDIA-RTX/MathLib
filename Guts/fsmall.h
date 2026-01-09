// © 2021 NVIDIA Corporation

#pragma once

// IEEE 754-2008 Half Precision
// https://ieeexplore.ieee.org/document/4610935
// https://en.wikipedia.org/wiki/Half-precision_floating-point_format
struct fp16 {
    static constexpr uint32_t eBits = 5;
    static constexpr uint32_t mBits = 10;
    static constexpr bool sign = true;
    static constexpr bool inf = true;
};

// Brain Floating Point (Google/Intel)
// https://cloud.google.com/tpu/docs/bfloat16
// https://en.wikipedia.org/wiki/Bfloat16_floating-point_format
struct bf16 {
    static constexpr uint32_t eBits = 8;
    static constexpr uint32_t mBits = 7;
    static constexpr bool sign = true;
    static constexpr bool inf = true;
};

// Packed Float - R11G11B10F
struct fp11u {
    static constexpr uint32_t eBits = 5;
    static constexpr uint32_t mBits = 6;
    static constexpr bool sign = false;
    static constexpr bool inf = true;
};

struct fp10u {
    static constexpr uint32_t eBits = 5;
    static constexpr uint32_t mBits = 5;
    static constexpr bool sign = false;
    static constexpr bool inf = true;
};

// https://en.wikipedia.org/wiki/Minifloat
struct fp8_e5m2 {
    static constexpr uint32_t eBits = 5;
    static constexpr uint32_t mBits = 2;
    static constexpr bool sign = true;
    static constexpr bool inf = true;
};

// NVIDIA/OCP (i.e. fn, not fnuz)
struct fp8_e4m3 {
    static constexpr uint32_t eBits = 4;
    static constexpr uint32_t mBits = 3;
    static constexpr bool sign = true;
    static constexpr bool inf = false;
};

//======================================================================================================================
// float16
//======================================================================================================================

#ifndef __ARM_NEON
struct float16_t {
    uint16_t x;

    ML_INLINE float16_t() = default;
    ML_INLINE float16_t(const float16_t&) = default;
    ML_INLINE float16_t& operator=(const float16_t&) = default;

    // Conversion

    ML_INLINE float16_t(float v);
    ML_INLINE operator float() const;
};
#endif

union float16_t2 {
    uint32_t xy;

    struct {
        float16_t x, y;
    };

    ML_INLINE float16_t2(const float16_t& x, const float16_t& y)
        : x(x), y(y) {
    }

    ML_INLINE float16_t2() = default;
    ML_INLINE float16_t2(const float16_t2&) = default;
    ML_INLINE float16_t2& operator=(const float16_t2&) = default;

    // Conversion

    ML_INLINE operator float2() const;
};

union float16_t4 {
    uint64_t xyzw;

    struct {
        float16_t2 xy, zw;
    };

    struct {
        float16_t x, y, z, w;
    };

    ML_INLINE float16_t4(const float16_t& x, const float16_t& y, const float16_t& z, const float16_t& w)
        : x(x), y(y), z(z), w(w) {
    }

    ML_INLINE float16_t4(const float16_t2& xy, const float16_t2& zw)
        : xy(xy), zw(zw) {
    }

    ML_INLINE float16_t4() = default;
    ML_INLINE float16_t4(const float16_t4&) = default;
    ML_INLINE float16_t4& operator=(const float16_t4&) = default;

    // Conversion

    ML_INLINE operator float4() const;
};

union float16_t8 {
    v4i A_B;

    struct {
        float16_t4 A, B;
    };

    struct {
        float16_t2 Axy, Azw, Bxy, Bzw;
    };

    struct {
        float16_t Ax, Ay, Az, Aw, Bx, By, Bz, Bw;
    };

    ML_INLINE float16_t8(const float16_t& Ax, const float16_t& Ay, const float16_t& Az, const float16_t& Aw,
        const float16_t& Bx, const float16_t& By, const float16_t& Bz, const float16_t& Bw)
        : Ax(Ax), Ay(Ay), Az(Az), Aw(Aw), Bx(Bx), By(By), Bz(Bz), Bw(Bw) {
    }

    ML_INLINE float16_t8(const float16_t2& Axy, const float16_t2& Azw, const float16_t2& Bxy, const float16_t2& Bzw)
        : Axy(Axy), Azw(Azw), Bxy(Bxy), Bzw(Bzw) {
    }

    ML_INLINE float16_t8(const float16_t4& a, const float16_t4& b)
        : A(a), B(b) {
    }

    ML_INLINE float16_t8() = default;
    ML_INLINE float16_t8(const float16_t8&) = default;
    ML_INLINE float16_t8& operator=(const float16_t8&) = default;

    // Conversion

    ML_INLINE float16_t8(const float4& a, const float4& b);
};

//======================================================================================================================
// float8_e4m3
//======================================================================================================================

struct float8_e4m3_t {
    uint8_t x;

    ML_INLINE float8_e4m3_t() = default;
    ML_INLINE float8_e4m3_t(const float8_e4m3_t&) = default;
    ML_INLINE float8_e4m3_t& operator=(const float8_e4m3_t&) = default;

    // Conversion

    ML_INLINE float8_e4m3_t(float v);
    ML_INLINE operator float() const;
};

union float8_e4m3_t2 {
    uint16_t xy;

    struct {
        float8_e4m3_t x, y;
    };

    ML_INLINE float8_e4m3_t2(const float8_e4m3_t& x, const float8_e4m3_t& y)
        : x(x), y(y) {
    }

    ML_INLINE float8_e4m3_t2() = default;
    ML_INLINE float8_e4m3_t2(const float8_e4m3_t2&) = default;
    ML_INLINE float8_e4m3_t2& operator=(const float8_e4m3_t2&) = default;

    // Conversion

    ML_INLINE operator float2() const;
};

union float8_e4m3_t4 {
    uint32_t xyzw;

    struct {
        float8_e4m3_t2 xy, zw;
    };

    struct {
        float8_e4m3_t x, y, z, w;
    };

    ML_INLINE float8_e4m3_t4(const float8_e4m3_t& x, const float8_e4m3_t& y, const float8_e4m3_t& z, const float8_e4m3_t& w)
        : x(x), y(y), z(z), w(w) {
    }

    ML_INLINE float8_e4m3_t4(const float8_e4m3_t2& xy, const float8_e4m3_t2& zw)
        : xy(xy), zw(zw) {
    }

    ML_INLINE float8_e4m3_t4() = default;
    ML_INLINE float8_e4m3_t4(const float8_e4m3_t4&) = default;
    ML_INLINE float8_e4m3_t4& operator=(const float8_e4m3_t4&) = default;

    // Conversion

    ML_INLINE operator float4() const;
};

union float8_e4m3_t8 {
    uint64_t A_B;

    struct {
        float8_e4m3_t4 A, B;
    };

    struct {
        float8_e4m3_t2 Axy, Azw, Bxy, Bzw;
    };

    struct {
        float8_e4m3_t Ax, Ay, Az, Aw, Bx, By, Bz, Bw;
    };

    ML_INLINE float8_e4m3_t8(const float8_e4m3_t& Ax, const float8_e4m3_t& Ay, const float8_e4m3_t& Az, const float8_e4m3_t& Aw,
        const float8_e4m3_t& Bx, const float8_e4m3_t& By, const float8_e4m3_t& Bz, const float8_e4m3_t& Bw)
        : Ax(Ax), Ay(Ay), Az(Az), Aw(Aw), Bx(Bx), By(By), Bz(Bz), Bw(Bw) {
    }

    ML_INLINE float8_e4m3_t8(const float8_e4m3_t2& Axy, const float8_e4m3_t2& Azw, const float8_e4m3_t2& Bxy, const float8_e4m3_t2& Bzw)
        : Axy(Axy), Azw(Azw), Bxy(Bxy), Bzw(Bzw) {
    }

    ML_INLINE float8_e4m3_t8(const float8_e4m3_t4& a, const float8_e4m3_t4& b)
        : A(a), B(b) {
    }

    ML_INLINE float8_e4m3_t8() = default;
    ML_INLINE float8_e4m3_t8(const float8_e4m3_t8&) = default;
    ML_INLINE float8_e4m3_t8& operator=(const float8_e4m3_t8&) = default;

    // Conversion

    ML_INLINE float8_e4m3_t8(const float4& a, const float4& b);
};

//======================================================================================================================
// float8_e5m2
//======================================================================================================================

struct float8_e5m2_t {
    uint8_t x;

    ML_INLINE float8_e5m2_t() = default;
    ML_INLINE float8_e5m2_t(const float8_e5m2_t&) = default;
    ML_INLINE float8_e5m2_t& operator=(const float8_e5m2_t&) = default;

    // Conversion

    ML_INLINE float8_e5m2_t(float v);
    ML_INLINE operator float() const;
};

union float8_e5m2_t2 {
    uint16_t xy;

    struct {
        float8_e5m2_t x, y;
    };

    ML_INLINE float8_e5m2_t2(const float8_e5m2_t& x, const float8_e5m2_t& y)
        : x(x), y(y) {
    }

    ML_INLINE float8_e5m2_t2() = default;
    ML_INLINE float8_e5m2_t2(const float8_e5m2_t2&) = default;
    ML_INLINE float8_e5m2_t2& operator=(const float8_e5m2_t2&) = default;

    // Conversion

    ML_INLINE operator float2() const;
};

union float8_e5m2_t4 {
    uint32_t xyzw;

    struct {
        float8_e5m2_t2 xy, zw;
    };

    struct {
        float8_e5m2_t x, y, z, w;
    };

    ML_INLINE float8_e5m2_t4(const float8_e5m2_t& x, const float8_e5m2_t& y, const float8_e5m2_t& z, const float8_e5m2_t& w)
        : x(x), y(y), z(z), w(w) {
    }

    ML_INLINE float8_e5m2_t4(const float8_e5m2_t2& xy, const float8_e5m2_t2& zw)
        : xy(xy), zw(zw) {
    }

    ML_INLINE float8_e5m2_t4() = default;
    ML_INLINE float8_e5m2_t4(const float8_e5m2_t4&) = default;
    ML_INLINE float8_e5m2_t4& operator=(const float8_e5m2_t4&) = default;

    // Conversion

    ML_INLINE operator float4() const;
};

union float8_e5m2_t8 {
    uint64_t A_B;

    struct {
        float8_e5m2_t4 A, B;
    };

    struct {
        float8_e5m2_t2 Axy, Azw, Bxy, Bzw;
    };

    struct {
        float8_e5m2_t Ax, Ay, Az, Aw, Bx, By, Bz, Bw;
    };

    ML_INLINE float8_e5m2_t8(const float8_e5m2_t& Ax, const float8_e5m2_t& Ay, const float8_e5m2_t& Az, const float8_e5m2_t& Aw,
        const float8_e5m2_t& Bx, const float8_e5m2_t& By, const float8_e5m2_t& Bz, const float8_e5m2_t& Bw)
        : Ax(Ax), Ay(Ay), Az(Az), Aw(Aw), Bx(Bx), By(By), Bz(Bz), Bw(Bw) {
    }

    ML_INLINE float8_e5m2_t8(const float8_e5m2_t2& Axy, const float8_e5m2_t2& Azw, const float8_e5m2_t2& Bxy, const float8_e5m2_t2& Bzw)
        : Axy(Axy), Azw(Azw), Bxy(Bxy), Bzw(Bzw) {
    }

    ML_INLINE float8_e5m2_t8(const float8_e5m2_t4& a, const float8_e5m2_t4& b)
        : A(a), B(b) {
    }

    ML_INLINE float8_e5m2_t8() = default;
    ML_INLINE float8_e5m2_t8(const float8_e5m2_t8&) = default;
    ML_INLINE float8_e5m2_t8& operator=(const float8_e5m2_t8&) = default;

    // Conversion

    ML_INLINE float8_e5m2_t8(const float4& a, const float4& b);
};

//======================================================================================================================
// Conversion
//======================================================================================================================

// Scalar
template <typename FORMAT>
ML_INLINE uint32_t ToSmallFloat(float x) {
    constexpr uint32_t E_BITS = FORMAT::eBits;
    constexpr uint32_t M_BITS = FORMAT::mBits;
    constexpr int32_t E_BIAS = 127 - ((1 << (int32_t(E_BITS) - 1)) - 1);
    constexpr uint32_t E_MASK = (1u << E_BITS) - 1u;
    constexpr uint32_t M_MASK = (1u << M_BITS) - 1u;
    constexpr uint32_t M_SHIFT = 23u - M_BITS;
    constexpr uint32_t INF = E_MASK << M_BITS;

    static_assert(M_BITS > 0 && M_BITS < 23);
    static_assert(E_BITS > 0 && E_BITS < 8);

    uint32_t f32 = *(uint32_t*)&x;

    // Sign and abs
    uint32_t s = 0;
    if constexpr (FORMAT::sign)
        s = (f32 & 0x80000000) >> (31u - E_BITS - M_BITS);

    f32 &= 0x7FFFFFFF; // abs now

    // Extract exponent and mantissa
    int32_t e = (int32_t)(f32 >> 23) - E_BIAS;

    int32_t is_normal = (int32_t)(e > 0);
    uint32_t shift = is_normal ? M_SHIFT : M_SHIFT + 1 - e;
    if (shift > 25) // "min" in SSE
        return s;

    uint32_t m = (f32 & 0x007FFFFF) | 0x00800000; // with hidden bit

    // Rounding (ties to even)
    uint32_t bias = (1u << (shift - 1u)) - 1u; // 0.4999...
    bias += (m >> shift) & 1u;                 // add LSB to pull ties to even
    m += bias;                                 // can't overflow because "m" is 24 bits only

    m >>= shift;

    // Post-rounding carry
    uint32_t carry = m >> (M_BITS + is_normal); // 0 or 1 (if rounded up to a normal, "m" is now "1 << M_BITS")
    m >>= carry & is_normal;
    e = (e & -is_normal) + (int32_t)carry;

    // Combine
    m &= M_MASK; // clear mantissa overflow and the hidden bit
    uint32_t em = (uint32_t(e) << M_BITS) | m;

    // Final overflow check
    if constexpr (FORMAT::inf) {
        uint32_t special = INF | uint32_t(f32 > 0x7F800000); // quiet NAN is OK for DL/ML
        em = (e >= (int32_t)E_MASK) ? special : em;          // if "f32 == 0x7F800000" (INF), then "e >= E_MASK" too
    } else
        em = (e > (int32_t)E_MASK) ? (INF | M_MASK) : em; // if "e == E_MASK" it's "max value" (extended range)

    return s | em;
}

template <typename FORMAT>
ML_INLINE float FromSmallFloat(uint32_t x) {
    constexpr uint32_t E_BITS = FORMAT::eBits;
    constexpr uint32_t M_BITS = FORMAT::mBits;
    constexpr uint32_t E_BIAS = 127 - ((1 << (E_BITS - 1)) - 1);
    constexpr uint32_t E_MASK = (1u << E_BITS) - 1u;
    constexpr uint32_t M_MASK = (1u << M_BITS) - 1u;
    constexpr uint32_t M_SHIFT = 23 - M_BITS;
    constexpr uint32_t S_MASK = 1u << (E_BITS + M_BITS);
    constexpr uint32_t EM_MASK = (E_MASK << M_BITS) | M_MASK;

    uint32_t e = (x >> M_BITS) & E_MASK;
    uint32_t m = x & M_MASK;

    // Normal / special
    uint32_t norm_e = e + E_BIAS;
    if constexpr (FORMAT::inf)
        norm_e = (e == E_MASK) ? 0xFF : norm_e;

    uint32_t res = (norm_e << 23) | (m << M_SHIFT);
    if constexpr (!FORMAT::inf)
        res = (x & EM_MASK) == EM_MASK ? 0x7FC00000 : res;

    // Denorm
    if (e == 0) {
        // Slow path
#ifdef _MSC_VER
        uint32_t lz = __lzcnt(m);
#else
        uint32_t lz = __builtin_clz(m | 0x1);
#endif
        uint32_t shift = lz - (31 - M_BITS);
        uint32_t sub_e = (E_BIAS + 1 - shift) << 23;
        uint32_t sub_m = (m << (M_SHIFT + shift)) & 0x7FFFFF;
        res = !m ? 0 : (sub_e | sub_m);
    }

    // Sign
    if constexpr (FORMAT::sign)
        res |= (x & S_MASK) << (31 - E_BITS - M_BITS);

    return *(float*)&res;
}

// Vector4
template <typename FORMAT>
ML_INLINE v4i ToSmallFloat4(v4f x) {
    constexpr uint32_t E_BITS = FORMAT::eBits;
    constexpr uint32_t M_BITS = FORMAT::mBits;
    constexpr int32_t E_BIAS = 127 - ((1 << (int32_t(E_BITS) - 1)) - 1);
    constexpr uint32_t E_MASK = (1u << E_BITS) - 1u;
    constexpr uint32_t M_MASK = (1u << M_BITS) - 1u;
    constexpr uint32_t M_SHIFT = 23u - M_BITS;
    constexpr uint32_t INF = E_MASK << M_BITS;

    const v4i _1 = _mm_set1_epi32(1);
    const v4i _0 = _mm_setzero_si128();

    v4i f32 = _mm_castps_si128(x);

    // Sign and abs
    v4i s = _0;
    if constexpr (FORMAT::sign)
        s = _mm_srli_epi32(_mm_and_si128(f32, _mm_set1_epi32(0x80000000)), 31 - E_BITS - M_BITS);

    f32 = _mm_and_si128(f32, _mm_set1_epi32(0x7FFFFFFF)); // abs now

    // Extract exponent and mantissa
    v4i e = _mm_sub_epi32(_mm_srli_epi32(f32, 23), _mm_set1_epi32(E_BIAS));

    v4i is_normal = _mm_cmpgt_epi32(e, _0);
    v4i shift = _mm_add_epi32(_mm_set1_epi32(M_SHIFT), _mm_andnot_si128(is_normal, _mm_sub_epi32(_1, e)));
    shift = _mm_min_epi32(_mm_set1_epi32(25), shift);

    v4i m = _mm_or_si128(_mm_and_si128(f32, _mm_set1_epi32(0x7FFFFF)), _mm_set1_epi32(0x00800000));

    // Rounding (ties to even)
    v4i half_minus_one = _mm_sub_epi32(_mm_sllv_epi32(_1, _mm_sub_epi32(shift, _1)), _1);
    v4i lsb = _mm_and_si128(_mm_srlv_epi32(m, shift), _1);
    m = _mm_add_epi32(m, _mm_add_epi32(half_minus_one, lsb));
    m = _mm_srlv_epi32(m, shift);

    // Post-rounding carry
    // SSE Trick: Map boolean "is_normal" to 0 or 1 integer to unify carry detection logic
    v4i normal_one = _mm_and_si128(is_normal, _1);
    // SSE Trick: Shift m by M_BITS (denorm) or M_BITS+1 (norm) to extract the carry bit
    v4i carry_shift = _mm_add_epi32(_mm_set1_epi32(M_BITS), normal_one);
    v4i carry = _mm_srlv_epi32(m, carry_shift);
    m = _mm_srlv_epi32(m, _mm_and_si128(carry, normal_one));
    e = _mm_add_epi32(_mm_and_si128(e, is_normal), carry);

    // Combine
    m = _mm_and_si128(m, _mm_set1_epi32(M_MASK));
    v4i em = _mm_or_si128(_mm_slli_epi32(e, M_BITS), m);

    // Final overflow check
    if constexpr (FORMAT::inf) {
        v4i special_mask = _mm_cmpgt_epi32(e, _mm_set1_epi32(E_MASK - 1));
        v4i is_nan = _mm_cmpgt_epi32(f32, _mm_set1_epi32(0x7F800000));
        v4i special = _mm_or_si128(_mm_set1_epi32(INF), is_nan);

        em = _mm_blendv_epi8(em, special, special_mask);
    } else {
        v4i special_mask = _mm_cmpgt_epi32(e, _mm_set1_epi32(E_MASK));

        em = _mm_blendv_epi8(em, _mm_set1_epi32(INF | M_MASK), special_mask);
    }

    return _mm_or_si128(s, em);
}

template <typename FORMAT>
ML_INLINE v4f FromSmallFloat4(v4i x) {
    constexpr uint32_t E_BITS = FORMAT::eBits;
    constexpr uint32_t M_BITS = FORMAT::mBits;
    constexpr uint32_t E_BIAS = 127 - ((1 << (E_BITS - 1)) - 1);
    constexpr uint32_t E_MASK = (1u << E_BITS) - 1u;
    constexpr uint32_t M_MASK = (1u << M_BITS) - 1u;
    constexpr uint32_t M_SHIFT = 23 - M_BITS;
    constexpr uint32_t S_MASK = 1u << (E_BITS + M_BITS);
    constexpr uint32_t EM_MASK = (E_MASK << M_BITS) | M_MASK;

    const v4i _0 = _mm_setzero_si128();

    v4i e = _mm_and_si128(_mm_srli_epi32(x, M_BITS), _mm_set1_epi32(E_MASK));
    v4i m = _mm_and_si128(x, _mm_set1_epi32(M_MASK));

    // Normal / special
    v4i norm_e = _mm_add_epi32(e, _mm_set1_epi32(E_BIAS));
    if constexpr (FORMAT::inf) {
        v4i overflow_mask = _mm_cmpeq_epi32(e, _mm_set1_epi32(E_MASK));
        norm_e = _mm_blendv_epi8(norm_e, _mm_set1_epi32(0xFF), overflow_mask);
    }

    v4i res = _mm_or_si128(_mm_slli_epi32(norm_e, 23), _mm_slli_epi32(m, M_SHIFT));
    if constexpr (!FORMAT::inf) {
        v4i nan_mask = _mm_cmpeq_epi32(_mm_and_si128(x, _mm_set1_epi32(EM_MASK)), _mm_set1_epi32(EM_MASK));
        res = _mm_blendv_epi8(res, _mm_set1_epi32(0x7FC00000), nan_mask);
    }

    // Denorm
    // SSE Trick: Converting an integer to float uses the CPU's hardware
    // priority encoder to normalize the mantissa. The resulting float's
    // exponent (at bit 23) tells us exactly how many leading zeros
    // were skipped. This is faster than LZCNT in a SIMD loop.
    v4f m_f = _mm_cvtepi32_ps(m);
    v4i m_fi = _mm_castps_si128(m_f);
    v4i e_f = _mm_srli_epi32(m_fi, 23);
    v4i shift = _mm_sub_epi32(_mm_set1_epi32(127 + M_BITS), e_f);

    v4i sub_e = _mm_slli_epi32(_mm_sub_epi32(_mm_set1_epi32(E_BIAS + 1), shift), 23);
    v4i sub_m = _mm_and_si128(_mm_sllv_epi32(m, _mm_add_epi32(_mm_set1_epi32(M_SHIFT), shift)), _mm_set1_epi32(0x7FFFFF));
    v4i sub_res = _mm_blendv_epi8(_mm_or_si128(sub_e, sub_m), _0, _mm_cmpeq_epi32(m, _0));

    res = _mm_blendv_epi8(res, sub_res, _mm_cmpeq_epi32(e, _0));

    // Sign
    if constexpr (FORMAT::sign) {
        v4i s = _mm_and_si128(x, _mm_set1_epi32(S_MASK));
        s = _mm_slli_epi32(s, 31 - E_BITS - M_BITS);
        res = _mm_or_si128(res, s);
    }

    return _mm_castsi128_ps(res);
}

ML_INLINE uint32_t v4i_to_u32(v4i v) {
    const v4i mask = _mm_setr_epi8(
        0, 4, 8, 12,
        -1, -1, -1, -1,
        -1, -1, -1, -1,
        -1, -1, -1, -1);

    v = _mm_shuffle_epi8(v, mask);

    return (uint32_t)_mm_cvtsi128_si32(v);
}

ML_INLINE uint64_t v4i_to_u64(v4i v) {
    const v4i mask = _mm_setr_epi8(
        0, 1, 4, 5,
        8, 9, 12, 13,
        -1, -1, -1, -1,
        -1, -1, -1, -1);

    v = _mm_shuffle_epi8(v, mask);

    return (uint64_t)_mm_cvtsi128_si64(v);
}

ML_INLINE v4i u32_to_v4i(uint32_t p) {
    v4i v = _mm_cvtsi32_si128(p);

    const v4i mask = _mm_setr_epi8(
        0, -1, -1, -1,
        1, -1, -1, -1,
        2, -1, -1, -1,
        3, -1, -1, -1);

    return _mm_shuffle_epi8(v, mask);
}

ML_INLINE v4i u64_to_v4i(uint64_t p) {
    v4i v = _mm_cvtsi64_si128(p);

    const v4i mask = _mm_setr_epi8(
        0, 1, -1, -1,
        2, 3, -1, -1,
        4, 5, -1, -1,
        6, 7, -1, -1);

    return _mm_shuffle_epi8(v, mask);
}

// Vector8 (slow in emulation mode) // TODO: a few intrinsics missing emulation
#if (ML_INTRINSIC_LEVEL >= ML_INTRINSIC_AVX2)

template <typename FORMAT>
ML_INLINE v8i ToSmallFloat8(v8f x) {
    constexpr uint32_t E_BITS = FORMAT::eBits;
    constexpr uint32_t M_BITS = FORMAT::mBits;
    constexpr int32_t E_BIAS = 127 - ((1 << (int32_t(E_BITS) - 1)) - 1);
    constexpr uint32_t E_MASK = (1u << E_BITS) - 1u;
    constexpr uint32_t M_MASK = (1u << M_BITS) - 1u;
    constexpr uint32_t M_SHIFT = 23u - M_BITS;
    constexpr uint32_t INF = E_MASK << M_BITS;

    const v8i _1 = _mm256_set1_epi32(1);
    const v8i _0 = _mm256_setzero_si256();

    v8i f32 = _mm256_castps_si256(x);

    // Sign and abs
    v8i s = _0;
    if constexpr (FORMAT::sign)
        s = _mm256_srli_epi32(_mm256_and_si256(f32, _mm256_set1_epi32(0x80000000)), 31 - E_BITS - M_BITS);

    f32 = _mm256_and_si256(f32, _mm256_set1_epi32(0x7FFFFFFF));

    // Extract exponent and mantissa
    v8i e = _mm256_sub_epi32(_mm256_srli_epi32(f32, 23), _mm256_set1_epi32(E_BIAS));

    v8i is_normal = _mm256_cmpgt_epi32(e, _0);
    v8i shift = _mm256_add_epi32(_mm256_set1_epi32(M_SHIFT), _mm256_andnot_si256(is_normal, _mm256_sub_epi32(_1, e)));
    shift = _mm256_min_epi32(_mm256_set1_epi32(25), shift);

    v8i m = _mm256_or_si256(_mm256_and_si256(f32, _mm256_set1_epi32(0x7FFFFF)), _mm256_set1_epi32(0x00800000));

    // Rounding (ties to even)
    v8i half_minus_one = _mm256_sub_epi32(_mm256_sllv_epi32(_1, _mm256_sub_epi32(shift, _1)), _1);
    v8i lsb = _mm256_and_si256(_mm256_srlv_epi32(m, shift), _1);
    m = _mm256_add_epi32(m, _mm256_add_epi32(half_minus_one, lsb));
    m = _mm256_srlv_epi32(m, shift);

    // Post-rounding carry
    v8i normal_one = _mm256_and_si256(is_normal, _1);
    v8i carry_shift = _mm256_add_epi32(_mm256_set1_epi32(M_BITS), normal_one);
    v8i carry = _mm256_srlv_epi32(m, carry_shift);
    m = _mm256_srlv_epi32(m, _mm256_and_si256(carry, normal_one));
    e = _mm256_add_epi32(_mm256_and_si256(e, is_normal), carry);

    // Combine
    m = _mm256_and_si256(m, _mm256_set1_epi32(M_MASK));
    v8i em = _mm256_or_si256(_mm256_slli_epi32(e, M_BITS), m);

    // Final overflow check
    if constexpr (FORMAT::inf) {
        v8i special_mask = _mm256_cmpgt_epi32(e, _mm256_set1_epi32(E_MASK - 1));
        v8i is_nan = _mm256_cmpgt_epi32(f32, _mm256_set1_epi32(0x7F800000));
        v8i special = _mm256_or_si256(_mm256_set1_epi32(INF), is_nan);
        em = _mm256_blendv_epi8(em, special, special_mask);
    } else {
        v8i special_mask = _mm256_cmpgt_epi32(e, _mm256_set1_epi32(E_MASK));
        em = _mm256_blendv_epi8(em, _mm256_set1_epi32(INF | M_MASK), special_mask);
    }

    return _mm256_or_si256(s, em);
}

template <typename FORMAT>
ML_INLINE v8f FromSmallFloat8(v8i x) {
    constexpr uint32_t E_BITS = FORMAT::eBits;
    constexpr uint32_t M_BITS = FORMAT::mBits;
    constexpr uint32_t E_BIAS = 127 - ((1 << (E_BITS - 1)) - 1);
    constexpr uint32_t E_MASK = (1u << E_BITS) - 1u;
    constexpr uint32_t M_MASK = (1u << M_BITS) - 1u;
    constexpr uint32_t M_SHIFT = 23 - M_BITS;
    constexpr uint32_t S_MASK = 1u << (E_BITS + M_BITS);
    constexpr uint32_t EM_MASK = (E_MASK << M_BITS) | M_MASK;

    const v8i _0 = _mm256_setzero_si256();

    v8i e = _mm256_and_si256(_mm256_srli_epi32(x, M_BITS), _mm256_set1_epi32(E_MASK));
    v8i m = _mm256_and_si256(x, _mm256_set1_epi32(M_MASK));

    // Normal / special
    v8i norm_e = _mm256_add_epi32(e, _mm256_set1_epi32(E_BIAS));
    if constexpr (FORMAT::inf) {
        v8i overflow_mask = _mm256_cmpeq_epi32(e, _mm256_set1_epi32(E_MASK));
        norm_e = _mm256_blendv_epi8(norm_e, _mm256_set1_epi32(0xFF), overflow_mask);
    }

    v8i res = _mm256_or_si256(_mm256_slli_epi32(norm_e, 23), _mm256_slli_epi32(m, M_SHIFT));
    if constexpr (!FORMAT::inf) {
        v8i nan_mask = _mm256_cmpeq_epi32(_mm256_and_si256(x, _mm256_set1_epi32(EM_MASK)), _mm256_set1_epi32(EM_MASK));
        res = _mm256_blendv_epi8(res, _mm256_set1_epi32(0x7FC00000), nan_mask);
    }

    // Denorm
    v8f m_f = _mm256_cvtepi32_ps(m);
    v8i m_fi = _mm256_castps_si256(m_f);
    v8i e_f = _mm256_srli_epi32(m_fi, 23);
    v8i shift = _mm256_sub_epi32(_mm256_set1_epi32(127 + M_BITS), e_f);

    v8i sub_e = _mm256_slli_epi32(_mm256_sub_epi32(_mm256_set1_epi32(E_BIAS + 1), shift), 23);
    v8i sub_m = _mm256_and_si256(_mm256_sllv_epi32(m, _mm256_add_epi32(_mm256_set1_epi32(M_SHIFT), shift)), _mm256_set1_epi32(0x7FFFFF));
    v8i sub_res = _mm256_blendv_epi8(_mm256_or_si256(sub_e, sub_m), _0, _mm256_cmpeq_epi32(m, _0));

    res = _mm256_blendv_epi8(res, sub_res, _mm256_cmpeq_epi32(e, _0));

    // Sign
    if constexpr (FORMAT::sign) {
        v8i s = _mm256_and_si256(x, _mm256_set1_epi32(S_MASK));
        s = _mm256_slli_epi32(s, 31 - E_BITS - M_BITS);
        res = _mm256_or_si256(res, s);
    }

    return _mm256_castsi256_ps(res);
}

ML_INLINE uint64_t v8i_to_u64(v8i v) {
    const v8i mask = _mm256_setr_epi8(
        0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);

    v = _mm256_shuffle_epi8(v, mask);
    v = _mm256_permutevar8x32_epi32(v, _mm256_setr_epi32(0, 4, 0, 0, 0, 0, 0, 0));

    return (uint64_t)_mm256_extract_epi64(v, 0);
}

ML_INLINE v4i v8i_to_u64x2(v8i v) {
    const v8i mask = _mm256_setr_epi8(
        0, 1, 4, 5, 8, 9, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1,
        0, 1, 4, 5, 8, 9, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1);

    v = _mm256_shuffle_epi8(v, mask);

    v4i lo = _mm256_castsi256_si128(v);
    v4i hi = _mm256_extracti128_si256(v, 1);

    return _mm_unpacklo_epi64(lo, hi);
}

ML_INLINE v8i u64_to_v8i(uint64_t p) {
    v8i v = _mm256_set1_epi64x(p);

    const v8i mask = _mm256_setr_epi8(
        0, -1, -1, -1, 1, -1, -1, -1, 2, -1, -1, -1, 3, -1, -1, -1,
        4, -1, -1, -1, 5, -1, -1, -1, 6, -1, -1, -1, 7, -1, -1, -1);

    return _mm256_shuffle_epi8(v, mask);
}

ML_INLINE v8i u64x2_to_v8i(v4i p) {
    v8i v = _mm256_castsi128_si256(p);
    v = _mm256_permute2x128_si256(v, v, 0x00);

    const v8i mask = _mm256_setr_epi8(
        0, 1, -1, -1, 2, 3, -1, -1, 4, 5, -1, -1, 6, 7, -1, -1,
        8, 9, -1, -1, 10, 11, -1, -1, 12, 13, -1, -1, 14, 15, -1, -1);

    return _mm256_shuffle_epi8(v, mask);
}

#endif

// HLSL
ML_INLINE uint32_t f32tof16(float x) {
#if (ML_INTRINSIC_LEVEL >= ML_INTRINSIC_AVX1 && ML_ALLOW_HW_FP16)
    v4f v = v4f_set(x, 0.0f, 0.0f, 0.0f);
    v4i p = _mm_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT);

    uint32_t r = _mm_cvtsi128_si32(p);
#else
    uint32_t r = ToSmallFloat<fp16>(x);
#endif

    return r;
}

ML_INLINE float f16tof32(uint32_t x) {
#if (ML_INTRINSIC_LEVEL >= ML_INTRINSIC_AVX1 && ML_ALLOW_HW_FP16)
    v4i p = _mm_cvtsi32_si128(x);
    v4f f = _mm_cvtph_ps(p);

    return _mm_cvtss_f32(f);
#else
    return FromSmallFloat<fp16>(x);
#endif
}

//===============================================================

struct FLOAT32 {
    static constexpr uint32_t EXPONENT_WIDTH = 8;
    static constexpr uint32_t MANTISSA_WIDTH = 23;
    static constexpr bool SUPPORTS_INF = true;
};

// Reference: https://en.wikipedia.org/wiki/Half-precision_floating-point_format
struct FLOAT16 {
    static constexpr uint32_t EXPONENT_WIDTH = 5;
    static constexpr uint32_t MANTISSA_WIDTH = 10;
    static constexpr bool SUPPORTS_INF = true;
};

// Reference: "FP8 Formats for Deep Learning" by Paulius Micikevicius et al.
// https://arxiv.org/pdf/2209.05433
struct FLOAT8E4M3 {
    static constexpr uint32_t EXPONENT_WIDTH = 4;
    static constexpr uint32_t MANTISSA_WIDTH = 3;
    static constexpr bool SUPPORTS_INF = false;
};

// Reference: See above for FLOAT8E4M3
struct FLOAT8E5M2 {
    static constexpr uint32_t EXPONENT_WIDTH = 5;
    static constexpr uint32_t MANTISSA_WIDTH = 2;
    static constexpr bool SUPPORTS_INF = true;
};

template <typename T>
struct HELPER : public T {
    static constexpr uint32_t EXPONENT_BIAS = (1U << (T::EXPONENT_WIDTH - 1U)) - 1U;
    static constexpr uint32_t EXPONENT_SHIFT = T::MANTISSA_WIDTH;
    static constexpr uint32_t MAX_EXPONENT = (1U << T::EXPONENT_WIDTH) - 1U;
    static constexpr uint32_t SIGN_BIT_SHIFT = T::MANTISSA_WIDTH + T::EXPONENT_WIDTH;
    static constexpr uint32_t SIGN_BIT = 1U << SIGN_BIT_SHIFT;
    static constexpr uint32_t EXPONENT_MASK = MAX_EXPONENT << EXPONENT_SHIFT;
    static constexpr uint32_t MANTISSA_MASK = (1U << T::MANTISSA_WIDTH) - 1U;
    static constexpr uint32_t TOTAL_WIDTH = 1 + T::EXPONENT_WIDTH + T::MANTISSA_WIDTH;
    static constexpr uint32_t EXPONENT_MANTISSA_MASK = SIGN_BIT - 1;
};

template <typename OUT_FORMAT, typename OUT_TYPE>
OUT_TYPE DownConvert(float const x) {
    using IN_FORMAT = HELPER<FLOAT32>;

    static_assert(OUT_FORMAT::MANTISSA_WIDTH < IN_FORMAT::MANTISSA_WIDTH);
    static_assert(OUT_FORMAT::EXPONENT_WIDTH < IN_FORMAT::EXPONENT_WIDTH);
    static_assert(OUT_FORMAT::EXPONENT_BIAS + OUT_FORMAT::MANTISSA_WIDTH < IN_FORMAT::EXPONENT_BIAS);
    static_assert(OUT_FORMAT::TOTAL_WIDTH == sizeof(OUT_TYPE) * 8);

    uint32_t const inBits = *(uint32_t*)&x;
    uint32_t const inSign = inBits & IN_FORMAT::SIGN_BIT;
    uint32_t const inExponent = inBits & IN_FORMAT::EXPONENT_MASK;
    uint32_t const inMantissa = inBits & IN_FORMAT::MANTISSA_MASK;

    OUT_TYPE const outSign = OUT_TYPE(inSign >> (IN_FORMAT::SIGN_BIT_SHIFT - OUT_FORMAT::SIGN_BIT_SHIFT));
    OUT_TYPE outExponent = 0;
    OUT_TYPE outMantissa = 0;

    if (inExponent == IN_FORMAT::EXPONENT_MASK) // Inf or NaN
    {
        outExponent = OUT_FORMAT::EXPONENT_MASK;
        if (inMantissa != 0 || !OUT_FORMAT::SUPPORTS_INF) // Input NaN, or Inf is not supported - force NaN
            outMantissa = OUT_FORMAT::MANTISSA_MASK;
    } else if (inExponent == 0) // Zero or denormal
    {
        // Input denormals are too small for the output formats, return zero.
        // outExponent and outMantissa are already 0, do nothing.
    } else // Normal number that might turn into a denormal or Inf/NaN after down-conversion
    {
        constexpr uint32_t inHiddenOne = (1u << IN_FORMAT::MANTISSA_WIDTH);
        constexpr uint32_t outHiddenOne = (1u << OUT_FORMAT::MANTISSA_WIDTH);

        // Restore the hidden 1 for now
        uint32_t mantissa = inMantissa | inHiddenOne;

        int adjustedExponent = int(inExponent >> IN_FORMAT::EXPONENT_SHIFT)
            - IN_FORMAT::EXPONENT_BIAS + OUT_FORMAT::EXPONENT_BIAS;

        uint32_t remainderBits = IN_FORMAT::MANTISSA_WIDTH - OUT_FORMAT::MANTISSA_WIDTH;
        if (adjustedExponent <= 0) // Convert to denormal
        {
            remainderBits += 1 - adjustedExponent;
            adjustedExponent = 0;
        }

        if (remainderBits > IN_FORMAT::MANTISSA_WIDTH + 1) // Shifting too far to the right
        {
            mantissa = 0;
        } else {
            uint32_t remainderMask = (1u << remainderBits) - 1u;
            uint32_t remainderHalf = 1u << (remainderBits - 1u);
            uint32_t remainder = mantissa & remainderMask;
            mantissa >>= remainderBits;

            // Rounding to nearest, ties to even
            if (remainder > remainderHalf || (remainder == remainderHalf && (mantissa & 1u) != 0)) {
                ++mantissa;
            }
        }

        // Mantissa is greater than the output format including the hidden one
        if ((mantissa & (outHiddenOne << 1u)) != 0) {
            mantissa >>= 1;
            ++adjustedExponent;
        } else if ((mantissa & outHiddenOne) == 0) {
            // Hidden one turned into zero, it's a denormal now
            adjustedExponent = 0;
        } else if (adjustedExponent == 0) {
            // There is a hidden 1, so it cannot be a denormal - set the exponent to 1
            adjustedExponent = 1;
        }

        mantissa &= OUT_FORMAT::MANTISSA_MASK;
        if constexpr (OUT_FORMAT::SUPPORTS_INF) {
            // The number turned into Inf, clear the mantissa to make sure it's not a NaN
            if (adjustedExponent >= OUT_FORMAT::MAX_EXPONENT) {
                adjustedExponent = OUT_FORMAT::MAX_EXPONENT;
                mantissa = 0;
            }
        } else {
            // Inf not supported - turn any numbers with out-of-range exponents into NaN
            if (adjustedExponent > OUT_FORMAT::MAX_EXPONENT) {
                adjustedExponent = OUT_FORMAT::MAX_EXPONENT;
                mantissa = OUT_FORMAT::MANTISSA_MASK;
            }
        }

        outExponent = OUT_TYPE(adjustedExponent) << OUT_FORMAT::EXPONENT_SHIFT;
        outMantissa = OUT_TYPE(mantissa);
    }

    return outSign | outExponent | outMantissa;
}

template <typename IN_FORMAT, typename IN_TYPE>
float UpConvert(IN_TYPE const bits) {
    using OUT_FORMAT = HELPER<FLOAT32>;

    static_assert(OUT_FORMAT::MANTISSA_WIDTH > IN_FORMAT::MANTISSA_WIDTH);
    static_assert(OUT_FORMAT::EXPONENT_WIDTH > IN_FORMAT::EXPONENT_WIDTH);
    static_assert(OUT_FORMAT::EXPONENT_BIAS > IN_FORMAT::EXPONENT_BIAS + IN_FORMAT::MANTISSA_WIDTH);
    static_assert(IN_FORMAT::TOTAL_WIDTH == sizeof(IN_TYPE) * 8);

    IN_TYPE const inSign = bits & IN_FORMAT::SIGN_BIT;
    IN_TYPE const inExponent = bits & IN_FORMAT::EXPONENT_MASK;
    IN_TYPE const inMantissa = bits & IN_FORMAT::MANTISSA_MASK;

    uint32_t const outSign = uint32_t(inSign) << (OUT_FORMAT::SIGN_BIT_SHIFT - IN_FORMAT::SIGN_BIT_SHIFT);
    uint32_t outExponent = uint32_t(inExponent) << (OUT_FORMAT::EXPONENT_SHIFT - IN_FORMAT::EXPONENT_SHIFT);
    outExponent += (OUT_FORMAT::EXPONENT_BIAS - IN_FORMAT::EXPONENT_BIAS) << OUT_FORMAT::EXPONENT_SHIFT;
    uint32_t outMantissa = uint32_t(inMantissa) << (OUT_FORMAT::MANTISSA_WIDTH - IN_FORMAT::MANTISSA_WIDTH);

    if (inExponent == IN_FORMAT::EXPONENT_MASK && (IN_FORMAT::SUPPORTS_INF || inMantissa == IN_FORMAT::MANTISSA_MASK)) // Inf or NaN
    {
        outExponent = OUT_FORMAT::EXPONENT_MASK;
    } else if (inExponent == 0) // Denormal
    {
        if (inMantissa != 0) // Non-zero denormal - convert to a normal number, there must be enough exponent bits for that
        {
#ifdef _MSC_VER
            uint32_t leadingZeros = __lzcnt(uint32_t(inMantissa));
#else
            uint32_t leadingZeros = __builtin_clz(uint32_t(inMantissa));
#endif
            // Don't count the bits from uint32_t higher than the mantissa
            leadingZeros -= 32 - IN_FORMAT::MANTISSA_WIDTH;

            // Shift the mantissa so that its highest "one" bit becomes the hidden "one"
            outMantissa = (outMantissa << (leadingZeros + 1)) & OUT_FORMAT::MANTISSA_MASK;
            // Correct the exponent after that mantissa shift.
            // This assumes that the result is going to be positive, and the static_assert on EXPONENT_BIAS above
            // makes sure that's always true.
            outExponent -= leadingZeros << OUT_FORMAT::EXPONENT_SHIFT;
        } else // Zero
        {
            outExponent = 0;
        }
    }

    uint32_t r = outSign | outExponent | outMantissa;

    return *(float*)&r;
}