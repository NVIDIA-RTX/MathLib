// © 2021 NVIDIA Corporation

#pragma once

template <class C, typename T, uint32_t... Indices>
class swizzle {
private:
    // Based on: https://kiorisyshen.github.io/2018/08/27/Vector%20Swizzling%20and%20Parameter%20Pack%20in%20C++/
    T a[sizeof...(Indices)];

public:
    static constexpr uint32_t i[] = {Indices...};
    static constexpr size_t N = sizeof...(Indices);

    ML_INLINE void operator=(const C& rhs) {
        for (size_t n = 0; n < N; n++)
            a[i[n]] = rhs[n];
    }

    ML_INLINE operator C() const {
        return C(a[Indices]...);
    }
};

// Swizzle ops

#define ML_SWIZZLE_2_OP(op, f, swizzle) \
    ML_INLINE void operator op(const C& v) { \
        ML_StaticAssertMsg(X != Y, "Wrong swizzle in " ML_Stringify(op)); \
        a[X] op v.x; \
        a[Y] op v.y; \
    }

#define ML_SWIZZLE_3_OP(op, f, swizzle) \
    ML_INLINE void operator op(const C& v) { \
        ML_StaticAssertMsg(X != Y && Y != Z && Z != X, "Wrong swizzle in " ML_Stringify(op)); \
        a[X] op v.x; \
        a[Y] op v.y; \
        a[Z] op v.z; \
    }

#if 0
#    define ML_SWIZZLE_4_OP(op, f, swizzle) \
        ML_INLINE void operator op(const C& v) { \
            ML_StaticAssertMsg(X + Y + Z + W == 6, "Wrong swizzle in " ML_Stringify(op)); \
            a[X] op v.x; \
            a[Y] op v.y; \
            a[Z] op v.z; \
            a[W] op v.w; \
        }
#else
#    define ML_SWIZZLE_4_OP(op, f, swizzle) \
        ML_INLINE void operator op(const C& v) { \
            ML_StaticAssertMsg(X + Y + Z + W == 6, "Wrong swizzle in " ML_Stringify(op)); \
            vec = f(swizzle(vec, X, Y, Z, W), v); \
        }
#endif

// v4i

template <class C, uint32_t X, uint32_t Y>
class v4i_swizzle2 {
private:
    union {
        struct {
            v4i vec;
        };

        struct {
            int32_t a[COORD_4D];
        };
    };

public:
    // Read-only: fast
    ML_INLINE operator C() const {
        return C(a[X], a[Y]);
    }

    // Read-write: most likely slow
    ML_SWIZZLE_2_OP(=, _mm_copy, v4i_swizzle)
    ML_SWIZZLE_2_OP(-=, _mm_sub_epi32, v4i_swizzle)
    ML_SWIZZLE_2_OP(+=, _mm_add_epi32, v4i_swizzle)
    ML_SWIZZLE_2_OP(*=, _mm_mullo_epi32, v4i_swizzle)
    ML_SWIZZLE_2_OP(/=, _mm_div_epi32, v4i_swizzle)
    ML_SWIZZLE_2_OP(%=, v4i_mod, v4i_swizzle)
    ML_SWIZZLE_2_OP(<<=, _mm_sllv_epi32, v4i_swizzle)
    ML_SWIZZLE_2_OP(>>=, _mm_srlv_epi32, v4i_swizzle)
    ML_SWIZZLE_2_OP(&=, _mm_and_si128, v4i_swizzle)
    ML_SWIZZLE_2_OP(|=, _mm_or_si128, v4i_swizzle)
    ML_SWIZZLE_2_OP(^=, _mm_xor_si128, v4i_swizzle)
};

template <class C, uint32_t X, uint32_t Y, uint32_t Z>
class v4i_swizzle3 {
private:
    union {
        struct {
            v4i vec;
        };

        struct {
            int32_t a[COORD_4D];
        };
    };

public:
    // Read-only: fast
    ML_INLINE operator C() const {
        return v4i_swizzle(vec, X, Y, Z, 3);
    }

    // Read-write: most likely slow
    ML_SWIZZLE_3_OP(=, _mm_copy, v4i_swizzle)
    ML_SWIZZLE_3_OP(-=, _mm_sub_epi32, v4i_swizzle)
    ML_SWIZZLE_3_OP(+=, _mm_add_epi32, v4i_swizzle)
    ML_SWIZZLE_3_OP(*=, _mm_mullo_epi32, v4i_swizzle)
    ML_SWIZZLE_3_OP(/=, _mm_div_epi32, v4i_swizzle)
    ML_SWIZZLE_3_OP(%=, v4i_mod, v4i_swizzle)
    ML_SWIZZLE_3_OP(<<=, _mm_sllv_epi32, v4i_swizzle)
    ML_SWIZZLE_3_OP(>>=, _mm_srlv_epi32, v4i_swizzle)
    ML_SWIZZLE_3_OP(&=, _mm_and_si128, v4i_swizzle)
    ML_SWIZZLE_3_OP(|=, _mm_or_si128, v4i_swizzle)
    ML_SWIZZLE_3_OP(^=, _mm_xor_si128, v4i_swizzle)
};

template <class C, uint32_t X, uint32_t Y, uint32_t Z, uint32_t W>
class v4i_swizzle4 {
private:
    union {
        struct {
            v4i vec;
        };

        struct {
            int32_t a[COORD_4D];
        };
    };

public:
    // Read-only: fast
    ML_INLINE operator C() const {
        return v4i_swizzle(vec, X, Y, Z, W);
    }

    // Read-write: most likely slow
    ML_SWIZZLE_4_OP(=, _mm_copy, v4i_swizzle)
    ML_SWIZZLE_4_OP(-=, _mm_sub_epi32, v4i_swizzle)
    ML_SWIZZLE_4_OP(+=, _mm_add_epi32, v4i_swizzle)
    ML_SWIZZLE_4_OP(*=, _mm_mullo_epi32, v4i_swizzle)
    ML_SWIZZLE_4_OP(/=, _mm_div_epi32, v4i_swizzle)
    ML_SWIZZLE_4_OP(%=, v4i_mod, v4i_swizzle)
    ML_SWIZZLE_4_OP(<<=, _mm_sllv_epi32, v4i_swizzle)
    ML_SWIZZLE_4_OP(>>=, _mm_srlv_epi32, v4i_swizzle)
    ML_SWIZZLE_4_OP(&=, _mm_and_si128, v4i_swizzle)
    ML_SWIZZLE_4_OP(|=, _mm_or_si128, v4i_swizzle)
    ML_SWIZZLE_4_OP(^=, _mm_xor_si128, v4i_swizzle)
};

// v4u

template <class C, uint32_t X, uint32_t Y>
class v4u_swizzle2 {
private:
    union {
        struct {
            v4i vec;
        };

        struct {
            uint32_t a[COORD_4D];
        };
    };

public:
    // Read-only: fast
    ML_INLINE operator C() const {
        return C(a[X], a[Y]);
    }

    // Read-write: most likely slow
    ML_SWIZZLE_2_OP(=, _mm_copy, v4i_swizzle)
    ML_SWIZZLE_2_OP(-=, _mm_sub_epi32, v4i_swizzle)
    ML_SWIZZLE_2_OP(+=, _mm_add_epi32, v4i_swizzle)
    ML_SWIZZLE_2_OP(*=, _mm_mullo_epi32, v4i_swizzle)
    ML_SWIZZLE_2_OP(/=, _mm_div_epu32, v4i_swizzle)
    ML_SWIZZLE_2_OP(%=, v4u_mod, v4i_swizzle)
    ML_SWIZZLE_2_OP(<<=, _mm_sllv_epi32, v4i_swizzle)
    ML_SWIZZLE_2_OP(>>=, _mm_srlv_epi32, v4i_swizzle)
    ML_SWIZZLE_2_OP(&=, _mm_and_si128, v4i_swizzle)
    ML_SWIZZLE_2_OP(|=, _mm_or_si128, v4i_swizzle)
    ML_SWIZZLE_2_OP(^=, _mm_xor_si128, v4i_swizzle)
};

template <class C, uint32_t X, uint32_t Y, uint32_t Z>
class v4u_swizzle3 {
private:
    union {
        struct {
            v4i vec;
        };

        struct {
            uint32_t a[COORD_4D];
        };
    };

public:
    // Read-only: fast
    ML_INLINE operator C() const {
        return v4i_swizzle(vec, X, Y, Z, 3);
    }

    // Read-write: most likely slow
    ML_SWIZZLE_3_OP(=, _mm_copy, v4i_swizzle)
    ML_SWIZZLE_3_OP(-=, _mm_sub_epi32, v4i_swizzle)
    ML_SWIZZLE_3_OP(+=, _mm_add_epi32, v4i_swizzle)
    ML_SWIZZLE_3_OP(*=, _mm_mullo_epi32, v4i_swizzle)
    ML_SWIZZLE_3_OP(/=, _mm_div_epu32, v4i_swizzle)
    ML_SWIZZLE_3_OP(%=, v4u_mod, v4i_swizzle)
    ML_SWIZZLE_3_OP(<<=, _mm_sllv_epi32, v4i_swizzle)
    ML_SWIZZLE_3_OP(>>=, _mm_srlv_epi32, v4i_swizzle)
    ML_SWIZZLE_3_OP(&=, _mm_and_si128, v4i_swizzle)
    ML_SWIZZLE_3_OP(|=, _mm_or_si128, v4i_swizzle)
    ML_SWIZZLE_3_OP(^=, _mm_xor_si128, v4i_swizzle)
};

template <class C, uint32_t X, uint32_t Y, uint32_t Z, uint32_t W>
class v4u_swizzle4 {
private:
    union {
        struct {
            v4i vec;
        };

        struct {
            uint32_t a[COORD_4D];
        };
    };

public:
    // Read-only: fast
    ML_INLINE operator C() const {
        return v4i_swizzle(vec, X, Y, Z, W);
    }

    // Read-write: most likely slow
    ML_SWIZZLE_4_OP(=, _mm_copy, v4i_swizzle)
    ML_SWIZZLE_4_OP(-=, _mm_sub_epi32, v4i_swizzle)
    ML_SWIZZLE_4_OP(+=, _mm_add_epi32, v4i_swizzle)
    ML_SWIZZLE_4_OP(*=, _mm_mullo_epi32, v4i_swizzle)
    ML_SWIZZLE_4_OP(/=, _mm_div_epu32, v4i_swizzle)
    ML_SWIZZLE_4_OP(%=, v4u_mod, v4i_swizzle)
    ML_SWIZZLE_4_OP(<<=, _mm_sllv_epi32, v4i_swizzle)
    ML_SWIZZLE_4_OP(>>=, _mm_srlv_epi32, v4i_swizzle)
    ML_SWIZZLE_4_OP(&=, _mm_and_si128, v4i_swizzle)
    ML_SWIZZLE_4_OP(|=, _mm_or_si128, v4i_swizzle)
    ML_SWIZZLE_4_OP(^=, _mm_xor_si128, v4i_swizzle)
};

// v4f

template <class C, uint32_t X, uint32_t Y>
class v4f_swizzle2 {
private:
    union {
        struct {
            v4f vec;
        };

        struct {
            float a[COORD_4D];
        };
    };

public:
    // Read-only: fast
    ML_INLINE operator C() const {
        return C(a[X], a[Y]);
    }

    // Read-write: most likely slow
    ML_SWIZZLE_2_OP(=, _mm_copy, v4f_swizzle)
    ML_SWIZZLE_2_OP(-=, _mm_sub_ps, v4f_swizzle)
    ML_SWIZZLE_2_OP(+=, _mm_add_ps, v4f_swizzle)
    ML_SWIZZLE_2_OP(*=, _mm_mul_ps, v4f_swizzle)
    ML_SWIZZLE_2_OP(/=, _mm_div_ps, v4f_swizzle)
};

template <class C, uint32_t X, uint32_t Y, uint32_t Z>
class v4f_swizzle3 {
private:
    union {
        struct {
            v4f vec;
        };

        struct {
            float a[COORD_4D];
        };
    };

public:
    // Read-only: fast
    ML_INLINE operator C() const {
        return v4f_swizzle(vec, X, Y, Z, 3);
    }

    // Read-write: most likely slow
    ML_SWIZZLE_3_OP(=, _mm_copy, v4f_swizzle)
    ML_SWIZZLE_3_OP(-=, _mm_sub_ps, v4f_swizzle)
    ML_SWIZZLE_3_OP(+=, _mm_add_ps, v4f_swizzle)
    ML_SWIZZLE_3_OP(*=, _mm_mul_ps, v4f_swizzle)
    ML_SWIZZLE_3_OP(/=, _mm_div_ps, v4f_swizzle)
};

template <class C, uint32_t X, uint32_t Y, uint32_t Z, uint32_t W>
class v4f_swizzle4 {
private:
    union {
        struct {
            v4f vec;
        };

        struct {
            float a[COORD_4D];
        };
    };

public:
    // Read-only: fast
    ML_INLINE operator C() const {
        return v4f_swizzle(vec, X, Y, Z, W);
    }

    // Read-write: most likely slow
    ML_SWIZZLE_4_OP(=, _mm_copy, v4f_swizzle)
    ML_SWIZZLE_4_OP(-=, _mm_sub_ps, v4f_swizzle)
    ML_SWIZZLE_4_OP(+=, _mm_add_ps, v4f_swizzle)
    ML_SWIZZLE_4_OP(*=, _mm_mul_ps, v4f_swizzle)
    ML_SWIZZLE_4_OP(/=, _mm_div_ps, v4f_swizzle)
};

// v4d

template <class C, uint32_t X, uint32_t Y>
class v4d_swizzle2 {
private:
    union {
        struct {
            v4d vec;
        };

        struct {
            double a[COORD_4D];
        };
    };

public:
    // Read-only: fast
    ML_INLINE operator C() const {
        return C(a[X], a[Y]);
    }

    // Read-write: most likely slow
    ML_SWIZZLE_2_OP(=, _mm_copy, v4d_swizzle)
    ML_SWIZZLE_2_OP(-=, _mm256_sub_pd, v4d_swizzle)
    ML_SWIZZLE_2_OP(+=, _mm256_add_pd, v4d_swizzle)
    ML_SWIZZLE_2_OP(*=, _mm256_mul_pd, v4d_swizzle)
    ML_SWIZZLE_2_OP(/=, _mm256_div_pd, v4d_swizzle)
};

template <class C, uint32_t X, uint32_t Y, uint32_t Z>
class v4d_swizzle3 {
private:
    union {
        struct {
            v4d vec;
        };

        struct {
            double a[COORD_4D];
        };
    };

public:
    // Read-only: fast
    ML_INLINE operator C() const {
        return v4d_swizzle(vec, X, Y, Z, 3);
    }

    // Read-write: most likely slow
    ML_SWIZZLE_3_OP(=, _mm_copy, v4d_swizzle)
    ML_SWIZZLE_3_OP(-=, _mm256_sub_pd, v4d_swizzle)
    ML_SWIZZLE_3_OP(+=, _mm256_add_pd, v4d_swizzle)
    ML_SWIZZLE_3_OP(*=, _mm256_mul_pd, v4d_swizzle)
    ML_SWIZZLE_3_OP(/=, _mm256_div_pd, v4d_swizzle)
};

template <class C, uint32_t X, uint32_t Y, uint32_t Z, uint32_t W>
class v4d_swizzle4 {
private:
    union {
        struct {
            v4d vec;
        };

        struct {
            double a[COORD_4D];
        };
    };

public:
    // Read-only: fast
    ML_INLINE operator C() const {
        return v4d_swizzle(vec, X, Y, Z, W);
    }

    // Read-write: most likely slow
    ML_SWIZZLE_4_OP(=, _mm_copy, v4d_swizzle)
    ML_SWIZZLE_4_OP(-=, _mm256_sub_pd, v4d_swizzle)
    ML_SWIZZLE_4_OP(+=, _mm256_add_pd, v4d_swizzle)
    ML_SWIZZLE_4_OP(*=, _mm256_mul_pd, v4d_swizzle)
    ML_SWIZZLE_4_OP(/=, _mm256_div_pd, v4d_swizzle)
};

#undef ML_SWIZZLE_2_OP
#undef ML_SWIZZLE_3_OP
#undef ML_SWIZZLE_4_OP

// swizzles

#define ML_SWIZZLE_2(C, T) \
    swizzle<C, T, 0, 0> xx; \
    swizzle<C, T, 0, 1> xy; \
    swizzle<C, T, 1, 0> yx; \
    swizzle<C, T, 1, 1> yy

#define ML_SWIZZLE_3(S2, C2, S3, C3) \
    S2<C2, 0, 0> xx; \
    S2<C2, 0, 1> xy; \
    S2<C2, 0, 2> xz; \
    S2<C2, 1, 0> yx; \
    S2<C2, 1, 1> yy; \
    S2<C2, 1, 2> yz; \
    S2<C2, 2, 0> zx; \
    S2<C2, 2, 1> zy; \
    S2<C2, 2, 2> zz; \
    S3<C3, 0, 0, 0> xxx; \
    S3<C3, 0, 0, 1> xxy; \
    S3<C3, 0, 0, 2> xxz; \
    S3<C3, 0, 1, 0> xyx; \
    S3<C3, 0, 1, 1> xyy; \
    S3<C3, 0, 1, 2> xyz; \
    S3<C3, 0, 2, 0> xzx; \
    S3<C3, 0, 2, 1> xzy; \
    S3<C3, 0, 2, 2> xzz; \
    S3<C3, 1, 0, 0> yxx; \
    S3<C3, 1, 0, 1> yxy; \
    S3<C3, 1, 0, 2> yxz; \
    S3<C3, 1, 1, 0> yyx; \
    S3<C3, 1, 1, 1> yyy; \
    S3<C3, 1, 1, 2> yyz; \
    S3<C3, 1, 2, 0> yzx; \
    S3<C3, 1, 2, 1> yzy; \
    S3<C3, 1, 2, 2> yzz; \
    S3<C3, 2, 0, 0> zxx; \
    S3<C3, 2, 0, 1> zxy; \
    S3<C3, 2, 0, 2> zxz; \
    S3<C3, 2, 1, 0> zyx; \
    S3<C3, 2, 1, 1> zyy; \
    S3<C3, 2, 1, 2> zyz; \
    S3<C3, 2, 2, 0> zzx; \
    S3<C3, 2, 2, 1> zzy; \
    S3<C3, 2, 2, 2> zzz

#define ML_SWIZZLE_4(S2, C2, S3, C3, S4, C4) \
    S2<C2, 0, 0> xx; \
    S2<C2, 0, 1> xy; \
    S2<C2, 0, 2> xz; \
    S2<C2, 0, 3> xw; \
    S2<C2, 1, 0> yx; \
    S2<C2, 1, 1> yy; \
    S2<C2, 1, 2> yz; \
    S2<C2, 1, 3> yw; \
    S2<C2, 2, 0> zx; \
    S2<C2, 2, 1> zy; \
    S2<C2, 2, 2> zz; \
    S2<C2, 2, 3> zw; \
    S2<C2, 3, 0> wx; \
    S2<C2, 3, 1> wy; \
    S2<C2, 3, 2> wz; \
    S2<C2, 3, 3> ww; \
    S3<C3, 0, 0, 0> xxx; \
    S3<C3, 0, 0, 1> xxy; \
    S3<C3, 0, 0, 2> xxz; \
    S3<C3, 0, 0, 3> xxw; \
    S3<C3, 0, 1, 0> xyx; \
    S3<C3, 0, 1, 1> xyy; \
    S3<C3, 0, 1, 2> xyz; \
    S3<C3, 0, 1, 3> xyw; \
    S3<C3, 0, 2, 0> xzx; \
    S3<C3, 0, 2, 1> xzy; \
    S3<C3, 0, 2, 2> xzz; \
    S3<C3, 0, 2, 3> xzw; \
    S3<C3, 0, 3, 0> xwx; \
    S3<C3, 0, 3, 1> xwy; \
    S3<C3, 0, 3, 2> xwz; \
    S3<C3, 0, 3, 3> xww; \
    S3<C3, 1, 0, 0> yxx; \
    S3<C3, 1, 0, 1> yxy; \
    S3<C3, 1, 0, 2> yxz; \
    S3<C3, 1, 0, 3> yxw; \
    S3<C3, 1, 1, 0> yyx; \
    S3<C3, 1, 1, 1> yyy; \
    S3<C3, 1, 1, 2> yyz; \
    S3<C3, 1, 1, 3> yyw; \
    S3<C3, 1, 2, 0> yzx; \
    S3<C3, 1, 2, 1> yzy; \
    S3<C3, 1, 2, 2> yzz; \
    S3<C3, 1, 2, 3> yzw; \
    S3<C3, 1, 3, 0> ywx; \
    S3<C3, 1, 3, 1> ywy; \
    S3<C3, 1, 3, 2> ywz; \
    S3<C3, 1, 3, 3> yww; \
    S3<C3, 2, 0, 0> zxx; \
    S3<C3, 2, 0, 1> zxy; \
    S3<C3, 2, 0, 2> zxz; \
    S3<C3, 2, 0, 3> zxw; \
    S3<C3, 2, 1, 0> zyx; \
    S3<C3, 2, 1, 1> zyy; \
    S3<C3, 2, 1, 2> zyz; \
    S3<C3, 2, 1, 3> zyw; \
    S3<C3, 2, 2, 0> zzx; \
    S3<C3, 2, 2, 1> zzy; \
    S3<C3, 2, 2, 2> zzz; \
    S3<C3, 2, 2, 3> zzw; \
    S3<C3, 2, 3, 0> zwx; \
    S3<C3, 2, 3, 1> zwy; \
    S3<C3, 2, 3, 2> zwz; \
    S3<C3, 2, 3, 3> zww; \
    S3<C3, 3, 0, 0> wxx; \
    S3<C3, 3, 0, 1> wxy; \
    S3<C3, 3, 0, 2> wxz; \
    S3<C3, 3, 0, 3> wxw; \
    S3<C3, 3, 1, 0> wyx; \
    S3<C3, 3, 1, 1> wyy; \
    S3<C3, 3, 1, 2> wyz; \
    S3<C3, 3, 1, 3> wyw; \
    S3<C3, 3, 2, 0> wzx; \
    S3<C3, 3, 2, 1> wzy; \
    S3<C3, 3, 2, 2> wzz; \
    S3<C3, 3, 2, 3> wzw; \
    S3<C3, 3, 3, 0> wwx; \
    S3<C3, 3, 3, 1> wwy; \
    S3<C3, 3, 3, 2> wwz; \
    S3<C3, 3, 3, 3> www; \
    S4<C4, 0, 0, 0, 0> xxxx; \
    S4<C4, 0, 0, 0, 1> xxxy; \
    S4<C4, 0, 0, 0, 2> xxxz; \
    S4<C4, 0, 0, 0, 3> xxxw; \
    S4<C4, 0, 0, 1, 0> xxyx; \
    S4<C4, 0, 0, 1, 1> xxyy; \
    S4<C4, 0, 0, 1, 2> xxyz; \
    S4<C4, 0, 0, 1, 3> xxyw; \
    S4<C4, 0, 0, 2, 0> xxzx; \
    S4<C4, 0, 0, 2, 1> xxzy; \
    S4<C4, 0, 0, 2, 2> xxzz; \
    S4<C4, 0, 0, 2, 3> xxzw; \
    S4<C4, 0, 0, 3, 0> xxwx; \
    S4<C4, 0, 0, 3, 1> xxwy; \
    S4<C4, 0, 0, 3, 2> xxwz; \
    S4<C4, 0, 0, 3, 3> xxww; \
    S4<C4, 0, 1, 0, 0> xyxx; \
    S4<C4, 0, 1, 0, 1> xyxy; \
    S4<C4, 0, 1, 0, 2> xyxz; \
    S4<C4, 0, 1, 0, 3> xyxw; \
    S4<C4, 0, 1, 1, 0> xyyx; \
    S4<C4, 0, 1, 1, 1> xyyy; \
    S4<C4, 0, 1, 1, 2> xyyz; \
    S4<C4, 0, 1, 1, 3> xyyw; \
    S4<C4, 0, 1, 2, 0> xyzx; \
    S4<C4, 0, 1, 2, 1> xyzy; \
    S4<C4, 0, 1, 2, 2> xyzz; \
    S4<C4, 0, 1, 2, 3> xyzw; \
    S4<C4, 0, 1, 3, 0> xywx; \
    S4<C4, 0, 1, 3, 1> xywy; \
    S4<C4, 0, 1, 3, 2> xywz; \
    S4<C4, 0, 1, 3, 3> xyww; \
    S4<C4, 0, 2, 0, 0> xzxx; \
    S4<C4, 0, 2, 0, 1> xzxy; \
    S4<C4, 0, 2, 0, 2> xzxz; \
    S4<C4, 0, 2, 0, 3> xzxw; \
    S4<C4, 0, 2, 1, 0> xzyx; \
    S4<C4, 0, 2, 1, 1> xzyy; \
    S4<C4, 0, 2, 1, 2> xzyz; \
    S4<C4, 0, 2, 1, 3> xzyw; \
    S4<C4, 0, 2, 2, 0> xzzx; \
    S4<C4, 0, 2, 2, 1> xzzy; \
    S4<C4, 0, 2, 2, 2> xzzz; \
    S4<C4, 0, 2, 2, 3> xzzw; \
    S4<C4, 0, 2, 3, 0> xzwx; \
    S4<C4, 0, 2, 3, 1> xzwy; \
    S4<C4, 0, 2, 3, 2> xzwz; \
    S4<C4, 0, 2, 3, 3> xzww; \
    S4<C4, 0, 3, 0, 0> xwxx; \
    S4<C4, 0, 3, 0, 1> xwxy; \
    S4<C4, 0, 3, 0, 2> xwxz; \
    S4<C4, 0, 3, 0, 3> xwxw; \
    S4<C4, 0, 3, 1, 0> xwyx; \
    S4<C4, 0, 3, 1, 1> xwyy; \
    S4<C4, 0, 3, 1, 2> xwyz; \
    S4<C4, 0, 3, 1, 3> xwyw; \
    S4<C4, 0, 3, 2, 0> xwzx; \
    S4<C4, 0, 3, 2, 1> xwzy; \
    S4<C4, 0, 3, 2, 2> xwzz; \
    S4<C4, 0, 3, 2, 3> xwzw; \
    S4<C4, 0, 3, 3, 0> xwwx; \
    S4<C4, 0, 3, 3, 1> xwwy; \
    S4<C4, 0, 3, 3, 2> xwwz; \
    S4<C4, 0, 3, 3, 3> xwww; \
    S4<C4, 1, 0, 0, 0> yxxx; \
    S4<C4, 1, 0, 0, 1> yxxy; \
    S4<C4, 1, 0, 0, 2> yxxz; \
    S4<C4, 1, 0, 0, 3> yxxw; \
    S4<C4, 1, 0, 1, 0> yxyx; \
    S4<C4, 1, 0, 1, 1> yxyy; \
    S4<C4, 1, 0, 1, 2> yxyz; \
    S4<C4, 1, 0, 1, 3> yxyw; \
    S4<C4, 1, 0, 2, 0> yxzx; \
    S4<C4, 1, 0, 2, 1> yxzy; \
    S4<C4, 1, 0, 2, 2> yxzz; \
    S4<C4, 1, 0, 2, 3> yxzw; \
    S4<C4, 1, 0, 3, 0> yxwx; \
    S4<C4, 1, 0, 3, 1> yxwy; \
    S4<C4, 1, 0, 3, 2> yxwz; \
    S4<C4, 1, 0, 3, 3> yxww; \
    S4<C4, 1, 1, 0, 0> yyxx; \
    S4<C4, 1, 1, 0, 1> yyxy; \
    S4<C4, 1, 1, 0, 2> yyxz; \
    S4<C4, 1, 1, 0, 3> yyxw; \
    S4<C4, 1, 1, 1, 0> yyyx; \
    S4<C4, 1, 1, 1, 1> yyyy; \
    S4<C4, 1, 1, 1, 2> yyyz; \
    S4<C4, 1, 1, 1, 3> yyyw; \
    S4<C4, 1, 1, 2, 0> yyzx; \
    S4<C4, 1, 1, 2, 1> yyzy; \
    S4<C4, 1, 1, 2, 2> yyzz; \
    S4<C4, 1, 1, 2, 3> yyzw; \
    S4<C4, 1, 1, 3, 0> yywx; \
    S4<C4, 1, 1, 3, 1> yywy; \
    S4<C4, 1, 1, 3, 2> yywz; \
    S4<C4, 1, 1, 3, 3> yyww; \
    S4<C4, 1, 2, 0, 0> yzxx; \
    S4<C4, 1, 2, 0, 1> yzxy; \
    S4<C4, 1, 2, 0, 2> yzxz; \
    S4<C4, 1, 2, 0, 3> yzxw; \
    S4<C4, 1, 2, 1, 0> yzyx; \
    S4<C4, 1, 2, 1, 1> yzyy; \
    S4<C4, 1, 2, 1, 2> yzyz; \
    S4<C4, 1, 2, 1, 3> yzyw; \
    S4<C4, 1, 2, 2, 0> yzzx; \
    S4<C4, 1, 2, 2, 1> yzzy; \
    S4<C4, 1, 2, 2, 2> yzzz; \
    S4<C4, 1, 2, 2, 3> yzzw; \
    S4<C4, 1, 2, 3, 0> yzwx; \
    S4<C4, 1, 2, 3, 1> yzwy; \
    S4<C4, 1, 2, 3, 2> yzwz; \
    S4<C4, 1, 2, 3, 3> yzww; \
    S4<C4, 1, 3, 0, 0> ywxx; \
    S4<C4, 1, 3, 0, 1> ywxy; \
    S4<C4, 1, 3, 0, 2> ywxz; \
    S4<C4, 1, 3, 0, 3> ywxw; \
    S4<C4, 1, 3, 1, 0> ywyx; \
    S4<C4, 1, 3, 1, 1> ywyy; \
    S4<C4, 1, 3, 1, 2> ywyz; \
    S4<C4, 1, 3, 1, 3> ywyw; \
    S4<C4, 1, 3, 2, 0> ywzx; \
    S4<C4, 1, 3, 2, 1> ywzy; \
    S4<C4, 1, 3, 2, 2> ywzz; \
    S4<C4, 1, 3, 2, 3> ywzw; \
    S4<C4, 1, 3, 3, 0> ywwx; \
    S4<C4, 1, 3, 3, 1> ywwy; \
    S4<C4, 1, 3, 3, 2> ywwz; \
    S4<C4, 1, 3, 3, 3> ywww; \
    S4<C4, 2, 0, 0, 0> zxxx; \
    S4<C4, 2, 0, 0, 1> zxxy; \
    S4<C4, 2, 0, 0, 2> zxxz; \
    S4<C4, 2, 0, 0, 3> zxxw; \
    S4<C4, 2, 0, 1, 0> zxyx; \
    S4<C4, 2, 0, 1, 1> zxyy; \
    S4<C4, 2, 0, 1, 2> zxyz; \
    S4<C4, 2, 0, 1, 3> zxyw; \
    S4<C4, 2, 0, 2, 0> zxzx; \
    S4<C4, 2, 0, 2, 1> zxzy; \
    S4<C4, 2, 0, 2, 2> zxzz; \
    S4<C4, 2, 0, 2, 3> zxzw; \
    S4<C4, 2, 0, 3, 0> zxwx; \
    S4<C4, 2, 0, 3, 1> zxwy; \
    S4<C4, 2, 0, 3, 2> zxwz; \
    S4<C4, 2, 0, 3, 3> zxww; \
    S4<C4, 2, 1, 0, 0> zyxx; \
    S4<C4, 2, 1, 0, 1> zyxy; \
    S4<C4, 2, 1, 0, 2> zyxz; \
    S4<C4, 2, 1, 0, 3> zyxw; \
    S4<C4, 2, 1, 1, 0> zyyx; \
    S4<C4, 2, 1, 1, 1> zyyy; \
    S4<C4, 2, 1, 1, 2> zyyz; \
    S4<C4, 2, 1, 1, 3> zyyw; \
    S4<C4, 2, 1, 2, 0> zyzx; \
    S4<C4, 2, 1, 2, 1> zyzy; \
    S4<C4, 2, 1, 2, 2> zyzz; \
    S4<C4, 2, 1, 2, 3> zyzw; \
    S4<C4, 2, 1, 3, 0> zywx; \
    S4<C4, 2, 1, 3, 1> zywy; \
    S4<C4, 2, 1, 3, 2> zywz; \
    S4<C4, 2, 1, 3, 3> zyww; \
    S4<C4, 2, 2, 0, 0> zzxx; \
    S4<C4, 2, 2, 0, 1> zzxy; \
    S4<C4, 2, 2, 0, 2> zzxz; \
    S4<C4, 2, 2, 0, 3> zzxw; \
    S4<C4, 2, 2, 1, 0> zzyx; \
    S4<C4, 2, 2, 1, 1> zzyy; \
    S4<C4, 2, 2, 1, 2> zzyz; \
    S4<C4, 2, 2, 1, 3> zzyw; \
    S4<C4, 2, 2, 2, 0> zzzx; \
    S4<C4, 2, 2, 2, 1> zzzy; \
    S4<C4, 2, 2, 2, 2> zzzz; \
    S4<C4, 2, 2, 2, 3> zzzw; \
    S4<C4, 2, 2, 3, 0> zzwx; \
    S4<C4, 2, 2, 3, 1> zzwy; \
    S4<C4, 2, 2, 3, 2> zzwz; \
    S4<C4, 2, 2, 3, 3> zzww; \
    S4<C4, 2, 3, 0, 0> zwxx; \
    S4<C4, 2, 3, 0, 1> zwxy; \
    S4<C4, 2, 3, 0, 2> zwxz; \
    S4<C4, 2, 3, 0, 3> zwxw; \
    S4<C4, 2, 3, 1, 0> zwyx; \
    S4<C4, 2, 3, 1, 1> zwyy; \
    S4<C4, 2, 3, 1, 2> zwyz; \
    S4<C4, 2, 3, 1, 3> zwyw; \
    S4<C4, 2, 3, 2, 0> zwzx; \
    S4<C4, 2, 3, 2, 1> zwzy; \
    S4<C4, 2, 3, 2, 2> zwzz; \
    S4<C4, 2, 3, 2, 3> zwzw; \
    S4<C4, 2, 3, 3, 0> zwwx; \
    S4<C4, 2, 3, 3, 1> zwwy; \
    S4<C4, 2, 3, 3, 2> zwwz; \
    S4<C4, 2, 3, 3, 3> zwww; \
    S4<C4, 3, 0, 0, 0> wxxx; \
    S4<C4, 3, 0, 0, 1> wxxy; \
    S4<C4, 3, 0, 0, 2> wxxz; \
    S4<C4, 3, 0, 0, 3> wxxw; \
    S4<C4, 3, 0, 1, 0> wxyx; \
    S4<C4, 3, 0, 1, 1> wxyy; \
    S4<C4, 3, 0, 1, 2> wxyz; \
    S4<C4, 3, 0, 1, 3> wxyw; \
    S4<C4, 3, 0, 2, 0> wxzx; \
    S4<C4, 3, 0, 2, 1> wxzy; \
    S4<C4, 3, 0, 2, 2> wxzz; \
    S4<C4, 3, 0, 2, 3> wxzw; \
    S4<C4, 3, 0, 3, 0> wxwx; \
    S4<C4, 3, 0, 3, 1> wxwy; \
    S4<C4, 3, 0, 3, 2> wxwz; \
    S4<C4, 3, 0, 3, 3> wxww; \
    S4<C4, 3, 1, 0, 0> wyxx; \
    S4<C4, 3, 1, 0, 1> wyxy; \
    S4<C4, 3, 1, 0, 2> wyxz; \
    S4<C4, 3, 1, 0, 3> wyxw; \
    S4<C4, 3, 1, 1, 0> wyyx; \
    S4<C4, 3, 1, 1, 1> wyyy; \
    S4<C4, 3, 1, 1, 2> wyyz; \
    S4<C4, 3, 1, 1, 3> wyyw; \
    S4<C4, 3, 1, 2, 0> wyzx; \
    S4<C4, 3, 1, 2, 1> wyzy; \
    S4<C4, 3, 1, 2, 2> wyzz; \
    S4<C4, 3, 1, 2, 3> wyzw; \
    S4<C4, 3, 1, 3, 0> wywx; \
    S4<C4, 3, 1, 3, 1> wywy; \
    S4<C4, 3, 1, 3, 2> wywz; \
    S4<C4, 3, 1, 3, 3> wyww; \
    S4<C4, 3, 2, 0, 0> wzxx; \
    S4<C4, 3, 2, 0, 1> wzxy; \
    S4<C4, 3, 2, 0, 2> wzxz; \
    S4<C4, 3, 2, 0, 3> wzxw; \
    S4<C4, 3, 2, 1, 0> wzyx; \
    S4<C4, 3, 2, 1, 1> wzyy; \
    S4<C4, 3, 2, 1, 2> wzyz; \
    S4<C4, 3, 2, 1, 3> wzyw; \
    S4<C4, 3, 2, 2, 0> wzzx; \
    S4<C4, 3, 2, 2, 1> wzzy; \
    S4<C4, 3, 2, 2, 2> wzzz; \
    S4<C4, 3, 2, 2, 3> wzzw; \
    S4<C4, 3, 2, 3, 0> wzwx; \
    S4<C4, 3, 2, 3, 1> wzwy; \
    S4<C4, 3, 2, 3, 2> wzwz; \
    S4<C4, 3, 2, 3, 3> wzww; \
    S4<C4, 3, 3, 0, 0> wwxx; \
    S4<C4, 3, 3, 0, 1> wwxy; \
    S4<C4, 3, 3, 0, 2> wwxz; \
    S4<C4, 3, 3, 0, 3> wwxw; \
    S4<C4, 3, 3, 1, 0> wwyx; \
    S4<C4, 3, 3, 1, 1> wwyy; \
    S4<C4, 3, 3, 1, 2> wwyz; \
    S4<C4, 3, 3, 1, 3> wwyw; \
    S4<C4, 3, 3, 2, 0> wwzx; \
    S4<C4, 3, 3, 2, 1> wwzy; \
    S4<C4, 3, 3, 2, 2> wwzz; \
    S4<C4, 3, 3, 2, 3> wwzw; \
    S4<C4, 3, 3, 3, 0> wwwx; \
    S4<C4, 3, 3, 3, 1> wwwy; \
    S4<C4, 3, 3, 3, 2> wwwz; \
    S4<C4, 3, 3, 3, 3> wwww
