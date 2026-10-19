// © 2021 NVIDIA Corporation

#define Test_Eps         0.00001
#define Test_ConstantEps 0.001 // hack for "fmod"

#define TestEqual_x2(C, T) ML_Assert(all(C((T)x1, (T)y1) == C((T)x1, (T)y1)))
#define TestEqual_x3(C, T) ML_Assert(all(C((T)x1, (T)y1, (T)z1) == C((T)x1, (T)y1, (T)z1)))
#define TestEqual_x4(C, T) ML_Assert(all(C((T)x1, (T)y1, (T)z1, (T)w1) == C((T)x1, (T)y1, (T)z1, (T)w1)))

#define TestNotEqual_x2(C, T) ML_Assert(any(C((T)x1, (T)y1) != C((T)y1, (T)x1)))
#define TestNotEqual_x3(C, T) ML_Assert(any(C((T)x1, (T)y1, (T)z1) != C((T)z1, (T)x1, (T)y1)))
#define TestNotEqual_x4(C, T) ML_Assert(any(C((T)x1, (T)y1, (T)z1, (T)w1) != C((T)w1, (T)z1, (T)y1, (T)x1)))

#define TestOp_x2(C, T, op) ML_Assert(all((C((T)x1, (T)y1) op C((T)x2, (T)y2)) == C((T)x1 op(T) x2, (T)y1 op(T) y2)))
#define TestOp_x3(C, T, op) ML_Assert(all((C((T)x1, (T)y1, (T)z1) op C((T)x2, (T)y2, (T)z2)) == C((T)x1 op(T) x2, (T)y1 op(T) y2, (T)z1 op(T) z2)))
#define TestOp_x4(C, T, op) ML_Assert(all((C((T)x1, (T)y1, (T)z1, (T)w1) op C((T)x2, (T)y2, (T)z2, (T)w2)) == C((T)x1 op(T) x2, (T)y1 op(T) y2, (T)z1 op(T) z2, (T)w1 op(T) w2)))

#define Test1_x2(C, T, func) ML_Assert(all(func(C((T)x1, (T)y1)) == C(func((T)x1), func((T)y1))))
#define Test1_x3(C, T, func) ML_Assert(all(func(C((T)x1, (T)y1, (T)z1)) == C(func((T)x1), func((T)y1), func((T)z1))))
#define Test1_x4(C, T, func) ML_Assert(all(func(C((T)x1, (T)y1, (T)z1, (T)w1)) == C(func((T)x1), func((T)y1), func((T)z1), func((T)w1))))

#define Test2_x2(C, T, func) ML_Assert(all(func(C((T)x1, (T)y1), C((T)x2, (T)y2)) == C(func((T)x1, (T)x2), func((T)y1, (T)y2))))
#define Test2_x3(C, T, func) ML_Assert(all(func(C((T)x1, (T)y1, (T)z1), C((T)x2, (T)y2, (T)z2)) == C(func((T)x1, (T)x2), func((T)y1, (T)y2), func((T)z1, (T)z2))))

#define Test2_x4(C, T, func) \
    ML_Assert(all(func(C((T)x1, (T)y1, (T)z1, (T)w1), C((T)x2, (T)y2, (T)z2, (T)w2)) == C(func((T)x1, (T)x2), func((T)y1, (T)y2), func((T)z1, (T)z2), func((T)w1, (T)w2))))

#define Test1_x3_eps(C, T, func) \
    ML_Assert(all(abs(func(C((T)x1, (T)y1, (T)z1)) - C(func((T)x1), func((T)y1), func((T)z1))) <= abs(C(func((T)x1), func((T)y1), func((T)z1))) * (T)Test_Eps))

#define Test1_x4_eps(C, T, func) \
    ML_Assert(all(abs(func(C((T)x1, (T)y1, (T)z1, (T)w1)) - C(func((T)x1), func((T)y1), func((T)z1), func((T)w1))) <= abs(C(func((T)x1), func((T)y1), func((T)z1), func((T)w1))) * (T)Test_Eps))

#define Test2_x3_eps(C, T, func) \
    ML_Assert(all(abs(func(C((T)x1, (T)y1, (T)z1), C((T)x2, (T)y2, (T)z2)) - C(func((T)x1, (T)x2), func((T)y1, (T)y2), func((T)z1, (T)z2))) <= abs(C(func((T)x1, (T)x2), func((T)y1, (T)y2), func((T)z1, (T)z2))) * (T)Test_Eps))
#define Test2_x4_eps(C, T, func) \
    ML_Assert(all(abs(func(C((T)x1, (T)y1, (T)z1, (T)w1), C((T)x2, (T)y2, (T)z2, (T)w2)) - C(func((T)x1, (T)x2), func((T)y1, (T)y2), func((T)z1, (T)z2), func((T)w1, (T)w2))) <= abs(C(func((T)x1, (T)x2), func((T)y1, (T)y2), func((T)z1, (T)z2), func((T)w1, (T)w2))) * (T)Test_Eps))

#define Test1_x3_ceps(C, T, func) \
    ML_Assert(all(abs(func(C((T)x1, (T)y1, (T)z1)) - C(func((T)x1), func((T)y1), func((T)z1))) <= abs(C(func((T)x1), func((T)y1), func((T)z1))) * (T)Test_Eps))

#define Test1_x4_ceps(C, T, func) \
    ML_Assert(all(abs(func(C((T)x1, (T)y1, (T)z1, (T)w1)) - C(func((T)x1), func((T)y1), func((T)z1), func((T)w1))) <= abs(C(func((T)x1), func((T)y1), func((T)z1), func((T)w1))) * (T)Test_Eps))

#define Test2_x3_ceps(C, T, func) \
    ML_Assert(all(abs(func(C((T)x1, (T)y1, (T)z1), C((T)x2, (T)y2, (T)z2)) - C(func((T)x1, (T)x2), func((T)y1, (T)y2), func((T)z1, (T)z2))) <= (T)Test_ConstantEps))
#define Test2_x4_ceps(C, T, func) \
    ML_Assert(all(abs(func(C((T)x1, (T)y1, (T)z1, (T)w1), C((T)x2, (T)y2, (T)z2, (T)w2)) - C(func((T)x1, (T)x2), func((T)y1, (T)y2), func((T)z1, (T)z2), func((T)w1, (T)w2))) <= (T)Test_ConstantEps))

#include "../ml.hlsli"

#ifdef ML_NAMESPACE
namespace ml {
#endif

void ML_Tests() {
    { // Quatf arithmetic and construction
        const Quatf identity(1.0f);
        const Quatf a(1.0f, 2.0f, 3.0f, 4.0f);
        const Quatf b(-2.0f, 1.0f, 0.5f, 3.0f);
        ML_Assert(all(Quatf(float3(1.0f)) == Quatf(1.0f, 1.0f, 1.0f, 0.0f)));
        ML_Assert(all(Quatf(float3(1.0f, 1.0f, 1.0f)) == Quatf(float3(1.0f))));
        ML_Assert(all((a + 2.0f) == (a + Quatf(2.0f))));
        ML_Assert(all((2.0f + a) == (Quatf(2.0f) + a)));
        ML_Assert(all((a - 2.0f) == (a - Quatf(2.0f))));
        ML_Assert(all((2.0f - a) == (Quatf(2.0f) - a)));
        ML_Assert(abs(a.Norm() - 30.0f) < Test_Eps);
        ML_Assert(all(abs((a * a.Inverse()).q - identity.q) < float4(Test_Eps)));
        ML_Assert(all(abs((a / b).q - (a * b.Inverse()).q) < float4(Test_Eps)));
        ML_Assert(all(abs((2.0f / a).q - (Quatf(2.0f) / a).q) < float4(Test_Eps)));

        Quatf c = a;
        c += 2.0f;
        ML_Assert(all(c == (a + 2.0f)));
        c -= 2.0f;
        ML_Assert(all(c == a));
        c *= b;
        ML_Assert(all(abs(c.q - (a * b).q) < float4(Test_Eps)));
        c /= b;
        ML_Assert(all(abs(c.q - a.q) < float4(Test_Eps)));
        c = a;
        c *= c;
        ML_Assert(all(abs(c.q - (a * a).q) < float4(Test_Eps)));
        c = a;
        c /= c;
        ML_Assert(all(abs(c.q - identity.q) < float4(Test_Eps)));
        c = a;
        c *= 2.0f;
        c /= 2.0f;
        ML_Assert(all(c == a));
    }

    { // Quatf interpolation and rotation-vector limits
        const Quatf identity(1.0f);
        const Quatf quarterTurn(1.5707963267948966f, float3(0.0f, 0.0f, 1.0f));
        const float3 v(1.0f, 0.0f, 0.0f);
        ML_Assert(all(identity.GetRotation() == float3(0.0f)));
        ML_Assert(all((-identity).GetRotation() == float3(0.0f)));
        ML_Assert(all(abs(quarterTurn.GetRotation() - float3(0.0f, 0.0f, 1.5707963267948966f)) < float3(Test_Eps)));
        const Quatf tiny(1e-8f, float3(0.0f, 1.0f, 0.0f));
        ML_Assert(abs(tiny.GetRotation().y - 1e-8f) < 1e-12f);

        for (uint32_t i = 0; i <= 16; i++) {
            const float t = float(i) / 16.0f;
            const float3 expected = Quatf(t * 1.5707963267948966f, float3(0.0f, 0.0f, 1.0f)) * v;
            ML_Assert(all(abs(Slerp(identity, quarterTurn, t) * v - expected) < float3(Test_Eps)));
            ML_Assert(all(abs(Slerp(identity, -quarterTurn, t) * v - expected) < float3(Test_Eps)));
            ML_Assert(abs(Slerp(identity, -identity, t).Norm() - 1.0f) < Test_Eps);
            ML_Assert(abs(lerp(identity, -identity, t).Norm() - 1.0f) < Test_Eps);
            ML_Assert(all(abs(Slerp(identity, tiny, t).GetRotation() - float3(0.0f, t * 1e-8f, 0.0f)) < float3(1e-12f)));
        }

        const Quatf slightlyLong = -identity * 1.0000001f;
        ML_Assert(abs(Slerp(identity, slightlyLong, 0.5f).Norm() - 1.0f) < Test_Eps);
        ML_Assert(all(Slerp(identity, quarterTurn, -1.0f) == identity));
        ML_Assert(all(Slerp(identity, quarterTurn, 2.0f) == quarterTurn));
        ML_Assert(all(lerp(identity, quarterTurn, -1.0f) == identity));
        ML_Assert(all(lerp(identity, quarterTurn, 2.0f) == quarterTurn));
    }

    { // Quatf composition and TQS use local scale followed by rotation and translation
        const Quatf a(0.7f, normalize(float3(1.0f, 2.0f, -3.0f)));
        const Quatf b(-1.1f, normalize(float3(-2.0f, 1.0f, 3.0f)));
        const Quatf rotation = a * b;
        float4x4 identityMatrix;
        identityMatrix.SetupByQuaternion({0.0f, 0.0f, 0.0f, 1.0f});
        ML_Assert(all(identityMatrix * float3(1.0f, 2.0f, 3.0f) == float3(1.0f, 2.0f, 3.0f)));
        float4x4 rotationMatrix;
        rotationMatrix.SetupByQuaternion(rotation);
        ML_Assert(all(abs(rotationMatrix * float3(1.0f, 2.0f, 3.0f) - rotation * float3(1.0f, 2.0f, 3.0f)) < float3(Test_Eps)));
        const float3 translation(10.0f, 20.0f, 30.0f);
        const float3 scales[] = {float3(2.0f), float3(2.0f, 3.0f, 4.0f), float3(-2.0f, 0.0f, 4.0f)};

        for (uint32_t i = 0; i < 3; i++) {
            float4x4 matrix;
            matrix.SetupByTQS(translation, rotation, scales[i]);

            for (uint32_t j = 0; j < 3; j++) {
                const float3 v(float(j + 1), float(j) - 2.0f, 0.5f - float(j));
                ML_Assert(all(abs(rotation * v - a * (b * v)) < float3(Test_Eps)));
                ML_Assert(all(abs(matrix * v - (translation + rotation * (scales[i] * v))) < float3(Test_Eps)));
            }
        }
    }

    const double quatdEps = 1e-12;

    { // Quatd arithmetic and construction
        const Quatd identity(1.0);
        const Quatd a(1.0, 2.0, 3.0, 4.0);
        const Quatd b(-2.0, 1.0, 0.5, 3.0);
        ML_Assert(all(Quatd(double3(1.0)) == Quatd(1.0, 1.0, 1.0, 0.0)));
        ML_Assert(all(Quatd(double3(1.0, 1.0, 1.0)) == Quatd(double3(1.0))));
        ML_Assert(all((a + 2.0) == (a + Quatd(2.0))));
        ML_Assert(all((2.0 + a) == (Quatd(2.0) + a)));
        ML_Assert(all((a - 2.0) == (a - Quatd(2.0))));
        ML_Assert(all((2.0 - a) == (Quatd(2.0) - a)));
        ML_Assert(abs(a.Norm() - 30.0) < quatdEps);
        ML_Assert(all(abs((a * a.Inverse()).q - identity.q) < double4(quatdEps)));
        ML_Assert(all(abs((a / b).q - (a * b.Inverse()).q) < double4(quatdEps)));
        ML_Assert(all(abs((2.0 / a).q - (Quatd(2.0) / a).q) < double4(quatdEps)));

        Quatd c = a;
        c += 2.0;
        ML_Assert(all(c == (a + 2.0)));
        c -= 2.0;
        ML_Assert(all(c == a));
        c *= b;
        ML_Assert(all(abs(c.q - (a * b).q) < double4(quatdEps)));
        c /= b;
        ML_Assert(all(abs(c.q - a.q) < double4(quatdEps)));
        c = a;
        c *= c;
        ML_Assert(all(abs(c.q - (a * a).q) < double4(quatdEps)));
        c = a;
        c /= c;
        ML_Assert(all(abs(c.q - identity.q) < double4(quatdEps)));
        c = a;
        c *= 2.0;
        c /= 2.0;
        ML_Assert(all(c == a));
    }

    { // Quatd interpolation and rotation-vector limits
        const Quatd identity(1.0);
        const Quatd quarterTurn(1.5707963267948966, double3(0.0, 0.0, 1.0));
        const double3 v(1.0, 0.0, 0.0);
        ML_Assert(all(identity.GetRotation() == double3(0.0)));
        ML_Assert(all((-identity).GetRotation() == double3(0.0)));
        ML_Assert(all(abs(quarterTurn.GetRotation() - double3(0.0, 0.0, 1.5707963267948966)) < double3(quatdEps)));
        const Quatd tiny(1e-16, double3(0.0, 1.0, 0.0));
        ML_Assert(abs(tiny.GetRotation().y - 1e-16) < 1e-28);

        for (uint32_t i = 0; i <= 16; i++) {
            const double t = double(i) / 16.0;
            const double3 expected = Quatd(t * 1.5707963267948966, double3(0.0, 0.0, 1.0)) * v;
            ML_Assert(all(abs(Slerp(identity, quarterTurn, t) * v - expected) < double3(quatdEps)));
            ML_Assert(all(abs(Slerp(identity, -quarterTurn, t) * v - expected) < double3(quatdEps)));
            ML_Assert(abs(Slerp(identity, -identity, t).Norm() - 1.0) < quatdEps);
            ML_Assert(abs(lerp(identity, -identity, t).Norm() - 1.0) < quatdEps);
            ML_Assert(all(abs(Slerp(identity, tiny, t).GetRotation() - double3(0.0, t * 1e-16, 0.0)) < double3(1e-28)));
        }

        const Quatd slightlyLong = -identity * 1.0000000000000002;
        ML_Assert(abs(Slerp(identity, slightlyLong, 0.5).Norm() - 1.0) < quatdEps);
        ML_Assert(all(Slerp(identity, quarterTurn, -1.0) == identity));
        ML_Assert(all(Slerp(identity, quarterTurn, 2.0) == quarterTurn));
        ML_Assert(all(lerp(identity, quarterTurn, -1.0) == identity));
        ML_Assert(all(lerp(identity, quarterTurn, 2.0) == quarterTurn));
    }

    { // Quatd composition and TQS use local scale followed by rotation and translation
        const Quatd a(0.7, normalize(double3(1.0, 2.0, -3.0)));
        const Quatd b(-1.1, normalize(double3(-2.0, 1.0, 3.0)));
        const Quatd rotation = a * b;
        double4x4 identityMatrix;
        identityMatrix.SetupByQuaternion({0.0, 0.0, 0.0, 1.0});
        ML_Assert(all(identityMatrix * double3(1.0, 2.0, 3.0) == double3(1.0, 2.0, 3.0)));
        double4x4 rotationMatrix;
        rotationMatrix.SetupByQuaternion(rotation);
        ML_Assert(all(abs(rotationMatrix * double3(1.0, 2.0, 3.0) - rotation * double3(1.0, 2.0, 3.0)) < double3(quatdEps)));
        const double3 translation(10.0, 20.0, 30.0);
        const double3 scales[] = {double3(2.0), double3(2.0, 3.0, 4.0), double3(-2.0, 0.0, 4.0)};

        for (uint32_t i = 0; i < 3; i++) {
            double4x4 matrix;
            matrix.SetupByTQS(translation, rotation, scales[i]);

            for (uint32_t j = 0; j < 3; j++) {
                const double3 v(double(j + 1), double(j) - 2.0, 0.5 - double(j));
                ML_Assert(all(abs(rotation * v - a * (b * v)) < double3(quatdEps)));
                ML_Assert(all(abs(matrix * v - (translation + rotation * (scales[i] * v))) < double3(quatdEps)));
            }
        }
    }

    { // Double-precision interpolation near the linear fallback threshold
        const Quatd identity(1.0);
        const double3 axis = normalize(double3(1.0 + 1e-10, 2.0 - 1e-10, -3.0 + 1e-9));
        const double angles[] = {0.02, 0.00028, 0.00029, 1e-5, 1e-8};

        for (uint32_t i = 0; i < 5; i++) {
            const Quatd end(angles[i], axis);
            const Quatd midpoint = Slerp(identity, -end, 0.25);
            const double3 expected = axis * (angles[i] * 0.25);
            ML_Assert(all(abs(midpoint.GetRotation() - expected) < double3(quatdEps)));
        }
    }

    { // float rotation between parallel and nearly parallel directions
        const float3 from(0.0f, 0.0f, 1.0f);
        const float angles[] = {0.0f, 1e-4f, 0.8f, 2.8f};
        for (uint32_t i = 0; i < 4; i++) {
            const float3 to(sin(angles[i]), 0.0f, cos(angles[i]));
            float4x4 rotation;
            rotation.SetupByRotation(from, to);
            ML_Assert(abs(rotation.a00 - to.z) < Test_Eps);
            ML_Assert(abs(rotation.a11 - 1.0f) < Test_Eps);
            ML_Assert(abs(rotation.a22 - to.z) < Test_Eps);
            ML_Assert(abs(rotation.a02 - to.x) < Test_Eps);
            ML_Assert(abs(rotation.a20 + to.x) < Test_Eps);
        }
    }

    { // float vector Slerp follows a great circle, including near-antipodal directions
        const float4 from(1.0f, 0.0f, 0.0f, 0.0f);
        const float4 to(0.0f, 1.0f, 0.0f, 0.0f);
        ML_Assert(all(Slerp(from, to, -1.0f) == from));
        ML_Assert(all(Slerp(from, to, 1.0f + 1.0f) == to));
        ML_Assert(all(Slerp(from, to, 0.0f) == from));
        ML_Assert(all(Slerp(from, to, 1.0f) == to));
        ML_Assert(all(abs(Slerp(from, -from, 0.0f) - from) < float4(Test_Eps)));
        ML_Assert(all(abs(Slerp(from, -from, 1.0f) + from) < float4(Test_Eps)));
        const float angles[] = {0.0f, 0.05f, 0.8f, 2.8f, 3.13f};
        for (uint32_t i = 0; i < 5; i++) {
            const float4 to(cos(angles[i]), sin(angles[i]), 0.0f, 0.0f);
            for (uint32_t j = 0; j <= 4; j++) {
                const float t = float(j) / 4.0f;
                const float4 expected(cos(angles[i] * t), sin(angles[i] * t), 0.0f, 0.0f);
                // Near-antipodal interpolation amplifies rounding in the input dot product.
                const float eps = i == 4 ? 1e-4f : float(Test_Eps);
                ML_Assert(all(abs(Slerp(from, to, t) - expected) < float4(eps)));
            }
        }
    }

    { // float remainder truncates positive and negative quotients toward zero
        const float4 x(5.5f, -5.5f, 4.0f, -0.25f);
        const float4 y(2.0f, 2.0f, -2.0f, -2.0f);
        ML_Assert(all(fmod(x, y) == float4(1.5f, -1.5f, 0.0f, -0.25f)));
    }

    { // double rotation between parallel and nearly parallel directions
        const double3 from(0.0, 0.0, 1.0);
        const double angles[] = {0.0, 1e-4, 0.8, 2.8};
        for (uint32_t i = 0; i < 4; i++) {
            const double3 to(sin(angles[i]), 0.0, cos(angles[i]));
            double4x4 rotation;
            rotation.SetupByRotation(from, to);
            ML_Assert(abs(rotation.a00 - to.z) < 1e-12);
            ML_Assert(abs(rotation.a11 - 1.0) < 1e-12);
            ML_Assert(abs(rotation.a22 - to.z) < 1e-12);
            ML_Assert(abs(rotation.a02 - to.x) < 1e-12);
            ML_Assert(abs(rotation.a20 + to.x) < 1e-12);
        }
    }

    { // double vector Slerp follows a great circle, including near-antipodal directions
        const double4 from(1.0, 0.0, 0.0, 0.0);
        const double4 to(0.0, 1.0, 0.0, 0.0);
        ML_Assert(all(Slerp(from, to, -1.0) == from));
        ML_Assert(all(Slerp(from, to, 1.0 + 1.0) == to));
        ML_Assert(all(Slerp(from, to, 0.0) == from));
        ML_Assert(all(Slerp(from, to, 1.0) == to));
        ML_Assert(all(abs(Slerp(from, -from, 0.0) - from) < double4(1e-12)));
        ML_Assert(all(abs(Slerp(from, -from, 1.0) + from) < double4(1e-12)));
        const double angles[] = {0.0, 0.00014, 0.00015, 0.05, 0.8, 2.8, 3.13};
        for (uint32_t i = 0; i < 7; i++) {
            const double4 to(cos(angles[i]), sin(angles[i]), 0.0, 0.0);
            for (uint32_t j = 0; j <= 4; j++) {
                const double t = double(j) / 4.0;
                const double4 expected(cos(angles[i] * t), sin(angles[i] * t), 0.0, 0.0);
                ML_Assert(all(abs(Slerp(from, to, t) - expected) < double4(1e-12)));
            }
        }
    }

    { // double remainder truncates positive and negative quotients toward zero
        const double4 x(5.5, -5.5, 4.0, -0.25);
        const double4 y(2.0, 2.0, -2.0, -2.0);
        ML_Assert(all(fmod(x, y) == double4(1.5, -1.5, 0.0, -0.25)));
    }

    { // Full-precision reciprocal preserves zero/infinity and avoids intermediate overflow
        const double infinity = double4(c_v4d_Inf).x;
        const double inputs[] = {0.0, -0.0, infinity, -infinity, 1e-308, -1e-308, 3.0, -7.0};
        for (uint32_t i = 0; i < 8; i++) {
            const double4 result(v4d_rcp(double4(inputs[i]).ymm));
            ML_Assert(all(result == double4(1.0 / inputs[i])));
        }
    }

    const uint32_t N = 10000;
    const float R = 10000.0f;

    uint32_t rngState = 1983;
    Rng::Hash::Initialize(rngState, 0, 0);

    for (uint32_t i = 0; i < N; i++) {
        { // Ops
            float x1 = (Rng::Hash::GetFloat(rngState) - 0.5f) * R;
            float y1 = (Rng::Hash::GetFloat(rngState) - 0.5f) * R;
            float z1 = (Rng::Hash::GetFloat(rngState) - 0.5f) * R;
            float w1 = (Rng::Hash::GetFloat(rngState) - 0.5f) * R;

            float x2 = (Rng::Hash::GetFloat(rngState) - 0.5f) * R;
            float y2 = (Rng::Hash::GetFloat(rngState) - 0.5f) * R;
            float z2 = (Rng::Hash::GetFloat(rngState) - 0.5f) * R;
            float w2 = (Rng::Hash::GetFloat(rngState) - 0.5f) * R;

            TestOp_x2(float2, float, -);
            TestOp_x2(float2, float, +);
            TestOp_x2(float2, float, *);
            TestOp_x2(float2, float, /);

            TestOp_x3(float3, float, -);
            TestOp_x3(float3, float, +);
            TestOp_x3(float3, float, *);
            TestOp_x3(float3, float, /);

            TestOp_x4(float4, float, -);
            TestOp_x4(float4, float, +);
            TestOp_x4(float4, float, *);
            TestOp_x4(float4, float, /);

            TestOp_x2(double2, double, -);
            TestOp_x2(double2, double, +);
            TestOp_x2(double2, double, *);
            TestOp_x2(double2, double, /);

            TestOp_x3(double3, double, -);
            TestOp_x3(double3, double, +);
            TestOp_x3(double3, double, *);
            TestOp_x3(double3, double, /);

            TestOp_x4(double4, double, +);
            TestOp_x4(double4, double, -);
            TestOp_x4(double4, double, *);
            TestOp_x4(double4, double, /);

            // Avoid division by "0" for integers
            if (x2 > -1 && x2 < 1)
                x2 = 1;

            if (y2 > -1 && y2 < 1)
                y2 = 1;

            if (z2 > -1 && z2 < 1)
                z2 = 1;

            if (w2 > -1 && w2 < 1)
                w2 = 1;

            TestOp_x2(int2, int32_t, -);
            TestOp_x2(int2, int32_t, +);
            TestOp_x2(int2, int32_t, *);
            TestOp_x2(int2, int32_t, /);

            TestOp_x3(int3, int32_t, -);
            TestOp_x3(int3, int32_t, +);
            TestOp_x3(int3, int32_t, *);
            TestOp_x3(int3, int32_t, /);

            TestOp_x4(int4, int32_t, -);
            TestOp_x4(int4, int32_t, +);
            TestOp_x4(int4, int32_t, *);
            TestOp_x4(int4, int32_t, /);

            TestOp_x2(uint2, uint32_t, -);
            TestOp_x2(uint2, uint32_t, +);
            TestOp_x2(uint2, uint32_t, *);
            TestOp_x2(uint2, uint32_t, /);

            TestOp_x3(uint3, uint32_t, -);
            TestOp_x3(uint3, uint32_t, +);
            TestOp_x3(uint3, uint32_t, *);
            TestOp_x3(uint3, uint32_t, /);

            TestOp_x4(uint4, uint32_t, -);
            TestOp_x4(uint4, uint32_t, +);
            TestOp_x4(uint4, uint32_t, *);
            TestOp_x4(uint4, uint32_t, /);
        }

        { // Integer ops
            uint32_t x1 = Rng::Hash::GetUint(rngState);
            uint32_t y1 = Rng::Hash::GetUint(rngState);
            uint32_t z1 = Rng::Hash::GetUint(rngState);
            uint32_t w1 = Rng::Hash::GetUint(rngState);

            uint32_t x2 = Rng::Hash::GetUint(rngState);
            uint32_t y2 = Rng::Hash::GetUint(rngState);
            uint32_t z2 = Rng::Hash::GetUint(rngState);
            uint32_t w2 = Rng::Hash::GetUint(rngState);

            TestOp_x2(int2, int32_t, &);
            TestOp_x2(int2, int32_t, |);
            TestOp_x2(int2, int32_t, ^);

            TestOp_x3(int3, int32_t, &);
            TestOp_x3(int3, int32_t, |);
            TestOp_x3(int3, int32_t, ^);

            TestOp_x4(int4, int32_t, &);
            TestOp_x4(int4, int32_t, |);
            TestOp_x4(int4, int32_t, ^);

            TestOp_x2(uint2, uint32_t, &);
            TestOp_x2(uint2, uint32_t, |);
            TestOp_x2(uint2, uint32_t, ^);

            TestOp_x3(uint3, uint32_t, &);
            TestOp_x3(uint3, uint32_t, |);
            TestOp_x3(uint3, uint32_t, ^);

            TestOp_x4(uint4, uint32_t, &);
            TestOp_x4(uint4, uint32_t, |);
            TestOp_x4(uint4, uint32_t, ^);

            // Shifts and mod: use sane 2nd arg
            x1 &= 0x7FFFFFFF;
            y1 &= 0x7FFFFFFF;
            z1 &= 0x7FFFFFFF;
            w1 &= 0x7FFFFFFF;

            x2 &= 31;
            y2 &= 31;
            z2 &= 31;
            w2 &= 31;

            TestOp_x2(int2, int32_t, <<);
            TestOp_x2(int2, int32_t, >>);

            TestOp_x3(int3, int32_t, <<);
            TestOp_x3(int3, int32_t, >>);

            TestOp_x4(int4, int32_t, <<);
            TestOp_x4(int4, int32_t, >>);

            TestOp_x2(uint2, uint32_t, <<);
            TestOp_x2(uint2, uint32_t, >>);

            TestOp_x3(uint3, uint32_t, <<);
            TestOp_x3(uint3, uint32_t, >>);

            TestOp_x4(uint4, uint32_t, <<);
            TestOp_x4(uint4, uint32_t, >>);

            // Avoid division by "0"
            if (!x2)
                x2 = 1;

            if (!y2)
                y2 = 1;

            if (!z2)
                z2 = 1;

            if (!w2)
                w2 = 1;

            TestOp_x2(int2, int32_t, %);
            TestOp_x3(int3, int32_t, %);
            TestOp_x4(int4, int32_t, %);
            TestOp_x2(uint2, uint32_t, %);
            TestOp_x3(uint3, uint32_t, %);
            TestOp_x4(uint4, uint32_t, %);
        }

        { // Math [-INF, INF]
            float x1 = (Rng::Hash::GetFloat(rngState) - 0.5f) * R;
            float y1 = (Rng::Hash::GetFloat(rngState) - 0.5f) * R;
            float z1 = (Rng::Hash::GetFloat(rngState) - 0.5f) * R;
            float w1 = (Rng::Hash::GetFloat(rngState) - 0.5f) * R;

            float x2 = (Rng::Hash::GetFloat(rngState) - 0.5f) * R;
            float y2 = (Rng::Hash::GetFloat(rngState) - 0.5f) * R;
            float z2 = (Rng::Hash::GetFloat(rngState) - 0.5f) * R;
            float w2 = (Rng::Hash::GetFloat(rngState) - 0.5f) * R;

            Test1_x2(float2, float, degrees);
            Test1_x2(float2, float, radians);
            Test1_x2(float2, float, sign);
            Test1_x2(float2, float, abs);
            Test1_x2(float2, float, floor);
            Test1_x2(float2, float, ceil);
            Test1_x2(float2, float, frac);
            Test1_x2(float2, float, saturate);
            Test2_x2(float2, float, min);
            Test2_x2(float2, float, max);
            Test2_x2(float2, float, step);

            Test1_x2(float2, float, rcp);
            Test1_x2(float2, float, sin);
            Test1_x2(float2, float, cos);
            Test1_x2(float2, float, tan);
            Test1_x2(float2, float, atan);
            Test2_x2(float2, float, fmod);
            Test2_x2(float2, float, atan2);

            Test1_x2(double2, double, degrees);
            Test1_x2(double2, double, radians);
            Test1_x2(double2, double, sign);
            Test1_x2(double2, double, abs);
            Test1_x2(double2, double, floor);
            Test1_x2(double2, double, ceil);
            Test1_x2(double2, double, frac);
            Test1_x2(double2, double, saturate);
            Test2_x2(double2, double, min);
            Test2_x2(double2, double, max);
            Test2_x2(double2, double, step);

            Test1_x2(double2, double, rcp);
            Test1_x2(double2, double, sin);
            Test1_x2(double2, double, cos);
            Test1_x2(double2, double, tan);
            Test1_x2(double2, double, atan);
            Test2_x2(double2, double, fmod);
            Test2_x2(double2, double, atan2);

            Test1_x3(float3, float, degrees);
            Test1_x3(float3, float, radians);
            Test1_x3(float3, float, sign);
            Test1_x3(float3, float, abs);
            Test1_x3(float3, float, floor);
            Test1_x3(float3, float, ceil);
            Test1_x3(float3, float, frac);
            Test1_x3(float3, float, saturate);
            Test2_x3(float3, float, min);
            Test2_x3(float3, float, max);
            Test2_x3(float3, float, step);

            Test1_x3_eps(float3, float, rcp);
            Test1_x3_eps(float3, float, sin);
            Test1_x3_eps(float3, float, cos);
            Test1_x3_eps(float3, float, tan);
            Test1_x3_eps(float3, float, atan);
            Test2_x3_ceps(float3, float, fmod);
            Test2_x3_eps(float3, float, atan2);

            Test1_x3(double3, double, degrees);
            Test1_x3(double3, double, radians);
            Test1_x3(double3, double, sign);
            Test1_x3(double3, double, abs);
            Test1_x3(double3, double, floor);
            Test1_x3(double3, double, ceil);
            Test1_x3(double3, double, frac);
            Test1_x3(double3, double, saturate);
            Test2_x3(double3, double, min);
            Test2_x3(double3, double, max);
            Test2_x3(double3, double, step);

            Test1_x3_eps(double3, double, rcp);
            Test1_x3_eps(double3, double, sin);
            Test1_x3_eps(double3, double, cos);
            Test1_x3_eps(double3, double, tan);
            Test1_x3_eps(double3, double, atan);
            Test2_x3_ceps(double3, double, fmod);
            Test2_x3_eps(double3, double, atan2);

            Test1_x4(float4, float, degrees);
            Test1_x4(float4, float, radians);
            Test1_x4(float4, float, sign);
            Test1_x4(float4, float, abs);
            Test1_x4(float4, float, floor);
            Test1_x4(float4, float, ceil);
            Test1_x4(float4, float, frac);
            Test1_x4(float4, float, saturate);
            Test2_x4(float4, float, min);
            Test2_x4(float4, float, max);
            Test2_x4(float4, float, step);

            Test1_x4_eps(float4, float, rcp);
            Test1_x4_eps(float4, float, sin);
            Test1_x4_eps(float4, float, cos);
            Test1_x4_eps(float4, float, tan);
            Test1_x4_eps(float4, float, atan);
            Test2_x4_ceps(float4, float, fmod);
            Test2_x4_eps(float4, float, atan2);

            Test1_x4(double4, double, degrees);
            Test1_x4(double4, double, radians);
            Test1_x4(double4, double, sign);
            Test1_x4(double4, double, abs);
            Test1_x4(double4, double, floor);
            Test1_x4(double4, double, ceil);
            Test1_x4(double4, double, frac);
            Test1_x4(double4, double, saturate);
            Test2_x4(double4, double, min);
            Test2_x4(double4, double, max);
            Test2_x4(double4, double, step);

            Test1_x4_eps(double4, double, rcp);
            Test1_x4_eps(double4, double, sin);
            Test1_x4_eps(double4, double, cos);
            Test1_x4_eps(double4, double, tan);
            Test1_x4_eps(double4, double, atan);
            Test2_x4_ceps(double4, double, fmod);
            Test2_x4_eps(double4, double, atan2);

            // round: avoid fractional part = 0.5
            if (frac(x1) == 0.5f)
                x1 = uFloat(uFloat(x1).i + 1).f;

            if (frac(y1) == 0.5f)
                y1 = uFloat(uFloat(y1).i + 1).f;

            if (frac(z1) == 0.5f)
                z1 = uFloat(uFloat(z1).i + 1).f;

            if (frac(w1) == 0.5f)
                w1 = uFloat(uFloat(w1).i + 1).f;

            Test1_x2(float2, float, round);
            Test1_x3(float3, float, round);
            Test1_x4(float4, float, round);

            Test1_x2(double2, double, round);
            Test1_x3(double3, double, round);
            Test1_x4(double4, double, round);

            // pow/exp/exp2: do not use to large "x" and "y" to avoid "INF"
            x1 *= 32.0f / R;
            y1 *= 32.0f / R;
            z1 *= 32.0f / R;
            w1 *= 32.0f / R;

            x2 *= 32.0f / R;
            y2 *= 32.0f / R;
            z2 *= 32.0f / R;
            w2 *= 32.0f / R;

            Test1_x2(float2, float, exp);
            Test1_x3_eps(float3, float, exp);
            Test1_x4_eps(float4, float, exp);

            Test1_x2(double2, double, exp);
            Test1_x3_eps(double3, double, exp);
            Test1_x4_eps(double4, double, exp);

            Test1_x2(float2, float, exp2);
            Test1_x3_eps(float3, float, exp2);
            Test1_x4_eps(float4, float, exp2);

            Test1_x2(double2, double, exp2);
            Test1_x3_eps(double3, double, exp2);
            Test1_x4_eps(double4, double, exp2);

            // pow: "x" must be positive
            x1 = abs(x1);
            y1 = abs(y1);
            z1 = abs(z1);
            w1 = abs(w1);

            Test2_x2(float2, float, pow);
            Test2_x3_eps(float3, float, pow);
            Test2_x4_eps(float4, float, pow);

            Test2_x2(double2, double, pow);
            Test2_x3_eps(double3, double, pow);
            Test2_x4_eps(double4, double, pow);
        }

        { // Math (> 0)
            float x1 = Rng::Hash::GetFloat(rngState) * R;
            float y1 = Rng::Hash::GetFloat(rngState) * R;
            float z1 = Rng::Hash::GetFloat(rngState) * R;
            float w1 = Rng::Hash::GetFloat(rngState) * R;

            Test1_x2(float2, float, rsqrt);
            Test1_x2(float2, float, sqrt);
            Test1_x2(float2, float, log);
            Test1_x2(float2, float, log2);

            Test1_x2(double2, double, rsqrt);
            Test1_x2(double2, double, sqrt);
            Test1_x2(double2, double, log);
            Test1_x2(double2, double, log2);

            Test1_x3_eps(float3, float, rsqrt);
            Test1_x3_eps(float3, float, sqrt);
            Test1_x3_eps(float3, float, log);
            Test1_x3_eps(float3, float, log2);

            Test1_x3_eps(double3, double, rsqrt);
            Test1_x3_eps(double3, double, sqrt);
            Test1_x3_eps(double3, double, log);
            Test1_x3_eps(double3, double, log2);

            Test1_x4_eps(float4, float, rsqrt);
            Test1_x4_eps(float4, float, sqrt);
            Test1_x4_eps(float4, float, log);
            Test1_x4_eps(float4, float, log2);

            Test1_x4_eps(double4, double, rsqrt);
            Test1_x4_eps(double4, double, sqrt);
            Test1_x4_eps(double4, double, log);
            Test1_x4_eps(double4, double, log2);
        }

        { // Math [-1; 1]
            float x1 = (Rng::Hash::GetFloat(rngState) - 0.5f) * 2.0f;
            float y1 = (Rng::Hash::GetFloat(rngState) - 0.5f) * 2.0f;
            float z1 = (Rng::Hash::GetFloat(rngState) - 0.5f) * 2.0f;
            float w1 = (Rng::Hash::GetFloat(rngState) - 0.5f) * 2.0f;

            Test1_x2(float2, float, asin);
            Test1_x2(float2, float, acos);

            Test1_x2(double2, double, asin);
            Test1_x2(double2, double, acos);

            Test1_x3_eps(float3, float, asin);
            Test1_x3_eps(float3, float, acos);

            Test1_x3_eps(double3, double, asin);
            Test1_x3_eps(double3, double, acos);

            Test1_x4_eps(float4, float, asin);
            Test1_x4_eps(float4, float, acos);

            Test1_x4_eps(double4, double, asin);
            Test1_x4_eps(double4, double, acos);
        }
    }

    { // == and !=
        float x1 = (Rng::Hash::GetFloat(rngState) - 0.5f) * R;
        float y1 = (Rng::Hash::GetFloat(rngState) - 0.5f) * R;
        float z1 = (Rng::Hash::GetFloat(rngState) - 0.5f) * R;
        float w1 = (Rng::Hash::GetFloat(rngState) - 0.5f) * R;

        TestEqual_x2(int2, int32_t);
        TestEqual_x3(int3, int32_t);
        TestEqual_x4(int4, int32_t);

        TestEqual_x2(uint2, uint32_t);
        TestEqual_x3(uint3, uint32_t);
        TestEqual_x4(uint4, uint32_t);

        TestEqual_x2(float2, float);
        TestEqual_x3(float3, float);
        TestEqual_x4(float4, float);

        TestEqual_x2(double2, double);
        TestEqual_x3(double3, double);
        TestEqual_x4(double4, double);

        TestNotEqual_x2(int2, int32_t);
        TestNotEqual_x3(int3, int32_t);
        TestNotEqual_x4(int4, int32_t);

        TestNotEqual_x2(uint2, uint32_t);
        TestNotEqual_x3(uint3, uint32_t);
        TestNotEqual_x4(uint4, uint32_t);

        TestNotEqual_x2(float2, float);
        TestNotEqual_x3(float3, float);
        TestNotEqual_x4(float4, float);

        TestNotEqual_x2(double2, double);
        TestNotEqual_x3(double3, double);
        TestNotEqual_x4(double4, double);
    }

    { // +0 == -0
        ML_Assert(all(+0 == -0));
        ML_Assert(all(int2(+0) == int2(-0)));
        ML_Assert(all(int3(+0) == int3(-0)));
        ML_Assert(all(int4(+0) == int4(-0)));

        ML_Assert(all(+0.0f == -0.0f));
        ML_Assert(all(float2(+0.0f) == float2(-0.0f)));
        ML_Assert(all(float3(+0.0f) == float3(-0.0f)));
        ML_Assert(all(float4(+0.0f) == float4(-0.0f)));

        ML_Assert(all(+0.0 == -0.0));
        ML_Assert(all(float2(+0.0) == float2(-0.0)));
        ML_Assert(all(float3(+0.0) == float3(-0.0)));
        ML_Assert(all(float4(+0.0) == float4(-0.0)));
    }

    { // NAN != NAN
        float myNanf = log(0.0f) * 0.0f;
        ML_Assert(myNanf != myNanf);
        ML_Assert(any(float2(myNanf) != float2(myNanf)));
        ML_Assert(any(float3(myNanf) != float3(myNanf)));
        ML_Assert(any(float4(myNanf) != float4(myNanf)));

        double myNan = log(0.0) * 0.0;
        ML_Assert(myNan != myNan);
        ML_Assert(any(double2(myNan) != double2(myNan)));
        ML_Assert(any(double3(myNan) != double3(myNan)));
        ML_Assert(any(double4(myNan) != double4(myNan)));
    }
}

#ifdef ML_NAMESPACE
}
#endif
