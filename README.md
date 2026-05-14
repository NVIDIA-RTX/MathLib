# MathLib (ML)

*MathLib (ML)* is a high-performance, header-only math library optimized for computer graphics. It provides a seamless bridge between *C++* and *HLSL*, allowing the same mathematical logic to be used across both CPU and GPU codebases. It consists of two primary components:

- `ml.hlsli` - a comprehensive *HLSL* header covering common computer graphics operations (geometry, color, packing, filtering, importance sampling, etc.);
- `ml.h` - a cross-platform, *SSE/AVX/NEON*-accelerated C++ library. It is designed to be *HLSL*-compatible, enabling `ml.hlsli` logic to run directly in *C++* with hardware-accelerated performance.

*ML* shares similar goals with [HLSL++](https://github.com/redorav/hlslpp). Key differences:
- *HLSL++* covers more *HLSL* features (and that's a big plus);
- *ML* supports HW-accelerated `double` types (like `double[2,3,4,4x4]`);
- *ML* provides broader coverage for "small" data types (FP16, FP8, packed, etc.);
- *ML* includes `ml.hlsli`, which has been proven to be very convenient in any 3D application development.

It would be good to exchange missing functionality one day.

## HLSL PART

`ml.hlsli` sections (`namespaces`):
 - `Math` - common mathematical operations;
 - `Geometry` - transformations, rotations and 3D world-space math;
 - `Color` - color spaces, transfer functions, LDR/HDR, blending, and ramps;
 - `Packing` - scalar and vector data packing utilities;
 - `Filtering` - nearest, linear and Catmull-Rom filters;
 - `Sequence` - low-discrepancy sampling sequences;
 - `Rng` - random number generators;
 - `BRDF` - BRDF implementations;
 - `ImportanceSampling` - BRDF-lobe importance sampling;
 - `SphericalHarmonics` - Spherical Harmonics (SH) math;
 - `Text` - resource-less text printing.

## C++ PART

`ml.h` features:
- **Hardware acceleration**: compile-time optimization levels: SSE3 (and below), +SSE4, +AVX1, +AVX2. Emulation of unsupported higher level intrinsics using lower level ones. ARM supported via [*sse2neon*](https://github.com/DLTcollab/sse2neon)
- **Core types**:
  - `int[2,3,4]`, `uint[2,3,4]`, `bool[2,3,4]`
  - `float[2,3,4,4x4]`
  - `double[2,3,4,4x4]`
- **DL/ML types**:
  - `float8_e4m3_t[2,4,8]`, `float8_e5m2_t[2,4,8]`, `float16_t[2,4,8]`
- **Supported HLSL functions**:
  - common functions: `rcp`, `sqrt`, `rsqrt`, `abs`, `sign`, `min`, `max`, `clamp`, `saturate`
  - rounding and modulo: `floor`, `round`, `ceil`, `frac`, `fmod`
  - transcendental functions: `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`, `pow`, `log`, `log2`, `exp`, `exp2`
  - special: `lerp`, `step`, `smoothstep`, `linearstep`, `all`, `any`
- **Data conversion:**
  - optimized packing from `fp32` to `fp16`, `fp11`, `fp10`, `fp8_e4m3`, `fp8_e5m2`, `SNORM`, `UNORM` and back
- **Linear algebra:**
  - vectors, matrices, overloaded operators, vector swizzling and related utilities
- **Other:**
  - projective math miscellaneous functionality
  - frustum & AABB primitives
  - random numbers generation
  - sorting algorithms

IMPORTANT:
- in C++, `sizeof(int3/uint3/float3) == sizeof(float4)` and `sizeof(double3) == sizeof(double4)` to honor *SIMD* alignment
- `float3x3` and `double3x3` are not implemented
- only 128-bit (`xmm`) and 256-bit (`ymm`) *SIMD* registers are used for acceleration
- `using namespace std` may lead to name collisions with library functions
- including `<cmath>` and/or `<cstdlib>` (even implicitly) after `ml.h` leads to name collisions. Ensure `ml.h` is included after standard library headers if necessary

## FUTURE

- supporting additional *HLSL* functionality
- improvements to unsupported intrinsic emulation

## LICENSE

*ML* is licensed under the MIT License.
