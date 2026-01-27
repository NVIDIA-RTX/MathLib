# MathLib (ML)

*ML* is a high-performance math library designed for computer graphics. It consists of two primary components:
- `ml.hlsli` - an HLSL header covering various computer graphics topics
- `ml.h` - a cross-platform, header-only, HLSL-compatible math library. It is *SSE/AVX/NEON*-accelerated, allowing `ml.hlsli` logic to be used directly in C++ code


## HLSL PART

`ml.hlsli` sections (`namespaces`):
 - `Math` - common mathematical operations
 - `Geometry` - transformations, rotations and 3D world-space math
 - `Color` - color spaces, transfer functions, LDR/HDR, blending, clamping, and color ramps
 - `Packing` - scalar and vector data packing utilities
 - `Filtering` - nearest, linear and Catmull-Rom filters
 - `Sequence` - low-discrepancy sampling sequences
 - `Rng` - random number generators
 - `BRDF` - BRDF implementations
 - `ImportanceSampling` - BRDF-lobes importance sampling
 - `SphericalHarmonics` - Spherical Harmonics (SH) math
 - `Text` - resource-less text printing

## C++ PART

`ml.h` features:
- **Hardware acceleration**: compile-time optimization level specialization: SSE3 (and below), +SSE4, +AVX1, +AVX2. Emulation of unsupported higher level intrinsics using lower level intrinsics. ARM supported via [*sse2neon*](https://github.com/DLTcollab/sse2neon)
- **Core types**:
  - `int[2,3,4]`, `uint[2,3,4]`, `bool[2,3,4]`
  - `float[2,3,4]`, `float4x4`
  - `double[2,3,4]`, `double4x4`
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
- in C++ code `sizeof(int3/uint3/float3) == sizeof(float4)` and `sizeof(double3) == sizeof(double4)` to honor *SIMD* alignment
- `float3x3` and `double3x3` are not implemented
- only 128-bit (`xmm`) and 256-bit (`ymm`) *SIMD* registers are used for acceleration
- `using namespace std` may lead to name collisions with library functions
- including `<cmath>` and/or `<cstdlib>` (even implicitly) after `ml.h` leads to name collisions. Ensure `ml.h` is included after standard library headers if necessary

## FUTURE

- supporting additional HLSL functionality
- improvements to unsupported intrinsic emulation

## LICENSE

*ML* is licensed under the MIT License.
