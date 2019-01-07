#include "asvgf/asvgf.h"

#include "kernel/pt_common.h"

#include "cuda/cudadefs.h"
#include "cuda/helper_math.h"
#include "cuda/cudautil.h"
#include "cuda/cudamemory.h"

#include "aten4idaten.h"

// TEA = Tiny Encryption Algorithm.
// https://en.wikipedia.org/wiki/Tiny_Encryption_Algorithm
inline __device__ void encryptTea(uint2& arg)
{
    const uint32_t key[] = {
        0xa341316c, 
        0xc8013ea4, 
        0xad90777d, 
        0x7e95761e,
    };

    uint32_t v0 = arg.x;
    uint32_t v1 = arg.y;
    uint32_t sum = 0;
    uint32_t delta = 0x9e3779b9;

    for (int i = 0; i < 16; i++) {
        sum += delta;
        v0 += ((v1 << 4) + key[0]) ^ (v1 + sum) ^ ((v1 >> 5) + key[1]);
        v1 += ((v0 << 4) + key[2]) ^ (v0 + sum) ^ ((v0 >> 5) + key[3]);
    }

    arg.x = v0;
    arg.y = v1;
}

inline __device__ bool testReprojectedDepth(float z1, float z2, float dz)
{
    float diffZ = abs(z1 - z2);
    return diffZ < 2.0 * (dz + 1e-3f);
}

#define AT_IS_INBOUND(x, a, b)  (((a) <= (x)) && ((x) <= (b)))

__global__ void doForwardProjection(
    int4* gradientSample,
    const float4* __restrict__ curAovNormalDepth,
    const float4* __restrict__ prevAovNormalDepth,
    int frame,
    int width, int height,
    int gradientTileSize,
    float cameraDistance,
    cudaSurfaceObject_t motionDetphBuffer,
    int* executedIdxArray)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix >= width || iy >= height) {
        return;
    }

    int idx = getIdx(ix, iy, width);

    // Compute randomized position as previous position.
    uint2 teaArg = make_uint2(idx, frame);
    encryptTea(teaArg);
    teaArg.x %= gradientTileSize;
    teaArg.y %= gradientTileSize;
    int2 prevPos = make_int2(
        ix * gradientTileSize + teaArg.x,
        iy * gradientTileSize + teaArg.y);

    float4 motionDepth;
    surf2Dread(&motionDepth, motionDetphBuffer, prevPos.x * sizeof(float4), prevPos.y);

    // NOTE
    // motion = prev - cur
    //  => -motion = cur - prev
    //  => prev + (-motion) = prev + (cur - prev) = cur
    int2 curPos = make_int2(prevPos.x - motionDepth.x, prevPos.y - motionDepth.y);

    // Check if position is in screen.
    if (!AT_IS_INBOUND(curPos.x, 0, width)
        || !AT_IS_INBOUND(curPos.y, 0, height))
    {
        return;
    }

    int curIdx = getIdx(curPos.x, curPos.y, width);
    int prevIdx = getIdx(prevPos.x, prevPos.y, width);

    float4 curNmlDepth = curAovNormalDepth[curIdx];
    float4 prevNmlDepth = curAovNormalDepth[prevIdx];

    float pixelDistanceRatio = (curNmlDepth.w / cameraDistance) * height;

    bool accept = testReprojectedDepth(curNmlDepth.w, prevNmlDepth.w, pixelDistanceRatio);
    if (!accept) {
        return;
    }

    // Remove depth.
    curNmlDepth.w = prevNmlDepth.w = 0;

    accept = (dot(curNmlDepth, prevNmlDepth) > 0.9f);
    if (!accept) {
        return;
    }

    int2 tilePos = make_int2(
        curPos.x % gradientTileSize,
        curPos.y % gradientTileSize);

    // NOTE
    // Atomic functions for CUDA.
    // http://www.slis.tsukuba.ac.jp/~fujisawa.makoto.fu/cgi-bin/wiki/index.php?CUDA%A5%A2%A5%C8%A5%DF%A5%C3%A5%AF%B4%D8%BF%F4

    int res = atomicCAS(&executedIdxArray[idx], -1, idx);
    if (res < 0) {
        // NOTE
        // w is not used.
        gradientSample[idx] = make_int4(tilePos.x, tilePos.y, prevIdx, 0);

        // Rng seed.

        // Mesh id.

        // Albedo.
    }
}
