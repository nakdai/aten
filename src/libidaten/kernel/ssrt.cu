#include "kernel/ssrt.h"
#include "kernel/context.cuh"
#include "kernel/light.cuh"
#include "kernel/material.cuh"
#include "kernel/intersect.cuh"
#include "kernel/accelerator.cuh"
#include "kernel/compaction.h"
#include "kernel/pt_common.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "cuda/helper_math.h"
#include "cuda/cudautil.h"
#include "cuda/cudamemory.h"

#include "aten4idaten.h"

#define SEPARATE_SHADOWRAY_HITTEST

//#define ENABLE_PROGRESSIVE

__global__ void genPath(
	idaten::SSRT::Path* paths,
	aten::ray* rays,
	int width, int height,
	int sample, int maxSamples,
	unsigned int frame,
	const aten::CameraParameter* __restrict__ camera,
	const unsigned int* sobolmatrices,
	const unsigned int* random)
{
	const auto ix = blockIdx.x * blockDim.x + threadIdx.x;
	const auto iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix >= width && iy >= height) {
		return;
	}

	const auto idx = getIdx(ix, iy, width);

	auto& path = paths[idx];
	path.isHit = false;

	if (path.isKill) {
		path.isTerminate = true;
		return;
	}

#ifdef ENABLE_PROGRESSIVE
#if IDATEN_SAMPLER == IDATEN_SAMPLER_SOBOL
	auto scramble = random[idx] * 0x1fe3434f;
	path.sampler.init(frame, 0, scramble, sobolmatrices);
#elif IDATEN_SAMPLER == IDATEN_SAMPLER_CMJ
	auto rnd = random[idx];
	auto scramble = rnd * 0x1fe3434f * ((frame + 133 * rnd) / (aten::CMJ::CMJ_DIM * aten::CMJ::CMJ_DIM));
	path.sampler.init(frame % (aten::CMJ::CMJ_DIM * aten::CMJ::CMJ_DIM), 0, scramble);
#endif
#else
	auto scramble = (iy * height * 4 + ix * 4) * maxSamples + sample + 1 + frame;
	path.sampler.init(frame, 0, scramble, sobolmatrices);
#endif

	float s = (ix + path.sampler.nextSample()) / (float)(camera->width);
	float t = (iy + path.sampler.nextSample()) / (float)(camera->height);

	AT_NAME::CameraSampleResult camsample;
	AT_NAME::PinholeCamera::sample(&camsample, camera, s, t);

	rays[idx] = camsample.r;

	path.throughput = aten::vec3(1);
	path.pdfb = 0.0f;
	path.isTerminate = false;
	path.isSingular = false;

	path.samples += 1;

	// Accumulate value, so do not reset.
	//path.contrib = aten::vec3(0);
}

__global__ void hitTestPrimaryRayInScreenSpace(
	cudaSurfaceObject_t gbuffer,
	idaten::SSRT::Path* paths,
	aten::Intersection* isects,
	int* hitbools,
	int width, int height,
	const aten::vec4 camPos,
	const aten::GeomParameter* __restrict__ geoms,
	const aten::PrimitiveParamter* __restrict__ prims,
	const aten::mat4* __restrict__ matrices,
	cudaTextureObject_t vtxPos)
{
	const auto ix = blockIdx.x * blockDim.x + threadIdx.x;
	const auto iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix >= width && iy >= height) {
		return;
	}

	const auto idx = getIdx(ix, iy, width);

	auto& path = paths[idx];
	path.isHit = false;

	hitbools[idx] = 0;

	if (path.isTerminate) {
		return;
	}

	// Sample data from texture.
	float4 data;
	surf2Dread(&data, gbuffer, ix * sizeof(float4), iy);

	// NOTE
	// x : objid
	// y : primid
	// zw : bary centroid

	int objid = __float_as_int(data.x);
	int primid = __float_as_int(data.y);

	isects[idx].objid = objid;
	isects[idx].primid = primid;

	// bary centroid.
	isects[idx].a = data.z;
	isects[idx].b = data.w;

	if (objid >= 0) {
		aten::PrimitiveParamter prim;
		prim.v0 = ((aten::vec4*)prims)[primid * aten::PrimitiveParamter_float4_size + 0];
		prim.v1 = ((aten::vec4*)prims)[primid * aten::PrimitiveParamter_float4_size + 1];

		isects[idx].mtrlid = prim.mtrlid;
		isects[idx].meshid = prim.gemoid;

		const auto* obj = &geoms[objid];

		float4 p0 = tex1Dfetch<float4>(vtxPos, prim.idx[0]);
		float4 p1 = tex1Dfetch<float4>(vtxPos, prim.idx[1]);
		float4 p2 = tex1Dfetch<float4>(vtxPos, prim.idx[2]);

		real a = data.z;
		real b = data.w;
		real c = 1 - a - b;

		// �d�S���W�n(barycentric coordinates).
		// v0�.
		// p = (1 - a - b)*v0 + a*v1 + b*v2
		auto p = c * p0 + a * p1 + b * p2;
		aten::vec4 vp(p.x, p.y, p.z, 1.0f);

		if (obj->mtxid >= 0) {
			auto mtxL2W = matrices[obj->mtxid * 2 + 0];
			vp = mtxL2W.apply(vp);
		}

		isects[idx].t = (camPos - vp).length();

		path.isHit = true;
		hitbools[idx] = 1;
	}
	else {
		path.isHit = false;
		hitbools[idx] = 0;
	}
}

inline __device__ bool intersectsDepthBuffer(float z, float minZ, float maxZ, float zThickness)
{
	// �w��͈͓��i���C�̎n�_�ƏI�_�j�� z ������΁A����̓��C�Ƀq�b�g�����Ƃ݂Ȃ���.
	z += zThickness;
	return (maxZ >= z) && (minZ - zThickness <= z);
}

inline __device__ bool traceScreenSpaceRay(
	cudaSurfaceObject_t depth,
	const aten::vec3& csOrig,
	const aten::vec3& csDir,
	const aten::mat4& mtxV2C,
	int width, int height,
	float nearPlaneZ,
	float stride,
	float jitter,
	aten::vec3& hitPixel)
{
	static const float zThickness = 1.0f;
	static const float maxDistance = 1000.0f;

	// Clip to the near plane.
	float rayLength = (csOrig.z + csDir.z * maxDistance) > -nearPlaneZ
		? (nearPlaneZ - csOrig.z) / csDir.z
		: maxDistance;

	aten::vec3 csEndPoint = csOrig + csDir * rayLength;

	// Project into homogeneous clip space.
	aten::vec4 H0 = mtxV2C.apply(aten::vec4(csOrig, 1));
	aten::vec4 H1 = mtxV2C.apply(aten::vec4(csEndPoint, 1));

	float k0 = 1.0 / H0.w;
	float k1 = 1.0 / H1.w;

	// The interpolated homogeneous version of the camera-space points.
	aten::vec3 Q0 = csOrig * k0;
	aten::vec3 Q1 = csEndPoint * k1;

	// Screen space point.
	aten::vec3 P0 = H0 * k0;
	aten::vec3 P1 = H1 * k1;

	// [-1, 1] -> [0, 1]
	P0 = P0 * 0.5f + 0.5f;
	P1 = P1 * 0.5f + 0.5f;

	P0.x *= width;
	P0.y *= height;
	P0.z = 0.0f;

	P1.x *= width;
	P1.y *= height;
	P1.z = 0.0f;

	// If the line is degenerate, make it cover at least one pixel to avoid handling zero-pixel extent as a special case later.
	// 2�_�Ԃ̋�����������x�����悤�ɂ���.
	P1 += aten::squared_length(P0 - P1) < 0.0001f
		? aten::vec3(0.01f)
		: aten::vec3(0.0f);
	aten::vec3 delta = P1 - P0;

	// Permute so that the primary iteration is in x to collapse all quadrant-specific DDA cases later.
	bool permute = false;
	if (abs(delta.x) < abs(delta.y))
	{
		permute = true;

		aten::swapVal(delta.x, delta.y);
		aten::swapVal(P0.x, P0.y);
		aten::swapVal(P1.x, P1.y);
	}

	float stepDir = 0.0f;
	if (delta.x < 0.0f) {
		stepDir = -1.0f;
	}
	else if (delta.x > 0.0f) {
		stepDir = 1.0f;
	}
	float invdx = stepDir / delta.x;

	// Track the derivatives of Q and k.
	aten::vec3 dQ = (Q1 - Q0) * invdx;
	float dk = (k1 - k0) * invdx;

	// y is slope.
	// slope = (y1 - y0) / (x1 - x0)
	aten::vec3 dP = aten::vec3(stepDir, delta.y * invdx, 0.0f);

	// Adjust end condition for iteration direction
	float end = P1.x * stepDir;

	int stepCount = 0;

	float prevZMaxEstimate = -csOrig.z;

	float rayZMin = prevZMaxEstimate;
	float rayZMax = prevZMaxEstimate;

	float sceneZMax = rayZMax + 100.0f;

	dP *= stride;
	dQ *= stride;
	dk *= stride;

	P0 += dP * jitter;
	Q0 += dQ * jitter;
	k0 += dk * jitter;

	aten::vec4 PQk = aten::vec4(P0.x, P0.y, Q0.z, k0);
	aten::vec4 dPQk = aten::vec4(dP.x, dP.y, dQ.z, dk);
	aten::vec3 Q = Q0;

	static const int maxSteps = 50;

	for (;
		((PQk.x * stepDir) <= end)	// �I�_�ɓ��B������.
		&& (stepCount < maxSteps)	// �ő又�����ɓ��B������.
		&& (sceneZMax != 0.0);	// �����Ȃ��Ƃ���ɓ��B���ĂȂ���.
		++stepCount)
	{
		// �O���Z�̍ő�l�����̍ŏ��l�ɂȂ�.
		rayZMin = prevZMaxEstimate;

		// ����Z�̍ő�l���v�Z����.
		// �������A1/2 pixel�� �]�T����������.
		// Q��w�����ŏ��Z����Ă��āA������1/w�ŏ��Z����̂ŁA���iView���W�n�j�ɖ߂邱�ƂɂȂ�.
		rayZMax = -(PQk.z + dPQk.z * 0.5) / (PQk.w + dPQk.w * 0.5);

		// ���Ɍ����čő�l��ێ�.
		prevZMaxEstimate = rayZMax;

		if (rayZMin > rayZMax) {
			// �O�̂���.
			float tmp = rayZMin;
			rayZMin = rayZMax;
			rayZMax = tmp;
		}

		hitPixel = permute ? aten::vec3(PQk.y, PQk.x, 0.0f) : aten::vec3(PQk.x, PQk.y, 0.0f);

		int ix = (int)hitPixel.x;
		int iy = (int)hitPixel.y;

		if (ix < 0 || ix >= width || iy < 0 || iy >= height) {
			return false;
		}

		// �V�[�����̌����_�ł̐[�x�l���擾.
		float4 data;
		surf2Dread(&data, depth, ix * sizeof(float4), iy);

		sceneZMax = data.x;

		if (intersectsDepthBuffer(sceneZMax, rayZMin, rayZMax, zThickness)) {
			break;
		}

		PQk += dPQk;
	}

	if (sceneZMax <= 0) {
		return false;
	}
	return intersectsDepthBuffer(sceneZMax, rayZMin, rayZMax, zThickness);
}

__global__ void hitTestPrimaryRayInScreenSpaceEx(
	cudaSurfaceObject_t gbuffer,
	cudaSurfaceObject_t depth,
	idaten::SSRT::Path* paths,
	aten::Intersection* isects,
	int* hitbools,
	int width, int height,
	const aten::vec4 camPos,
	float cameraNearPlaneZ,
	const aten::mat4 mtxW2V,
	const aten::mat4 mtxV2C,
	const aten::ray* __restrict__ rays,
	const aten::GeomParameter* __restrict__ geoms,
	const aten::PrimitiveParamter* __restrict__ prims,
	const aten::mat4* __restrict__ matrices,
	cudaTextureObject_t vtxPos,
	cudaTextureObject_t vtxNml)
{
	const auto ix = blockIdx.x * blockDim.x + threadIdx.x;
	const auto iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix >= width && iy >= height) {
		return;
	}

	const auto idx = getIdx(ix, iy, width);

	auto& path = paths[idx];
	path.isHit = false;

	hitbools[idx] = 0;

	if (path.isTerminate) {
		return;
	}

	// Sample data from texture.
	float4 data;
	surf2Dread(&data, gbuffer, ix * sizeof(float4), iy);

	// NOTE
	// x : objid
	// y : primid
	// zw : bary centroid

	int objid = __float_as_int(data.x);
	int primid = __float_as_int(data.y);

	isects[idx].objid = objid;
	isects[idx].primid = primid;

	// bary centroid.
	isects[idx].a = data.z;
	isects[idx].b = data.w;

	if (objid >= 0) {
		aten::vec3 vsOrig(0);	// �J��������̃��C�Ȃ̂�.
		aten::vec3 vsDir = mtxW2V.apply(rays[idx].dir);

		// TODO
		static const float stride = 5.0f;

		float c = (ix + iy) * 0.25f;
		float jitter = stride > 1.0f ? fmod(c, 1.0f) : 0.0f;

		aten::vec3 hitPixel(0);

		bool isIntersect = traceScreenSpaceRay(
			depth,
			vsOrig, vsDir,
			mtxV2C,
			width, height,
			cameraNearPlaneZ,
			stride, jitter,
			hitPixel);

		int x = (int)hitPixel.x;
		int y = (int)hitPixel.y;

		isIntersect = (0 <= x && x < width && 0 <= y && y < height);

		if (isIntersect) {
			path.isHit = true;
			hitbools[idx] = 1;

			aten::PrimitiveParamter prim;
			prim.v0 = ((aten::vec4*)prims)[primid * aten::PrimitiveParamter_float4_size + 0];
			prim.v1 = ((aten::vec4*)prims)[primid * aten::PrimitiveParamter_float4_size + 1];

			isects[idx].mtrlid = prim.mtrlid;
			isects[idx].meshid = prim.gemoid;
		}
	}
	else {
		path.isHit = false;
		hitbools[idx] = 0;
	}
}

__global__ void hitTest(
	idaten::SSRT::Path* paths,
	aten::Intersection* isects,
	aten::ray* rays,
	int* hitbools,
	int width, int height,
	const aten::GeomParameter* __restrict__ shapes, int geomnum,
	const aten::LightParameter* __restrict__ lights, int lightnum,
	cudaTextureObject_t* nodes,
	const aten::PrimitiveParamter* __restrict__ prims,
	cudaTextureObject_t vtxPos,
	aten::mat4* matrices)
{
	const auto ix = blockIdx.x * blockDim.x + threadIdx.x;
	const auto iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix >= width && iy >= height) {
		return;
	}

	const auto idx = getIdx(ix, iy, width);

	auto& path = paths[idx];
	path.isHit = false;

	hitbools[idx] = 0;

	if (path.isTerminate) {
		return;
	}

	Context ctxt;
	{
		ctxt.geomnum = geomnum;
		ctxt.shapes = shapes;
		ctxt.lightnum = lightnum;
		ctxt.lights = lights;
		ctxt.nodes = nodes;
		ctxt.prims = prims;
		ctxt.vtxPos = vtxPos;
		ctxt.matrices = matrices;
	}

	aten::Intersection isect;

	bool isHit = intersectClosest(&ctxt, rays[idx], &isect);

	isects[idx].t = isect.t;
	isects[idx].objid = isect.objid;
	isects[idx].mtrlid = isect.mtrlid;
	isects[idx].meshid = isect.meshid;
	isects[idx].primid = isect.primid;
	isects[idx].a = isect.a;
	isects[idx].b = isect.b;

	path.isHit = isHit;

	hitbools[idx] = isHit ? 1 : 0;
}

template <bool isFirstBounce>
__global__ void shadeMiss(
	idaten::SSRT::Path* paths,
	int width, int height)
{
	const auto ix = blockIdx.x * blockDim.x + threadIdx.x;
	const auto iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix >= width && iy >= height) {
		return;
	}

	const auto idx = getIdx(ix, iy, width);

	auto& path = paths[idx];

	if (!path.isTerminate && !path.isHit) {
		// TODO
		auto bg = aten::vec3(0);

		if (isFirstBounce) {
			path.isKill = true;
		}

		path.contrib += path.throughput * bg;

		path.isTerminate = true;
	}
}

template <bool isFirstBounce>
__global__ void shadeMissWithEnvmap(
	cudaTextureObject_t* textures,
	int envmapIdx,
	real envmapAvgIllum,
	real envmapMultiplyer,
	idaten::SSRT::Path* paths,
	const aten::ray* __restrict__ rays,
	int width, int height)
{
	const auto ix = blockIdx.x * blockDim.x + threadIdx.x;
	const auto iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix >= width && iy >= height) {
		return;
	}

	const auto idx = getIdx(ix, iy, width);

	auto& path = paths[idx];

	if (!path.isTerminate && !path.isHit) {
		auto r = rays[idx];

		auto uv = AT_NAME::envmap::convertDirectionToUV(r.dir);

		auto bg = tex2D<float4>(textures[envmapIdx], uv.x, uv.y);
		auto emit = aten::vec3(bg.x, bg.y, bg.z);

		float misW = 1.0f;
		if (isFirstBounce) {
			path.isKill = true;
		}
		else {
			auto pdfLight = AT_NAME::ImageBasedLight::samplePdf(emit, envmapAvgIllum);
			misW = path.pdfb / (pdfLight + path.pdfb);

			emit *= envmapMultiplyer;
		}

		path.contrib += path.throughput * misW * emit;

		path.isTerminate = true;
	}
}

__global__ void shade(
	unsigned int frame,
	cudaSurfaceObject_t outSurface,
	int width, int height,
	idaten::SSRT::Path* paths,
	int* hitindices,
	int hitnum,
	const aten::Intersection* __restrict__ isects,
	aten::ray* rays,
	int bounce, int rrBounce,
	const aten::GeomParameter* __restrict__ shapes, int geomnum,
	aten::MaterialParameter* mtrls,
	const aten::LightParameter* __restrict__ lights, int lightnum,
	cudaTextureObject_t* nodes,
	const aten::PrimitiveParamter* __restrict__ prims,
	cudaTextureObject_t vtxPos,
	cudaTextureObject_t vtxNml,
	const aten::mat4* __restrict__ matrices,
	cudaTextureObject_t* textures,
	const unsigned int* random,
	idaten::SSRT::ShadowRay* shadowRays)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= hitnum) {
		return;
	}

	Context ctxt;
	{
		ctxt.geomnum = geomnum;
		ctxt.shapes = shapes;
		ctxt.mtrls = mtrls;
		ctxt.lightnum = lightnum;
		ctxt.lights = lights;
		ctxt.nodes = nodes;
		ctxt.prims = prims;
		ctxt.vtxPos = vtxPos;
		ctxt.vtxNml = vtxNml;
		ctxt.matrices = matrices;
		ctxt.textures = textures;
	}

	idx = hitindices[idx];

	auto& path = paths[idx];
	const auto& ray = rays[idx];

#ifdef ENABLE_PROGRESSIVE
#if IDATEN_SAMPLER == IDATEN_SAMPLER_SOBOL
	auto scramble = random[idx] * 0x1fe3434f;
	path.sampler.init(frame, 4 + bounce * 300, scramble);
#elif IDATEN_SAMPLER == IDATEN_SAMPLER_CMJ
	auto rnd = random[idx];
	auto scramble = rnd * 0x1fe3434f * ((frame + 331 * rnd) / (aten::CMJ::CMJ_DIM * aten::CMJ::CMJ_DIM));
	path.sampler.init(frame % (aten::CMJ::CMJ_DIM * aten::CMJ::CMJ_DIM), 4 + bounce * 300, scramble);
#endif
#endif

	aten::hitrecord rec;

	const auto& isect = isects[idx];

	auto obj = &ctxt.shapes[isect.objid];
	evalHitResult(&ctxt, obj, ray, &rec, &isect);

	aten::MaterialParameter mtrl = ctxt.mtrls[rec.mtrlid];

	bool isBackfacing = dot(rec.normal, -ray.dir) < 0.0f;

	// �����ʒu�̖@��.
	// ���̂���̃��C�̓��o���l��.
	aten::vec3 orienting_normal = rec.normal;

	if (mtrl.type != aten::MaterialType::Layer) {
		mtrl.albedoMap = (int)(mtrl.albedoMap >= 0 ? ctxt.textures[mtrl.albedoMap] : -1);
		mtrl.normalMap = (int)(mtrl.normalMap >= 0 ? ctxt.textures[mtrl.normalMap] : -1);
		mtrl.roughnessMap = (int)(mtrl.roughnessMap >= 0 ? ctxt.textures[mtrl.roughnessMap] : -1);
	}

	// Implicit conection to light.
	if (mtrl.attrib.isEmissive) {
		if (!isBackfacing) {
			float weight = 1.0f;

			if (bounce > 0 && !path.isSingular) {
				auto cosLight = dot(orienting_normal, -ray.dir);
				auto dist2 = aten::squared_length(rec.p - ray.org);

				if (cosLight >= 0) {
					auto pdfLight = 1 / rec.area;

					// Convert pdf area to sradian.
					// http://www.slideshare.net/h013/edubpt-v100
					// p31 - p35
					pdfLight = pdfLight * dist2 / cosLight;

					weight = path.pdfb / (pdfLight + path.pdfb);
				}
			}

			path.contrib += path.throughput * weight * mtrl.baseColor;
		}

		// When ray hit the light, tracing will finish.
		path.isTerminate = true;
		return;
	}

	if (!mtrl.attrib.isTranslucent && isBackfacing) {
		orienting_normal = -orienting_normal;
	}

	// Apply normal map.
	int normalMap = mtrl.normalMap;
	if (mtrl.type == aten::MaterialType::Layer) {
		// �ŕ\�w�� NormalMap ��K�p.
		auto* topmtrl = &ctxt.mtrls[mtrl.layer[0]];
		normalMap = (int)(topmtrl->normalMap >= 0 ? ctxt.textures[topmtrl->normalMap] : -1);
	}
	AT_NAME::material::applyNormalMap(normalMap, orienting_normal, orienting_normal, rec.u, rec.v);

#ifdef SEPARATE_SHADOWRAY_HITTEST
	shadowRays[idx].isActive = false;
#endif

	// Explicit conection to light.
	if (!mtrl.attrib.isSingular)
	{
		real lightSelectPdf = 1;
		aten::LightSampleResult sampleres;

		// TODO
		// Importance sampling.
		int lightidx = aten::cmpMin<int>(path.sampler.nextSample() * lightnum, lightnum - 1);
		lightSelectPdf = 1.0f / lightnum;

		aten::LightParameter light;
		light.pos = ((aten::vec4*)ctxt.lights)[lightidx * aten::LightParameter_float4_size + 0];
		light.dir = ((aten::vec4*)ctxt.lights)[lightidx * aten::LightParameter_float4_size + 1];
		light.le = ((aten::vec4*)ctxt.lights)[lightidx * aten::LightParameter_float4_size + 2];
		light.v0 = ((aten::vec4*)ctxt.lights)[lightidx * aten::LightParameter_float4_size + 3];
		light.v1 = ((aten::vec4*)ctxt.lights)[lightidx * aten::LightParameter_float4_size + 4];
		light.v2 = ((aten::vec4*)ctxt.lights)[lightidx * aten::LightParameter_float4_size + 5];
		//auto light = ctxt.lights[lightidx];

		sampleLight(&sampleres, &ctxt, &light, rec.p, orienting_normal, &path.sampler);

		const auto& posLight = sampleres.pos;
		const auto& nmlLight = sampleres.nml;
		real pdfLight = sampleres.pdf;

		auto lightobj = sampleres.obj;

		auto dirToLight = normalize(sampleres.dir);
		auto distToLight = length(posLight - rec.p);

		real distHitObjToRayOrg = AT_MATH_INF;

		// Ray aim to the area light.
		// So, if ray doesn't hit anything in intersectCloserBVH, ray hit the area light.
		auto hitobj = lightobj;

		aten::Intersection isectTmp;

		auto shadowRayOrg = rec.p + AT_MATH_EPSILON * orienting_normal;
		auto tmp = rec.p + dirToLight - shadowRayOrg;
		auto shadowRayDir = normalize(tmp);

#ifdef SEPARATE_SHADOWRAY_HITTEST
		shadowRays[idx].isActive = true;
		shadowRays[idx].org = shadowRayOrg;
		shadowRays[idx].dir = shadowRayDir;
		shadowRays[idx].targetLightId = lightidx;
		shadowRays[idx].distToLight = distToLight;
#else
		aten::ray shadowRay(shadowRayOrg, shadowRayDir);

		bool isHit = intersectCloser(&ctxt, shadowRay, &isectTmp, distToLight - AT_MATH_EPSILON);

		if (isHit) {
			hitobj = (void*)&ctxt.shapes[isectTmp.objid];
		}

		isHit = AT_NAME::scene::hitLight(
			isHit,
			light.attrib,
			lightobj,
			distToLight,
			distHitObjToRayOrg,
			isectTmp.t,
			hitobj);

		if (isHit)
#endif
		{
			auto cosShadow = dot(orienting_normal, dirToLight);

			real pdfb = samplePDF(&ctxt, &mtrl, orienting_normal, ray.dir, dirToLight, rec.u, rec.v);
			auto bsdf = sampleBSDF(&ctxt, &mtrl, orienting_normal, ray.dir, dirToLight, rec.u, rec.v);

			bsdf *= path.throughput;

			// Get light color.
			auto emit = sampleres.finalColor;

			if (light.attrib.isSingular || light.attrib.isInfinite) {
				if (pdfLight > real(0) && cosShadow >= 0) {
					// TODO
					// �W�I���g���^�[���̈����ɂ���.
					// singular light �̏ꍇ�́AfinalColor �ɋ����̏��Z���܂܂�Ă���.
					// inifinite light �̏ꍇ�́A���������ɂȂ�ApdfLight�Ɋ܂܂�鋗�������Ƒł����������H.
					// �i�ł����������̂ŁApdfLight�ɂ͋��������͊܂�ł��Ȃ��j.
					auto misW = pdfLight / (pdfb + pdfLight);
#ifdef SEPARATE_SHADOWRAY_HITTEST
					shadowRays[idx].lightcontrib = 
#else
					path.contrib +=
#endif
						(misW * bsdf * emit * cosShadow / pdfLight) / lightSelectPdf;
				}
			}
			else {
				auto cosLight = dot(nmlLight, -dirToLight);

				if (cosShadow >= 0 && cosLight >= 0) {
					auto dist2 = aten::squared_length(sampleres.dir);
					auto G = cosShadow * cosLight / dist2;

					if (pdfb > real(0) && pdfLight > real(0)) {
						// Convert pdf from steradian to area.
						// http://www.slideshare.net/h013/edubpt-v100
						// p31 - p35
						pdfb = pdfb * cosLight / dist2;

						auto misW = pdfLight / (pdfb + pdfLight);
#ifdef SEPARATE_SHADOWRAY_HITTEST
						shadowRays[idx].lightcontrib =
#else
						path.contrib +=
#endif
							(misW * (bsdf * emit * G) / pdfLight) / lightSelectPdf;
					}
				}
			}
		}
	}

	real russianProb = real(1);

	if (bounce > rrBounce) {
		auto t = normalize(path.throughput);
		auto p = aten::cmpMax(t.r, aten::cmpMax(t.g, t.b));

		russianProb = path.sampler.nextSample();

		if (russianProb >= p) {
			//path.contrib = aten::vec3(0);
			path.isTerminate = true;
		}
		else {
			russianProb = p;
		}
	}
			
	AT_NAME::MaterialSampling sampling;

	sampleMaterial(
		&sampling,
		&ctxt,
		&mtrl,
		orienting_normal,
		ray.dir,
		rec.normal,
		&path.sampler,
		rec.u, rec.v);

	auto nextDir = normalize(sampling.dir);
	auto pdfb = sampling.pdf;
	auto bsdf = sampling.bsdf;

	real c = 1;
	if (!mtrl.attrib.isSingular) {
		// TODO
		// AMD�̂�abs���Ă��邪....
		//c = aten::abs(dot(orienting_normal, nextDir));
		c = dot(orienting_normal, nextDir);
	}

	if (pdfb > 0 && c > 0) {
		path.throughput *= bsdf * c / pdfb;
		path.throughput /= russianProb;
	}
	else {
		path.isTerminate = true;
	}

	// Make next ray.
	rays[idx] = aten::ray(rec.p, nextDir);

	path.pdfb = pdfb;
	path.isSingular = mtrl.attrib.isSingular;
}

__global__ void hitShadowRay(
	idaten::SSRT::Path* paths,
	int* hitindices,
	int hitnum,
	const idaten::SSRT::ShadowRay* __restrict__ shadowRays,
	const aten::GeomParameter* __restrict__ shapes, int geomnum,
	aten::MaterialParameter* mtrls,
	const aten::LightParameter* __restrict__ lights, int lightnum,
	cudaTextureObject_t* nodes,
	const aten::PrimitiveParamter* __restrict__ prims,
	cudaTextureObject_t vtxPos,
	const aten::mat4* __restrict__ matrices)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= hitnum) {
		return;
	}

	Context ctxt;
	{
		ctxt.geomnum = geomnum;
		ctxt.shapes = shapes;
		ctxt.mtrls = mtrls;
		ctxt.lightnum = lightnum;
		ctxt.lights = lights;
		ctxt.nodes = nodes;
		ctxt.prims = prims;
		ctxt.vtxPos = vtxPos;
		ctxt.matrices = matrices;
	}

	idx = hitindices[idx];

	auto& shadowRay = shadowRays[idx];

	if (shadowRay.isActive) {
		auto light = &ctxt.lights[shadowRay.targetLightId];
		auto lightobj = (light->objid >= 0 ? &ctxt.shapes[light->objid] : nullptr);

		real distHitObjToRayOrg = AT_MATH_INF;

		// Ray aim to the area light.
		// So, if ray doesn't hit anything in intersectCloserBVH, ray hit the area light.
		const aten::GeomParameter* hitobj = lightobj;

		aten::Intersection isectTmp;

		bool isHit = false;
		isHit = intersectCloser(&ctxt, shadowRay, &isectTmp, shadowRay.distToLight - AT_MATH_EPSILON);

		if (isHit) {
			hitobj = &ctxt.shapes[isectTmp.objid];
		}
		
		isHit = AT_NAME::scene::hitLight(
			isHit, 
			light->attrib,
			lightobj,
			shadowRay.distToLight,
			distHitObjToRayOrg,
			isectTmp.t,
			hitobj);

		if (isHit) {
			paths[idx].contrib += shadowRay.lightcontrib;
		}
	}
}

__global__ void gather(
	cudaSurfaceObject_t outSurface,
	const idaten::SSRT::Path* __restrict__ paths,
	int width, int height)
{
	const auto ix = blockIdx.x * blockDim.x + threadIdx.x;
	const auto iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix >= width && iy >= height) {
		return;
	}

	const auto idx = getIdx(ix, iy, width);

	const auto& path = paths[idx];

	int sample = path.samples;

	float4 data;
#ifdef ENABLE_PROGRESSIVE
	surf2Dread(&data, outSurface, ix * sizeof(float4), iy);

	// First data.w value is 0.
	int n = data.w;
	data = n * data + make_float4(path.contrib.x, path.contrib.y, path.contrib.z, 0) / sample;
	data /= (n + 1);
	data.w = n + 1;
#else
	data = make_float4(path.contrib.x, path.contrib.y, path.contrib.z, 0) / sample;
	data.w = sample;
#endif

	surf2Dwrite(
		data,
		outSurface,
		ix * sizeof(float4), iy,
		cudaBoundaryModeTrap);
}

namespace idaten {
	void SSRT::prepare()
	{
	}

	void SSRT::update(
		GLuint gltex,
		int width, int height,
		const aten::CameraParameter& camera,
		const std::vector<aten::GeomParameter>& shapes,
		const std::vector<aten::MaterialParameter>& mtrls,
		const std::vector<aten::LightParameter>& lights,
		const std::vector<std::vector<aten::GPUBvhNode>>& nodes,
		const std::vector<aten::PrimitiveParamter>& prims,
		const std::vector<aten::vertex>& vtxs,
		const std::vector<aten::mat4>& mtxs,
		const std::vector<TextureResource>& texs,
		const EnvmapResource& envmapRsc)
	{
		idaten::Renderer::update(
			gltex,
			width, height,
			camera,
			shapes,
			mtrls,
			lights,
			nodes,
			prims,
			vtxs,
			mtxs,
			texs, envmapRsc);

		m_hitbools.init(width * height);
		m_hitidx.init(width * height);

		m_sobolMatrices.init(AT_COUNTOF(sobol::Matrices::matrices));
		m_sobolMatrices.writeByNum(sobol::Matrices::matrices, m_sobolMatrices.maxNum());

		auto& r = aten::getRandom();

		m_random.init(width * height);
		m_random.writeByNum(&r[0], width * height);
	}

	void SSRT::setGBuffer(
		GLuint gltexGbuffer,
		GLuint gltexDepth)
	{
		m_gbuffer.init(gltexGbuffer, idaten::CudaGLRscRegisterType::ReadOnly);
		m_depth.init(gltexDepth, idaten::CudaGLRscRegisterType::ReadOnly);
	}

	static bool doneSetStackSize = false;

	void SSRT::render(
		int width, int height,
		int maxSamples,
		int maxBounce)
	{
#ifdef __AT_DEBUG__
		if (!doneSetStackSize) {
			size_t val = 0;
			cudaThreadGetLimit(&val, cudaLimitStackSize);
			cudaThreadSetLimit(cudaLimitStackSize, val * 4);
			doneSetStackSize = true;
		}
#endif

		int bounce = 0;

		m_paths.init(width * height);
		m_isects.init(width * height);
		m_rays.init(width * height);

#ifdef SEPARATE_SHADOWRAY_HITTEST
		m_shadowRays.init(width * height);
#endif

		cudaMemset(m_paths.ptr(), 0, m_paths.bytes());

		CudaGLResourceMap rscmap(&m_glimg);
		auto outputSurf = m_glimg.bind();

		auto vtxTexPos = m_vtxparamsPos.bind();
		auto vtxTexNml = m_vtxparamsNml.bind();

		{
			std::vector<cudaTextureObject_t> tmp;
			for (int i = 0; i < m_nodeparam.size(); i++) {
				auto nodeTex = m_nodeparam[i].bind();
				tmp.push_back(nodeTex);
			}
			m_nodetex.writeByNum(&tmp[0], tmp.size());
		}

		if (!m_texRsc.empty())
		{
			std::vector<cudaTextureObject_t> tmp;
			for (int i = 0; i < m_texRsc.size(); i++) {
				auto cudaTex = m_texRsc[i].bind();
				tmp.push_back(cudaTex);
			}
			m_tex.writeByNum(&tmp[0], tmp.size());
		}

		static const int rrBounce = 3;

		auto time = AT_NAME::timer::getSystemTime();

		for (int i = 0; i < maxSamples; i++) {
			onGenPath(
				width, height,
				i, maxSamples,
				vtxTexPos,
				vtxTexNml);

			bounce = 0;

			while (bounce < maxBounce) {
				onHitTest(
					width, height,
					bounce,
					vtxTexPos,
					vtxTexNml);
				
				onShadeMiss(width, height, bounce);

				int hitcount = 0;
				idaten::Compaction::compact(
					m_hitidx,
					m_hitbools,
					&hitcount);

				//AT_PRINTF("%d\n", hitcount);

				if (hitcount == 0) {
					break;
				}

				onShade(
					outputSurf,
					hitcount,
					width, height,
					bounce, rrBounce,
					vtxTexPos, vtxTexNml);

				bounce++;
			}
		}

		onGather(outputSurf, width, height, maxSamples);

		checkCudaErrors(cudaDeviceSynchronize());

		m_frame++;

		{
			m_vtxparamsPos.unbind();
			m_vtxparamsNml.unbind();

			for (int i = 0; i < m_nodeparam.size(); i++) {
				m_nodeparam[i].unbind();
			}
			m_nodetex.reset();

			for (int i = 0; i < m_texRsc.size(); i++) {
				m_texRsc[i].unbind();
			}
			m_tex.reset();
		}
	}

	void SSRT::onGenPath(
		int width, int height,
		int sample, int maxSamples,
		cudaTextureObject_t texVtxPos,
		cudaTextureObject_t texVtxNml)
	{
		dim3 block(BLOCK_SIZE, BLOCK_SIZE);
		dim3 grid(
			(width + block.x - 1) / block.x,
			(height + block.y - 1) / block.y);

		genPath << <grid, block >> > (
			m_paths.ptr(),
			m_rays.ptr(),
			width, height,
			sample, maxSamples,
			m_frame,
			m_cam.ptr(),
			m_sobolMatrices.ptr(),
			m_random.ptr());

		checkCudaKernel(genPath);
	}

	void SSRT::onHitTest(
		int width, int height,
		int bounce,
		cudaTextureObject_t texVtxPos,
		cudaTextureObject_t texVtxNml)
	{
		dim3 block(BLOCK_SIZE, BLOCK_SIZE);
		dim3 grid(
			(width + block.x - 1) / block.x,
			(height + block.y - 1) / block.y);

		if (bounce == 0) {
#if 0
			aten::vec4 campos = aten::vec4(m_camParam.origin, 1.0f);

			CudaGLResourceMap rscmap(&m_gbuffer);
			auto gbuffer = m_gbuffer.bind();

			hitTestPrimaryRayInScreenSpace << <grid, block >> > (
				gbuffer,
				m_paths.ptr(),
				m_isects.ptr(), 
				m_hitbools.ptr(), 
				width, height,
				campos,
				m_shapeparam.ptr(),
				m_primparams.ptr(),
				m_mtxparams.ptr(),
				texVtxPos);

			checkCudaKernel(hitTestPrimaryRayInScreenSpace);
#else
			aten::vec4 campos = aten::vec4(m_camParam.origin, 1.0f);

			aten::mat4 mtxW2V;
			aten::mat4 mtxV2C;

			mtxW2V.lookat(
				m_camParam.origin,
				m_camParam.center,
				m_camParam.up);

			mtxV2C.perspective(
				m_camParam.znear,
				m_camParam.zfar,
				m_camParam.vfov,
				m_camParam.aspect);

			CudaGLResourceMap rscmapGbuffer(&m_gbuffer);
			CudaGLResourceMap rscmapDepth(&m_depth);
			auto gbuffer = m_gbuffer.bind();
			auto depth = m_depth.bind();

			hitTestPrimaryRayInScreenSpaceEx << <grid, block >> > (
				gbuffer, depth,
				m_paths.ptr(),
				m_isects.ptr(),
				m_hitbools.ptr(),
				width, height,
				campos, m_camParam.znear,
				mtxW2V, mtxV2C,
				m_rays.ptr(),
				m_shapeparam.ptr(),
				m_primparams.ptr(),
				m_mtxparams.ptr(),
				texVtxPos,
				texVtxNml);
#endif
		}
		else {
			hitTest << <grid, block >> > (
				m_paths.ptr(),
				m_isects.ptr(),
				m_rays.ptr(),
				m_hitbools.ptr(),
				width, height,
				m_shapeparam.ptr(), m_shapeparam.num(),
				m_lightparam.ptr(), m_lightparam.num(),
				m_nodetex.ptr(),
				m_primparams.ptr(),
				texVtxPos,
				m_mtxparams.ptr());

			checkCudaKernel(hitTest);
		}
	}

	void SSRT::onShadeMiss(
		int width, int height,
		int bounce)
	{
		dim3 block(BLOCK_SIZE, BLOCK_SIZE);
		dim3 grid(
			(width + block.x - 1) / block.x,
			(height + block.y - 1) / block.y);

		if (m_envmapRsc.idx >= 0) {
			if (bounce == 0) {
				shadeMissWithEnvmap<true> << <grid, block >> > (
					m_tex.ptr(),
					m_envmapRsc.idx, m_envmapRsc.avgIllum, m_envmapRsc.multiplyer,
					m_paths.ptr(),
					m_rays.ptr(),
					width, height);
			}
			else {
				shadeMissWithEnvmap<false> << <grid, block >> > (
					m_tex.ptr(),
					m_envmapRsc.idx, m_envmapRsc.avgIllum, m_envmapRsc.multiplyer,
					m_paths.ptr(),
					m_rays.ptr(),
					width, height);
			}
		}
		else {
			if (bounce == 0) {
				shadeMiss<true> << <grid, block >> > (
					m_paths.ptr(),
					width, height);
			}
			else {
				shadeMiss<false> << <grid, block >> > (
					m_paths.ptr(),
					width, height);
			}
		}

		checkCudaKernel(shadeMiss);
	}

	void SSRT::onShade(
		cudaSurfaceObject_t outputSurf,
		int hitcount,
		int width, int height,
		int bounce, int rrBounce,
		cudaTextureObject_t texVtxPos,
		cudaTextureObject_t texVtxNml)
	{
		dim3 blockPerGrid((hitcount + 64 - 1) / 64);
		dim3 threadPerBlock(64);

		shade << <blockPerGrid, threadPerBlock >> > (
			m_frame,
			outputSurf,
			width, height,
			m_paths.ptr(),
			m_hitidx.ptr(), hitcount,
			m_isects.ptr(),
			m_rays.ptr(),
			bounce, rrBounce,
			m_shapeparam.ptr(), m_shapeparam.num(),
			m_mtrlparam.ptr(),
			m_lightparam.ptr(), m_lightparam.num(),
			m_nodetex.ptr(),
			m_primparams.ptr(),
			texVtxPos, texVtxNml,
			m_mtxparams.ptr(),
			m_tex.ptr(),
			m_random.ptr(),
			m_shadowRays.ptr());

		checkCudaKernel(shade);

#ifdef SEPARATE_SHADOWRAY_HITTEST
		hitShadowRay << <blockPerGrid, threadPerBlock >> > (
			//hitShadowRay << <1, 1 >> > (
			m_paths.ptr(),
			m_hitidx.ptr(), hitcount,
			m_shadowRays.ptr(),
			m_shapeparam.ptr(), m_shapeparam.num(),
			m_mtrlparam.ptr(),
			m_lightparam.ptr(), m_lightparam.num(),
			m_nodetex.ptr(),
			m_primparams.ptr(),
			texVtxPos,
			m_mtxparams.ptr());

		checkCudaKernel(hitShadowRay);
#endif
	}

	void SSRT::onGather(
		cudaSurfaceObject_t outputSurf,
		int width, int height,
		int maxSamples)
	{
		dim3 block(BLOCK_SIZE, BLOCK_SIZE);
		dim3 grid(
			(width + block.x - 1) / block.x,
			(height + block.y - 1) / block.y);

		gather << <grid, block >> > (
			outputSurf,
			m_paths.ptr(),
			width, height);
	}
}