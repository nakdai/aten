#include "kernel/raytracing.h"
#include "kernel/context.cuh"
#include "kernel/light.cuh"
#include "kernel/material.cuh"
#include "kernel/intersect.cuh"
#include "kernel/bvh.cuh"
#include "kernel/common.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "cuda/helper_math.h"
#include "cuda/cudautil.h"
#include "cuda/cudamemory.h"

#include "aten4idaten.h"

struct Path {
	aten::ray ray;
	aten::vec3 throughput;
	aten::hitrecord rec;
	bool isHit;
	bool isTerminate;
};

__global__ void genPathRayTracing(
	Path* paths,
	int width, int height,
	aten::CameraParameter* camera)
{
	const auto ix = blockIdx.x * blockDim.x + threadIdx.x;
	const auto iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix >= width && iy >= height) {
		return;
	}

	const auto idx = iy * camera->width + ix;

	float s = (ix + 0.5f) / (float)(camera->width - 1);
	float t = (iy + 0.5f) / (float)(camera->height - 1);

	AT_NAME::CameraSampleResult camsample;
	AT_NAME::PinholeCamera::sample(&camsample, camera, s, t);

	auto& path = paths[idx];

	path.ray = camsample.r;
	path.throughput = aten::make_float3(1);
	path.isHit = false;
	path.isTerminate = false;
}

__global__ void hitTestRayTracing(
	Path* paths,
	int width, int height,
	aten::ShapeParameter* shapes, int geomnum,
	aten::MaterialParameter* mtrls,
	aten::LightParameter* lights, int lightnum,
	cudaTextureObject_t* nodes,
	aten::PrimitiveParamter* prims,
	cudaTextureObject_t vertices)
{
	const auto ix = blockIdx.x * blockDim.x + threadIdx.x;
	const auto iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix >= width && iy >= height) {
		return;
	}

	const auto idx = iy * width + ix;

	auto& path = paths[idx];
	path.isHit = false;

	if (path.isTerminate) {
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
		ctxt.vertices = vertices;
	}
	
	aten::hitrecord rec;
	bool isHit = intersectBVH(&ctxt, path.ray, AT_MATH_EPSILON, AT_MATH_INF, &rec);

	path.isHit = isHit;
	path.rec = rec;
}

__global__ void raytracing(
	//float4* p,
	cudaSurfaceObject_t outSurface,
	Path* paths,
	int width, int height,
	aten::ShapeParameter* shapes, int geomnum,
	aten::MaterialParameter* mtrls,
	aten::LightParameter* lights, int lightnum,
	cudaTextureObject_t* nodes,
	aten::PrimitiveParamter* prims,
	cudaTextureObject_t vertices)
{
	const auto ix = blockIdx.x * blockDim.x + threadIdx.x;
	const auto iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix >= width && iy >= height) {
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
		ctxt.vertices = vertices;
	}

	const auto idx = iy * width + ix;

	auto& path = paths[idx];

	if (!path.isHit) {
		return;
	}
	if (path.isTerminate) {
		return;
	}

	aten::vec3 contrib = aten::make_float3(0);

	const aten::MaterialParameter* mtrl = &ctxt.mtrls[path.rec.mtrlid];

	if (mtrl->attrib.isEmissive) {
		contrib = path.throughput * mtrl->baseColor;

		path.isTerminate = true;
		//p[idx] = make_float4(contrib.x, contrib.y, contrib.z, 1);
		surf2Dwrite(make_float4(contrib.x, contrib.y, contrib.z, 1), outSurface, ix * sizeof(float4), iy, cudaBoundaryModeTrap);
		
		return;
	}

	// 交差位置の法線.
	// 物体からのレイの入出を考慮.
	const aten::vec3 orienting_normal = dot(path.rec.normal, path.ray.dir) < 0.0 ? path.rec.normal : -path.rec.normal;

	if (mtrl->attrib.isSingular || mtrl->attrib.isTranslucent) {
		AT_NAME::MaterialSampling sampling;
			
		sampleMaterial(
			&sampling,
			mtrl,
			orienting_normal, 
			path.ray.dir,
			path.rec.normal,
			nullptr,
			path.rec.u, path.rec.v);

		auto nextDir = normalize(sampling.dir);

		path.throughput *= sampling.bsdf;

		// Make next ray.
		path.ray = aten::ray(path.rec.p, nextDir);
	}
	else {
		// TODO
		auto light = lights[0];
		auto* sphere = &ctxt.shapes[light.object.idx];
		light.object.ptr = sphere;

		aten::LightSampleResult sampleres;
		sampleLight(&sampleres, &ctxt, &light, path.rec.p, nullptr);

		aten::vec3 dirToLight = sampleres.dir;
		auto len = dirToLight.length();

		dirToLight.normalize();

		aten::ray shadowRay(path.rec.p, dirToLight);

		aten::hitrecord tmpRec;

		bool isHit = intersectBVH(&ctxt, shadowRay, AT_MATH_EPSILON, AT_MATH_INF, &tmpRec);

		if (AT_NAME::scene::hitLight(isHit, &light, sampleres.pos, shadowRay, AT_MATH_EPSILON, AT_MATH_INF, &tmpRec)) {
			if (light.attrib.isInfinite) {
				len = 1.0f;
			}

			const auto c0 = max(0.0f, dot(orienting_normal, dirToLight));
			float c1 = 1.0f;

			if (!light.attrib.isSingular) {
				c1 = max(0.0f, dot(sampleres.nml, -dirToLight));
			}

			auto G = c0 * c1 / (len * len);

			contrib += path.throughput * (mtrl->baseColor * sampleres.finalColor) * G;
		}

		path.isTerminate = true;
		//p[idx] = make_float4(contrib.x, contrib.y, contrib.z, 1);
		surf2Dwrite(make_float4(contrib.x, contrib.y, contrib.z, 1), outSurface, ix * sizeof(float4), iy, cudaBoundaryModeTrap);
	}
}

namespace idaten {
	void RayTracing::prepare()
	{
		addFuncs();
	}

	void RayTracing::render(
		aten::vec4* image,
		int width, int height)
	{
		dim3 block(16, 16);
		dim3 grid(
			(width + block.x - 1) / block.x,
			(height + block.y - 1) / block.y);

		int depth = 0;

		idaten::TypedCudaMemory<Path> paths;
		paths.init(width * height);

		CudaGLResourceMap rscmap(&glimg);
		auto outputSurf = glimg.bind();

		auto vtxTex = vtxparams.bind();

		std::vector<cudaTextureObject_t> tmp;
		for (int i = 0; i < nodeparam.size(); i++) {
			auto nodeTex = nodeparam[i].bind();
			tmp.push_back(nodeTex);
		}
		nodetex.writeByNum(&tmp[0], tmp.size());

		genPathRayTracing << <grid, block >> > (
			paths.ptr(),
			width, height,
			cam.ptr());

		//checkCudaErrors(cudaDeviceSynchronize());

		while (depth < 5) {
			hitTestRayTracing << <grid, block >> > (
			//hitTestRayTracing << <1, 1 >> > (
				paths.ptr(),
				width, height,
				shapeparam.ptr(), shapeparam.num(),
				mtrlparam.ptr(),
				lightparam.ptr(), lightparam.num(),
				nodetex.ptr(),
				primparams.ptr(),
				vtxTex);

			auto err = cudaGetLastError();
			if (err != cudaSuccess) {
				AT_PRINTF("Cuda Kernel Err(hitTest) [%s]\n", cudaGetErrorString(err));
			}

			//checkCudaErrors(cudaDeviceSynchronize());

			raytracing << <grid, block >> > (
				//(float4*)dst.ptr(),
				outputSurf,
				paths.ptr(),
				width, height,
				shapeparam.ptr(), shapeparam.num(),
				mtrlparam.ptr(),
				lightparam.ptr(), lightparam.num(),
				nodetex.ptr(),
				primparams.ptr(),
				vtxTex);

			err = cudaGetLastError();
			if (err != cudaSuccess) {
				AT_PRINTF("Cuda Kernel Err(raytracing) [%s]\n", cudaGetErrorString(err));
			}

			//checkCudaErrors(cudaDeviceSynchronize());

			depth++;
		}

		checkCudaErrors(cudaDeviceSynchronize());

		vtxparams.unbind();
		for (int i = 0; i < nodeparam.size(); i++) {
			nodeparam[i].unbind();
		}
		nodetex.reset();

		//dst.read(image, sizeof(aten::vec4) * width * height);
	}
}