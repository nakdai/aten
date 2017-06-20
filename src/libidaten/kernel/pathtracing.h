#pragma once

#include "aten4idaten.h"
#include "cuda/cudamemory.h"
#include "cuda/cudaGLresource.h"

#include "kernel/renderer.h"

namespace idaten
{
	class PathTracing : public Renderer {
	public:
		struct ShadowRay : public aten::ray {
			aten::vec3 lightcontrib;
			real distToLight;
			int targetLightId;

			struct {
				uint32_t isActive : 1;
			};
		};

#ifdef __AT_CUDA__
		struct Path {
			aten::vec3 throughput;
			aten::vec3 contrib;
			aten::sampler sampler;

			real pdfb;
			int samples;

			bool isHit;
			bool isTerminate;
			bool isSingular;
			bool isKill;
		};
#else
		struct Path;
#endif

	public:
		PathTracing() {}
		virtual ~PathTracing() {}

	public:
		void prepare();

		virtual void render(
			aten::vec4* image,
			int width, int height,
			int maxSamples,
			int maxDepth) override final;

		virtual void update(
			GLuint gltex,
			int width, int height,
			const aten::CameraParameter& camera,
			const std::vector<aten::ShapeParameter>& shapes,
			const std::vector<aten::MaterialParameter>& mtrls,
			const std::vector<aten::LightParameter>& lights,
			const std::vector<std::vector<aten::BVHNode>>& nodes,
			const std::vector<aten::PrimitiveParamter>& prims,
			const std::vector<aten::vertex>& vtxs,
			const std::vector<aten::mat4>& mtxs,
			const std::vector<TextureResource>& texs,
			const EnvmapResource& envmapRsc) override;

	protected:
		inline void onGenPath(
			int width, int height,
			int sample, int maxSamples,
			int seed);

		virtual void onHitTest(
			int width, int height,
			cudaTextureObject_t texVtxPos,
			int depth,
			int sample, int maxSamples,
			int seed);

		inline void onShadeMiss(
			int width, int height,
			int depth);

		inline void onShade(
			cudaSurfaceObject_t outputSurf,
			int hitcount,
			int depth, int rrDepth,
			cudaTextureObject_t texVtxPos,
			cudaTextureObject_t texVtxNml);

		virtual void onGather(
			cudaSurfaceObject_t outputSurf,
			Path* path,
			int width, int height);

	private:
		idaten::TypedCudaMemory<int> m_hitbools;
		idaten::TypedCudaMemory<int> m_hitidx;
	};

	class PathTracingGeometryRendering : public PathTracing {
	public:
		PathTracingGeometryRendering() {}
		virtual ~PathTracingGeometryRendering() {}

	public:
		virtual void update(
			GLuint gltex,
			int width, int height,
			const aten::CameraParameter& camera,
			const std::vector<aten::ShapeParameter>& shapes,
			const std::vector<aten::MaterialParameter>& mtrls,
			const std::vector<aten::LightParameter>& lights,
			const std::vector<std::vector<aten::BVHNode>>& nodes,
			const std::vector<aten::PrimitiveParamter>& prims,
			const std::vector<aten::vertex>& vtxs,
			const std::vector<aten::mat4>& mtxs,
			const std::vector<TextureResource>& texs,
			const EnvmapResource& envmapRsc) override final;

	private:
		virtual void onHitTest(
			int width, int height,
			cudaTextureObject_t texVtxPos,
			int depth,
			int sample, int maxSamples,
			int seed) override final;

		virtual void onGather(
			cudaSurfaceObject_t outputSurf,
			Path* path,
			int width, int height) override final;

	private:
		idaten::TypedCudaMemory<int> m_mtrlIds;
	};
}
