#pragma once

#include "aten4idaten.h"
#include "cuda/cudamemory.h"
#include "cuda/cudaGLresource.h"

#include "kernel/renderer.h"

namespace idaten
{
	class PathTracing : public Renderer {
	public:
		struct ShadowRay {
			aten::ray r;
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
		AT_STATICASSERT((sizeof(Path) % 4) == 0);
#else
		struct Path;
#endif

	public:
		PathTracing() {}
		virtual ~PathTracing() {}

	public:
		void prepare();

		virtual void render(
			int width, int height,
			int maxSamples,
			int maxBounce) override final;

		virtual void update(
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
			const EnvmapResource& envmapRsc) override;

		void updateMaterial(const std::vector<aten::MaterialParameter>& mtrls);

		virtual void enableRenderAOV(
			GLuint gltexPosition,
			GLuint gltexNormal,
			GLuint gltexAlbedo,
			const aten::vec3& posRange) override;

		void enableProgressive(bool enable)
		{
			m_enableProgressive = enable;
		}
		bool isProgressive() const
		{
			return m_enableProgressive;
		}

	protected:
		virtual void onGenPath(
			int width, int height,
			int sample, int maxSamples,
			cudaTextureObject_t texVtxPos,
			cudaTextureObject_t texVtxNml);

		virtual void onHitTest(
			int width, int height,
			cudaTextureObject_t texVtxPos);

		virtual void onShadeMiss(
			int width, int height,
			int bounce);

		virtual void onShade(
			cudaSurfaceObject_t outputSurf,
			int hitcount,
			int width, int height,
			int bounce, int rrBounce,
			cudaTextureObject_t texVtxPos,
			cudaTextureObject_t texVtxNml);

		virtual void onGather(
			cudaSurfaceObject_t outputSurf,
			int width, int height,
			int maxSamples);

	protected:
		idaten::TypedCudaMemory<idaten::PathTracing::Path> m_paths;
		idaten::TypedCudaMemory<aten::Intersection> m_isects;
		idaten::TypedCudaMemory<aten::ray> m_rays;
		idaten::TypedCudaMemory<idaten::PathTracing::ShadowRay> m_shadowRays;

		idaten::TypedCudaMemory<int> m_hitbools;
		idaten::TypedCudaMemory<int> m_hitidx;

		idaten::TypedCudaMemory<unsigned int> m_sobolMatrices;
		idaten::TypedCudaMemory<unsigned int> m_random;

		bool m_enableAOV{ false };
		aten::vec3 m_posRange{ aten::vec3(real(1)) };
		idaten::TypedCudaMemory<cudaSurfaceObject_t> m_aovCudaRsc;
		std::vector<idaten::CudaGLSurface> m_aovs;

		uint32_t m_frame{ 1 };

		bool m_enableProgressive{ false };
	};
}
