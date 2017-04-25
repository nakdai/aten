#pragma once

#include "types.h"
#include "scene/bvh.h"
#include "material/material.h"
#include "math/mat4.h"
#include "shape/shape.h"
#include "object/vertex.h"

namespace aten
{
	class shape;

	class face : public bvhnode {
	public:
		face() {}
		virtual ~face() {}

	public:
		virtual bool hit(
			const ray& r,
			real t_min, real t_max,
			hitrecord& rec) const override;

		static bool hit(
			const ShapeParameter& param,
			const vertex& v0,
			const vertex& v1,
			const vertex& v2,
			const ray& r,
			real t_min, real t_max,
			hitrecord& rec);

		virtual aabb getBoundingbox() const override
		{
			return std::move(param.bbox);
		}

		virtual vec3 getRandomPosOn(sampler* sampler) const override;

		virtual SamplingPosNormalPdf getSamplePosNormalPdf(sampler* sampler) const override;

		virtual const ShapeParameter& getParam() const override final
		{
			return param;
		}

		void build();
	
		ShapeParameter param;

		shape* parent{ nullptr };
	};

	class shape : public bvhnode {
	public:
		shape() {}
		virtual ~shape() {}

		void build();

		virtual bool hit(
			const ray& r,
			real t_min, real t_max,
			hitrecord& rec) const override final;

		virtual vec3 getRandomPosOn(sampler* sampler) const override final
		{
			auto r = sampler->nextSample();
			int idx = (int)(r * (faces.size() - 1));
			auto face = faces[idx];
			return face->getRandomPosOn(sampler);
		}

		virtual SamplingPosNormalPdf getSamplePosNormalPdf(sampler* sampler) const override final
		{
			auto r = sampler->nextSample();
			int idx = (int)(r * (faces.size() - 1));
			auto face = faces[idx];
			return face->getSamplePosNormalPdf(sampler);
		}
		
		std::vector<face*> faces;
		material* mtrl{ nullptr };
		real area;

	private:
		bvhnode m_node;
	};

	template<typename T> class instance;

	class object {
		friend class instance<object>;

	public:
		object() {}
		~object() {}

	public:
		bool hit(
			const ray& r,
			const mat4& mtxL2W,
			real t_min, real t_max,
			hitrecord& rec);

	private:
		void build();

		vec3 getRandomPosOn(sampler* sampler) const
		{
			auto r = sampler->nextSample();
			int idx = (int)(r * (shapes.size() - 1));
			auto shape = shapes[idx];
			return shape->getRandomPosOn(sampler);
		}

		hitable::SamplingPosNormalPdf getSamplePosNormalPdf(const mat4& mtxL2W, sampler* sampler) const;

	public:
		std::vector<shape*> shapes;
		aabb bbox;

	private:
		bvhnode m_node;
		real m_area;
		uint32_t m_triangles{ 0 };
	};
}
