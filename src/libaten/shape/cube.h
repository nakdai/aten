#pragma once

#include "types.h"
#include "scene/bvh.h"
#include "math/mat4.h"

namespace aten
{
	template<typename T> class instance;

	class cube : public bvhnode {
		friend class instance<cube>;

	public:
		cube() {}
		cube(const vec3& c, real w, real h, real d, material* m);

		cube(real w, real h, real d, material* m)
			: cube(vec3(0), w, h, d, m)
		{}

		virtual ~cube() {}

	public:
		virtual bool hit(
			const ray& r,
			real t_min, real t_max,
			hitrecord& rec) const override final;

		virtual aabb getBoundingbox() const override final;

		const vec3& center() const
		{
			return m_center;
		}

		const vec3& size() const
		{
			return m_size;
		}

		virtual vec3 getRandomPosOn(sampler* sampler) const override final;

		virtual std::tuple<vec3, vec3> getSamplePosAndNormal(sampler* sampler) const override final;

	private:
		bool hit(
			const ray& r,
			const mat4& mtxL2W,
			real t_min, real t_max,
			hitrecord& rec) const;

	private:
		enum Face {
			POS_X,
			NEG_X,
			POS_Y,
			NEG_Y,
			POS_Z,
			NEG_Z,
		};

		Face onGetRandomPosOn(vec3& pos, sampler* sampler) const;

		static Face findFace(const vec3& d);

	private:
		vec3 m_center;
		vec3 m_size;
		aabb m_bbox;
		material* m_mtrl{ nullptr };
	};
}
