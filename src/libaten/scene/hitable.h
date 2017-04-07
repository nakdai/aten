#pragma once 

#include <vector>
#include "types.h"
#include "scene/aabb.h"
#include "math/vec3.h"
#include "material/material.h"
#include "sampler/sampler.h"

namespace aten {
	class hitable;

	struct hitrecord {
		real t{ AT_MATH_INF };

		vec3 p;

		vec3 normal;

		// tangent coordinate.
		vec3 du;
		vec3 dv;

		// texture coordinate.
		real u{ real(0) };
		real v{ real(0) };

		real area{ real(1) };

		hitable* obj{ nullptr };

		material* mtrl{ nullptr };
	};

	class hitable {
	public:
		hitable(const char* name = nullptr);
		virtual ~hitable() {}

	public:
		virtual bool hit(
			const ray& r,
			real t_min, real t_max,
			hitrecord& rec) const = 0;

		virtual aabb getBoundingbox() const = 0;

		virtual vec3 getRandomPosOn(sampler* sampler) const
		{
			AT_ASSERT(false);
			return std::move(vec3());
		}

		virtual std::tuple<vec3, vec3> getSamplePosAndNormal(sampler* sampler) const
		{
			AT_ASSERT(false);
			return std::move(std::tuple<vec3, vec3>(vec3(), vec3()));
		}

		uint32_t id() const
		{
			return m_id;
		}

	private:
		uint32_t m_id{ 0 };
		const char* m_name;
	};
}
