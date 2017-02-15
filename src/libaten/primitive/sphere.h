#pragma once

#include "types.h"
#include "scene/hitable.h"

namespace aten
{
	class sphere : public hitable {
	public:
		sphere() {}
		sphere(const vec3& c, real r, material* m)
			: m_center(c), m_radius(r), m_mtrl(m)
		{};

		virtual ~sphere() {}

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

		real radius() const
		{
			return m_radius;
		}

	private:
		vec3 m_center;
		real m_radius{ CONST_REAL(0.0) };
		material* m_mtrl{ nullptr };
	};
}
