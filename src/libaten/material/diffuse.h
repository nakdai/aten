#pragma once

#include "material/material.h"

namespace aten
{
	class diffuse : public material {
	public:
		diffuse() {}
		diffuse(const vec3& c)
			: m_color(c)
		{}

		virtual ~diffuse() {}

		virtual vec3 color() const override final
		{
			return m_color;
		}

		virtual real pdf(const vec3& normal, const vec3& dir) const override final;

		virtual vec3 sampleDirection(const vec3& normal, sampler* sampler) const override final;

		virtual vec3 brdf(const vec3& normal, const vec3& dir) const override final;

	private:
		vec3 m_color;
	};
}
