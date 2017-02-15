#pragma once

#include "scene/hitable.h"
#include "scene/accel.h"
#include "sampler/random.h"

namespace aten {
	class bvh : public accel {
	public:
		bvh() {}
		virtual ~bvh() {}

	private:
		bvh(
			hitable** list,
			uint32_t num)
		{
			build(list, num);
		}

	public:
		virtual void build(
			hitable** list,
			uint32_t num) override;

		virtual bool hit(
			const ray& r,
			real t_min, real t_max,
			hitrecord& rec) const override;

		virtual aabb getBoundingbox() const override
		{
			return std::move(m_aabb);
		}

	protected:
		hitable* m_left{ nullptr };
		hitable* m_right{ nullptr };
		aabb m_aabb;
	};
}
