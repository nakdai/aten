#pragma once

#include "aten4idaten.h"

__device__ void addIntersectFuncs();

AT_DEVICE_API bool intersectShape(
	const aten::ShapeParameter& shape,
	const aten::ray& r,
	float t_min, float t_max,
	aten::hitrecord& rec);