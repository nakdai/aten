#include "geometry/object.h"
#include "math/intersect.h"
#include "accelerator/accelerator.h"

#include <iterator>

//#define ENABLE_LINEAR_HITTEST

namespace AT_NAME
{
	void object::build()
	{
		if (m_triangles > 0) {
			// Builded already.
			return;
		}

		m_accel = aten::accelerator::createAccelerator();

		param.primid = shapes[0]->faces[0]->id;

		param.area = 0;
		m_triangles = 0;

		// Avoid sorting objshape list in bvh::build directly.
		std::vector<face*> tmp;

		bbox.empty();

		for (const auto s : shapes) {
			s->build();

			param.area += s->param.area;
			m_triangles += (uint32_t)s->faces.size();

			tmp.insert(tmp.end(), s->faces.begin(), s->faces.end());

			aabb::merge(bbox, s->m_aabb);
		}

		param.primnum = m_triangles;

		m_accel->asNested();
		m_accel->build((hitable**)&tmp[0], (uint32_t)tmp.size(), &bbox);

		bbox = m_accel->getBoundingbox();
	}

	bool object::hit(
		const aten::ray& r,
		real t_min, real t_max,
		aten::Intersection& isect) const
	{
		bool isHit = m_accel->hit(r, t_min, t_max, isect);

		if (isHit) {
			auto f = face::faces()[isect.objid];

			// 自身のIDを返す.
			isect.objid = id();
		}
		return isHit;
	}

	void object::evalHitResult(
		const aten::ray& r,
		const aten::mat4& mtxL2W,
		aten::hitrecord& rec,
		const aten::Intersection& isect) const
	{
		auto f = face::faces()[isect.primid];

		auto& vtxs = aten::VertexManager::getVertices();

		const auto& v0 = vtxs[f->param.idx[0]];
		const auto& v1 = vtxs[f->param.idx[1]];
		const auto& v2 = vtxs[f->param.idx[2]];

		//face::evalHitResult(v0, v1, v2, &rec, &isect);
		f->evalHitResult(r, rec, isect);

		real orignalLen = 0;
		{
			const auto& p0 = v0.pos;
			const auto& p1 = v1.pos;

			orignalLen = length(p1.v - p0.v);
		}

		real scaledLen = 0;
		{
			auto p0 = mtxL2W.apply(v0.pos);
			auto p1 = mtxL2W.apply(v1.pos);

			scaledLen = length(p1.v - p0.v);
		}

		real ratio = scaledLen / orignalLen;
		ratio = ratio * ratio;

		rec.area = param.area * ratio;

		rec.objid = isect.objid;
		rec.mtrlid = isect.mtrlid;
	}

	void object::getSamplePosNormalArea(
		aten::hitable::SamplePosNormalPdfResult* result,
		const aten::mat4& mtxL2W, 
		aten::sampler* sampler) const
	{
		auto r = sampler->nextSample();
		int shapeidx = (int)(r * (shapes.size() - 1));
		auto objshape = shapes[shapeidx];

		r = sampler->nextSample();
		int faceidx = (int)(r * (objshape->faces.size() - 1));
		auto f = objshape->faces[faceidx];

		const auto& v0 = aten::VertexManager::getVertex(f->param.idx[0]);
		const auto& v1 = aten::VertexManager::getVertex(f->param.idx[1]);

		real orignalLen = 0;
		{
			const auto& p0 = v0.pos;
			const auto& p1 = v1.pos;

			orignalLen = (p1 - p0).length();
		}

		real scaledLen = 0;
		{
			auto p0 = mtxL2W.apply(v0.pos);
			auto p1 = mtxL2W.apply(v1.pos);

			scaledLen = length(p1.v - p0.v);
		}

		real ratio = scaledLen / orignalLen;
		ratio = ratio * ratio;

		auto area = param.area * ratio;

		f->getSamplePosNormalArea(result, sampler);

		result->area = area;
	}

	void object::getPrimitives(aten::PrimitiveParamter* primparams) const
	{
		int cnt = 0;

		for (auto s : shapes) {
			const auto& shapeParam = s->param;
			
			auto mtrlid = material::findMaterialIdx((material*)shapeParam.mtrl.ptr);

			for (auto f : s->faces) {
				auto faceParam = f->param;
				faceParam.mtrlid = mtrlid;
				primparams[cnt++] = faceParam;
			}
		}
	}

	void object::draw(
		aten::hitable::FuncPreDraw func,
		const aten::mat4& mtxL2W,
		int parentId)
	{
		int objid = (parentId < 0 ? id() : parentId);

		for (auto s : shapes) {
			s->draw(func, mtxL2W, objid);
		}
	}

	void object::drawAABB(
		aten::hitable::FuncDrawAABB func,
		const aten::mat4& mtxL2W)
	{
		m_accel->drawAABB(func, mtxL2W);
	}

	bool object::exportInternalAccelTree(const char* path)
	{
		bool result = false;
		if (m_accel) {
			result = m_accel->exportTree(path);
		}
		return result;
	}
}
