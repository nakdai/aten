#pragma once

#include <atomic>

#include "types.h"
#include "material/material.h"
#include "math/mat4.h"
#include "geometry/geomparam.h"
#include "geometry/transformable.h"
#include "geometry/geombase.h"
#include "geometry/vertex.h"

namespace AT_NAME
{
    class face : public aten::hitable {
        friend class context;

    public:
        face() = default;
        virtual ~face() {}

        virtual bool hit(
            const aten::context& ctxt,
            const aten::ray& r,
            real t_min, real t_max,
            aten::Intersection& isect) const override;

        static bool hit(
            const aten::PrimitiveParamter* param,
            const aten::vec3& v0,
            const aten::vec3& v1,
            const aten::vec3& v2,
            const aten::ray& r,
            real t_min, real t_max,
            aten::Intersection* isect);

        virtual void evalHitResult(
            const aten::context& ctxt,
            const aten::ray& r,
            aten::hitrecord& rec,
            const aten::Intersection& isect) const final;

        static void evalHitResult(
            const aten::vertex& v0,
            const aten::vertex& v1,
            const aten::vertex& v2,
            aten::hitrecord* rec,
            const aten::Intersection* isect);

        virtual void getSamplePosNormalArea(
            const aten::context& ctxt,
            aten::hitable::SamplePosNormalPdfResult* result,
            aten::sampler* sampler) const override;

        virtual int geomid() const override;

        void build(
            const aten::context& ctxt,
            int mtrlid,
            int geomid);

        aten::aabb computeAABB(const aten::context& ctxt) const;

        const aten::PrimitiveParamter& getParam() const
        {
            return param;
        }

        void setParam(const aten::PrimitiveParamter& p)
        {
            param = p;
        }

        int getId() const
        {
            return m_id;
        }

    private:
        static std::shared_ptr<face> create(
            const aten::context& ctxt,
            const aten::PrimitiveParamter& param);

        template <typename T>
        auto updateIndex(T id)
            -> std::enable_if_t<(std::is_signed<T>::value && !std::is_floating_point<T>::value) || std::is_same<T, std::size_t>::value, void>
        {
            m_id = static_cast<decltype(m_id)>(id);
        }

    private:
        aten::PrimitiveParamter param;
        int m_id{ -1 };
    };
}
