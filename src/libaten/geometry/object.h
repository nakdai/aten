#pragma once

#include <atomic>

#include "types.h"
#include "material/material.h"
#include "math/mat4.h"
#include "geometry/face.h"
#include "geometry/objshape.h"
#include "geometry/transformable.h"

namespace AT_NAME
{
    class object : public aten::transformable {
    public:
        object() 
            : param(aten::GeometryType::Polygon), transformable(aten::GeometryType::Polygon)
        {}
        virtual ~object();

    public:
        virtual bool hit(
            const aten::ray& r,
            real t_min, real t_max,
            aten::Intersection& isect) const override final;

        virtual void evalHitResult(
            const aten::ray& r,
            const aten::mat4& mtxL2W,
            aten::hitrecord& rec,
            const aten::Intersection& isect) const override final;

        virtual const aten::GeomParameter& getParam() const override final
        {
            return param;
        }

        virtual aten::accelerator* getInternalAccelerator() override final
        {
            return m_accel;
        }

        virtual void draw(
            aten::hitable::FuncPreDraw func,
            const aten::mat4& mtxL2W,
            const aten::mat4& mtxPrevL2W,
            int parentId,
            uint32_t triOffset) override final;

        void draw(AT_NAME::FuncObjectMeshDraw func);

        virtual void drawAABB(
            aten::hitable::FuncDrawAABB func,
            const aten::mat4& mtxL2W) override final;

        bool exportInternalAccelTree(const char* path);

        bool importInternalAccelTree(const char* path, int offsetTriIdx = 0);

        void buildForRasterizeRendering();

        void gatherTrianglesAndMaterials(
            std::vector<std::vector<AT_NAME::face*>>& tris,
            std::vector<AT_NAME::material*>& mtrls);

        virtual void collectTriangles(std::vector<aten::PrimitiveParamter>& triangles) const override final;

        virtual uint32_t getTriangleCount() const override final
        {
            return m_triangles;
        }

        void build();

        virtual void getSamplePosNormalArea(
            aten::hitable::SamplePosNormalPdfResult* result,
            const aten::mat4& mtxL2W, 
            aten::sampler* sampler) const override final;

        void appendShape(objshape* shape)
        {
            shapes.push_back(shape);
        }

    private:
        std::vector<objshape*> shapes;
        aten::GeomParameter param;

        aten::accelerator* m_accel{ nullptr };
        uint32_t m_triangles{ 0 };
    };
}
