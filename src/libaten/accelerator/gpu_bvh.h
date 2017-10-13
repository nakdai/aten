#pragma once

#include "scene/hitable.h"
#include "accelerator/bvh.h"

namespace aten {
	// TODO
	// テスト用に bvh の継承クラスで作るが、インスタシエイトする必要がないので、あとで変更する.
	class GPUBvh : public accelerator {
	public:
		GPUBvh() {}
		virtual ~GPUBvh() {}

	public:
		virtual void build(
			hitable** list,
			uint32_t num) override;

		virtual bool hit(
			const ray& r,
			real t_min, real t_max,
			Intersection& isect) const override;

		std::vector<std::vector<GPUBvhNode>>& getNodes()
		{
			return m_listGpuBvhNode;
		}
		std::vector<aten::mat4>& getMatrices()
		{
			return m_mtxs;
		}

	private:
		struct GPUBvhNodeEntry {
			bvhnode* node;
			hitable* nestParent;
			aten::mat4 mtxL2W;

			GPUBvhNodeEntry(bvhnode* n, hitable* p, const aten::mat4& m)
				: node(n), nestParent(p), mtxL2W(m)
			{}
		};

		void registerBvhNodeToLinearList(
			bvhnode* root, 
			bvhnode* parentNode,
			hitable* nestParent,
			const aten::mat4& mtxL2W,
			std::vector<GPUBvhNodeEntry>& listBvhNode,
			std::map<hitable*, std::vector<accelerator*>>& nestedBvhMap);

		void registerGpuBvhNode(
			bool isPrimitiveLeaf,
			std::vector<GPUBvhNodeEntry>& listBvhNode,
			std::vector<GPUBvhNode>& listGpuBvhNode);

		void setOrderForLinearBVH(
			std::vector<GPUBvhNodeEntry>& listBvhNode,
			std::vector<GPUBvhNode>& listGpuBvhNode);

		bool hit(
			int exid,
			const std::vector<std::vector<GPUBvhNode>>& listGpuBvhNode,
			const ray& r,
			real t_min, real t_max,
			Intersection& isect) const;

		void dump(std::vector<GPUBvhNode>& nodes, const char* path);

	private:
		bvh m_bvh;

		std::vector<std::vector<GPUBvhNode>> m_listGpuBvhNode;
		std::vector<aten::mat4> m_mtxs;
	};
}
