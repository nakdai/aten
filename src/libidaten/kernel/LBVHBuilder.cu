#include "kernel/LBVHBuilder.h"
#include "kernel/RadixSort.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "cuda/helper_math.h"

#pragma optimize( "", off)

__device__  int computeLongestCommonPrefix(
	const uint32_t* sortedKeys,
	uint32_t numOfElems,
	int index1, int index2,
	uint32_t key1)
{
	// No need to check the upper bound, since i+1 will be at most numberOfElements - 1 (one 
	// thread per internal node)
	if (index2 < 0 || index2 >= numOfElems)
	{
		return 0;
	}

	auto key2 = sortedKeys[index2];

	if (key1 == key2)
	{

		return 32 + __clz(index1 ^ index2);
	}

	auto ret = __clz(key1 ^ key2);

	return ret;
}

__global__ void buildTree(
	uint32_t numOfElems,
	const uint32_t* __restrict__ sortedKeys,
	idaten::LBVHBuilder::LBVHNode* nodes)
{
	const auto i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= numOfElems - 1) {
		return;
	}

	const auto key1 = sortedKeys[i];

	const auto lcp1 = computeLongestCommonPrefix(sortedKeys, numOfElems, i, i + 1, key1);
	const auto lcp2 = computeLongestCommonPrefix(sortedKeys, numOfElems, i, i - 1, key1);

	auto d = (lcp1 - lcp2) < 0 ? -1 : 1;

	// Compute upper bound for the length of the range
	const auto minLcp = computeLongestCommonPrefix(sortedKeys, numOfElems, i, i - d, key1);
	int lMax = 2;
	while (computeLongestCommonPrefix(sortedKeys, numOfElems, i, i + lMax * d, key1) > minLcp)
	{
		lMax *= 2;
	}

	// Find other end using binary search
	int l = 0;
	int t = lMax;
	while (t > 1)
	{
		t = t / 2;
		auto lcp = computeLongestCommonPrefix(sortedKeys, numOfElems, i, i + (l + t) * d, key1);
		if (lcp > minLcp)
		{
			l += t;
		}
	}
	const auto j = i + l * d;

	// Find the split position using binary search
	const auto nodeLcp = computeLongestCommonPrefix(sortedKeys, numOfElems, i, j, key1);
	int s = 0;
	int divisor = 2;
	t = l;
	while (t > 1)
	{
		t = (l + divisor - 1) / divisor;
		auto lcp = computeLongestCommonPrefix(sortedKeys, numOfElems, i, i + (s + t) * d, key1);
		if (lcp > nodeLcp)
		{
			s += t;
		}
		divisor *= 2;
	}

	const auto splitPosition = i + s * d + min(d, 0);

	auto* node = nodes + i;
	if (i == 0) {
		node->parent = -1;
	}
	node->order = i;
	node->isLeaf = false;

	uint32_t leafBaseIdx = numOfElems - 1;

	if (min(i, j) == splitPosition) {
		node->left = leafBaseIdx + splitPosition;

		auto* leaf = nodes + node->left;
		leaf->order = node->left;
		leaf->parent = i;
		leaf->left = -1;
		leaf->right = -1;
		leaf->rangeMin = 0;
		leaf->rangeMax = 0;
		leaf->isLeaf = true;
	}
	else {
		node->left = splitPosition;

		auto* child = nodes + node->left;
		child->order = node->left;
		child->parent = i;
		child->isLeaf = false;
	}

	if (max(i, j) == splitPosition + 1) {
		node->right = leafBaseIdx + splitPosition + 1;

		auto* leaf = nodes + node->right;
		leaf->order = node->right;
		leaf->parent = i;
		leaf->left = -1;
		leaf->right = -1;
		leaf->rangeMin = 0;
		leaf->rangeMax = 0;
		leaf->isLeaf = true;
	}
	else {
		node->right = splitPosition + 1;

		auto* child = nodes + node->right;
		child->order = node->right;
		child->parent = i;
		child->isLeaf = false;
	}

	node->rangeMin = min(i, j);
	node->rangeMax = max(i, j);
}

__device__ __host__ inline void onApplyTraverseOrder(
	int idx,
	int numberOfTris,
	int shapeId,
	int triIdOffset,
	const idaten::LBVHBuilder::LBVHNode* src,
	aten::ThreadedBvhNode* dst)
{
	const auto* node = &src[idx];

	const idaten::LBVHBuilder::LBVHNode* next = node->left >= 0 ? &src[node->left] : nullptr;

	auto gpunode = &dst[idx];

	gpunode->shapeid = shapeId;
	gpunode->exid = -1;
	gpunode->meshid = -1;

	if (node->isLeaf) {
		// Base index to convert node index to triangle index.
		int leafBaseIdx = numberOfTris - 1;

		int leafId = node->order - leafBaseIdx;
		int triId = triIdOffset + leafId;

		gpunode->primid = triId;
	}
	else {
		gpunode->primid = -1;
	}

	gpunode->hit = -1;
	gpunode->miss = -1;

	bool isOrdered = false;

	if (node->isLeaf) {
		// Hit/Miss.
		// Always sibling.

		// The leaf has parent surely.
		auto parent = &src[node->parent];

		auto left = parent->left >= 0 ? &src[parent->left] : nullptr;
		auto right = parent->right >= 0 ? &src[parent->right] : nullptr;

		if (left == node) {
			// Sibling.
			gpunode->hit = (float)right->order;
			gpunode->miss = (float)right->order;

			isOrdered = true;
		}
	}
	else {
		// Hit.
		// Always the next node in the array.
		if (next) {
			gpunode->hit = (float)next->order;
		}
		else {
			gpunode->hit = -1;
		}
	}

	if (!isOrdered)
	{
		// Miss.

		// Search the parent.
		auto parentId = node->parent;
		const auto parent = (parentId >= 0
			? &src[parentId]
			: nullptr);

		if (parent) {
			const auto left = parent->left >= 0 ? &src[parent->left] : nullptr;
			const auto right = parent->right >= 0 ? &src[parent->right] : nullptr;

			if (left == node && right) {
				// Traverse to sibling (= parent's right)
				auto sibling = right;
				gpunode->miss = (float)sibling->order;
			}
			else {
				auto curParent = parent;

				// Traverse to ancester's right.
				for (;;) {
					// Search the grand parent.
					auto grandParentId = curParent->parent;
					const auto grandParent = (grandParentId >= 0
						? &src[grandParentId]
						: nullptr);

					if (grandParent) {
						const auto _left = grandParent->left >= 0 ? &src[grandParent->left] : nullptr;
						const auto _right = grandParent->right >= 0 ? &src[grandParent->right] : nullptr;

						auto sibling = _right;

						if (sibling) {
							if (sibling != curParent) {
								gpunode->miss = (float)sibling->order;

								if (node->isLeaf && gpunode->hit < 0) {
									gpunode->hit = (float)sibling->order;
								}

								break;
							}
						}
					}
					else {
						gpunode->miss = -1;
						break;
					}

					curParent = grandParent;
				}
			}
		}
		else {
			gpunode->miss = -1;
		}
	}
}

__global__ void applyTraverseOrder(
	uint32_t numberOfNodes,
	int numberOfTris,
	int shapeId,
	int triIdOffset,
	const idaten::LBVHBuilder::LBVHNode* __restrict__ src,
	aten::ThreadedBvhNode* dst)
{
	const auto idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= numberOfNodes) {
		return;
	}

	onApplyTraverseOrder(idx, numberOfTris, shapeId, triIdOffset, src, dst);
}

__device__ inline void computeBoundingBox(
	float4 v0,
	float4 v1,
	float4 v2,
	float4* aabbMin,
	float4* aabbMax)
{
	aabbMin->x = min(min(v0.x, v1.x), v2.x);
	aabbMin->y = min(min(v0.y, v1.y), v2.y);
	aabbMin->z = min(min(v0.z, v1.z), v2.z);

	aabbMax->x = max(max(v0.x, v1.x), v2.x);
	aabbMax->y = max(max(v0.y, v1.y), v2.y);
	aabbMax->z = max(max(v0.z, v1.z), v2.z);
}

__device__ inline void computeBoundingBox(
	float4 bboxMin_0,
	float4 bboxMax_0,
	float4 bboxMin_1,
	float4 bboxMax_1,
	float4* aabbMin,
	float4* aabbMax)
{
	aabbMin->x = min(bboxMin_0.x, bboxMin_1.x);
	aabbMin->y = min(bboxMin_0.y, bboxMin_1.y);
	aabbMin->z = min(bboxMin_0.z, bboxMin_1.z);

	aabbMax->x = max(bboxMax_0.x, bboxMax_1.x);
	aabbMax->y = max(bboxMax_0.y, bboxMax_1.y);
	aabbMax->z = max(bboxMax_0.z, bboxMax_1.z);
}

__global__ void computeBoudingBox(
	int numberOfTris,
	const idaten::LBVHBuilder::LBVHNode* __restrict__ src,
	aten::ThreadedBvhNode* dst,
	int triIdOffset,
	const aten::PrimitiveParamter* __restrict__ tris,
	cudaTextureObject_t vtxPos,
	uint32_t* executedIdxArray)
{
	const int i = threadIdx.x + blockIdx.x * blockDim.x;
	
	const int firstThreadIdxInBlock = blockIdx.x * blockDim.x;
	const int lastThreadIdxInBlock = firstThreadIdxInBlock + blockDim.x - 1;

	// Initialize cache of bounding boxes in shared memory
	extern __shared__ float4 sharedBboxMin[];
	__shared__ float4* sharedBboxMax;
	if (threadIdx.x == 0)
	{
		sharedBboxMax = sharedBboxMin + blockDim.x;
	}
	__syncthreads();

	// Check for valid threads
	if (i >= numberOfTris)
	{
		return;
	}

	// NOTE
	// Number of Internal Nodes = Number of Triangles - 1.
	int leafNodeIdx = i + numberOfTris - 1;

	// Base index to convert node index to triangle index.
	int leafBaseIdx = numberOfTris - 1;

	const auto* node = &src[leafNodeIdx];
	auto* gpunode = &dst[leafNodeIdx];

	// Calculate leaves bounding box.
	int leafId = node->order - leafBaseIdx;

	int triId = triIdOffset + leafId;

	aten::PrimitiveParamter prim;
	prim.v0 = ((aten::vec4*)tris)[triId * aten::PrimitiveParamter_float4_size + 0];

	float4 v0 = tex1Dfetch<float4>(vtxPos, prim.idx[0]);
	float4 v1 = tex1Dfetch<float4>(vtxPos, prim.idx[1]);
	float4 v2 = tex1Dfetch<float4>(vtxPos, prim.idx[2]);

	float4 aabbMin, aabbMax;
	computeBoundingBox(v0, v1, v2, &aabbMin, &aabbMax);

	// Keep bouding box to shared memory.
	sharedBboxMin[threadIdx.x] = aabbMin;
	sharedBboxMax[threadIdx.x] = aabbMax;

	gpunode->boxmin = aten::vec3(aabbMin.x, aabbMin.y, aabbMin.z);
	gpunode->boxmax = aten::vec3(aabbMax.x, aabbMax.y, aabbMax.z);

	__syncthreads();

	// リーフから親へたどっていく.

	int lastNode = leafId;
	int targetId = src->parent;

	while (targetId >= 0)
	{
		// ターゲットは親ノードで、ここでは子ノードを処理しているであろうスレッドのインデックスを取得する.
		// インデックスの配列は 0xffffffff で初期化されていて、処理されたらスレッドのインデックスで置換される.
		// つまり、配列内の値が 0xffffffff であったら、未処理ということになる.
		const auto childNodeThreadIdx = atomicExch(&executedIdxArray[targetId], i);

		if (childNodeThreadIdx == 0xffffffff) {
			// 未処理なので、これ以上は何もしない.
			return;
		}

		auto* targetSrc = &src[targetId];
		auto* targetDst = &dst[targetId];

		float4 childAABBMin, childAABBMax;

		// 子ノードを処理しているであろうスレッドが同一ブロックで処理しているかどうか.
		if (firstThreadIdxInBlock <= childNodeThreadIdx
			&& childNodeThreadIdx <= lastThreadIdxInBlock)
		{
			// 同一ブロックで処理されているので、shared memory にキャッシュされているデータを取得する.

			// ブロック内でのスレッドIDに変換.
			int threadIdxInBlock = childNodeThreadIdx - firstThreadIdxInBlock;

			childAABBMin = sharedBboxMin[threadIdxInBlock];
			childAABBMax = sharedBboxMax[threadIdxInBlock];
		}
		else {
			// 同一ブロックで処理されていないので、配列に格納されているデータを取得する.

			int childIdx = targetSrc->left;

			if (childIdx == lastNode) {
				childIdx = targetSrc->right;
			}

			auto* tmp = &dst[childIdx];

			childAABBMin = make_float4(tmp->boxmin.x, tmp->boxmin.y, tmp->boxmin.z, 0);
			childAABBMax = make_float4(tmp->boxmax.x, tmp->boxmax.y, tmp->boxmax.z, 0);
		}

		__syncthreads();

		computeBoundingBox(
			aabbMin, aabbMax,
			childAABBMin, childAABBMax,
			&aabbMin, &aabbMax);

		// Keep bouding box to shared memory.
		sharedBboxMin[threadIdx.x] = aabbMin;
		sharedBboxMax[threadIdx.x] = aabbMax;

		targetDst->boxmin = aten::vec3(aabbMin.x, aabbMin.y, aabbMin.z);
		targetDst->boxmax = aten::vec3(aabbMax.x, aabbMax.y, aabbMax.z);

		__syncthreads();

		// Update last processed node
		lastNode = targetId;

		// Update target node pointer
		targetId = targetSrc->parent;
	}
}

namespace idaten
{
	void LBVHBuilder::sort()
	{
		static const uint32_t keys[] = {
			1, 2, 4, 5, 19, 24, 25, 30,
		};

		std::vector<uint32_t> values;
		for (auto k : keys) {
			values.push_back(k);
		}

		TypedCudaMemory<uint32_t> sortedKeys;
		std::vector<uint32_t> v;
		RadixSort::sort(values, sortedKeys, &v);

		uint32_t numOfElems = values.size();

		uint32_t numInternalNode = numOfElems - 1;
		uint32_t numLeaves = numOfElems;

		m_nodesLbvh.init(numInternalNode + numLeaves);

		{
			dim3 block(256, 1, 1);
			dim3 grid((numOfElems + block.x - 1) / block.x, 1, 1);

			buildTree << <grid, block >> > (
				numOfElems,
				sortedKeys.ptr(),
				m_nodesLbvh.ptr());
		}

		m_nodes.init(numInternalNode + numLeaves);

#if 1
		{
			numOfElems = numInternalNode + numLeaves;

			dim3 block(256, 1, 1);
			dim3 grid((numOfElems + block.x - 1) / block.x, 1, 1);

			applyTraverseOrder << <grid, block >> > (
				numOfElems,
				numLeaves,
				0,
				0,
				m_nodesLbvh.ptr(),
				m_nodes.ptr());
		}
		std::vector<aten::ThreadedBvhNode> tmp1(m_nodes.maxNum());
		m_nodes.read(&tmp1[0], 0);
#else
		std::vector<LBVHNode> tmp(m_nodesLbvh.maxNum());
		m_nodesLbvh.read(&tmp[0], 0);

		std::vector<aten::ThreadedBvhNode> tmp1(m_nodes.maxNum());

		for (int n = 0; n < numInternalNode + numLeaves; n++) {
			onApplyTraverseOrder(n, &tmp[0], &tmp1[0]);
		}
#endif

		int xxx = 0;
	}
}