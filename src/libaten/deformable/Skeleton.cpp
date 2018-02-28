#include "deformable/Skeleton.h"
#include "misc/stream.h"
#include "misc/bitflag.h"
#include "math/quaternion.h"

namespace aten
{
	bool Skeleton::read(FileInputStream* stream)
	{
		AT_VRETURN_FALSE(AT_STREAM_READ(stream, &m_header, sizeof(m_header)));

		if (m_header.numJoint > 0) {
			m_joints.resize(m_header.numJoint);
			AT_VRETURN_FALSE(AT_STREAM_READ(stream, &m_joints[0], sizeof(JointParam) * m_header.numJoint));
		}
		
		return true;
	}


	///////////////////////////////////////////////////////

	void SkeletonController::init(Skeleton* skl)
	{
		m_skl = skl;

		auto numJoint = skl->getJointNum();
		const auto& joints = skl->getJoints();

		m_globalPose.resize(numJoint);
		m_globalMatrix.resize(numJoint);
		m_needUpdateJointFlag.resize(numJoint);

		// アニメーションパラメータの中で更新が必要なパラメータフラグ.
		for (uint32_t i = 0; i < numJoint; i++) {
			m_needUpdateJointFlag[i] = joints[i].validAnmParam;
			m_globalPose[i] = joints[i].pose;
		}

	}

	void SkeletonController::buildPose(const mat4& mtxL2W)
	{
		AT_ASSERT(m_skl);

		auto numJoint = m_skl->getJointNum();
		const auto& joints = m_skl->getJoints();

		for (uint32_t i = 0; i < numJoint; i++) {
			buildLocalMatrix(i);
		}

		// Apply parent's matrix.
		for (uint32_t i = 0; i < numJoint; i++) {
			auto parentIdx = joints[i].parent;

			auto& mtxJoint = m_globalMatrix[i];

			if (parentIdx >= 0) {
				AT_ASSERT((uint32_t)parentIdx < i);
				const auto& mtxParent = m_globalMatrix[parentIdx];
				mtxJoint = mtxParent * mtxJoint;
			}
			else {
				// ルートに対しては、L2Wマトリクスを計算する.
				mtxJoint = mtxL2W * mtxJoint;
			}
		}

		for (uint32_t i = 0; i < numJoint; i++) {
			auto& mtxJoint = m_globalMatrix[i];

			const auto& mtxInvBind = joints[i].mtxInvBind;

			mtxJoint = mtxJoint * mtxInvBind;
		}
	}

	void SkeletonController::buildLocalMatrix(uint32_t idx)
	{
		const auto& joints = m_skl->getJoints();

		const auto& joint = joints[idx];
		const auto& pose = m_globalPose[idx];

		// 計算するパラメータを判定するフラグ.
		Bit32Flag flag(
			joint.validParam
			| m_needUpdateJointFlag[idx]);

		// 念のため次に向けてリセットしておく.
		m_needUpdateJointFlag[idx] = 0;

		mat4& mtxJoint = m_globalMatrix[idx];
		mtxJoint.identity();

		if (flag.isOn((uint32_t)JointTransformType::Scale)) {
			mat4 scale;
			scale.asScale(pose.scale[0], pose.scale[1], pose.scale[2]);

			mtxJoint = mtxJoint * scale;
		}

		if (flag.isOn((uint32_t)JointTransformType::Quaternion)) {
			auto rot = pose.quat.getMatrix();

			mtxJoint = mtxJoint * rot;
		}

		if (flag.isOn((uint32_t)JointTransformType::Translate)) {
			mat4 translate;
			translate.asTrans(pose.trans[0], pose.trans[1], pose.trans[2]);

			mtxJoint = translate * mtxJoint;
		}
	}
}
