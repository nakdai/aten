#include "visualizer/RasterizeRenderer.h"
#include "visualizer/atengl.h"
#include "scene/scene.h"
#include "camera/camera.h"
#include "math/mat4.h"
#include "accelerator/accelerator.h"
#include "sampler/cmj.h"

namespace aten {
	bool RasterizeRenderer::init(
		int width, int height,
		const char* pathVS,
		const char* pathFS)
	{
		m_width = width;
		m_height = height;

		return m_shader.init(width, height, pathVS, pathFS);
	}

	bool RasterizeRenderer::init(
		int width, int height,
		const char* pathVS,
		const char* pathGS,
		const char* pathFS)
	{
		m_width = width;
		m_height = height;

		return m_shader.init(width, height, pathVS, pathGS, pathFS);
	}

	void RasterizeRenderer::draw(
		int frame,
		scene* scene,
		const camera* cam,
		FBO* fbo/*= nullptr*/)
	{
		auto camparam = cam->param();

		// TODO
		camparam.znear = real(0.1);
		camparam.zfar = real(10000.0);

		mat4 mtxW2V;
		mat4 mtxV2C;

		mtxW2V.lookat(
			camparam.origin,
			camparam.center,
			camparam.up);

		mtxV2C.perspective(
			camparam.znear,
			camparam.zfar,
			camparam.vfov,
			camparam.aspect);

		aten::mat4 mtxW2C = mtxV2C * mtxW2V;

		// TODO
		// For TAA.
		{
			CMJ sampler;
			auto rnd = getRandom(frame);
			auto scramble = rnd * 0x1fe3434f * ((frame + 331 * rnd) / (aten::CMJ::CMJ_DIM * aten::CMJ::CMJ_DIM));
			sampler.init(frame % (aten::CMJ::CMJ_DIM * aten::CMJ::CMJ_DIM), 4 + 300, scramble);

			auto smpl = sampler.nextSample2D();

			aten::mat4 mtxOffset;
			mtxOffset.asTrans(aten::vec3(
				smpl.x / m_width,
				smpl.y / m_height,
				real(0)));

			mtxW2C = mtxOffset * mtxW2C;
		}

		if (m_mtxPrevW2C.isIdentity()) {
			m_mtxPrevW2C = mtxW2C;
		}

		m_shader.prepareRender(nullptr, false);

		auto hMtxW2C = m_shader.getHandle("mtxW2C");
		CALL_GL_API(::glUniformMatrix4fv(hMtxW2C, 1, GL_TRUE, &mtxW2C.a[0]));

		auto hPrevMtxW2C = m_shader.getHandle("mtxPrevW2C");
		CALL_GL_API(::glUniformMatrix4fv(hPrevMtxW2C, 1, GL_TRUE, &m_mtxPrevW2C.a[0]));

		aten::vec4 invScreen(1.0f / m_width, 1.0f / m_height, 0.0f, 0.0f);
		auto hInvScr = m_shader.getHandle("invScreen");
		CALL_GL_API(::glUniform4fv(hInvScr, 1, invScreen.p));

		if (fbo) {
			AT_ASSERT(fbo->isValid());
			fbo->bindFBO(true);
		}
		else {
			// Set default frame buffer.
			CALL_GL_API(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0));
		}

		{
			CALL_GL_API(::glEnable(GL_DEPTH_TEST));
			//CALL_GL_API(::glEnable(GL_CULL_FACE));
		}

		// Clear.
		if (fbo) {
			float clr[] = { -1.0f, -1.0f, -1.0f, -1.0f };
			CALL_GL_API(glClearNamedFramebufferfv(fbo->getHandle(), GL_COLOR, 0, clr));
			CALL_GL_API(glClearNamedFramebufferfv(fbo->getHandle(), GL_COLOR, 1, clr));

			CALL_GL_API(::glClearDepthf(1.0f));
			CALL_GL_API(::glClearStencil(0));
			CALL_GL_API(::glClear(GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT));
		}
		else {
			CALL_GL_API(::glClearColor(0, 0.5f, 1.0f, 1.0f));
			CALL_GL_API(::glClearDepthf(1.0f));
			CALL_GL_API(::glClearStencil(0));
			CALL_GL_API(::glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT));
		}
		
		// TODO
		// ここでやる?
		static bool isInitVB = false;
		if (!isInitVB) {
			VertexManager::build();
			isInitVB = true;
		}

		scene->draw([&](const aten::mat4& mtxL2W, const aten::mat4& mtxPrevL2W, int objid, int primid) {
			auto hMtxL2W = m_shader.getHandle("mtxL2W");
			CALL_GL_API(::glUniformMatrix4fv(hMtxL2W, 1, GL_TRUE, &mtxL2W.a[0]));

			auto hPrevMtxL2W = m_shader.getHandle("mtxPrevL2W");
			CALL_GL_API(::glUniformMatrix4fv(hPrevMtxL2W, 1, GL_TRUE, &mtxPrevL2W.a[0]));

			auto hObjId = m_shader.getHandle("objid");
			CALL_GL_API(::glUniform1i(hObjId, objid));

			auto hPrimId = m_shader.getHandle("primid");
			CALL_GL_API(::glUniform1i(hPrimId, primid));
		});

		if (fbo) {
			// Set default frame buffer.
			CALL_GL_API(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0));
		}

		{
			CALL_GL_API(::glDisable(GL_DEPTH_TEST));
			CALL_GL_API(::glDisable(GL_CULL_FACE));
		}

		m_mtxPrevW2C = mtxW2C;
	}

	static const vertex boxvtx[] = {
		// 0
		{ { 0, 0, 0, 1 }, { 0, 0, 0 }, { 1, 0, 0 } },
		{ { 1, 0, 0, 1 }, { 0, 0, 0 }, { 1, 0, 0 } },

		// 1
		{ { 0, 0, 0, 1 },{ 0, 0, 0 },{ 1, 0, 0 } },
		{ { 0, 1, 0, 1 },{ 0, 0, 0 },{ 1, 0, 0 } },

		// 2
		{ { 0, 0, 0, 1 },{ 0, 0, 0 },{ 1, 0, 0 } },
		{ { 0, 0, 1, 1 },{ 0, 0, 0 },{ 1, 0, 0 } },

		// 3
		{ { 1, 0, 0, 1 },{ 0, 0, 0 },{ 1, 0, 0 } },
		{ { 1, 0, 1, 1 },{ 0, 0, 0 },{ 1, 0, 0 } },

		// 4
		{ { 1, 0, 0, 1 },{ 0, 0, 0 },{ 1, 0, 0 } },
		{ { 1, 1, 0, 1 },{ 0, 0, 0 },{ 1, 0, 0 } },

		// 5
		{ { 0, 1, 0, 1 },{ 0, 0, 0 },{ 1, 0, 0 } },
		{ { 1, 1, 0, 1 },{ 0, 0, 0 },{ 1, 0, 0 } },

		// 6
		{ { 0, 1, 0, 1 },{ 0, 0, 0 },{ 1, 0, 0 } },
		{ { 0, 1, 1, 1 },{ 0, 0, 0 },{ 1, 0, 0 } },

		// 7
		{ { 0, 0, 1, 1 },{ 0, 0, 0 },{ 1, 0, 0 } },
		{ { 1, 0, 1, 1 },{ 0, 0, 0 },{ 1, 0, 0 } },

		// 8
		{ { 0, 0, 1, 1 },{ 0, 0, 0 },{ 1, 0, 0 } },
		{ { 0, 1, 1, 1 },{ 0, 0, 0 },{ 1, 0, 0 } },

		// 9
		{ { 1, 0, 1, 1 },{ 0, 0, 0 },{ 1, 0, 0 } },
		{ { 1, 1, 1, 1 },{ 0, 0, 0 },{ 1, 0, 0 } },

		// 10
		{ { 1, 1, 0, 1 },{ 0, 0, 0 },{ 1, 0, 0 } },
		{ { 1, 1, 1, 1 },{ 0, 0, 0 },{ 1, 0, 0 } },

		{ { 0, 1, 1, 1 },{ 0, 0, 0 },{ 1, 0, 0 } },
		{ { 1, 1, 1, 1 },{ 0, 0, 0 },{ 1, 0, 0 } },
	};

	void RasterizeRenderer::drawAABB(
		const camera* cam,
		accelerator* accel)
	{
		// Initialize vb.
		static bool isInitVB = false;
		if (!isInitVB) {
			m_boxvb.init(
				sizeof(aten::vertex),
				AT_COUNTOF(boxvtx),
				0,
				(void*)boxvtx);

			isInitVB = true;
		}

		m_shader.prepareRender(nullptr, false);

		auto camparam = cam->param();

		// TODO
		camparam.znear = real(0.1);
		camparam.zfar = real(10000.0);

		mat4 mtxW2V;
		mat4 mtxV2C;

		mtxW2V.lookat(
			camparam.origin,
			camparam.center,
			camparam.up);

		mtxV2C.perspective(
			camparam.znear,
			camparam.zfar,
			camparam.vfov,
			camparam.aspect);

		aten::mat4 mtxW2C = mtxV2C * mtxW2V;

		auto hMtxW2C = m_shader.getHandle("mtxW2C");
		CALL_GL_API(::glUniformMatrix4fv(hMtxW2C, 1, GL_TRUE, &mtxW2C.a[0]));

		auto hMtxL2W = m_shader.getHandle("mtxL2W");

		accel->drawAABB([&](const aten::mat4& mtxL2W) {
			// Draw.
			CALL_GL_API(::glUniformMatrix4fv(hMtxL2W, 1, GL_TRUE, &mtxL2W.a[0]));
			m_boxvb.draw(aten::Primitive::Lines, 0, 12);
		}, aten::mat4::Identity);
	}

	void RasterizeRenderer::draw(
		object* obj, 
		const camera* cam,
		bool isWireFrame,
		FBO* fbo/*= nullptr*/,
		FuncSetUniform funcSetUniform/*= nullptr*/)
	{
		auto camparam = cam->param();

		// TODO
		camparam.znear = real(0.1);
		camparam.zfar = real(10000.0);

		mat4 mtxW2V;
		mat4 mtxV2C;

		mtxW2V.lookat(
			camparam.origin,
			camparam.center,
			camparam.up);

		mtxV2C.perspective(
			camparam.znear,
			camparam.zfar,
			camparam.vfov,
			camparam.aspect);

		aten::mat4 mtxW2C = mtxV2C * mtxW2V;

		m_shader.prepareRender(nullptr, false);

		// Not modify local to world matrix...
		auto hMtxL2W = m_shader.getHandle("mtxL2W");
		CALL_GL_API(::glUniformMatrix4fv(hMtxL2W, 1, GL_TRUE, &mat4::Identity.a[0]));

		auto hMtxW2C = m_shader.getHandle("mtxW2C");
		CALL_GL_API(::glUniformMatrix4fv(hMtxW2C, 1, GL_TRUE, &mtxW2C.a[0]));

		if (fbo) {
			AT_ASSERT(fbo->isValid());
			fbo->bindFBO(true);
		}
		else {
			// Set default frame buffer.
			CALL_GL_API(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0));
		}

		if (isWireFrame) {
			CALL_GL_API(::glPolygonMode(GL_FRONT_AND_BACK, GL_LINE));
		}
		else {
			CALL_GL_API(::glPolygonMode(GL_FRONT_AND_BACK, GL_FILL));
		}

		CALL_GL_API(::glEnable(GL_DEPTH_TEST));
		CALL_GL_API(::glEnable(GL_CULL_FACE));

		// TODO
		// Will modify to specify clear color.
		CALL_GL_API(::glClearColor(0, 0.5f, 1.0f, 1.0f));
		CALL_GL_API(::glClearDepthf(1.0f));
		CALL_GL_API(::glClearStencil(0));
		CALL_GL_API(::glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT));

		// TODO
		// ここでやる?
		static bool isInitGlobalVB = false;
		if (!isInitGlobalVB) {
			VertexManager::build();
			isInitGlobalVB = true;
		}

		auto hHasAlbedo = m_shader.getHandle("hasAlbedo");
		auto hColor = m_shader.getHandle("color");
		auto hMtrlId = m_shader.getHandle("materialId");

		obj->draw([&](const aten::vec3& color, const aten::texture* albedo, int mtrlid) {
			if (albedo) {
				albedo->bindAsGLTexture(0, &m_shader);
				CALL_GL_API(::glUniform1i(hHasAlbedo, true));
			}
			else {
				CALL_GL_API(::glUniform1i(hHasAlbedo, false));
			}

			CALL_GL_API(::glUniform4f(hColor, color.x, color.y, color.z, 1.0f));

			if (hMtrlId >= 0) {
				CALL_GL_API(::glUniform1i(hMtrlId, mtrlid));
			}

			if (funcSetUniform) {
				funcSetUniform(m_shader, color, albedo, mtrlid);
			}
		});

		// 戻す.
		CALL_GL_API(::glPolygonMode(GL_FRONT_AND_BACK, GL_FILL));
	}

	void RasterizeRenderer::initBuffer(
		uint32_t vtxStride,
		uint32_t vtxNum,
		const std::vector<uint32_t>& idxNums)
	{
		if (!m_isInitBuffer) {
			m_vb.init(
				sizeof(vertex),
				vtxNum,
				0,
				nullptr);

			m_ib.resize(idxNums.size());

			for (int i = 0; i < idxNums.size(); i++) {
				auto num = idxNums[i];
				m_ib[i].init(num, nullptr);
			}

			m_isInitBuffer = true;
		}
	}

	void RasterizeRenderer::draw(
		const std::vector<vertex>& vtxs,
		const std::vector<std::vector<int>>& idxs,
		const std::vector<material*>& mtrls,
		const camera* cam,
		bool isWireFrame,
		bool updateBuffer)
	{
		AT_ASSERT(idxs.size() == mtrls.size());

		auto camparam = cam->param();

		// TODO
		camparam.znear = real(0.1);
		camparam.zfar = real(10000.0);

		mat4 mtxW2V;
		mat4 mtxV2C;

		mtxW2V.lookat(
			camparam.origin,
			camparam.center,
			camparam.up);

		mtxV2C.perspective(
			camparam.znear,
			camparam.zfar,
			camparam.vfov,
			camparam.aspect);

		aten::mat4 mtxW2C = mtxV2C * mtxW2V;

		m_shader.prepareRender(nullptr, false);

		// Not modify local to world matrix...
		auto hMtxL2W = m_shader.getHandle("mtxL2W");
		CALL_GL_API(::glUniformMatrix4fv(hMtxL2W, 1, GL_TRUE, &mat4::Identity.a[0]));

		auto hMtxW2C = m_shader.getHandle("mtxW2C");
		CALL_GL_API(::glUniformMatrix4fv(hMtxW2C, 1, GL_TRUE, &mtxW2C.a[0]));

		// Set default frame buffer.
		CALL_GL_API(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0));

		if (isWireFrame) {
			CALL_GL_API(::glPolygonMode(GL_FRONT_AND_BACK, GL_LINE));
		}
		else {
			CALL_GL_API(::glPolygonMode(GL_FRONT_AND_BACK, GL_FILL));
		}

		CALL_GL_API(::glEnable(GL_DEPTH_TEST));
		CALL_GL_API(::glEnable(GL_CULL_FACE));

		// TODO
		// Will modify to specify clear color.
		CALL_GL_API(::glClearColor(0, 0.5f, 1.0f, 1.0f));
		CALL_GL_API(::glClearDepthf(1.0f));
		CALL_GL_API(::glClearStencil(0));
		CALL_GL_API(::glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT));

		if (!m_isInitBuffer) {
			m_vb.init(
				sizeof(vertex),
				vtxs.size(),
				0,
				&vtxs[0]);

			m_ib.resize(idxs.size());

			for (int i = 0; i < idxs.size(); i++) {
				if (!idxs[i].empty()) {
					m_ib[i].init((uint32_t)idxs[i].size(), &idxs[i][0]);
				}
			}

			m_isInitBuffer = true;
		}
		else if (updateBuffer) {
			// TODO
			// 最初に最大数バッファをアロケートしておかないといけない...

			m_vb.update(vtxs.size(), &vtxs[0]);

			for (int i = 0; i < idxs.size(); i++) {
				if (!idxs[i].empty()) {
					m_ib[i].update((uint32_t)idxs[i].size(), &idxs[i][0]);
				}
			}
		}

		auto hHasAlbedo = m_shader.getHandle("hasAlbedo");

		for (int i = 0; i < m_ib.size(); i++) {
			auto triNum = (uint32_t)idxs[i].size() / 3;

			if (triNum > 0) {
				auto m = mtrls[i];
				
				int albedoTexId = m ? m->param().albedoMap : -1;
				const aten::texture* albedo = albedoTexId >= 0 ? aten::texture::getTexture(albedoTexId) : nullptr;

				if (albedo) {
					albedo->bindAsGLTexture(0, &m_shader);
					CALL_GL_API(::glUniform1i(hHasAlbedo, true));
				}
				else {
					CALL_GL_API(::glUniform1i(hHasAlbedo, false));
				}

				m_ib[i].draw(m_vb, aten::Primitive::Triangles, 0, triNum);
			}
		}

		// 戻す.
		CALL_GL_API(::glPolygonMode(GL_FRONT_AND_BACK, GL_FILL));
	}
}
