#include "denoiser/atrous.h"

#include "visualizer/atengl.h"

namespace aten {
	bool ATrousDenoiser::init(
		int width, int height,
		const char* vsPath,
		const char* fsPath)
	{
		m_normal.init(width, height, 4);
		m_pos.init(width, height, 4);

		m_normal.initAsGLTexture();
		m_pos.initAsGLTexture();

		for (int i = 0; i < ITER; i++) {
			auto res = m_pass[i].init(
				width, height,
				vsPath, fsPath);
			AT_ASSERT(res);

			m_pass[i].m_body = this;
			m_pass[i].m_idx = i;

			addPass(&m_pass[i]);
		}

		return true;
	}

	void ATrousDenoiser::ATrousPass::prepareRender(
		const void* pixels,
		bool revert)
	{
		shader::prepareRender(pixels, revert);

		// Bind source tex handle.
		if (m_idx == 0)
		{
			GLuint srcTexHandle = visualizer::getTexHandle();
			auto prevPass = m_body->getPrevPass();
			if (prevPass) {
				srcTexHandle = prevPass->getFbo().getTexHandle();
			}

			texture::bindAsGLTexture(srcTexHandle, 0, this);
		}
		else {
			auto prevPass = getPrevPass();
			auto texHandle = prevPass->getFbo().getTexHandle();

			texture::bindAsGLTexture(texHandle, 0, this);
		}

		// Bind G-Buffer.
		m_body->m_normal.bindAsGLTexture(1, this);
		m_body->m_pos.bindAsGLTexture(2, this);

		int stepScale = 1 << m_idx;

		auto hStepScape = this->getHandle("");
		CALL_GL_API(::glUniform1i(hStepScape, stepScale));

		// TODO
		// Sigma.
	}
}