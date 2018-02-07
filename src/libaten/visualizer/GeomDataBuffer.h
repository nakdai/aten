#pragma once

#include "types.h"

namespace aten {
	enum Primitive {
		Triangles,
		Lines,
	};

	struct VertexAttrib {
		int type;
		int num;
		int size;
		int offset;
	};

	class GeomVertexBuffer {
		friend class GeomIndexBuffer;

	public:
		GeomVertexBuffer() {}
		virtual ~GeomVertexBuffer() {}

	public:
		void init(
			uint32_t stride,
			uint32_t vtxNum,
			uint32_t offset,
			const void* data);

		void init(
			uint32_t stride,
			uint32_t vtxNum,
			uint32_t offset,
			const VertexAttrib* attribs,
			uint32_t attribNum,
			const void* data);

		void update(
			uint32_t vtxNum,
			const void* data);

		void draw(
			Primitive mode,
			uint32_t idxOffset,
			uint32_t primNum);

	protected:
		uint32_t m_vbo{ 0 };
		uint32_t m_vao{ 0 };

		uint32_t m_vtxStride{ 0 };
		uint32_t m_vtxNum{ 0 };
		uint32_t m_vtxOffset{ 0 };

		uint32_t m_initVtxNum{ 0 };
	};

	//////////////////////////////////////////////////////////

	class GeomIndexBuffer {
	public:
		GeomIndexBuffer() {}
		virtual ~GeomIndexBuffer() {}

	public:
		void init(
			uint32_t idxNum,
			const void* data);

		void update(
			uint32_t idxNum,
			const void* data);

		void lock(void** dst);
		void unlock();

		void draw(
			GeomVertexBuffer& vb,
			Primitive mode,
			uint32_t idxOffset,
			uint32_t primNum);

	protected:
		uint32_t m_ibo{ 0 };

		uint32_t m_idxNum{ 0 };

		bool m_isLockedIBO{ false };

		uint32_t m_initIdxNum{ 0 };
	};
}