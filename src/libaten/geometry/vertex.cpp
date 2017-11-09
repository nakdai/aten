#include "geometry/vertex.h"

namespace aten
{
	std::vector<int> VertexManager::s_indices;
	std::vector<vertex> VertexManager::s_vertices;
	GeomVertexBuffer VertexManager::s_vb;

	void VertexManager::build()
	{
		s_vb.init(
			sizeof(vertex),
			s_vertices.size(),
			0,
			&s_vertices[0]);
	}
}