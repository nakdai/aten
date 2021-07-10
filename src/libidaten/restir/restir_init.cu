#include "restir/restir.h"

#include "cuda/cudadefs.h"
#include "cuda/helper_math.h"
#include "cuda/cudautil.h"
#include "cuda/cudamemory.h"

#include "aten4idaten.h"

__global__ void initReSTIR(
    idaten::Path* path,
    idaten::PathThroughput* throughput,
    idaten::PathContrib* contrib,
    idaten::PathAttribute* attrib,
    aten::sampler* sampler)
{
    path->throughput = throughput;
    path->contrib = contrib;
    path->attrib = attrib;
    path->sampler = sampler;
}

namespace idaten
{
    void ReSTIRPathTracing::onInit(int width, int height)
    {
        if (!m_isInitPash) {

            m_paths.init(1);

            m_pathThroughput.init(width * height);
            m_pathContrib.init(width * height);
            m_pathAttrib.init(width * height);
            m_pathSampler.init(width * height);

            for (int i = 0; i < 2; i++) {
                m_intermediates[i].init(width * height);
                m_reservoirs[i].init(width * height);
            }

            initReSTIR << <1, 1, 0, m_stream >> > (
                m_paths.ptr(),
                m_pathThroughput.ptr(),
                m_pathContrib.ptr(),
                m_pathAttrib.ptr(),
                m_pathSampler.ptr());

            m_isInitPash = true;
        }
    }

    void ReSTIRPathTracing::onClear()
    {
        cudaMemsetAsync(m_pathThroughput.ptr(), 0, m_pathThroughput.bytes(), m_stream);
        cudaMemsetAsync(m_pathContrib.ptr(), 0, m_pathContrib.bytes(), m_stream);
        cudaMemsetAsync(m_pathAttrib.ptr(), 0, m_pathAttrib.bytes(), m_stream);

        if (m_frame == 0) {
            cudaMemsetAsync(m_pathSampler.ptr(), 0, m_pathSampler.bytes(), m_stream);
        }
    }
}
