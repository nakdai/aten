#include "asvgf/asvgf.h"

namespace idaten
{
    bool AdvancedSVGFPathTracing::setBlueNoises(std::vector<aten::texture*>& noises)
    {
        const auto W = noises[0]->width();
        const auto H = noises[0]->height();

        // All noise texture have to be same size.
        {
            for (int i = 1; i < noises.size(); i++) {
                const auto n = noises[i];

                auto _w = n->width();
                auto _h = n->height();

                if (W != _w || H != _h) {
                    AT_ASSERT(false);
                    return false;
                }
            }
        }

        std::vector<const aten::vec4*> data;
        for (const auto n : noises) {
            data.push_back(n->colors());
        }

        m_bluenoise.init(data, W, H);

        return true;
    }

    void AdvancedSVGFPathTracing::render(
        const TileDomain& tileDomain,
        int maxSamples,
        int maxBounce)
    {
#if 0
        CudaGLResourceMapper rscmap(&m_glimg);
        auto outputSurf = m_glimg.bind();

        onDebug(tileDomain.w, tileDomain.h, outputSurf);
#else
        SVGFPathTracing::render(tileDomain, maxSamples, maxBounce);
#endif
    }
}