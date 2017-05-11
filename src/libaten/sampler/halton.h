#pragma once

#include <vector>
#include "types.h"
#include "sampler/sampler.h"

namespace aten {
	class Halton : public sampler {
	private:
		static std::vector<uint32_t> PrimeNumbers;

	public:
		static const uint32_t MaxPrimeNumbers = 10000000;

		// �f������.
		static void makePrimeNumbers(uint32_t maxNumber = MaxPrimeNumbers);

	public:
		Halton(uint32_t idx)
		{
			init(idx);
		}
		virtual ~Halton() {}

		virtual void init(uint32_t seed) override final
		{
			m_idx = (seed == 0 ? 1 : seed);
		}

		// [0, 1]
		virtual real nextSample() override final;

	private:
		uint32_t m_idx{ 1 };
		uint32_t m_dimension{ 0 };
	};
}
