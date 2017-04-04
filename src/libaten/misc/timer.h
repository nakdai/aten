#pragma once

#include "types.h"

namespace aten {
	struct SystemTime {
		uint16_t year{ 0 };
		uint16_t month{ 0 };
		uint16_t dayOfWeek{ 0 };
		uint16_t day{ 0 };
		uint16_t hour{ 0 };
		uint16_t minute{ 0 };
		uint16_t second{ 0 };
		uint16_t milliSeconds{ 0 };
	};

	class timer {
	public:
		timer() {}
		~timer() {}

	public:
		static void init();

		void begin();
		real end();

		static SystemTime getSystemTime();

	private:
		int64_t m_begin;
	};
}