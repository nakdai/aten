#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "aten.h"

namespace aten {
	class CudaMemory : public IStream {
	public:
		CudaMemory() {}

		CudaMemory(uint32_t bytes);
		CudaMemory(const void* p, uint32_t bytes);

		virtual ~CudaMemory();

	public:
		void init(uint32_t bytes);

		const void* ptr() const
		{
			return m_device;
		}
		void* ptr()
		{
			return m_device;
		}
		
		uint32_t bytes() const
		{
			return m_bytes;
		}

		virtual __host__ uint32_t write(const void* p, uint32_t size) override final;
		virtual __host__ uint32_t read(void* p, uint32_t size) override final;

		operator void*()
		{
			return m_device;
		}

		void reset();

		void free();

		static uint32_t getHeapSize();

	private:
		void* m_device{ nullptr };
		uint32_t m_bytes{ 0 };
		uint32_t m_pos{ 0 };
	};

	template <typename _T>
	class TypedCudaMemory : public CudaMemory {
	public:
		TypedCudaMemory() {}

		TypedCudaMemory(uint32_t num)
			: CudaMemory(sizeof(_T) * num)
		{
			m_num = num;
		}
		TypedCudaMemory(const _T* p, uint32_t num)
			: CudaMemory(p, sizeof(_T) * num)
		{
			m_num = num;
		}

		virtual ~TypedCudaMemory() {}

	public:
		__host__ uint32_t writeByNum(const _T* p, uint32_t num)
		{
			return CudaMemory::write(p, sizeof(_T) * num);
		}

		__host__ uint32_t readByNum(void* p, uint32_t num)
		{
			return CudaMemory::read(p, sizeof(_T) * num);
		}

		uint32_t num() const
		{
			return m_num;
		}

		const _T* ptr() const
		{
			return (const _T*)CudaMemory::ptr();
		}
		_T* ptr()
		{
			return (_T*)CudaMemory::ptr();
		}

	private:
		uint32_t m_num{ 0 };
	};
}