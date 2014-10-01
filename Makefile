CUDA_FLAGS=-arch=sm_30

test:
	make test_kernel

test_kernel: diffusion_kernel.cu test_diffusion_kernel.py
	make kernel
	nosetests test_diffusion_kernel.py
	rm *.pyc

kernel: diffusion_kernel.cu
	nvcc -cubin ${CUDA_FLAGS} diffusion_kernel.cu

