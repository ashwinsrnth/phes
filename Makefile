CUDA_FLAGS=-arch=sm_30


test:
	make test_kernel

kernel: diffusion_kernel.cu
	nvcc -cubin ${CUDA_FLAGS} diffusion_kernel.cu

test_kernel: diffusion_kernel.cu test_diffusion_kernel.py
	make kernel
	nosetests --nocapture test_diffusion_kernel.py
	rm *.pyc

clean:
	rm -f *.pyc
	rm -f *.cubin
