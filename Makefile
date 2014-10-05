CUDA_FLAGS=-arch=sm_30
MPI_FLAGS=--mca pml ob1 --mca btl_openib_cuda_want_gdr 0

test:
	make test_kernel
	make test_set_boundary

kernel: diffusion_kernel.cu
	nvcc -cubin ${CUDA_FLAGS} diffusion_kernel.cu

test_kernel: diffusion_kernel.cu test_diffusion_kernel.py
	make kernel
	nosetests --nocapture test_diffusion_kernel.py
	rm *.pyc

test_set_boundary: test_set_boundary.py
	mpiexec -n 27 ${MPI_FLAGS} nosetests test_set_boundary.py

clean:
	rm -f *.pyc
	rm -f *.cubin
