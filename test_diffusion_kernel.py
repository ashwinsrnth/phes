import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycuda import autoinit
import numpy as np

class TestDiffusionKernel(object):
    
    def setup(self):
        self.mod = cuda.module_from_file('diffusion_kernel.cubin')
        self.func = self.mod.get_function("temperature_update16x16")

    def test_prepared_call(self):        
        f = np.ones([3, 3, 3], dtype=np.float32)
        f = f*np.array([1,2,3], dtype=np.float32)
        f_gpu = gpuarray.to_gpu(f)

        dfdx = np.zeros([3, 3, 3], dtype=np.float32)
        dfdx_gpu = gpuarray.to_gpu(dfdx)
        
        # run the kernel:
        self.func.prepare([np.intp, np.intp, np.float32, np.float32,
                             np.intc, np.intc, np.intc, 
                                 np.float32, np.float32, np.float32])

        self.func.prepared_call((1,1,1), (16, 16, 1), 
                           f_gpu.gpudata, dfdx_gpu.gpudata, 1., 1., 3, 3, 3, 1, 1, 1)

        f = f_gpu.get()
        assert(f[1,1,1] == 2.0)
