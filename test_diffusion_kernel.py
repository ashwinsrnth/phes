import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycuda import autoinit
import numpy as np

class TestDiffusionKernel(object):
    
    def setup(self):
        self.mod = cuda.module_from_file('diffusion_kernel.cubin')
        self.func = self.mod.get_function("temperature_update16x16")

    def test_x_diffusion(self):        
        
        f = np.ones([3, 3, 3], dtype=np.float64)
        f = f*np.array([1,2,3], dtype=np.float64)
        f_gpu = gpuarray.to_gpu(f)

        f2 = np.zeros([3, 3, 3], dtype=np.float64)
        f2_gpu = gpuarray.to_gpu(f2)
        
        # run the kernel:
        self.func.prepare([np.intp, np.intp, np.float64, np.float64,
                             np.intc, np.intc, np.intc, 
                                 np.float64, np.float64, np.float64])

        self.func.prepared_call((1,1,1), (16, 16, 1), 
                           f_gpu.gpudata, f2_gpu.gpudata, 1., 1., 3, 3, 3, 1, 1, 1)

       
        f2 = f2_gpu.get()
        assert(f2[1,1,1] == 2.0)

    def test_z_diffusion(self):

        # test diffusion in the z-direction:
        
        f = np.ones([3, 3, 3], dtype=np.float64)
        f = f*np.array([1, 2, 3], dtype=np.float64)
        f = np.transpose(f, [2, 1, 0])
        f_gpu = gpuarray.to_gpu(f)

        f2 = np.zeros([3, 3, 3], dtype=np.float64)
        f2_gpu = gpuarray.to_gpu(f2)
        
        # run the kernel:
        self.func.prepare([np.intp, np.intp, np.float64, np.float64,
                             np.intc, np.intc, np.intc, 
                                 np.float64, np.float64, np.float64])

        self.func.prepared_call((1,1,1), (16, 16, 1), 
                           f_gpu.gpudata, f2_gpu.gpudata, 1., 1., 3, 3, 3, 1, 1, 1)
 
        f2 = f2_gpu.get()
        assert(f2[1,1,1] == 2.0)

    #TODO: pass this test.
    def test_inadequate_size(self):

        f = np.ones([3, 3, 3], dtype=np.float64)
        f = f*np.array([1,2,3], dtype=np.float64)
        f_gpu = gpuarray.to_gpu(f)

        f2 = np.zeros([3, 3, 3], dtype=np.float64)
        f2_gpu = gpuarray.to_gpu(f2)
        
        # run the kernel:
        self.func.prepare([np.intp, np.intp, np.float64, np.float64,
                             np.intc, np.intc, np.intc, 
                                 np.float64, np.float64, np.float64])

        #nose.tools.assert_raises(ValueError, 
        #        self.func.prepared_call, (1,1,1), (1, 1, 1), 
        #        f_gpu.gpudata, f2_gpu.gpudata, 1., 1., 3, 3, 3, 1, 1, 1)
