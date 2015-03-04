from createDA import *
from pycuda import autoinit
import nose


class TestGpuDA3d:

    @classmethod
    def setup_class(cls): 
        cls.comm = MPI.COMM_WORLD
        cls.rank = cls.comm.Get_rank()
        cls.size = cls.comm.Get_size()
        assert(cls.size == 27)
        cls.proc_sizes = [3, 3, 3]
        cls.local_dims = [4, 4, 4]
        
        cls.da = create_da(cls.proc_sizes, cls.local_dims)
 
    def test_gtol(self):

        nz, ny, nx = self.local_dims
        
        # fill a with rank
        a = np.zeros([nz,ny,nx], dtype=np.float64)
        a.fill(self.rank)
        a_gpu = gpuarray.to_gpu(a)

        # fill b with ones
        b = np.ones([nz+2,ny+2,nx+2], dtype=np.float64)
        b_gpu = gpuarray.to_gpu(b)

        self.da.global_to_local(a_gpu, b_gpu)

        # test gtol at the center
        if self.rank == 13:
            assert(np.all(b_gpu.get()[1:-1,1:-1,0] == 12))
            assert(np.all(b_gpu.get()[1:-1,1:-1,-1] == 14))
            assert(np.all(b_gpu.get()[1:-1,0,1:-1] == 10))
            assert(np.all(b_gpu.get()[1:-1,-1,1:-1] == 16))
            assert(np.all(b_gpu.get()[0,1:-1,1:-1] == 4))
            assert(np.all(b_gpu.get()[-1,1:-1,1:-1] == 22))
        
        # test that the boundaries remain unaffected:
        if self.rank == 22:
            # since we initially filled b with ones
            assert(np.all(b_gpu.get()[-1,:,:] == 1))

    def test_ltog(self):

        nz, ny, nx = self.local_dims

        # fill b with a sequence
        b = np.ones([nz+2, ny+2, nx+2], dtype=np.float64)
        b = b*np.arange((nx+2)*(ny+2)*(nz+2)).reshape([nz+2, ny+2, nx+2])
        b_gpu = gpuarray.to_gpu(b)

        # a is empty
        a_gpu = gpuarray.empty([nz,ny,nx], dtype=np.float64)

        self.da.local_to_global(b_gpu, a_gpu)

        # test ltog:
        if self.rank == 0:
            assert(np.all(a_gpu.get() == b_gpu.get()[1:-1,1:-1,1:-1]))

    @classmethod
    def teardown_class(cls):
        MPI.Finalize()
