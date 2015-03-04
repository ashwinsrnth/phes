from createDA import *
from pycuda import autoinit
import pygpu
from numpy.testing import *

class TestGpuDA3d:
    @classmethod
    def setup_class(cls): 
        cls.comm = MPI.COMM_WORLD
        cls.rank = cls.comm.Get_rank()
        cls.size = cls.comm.Get_size()
        assert(cls.size == 27)
        cls.proc_sizes = [3, 3, 3]
        cls.local_dims = [4, 4, 4]
        
        cls.da = create_da(cls.proc_sizes, cls.local_dims, dof=2)
    
    def test_gtol(self):

        nz, ny, nx = self.local_dims
        
        # fill a_gpu with rank
        a_gpu = self.da.createGlobalVec()
        a_gpu.set_value(self.rank)

        # fill b_gpu with ones
        b_gpu = self.da.createLocalVec()
        b_gpu.set_value(1.0)
        
        self.da.globalToLocal(a_gpu, b_gpu)

        # test gtol at the center
        if self.rank == 13:
            assert_equal(b_gpu.get()[1:-1,1:-1,0,:], 12)
            assert_equal(b_gpu.get()[1:-1,1:-1,-1,:], 14)
            assert_equal(b_gpu.get()[1:-1,0,1:-1,:], 10)
            assert_equal(b_gpu.get()[1:-1,-1,1:-1,:], 16)
            assert_equal(b_gpu.get()[0,1:-1,1:-1,:], 4)
            assert_equal(b_gpu.get()[-1,1:-1,1:-1,:], 22)
        
        # test that the boundaries remain unaffected:
        if self.rank == 22:
            # since we initially filled b with ones
            assert_equal(b_gpu.get()[-1,:,:,:], 1)
    
    def test_ltog(self):

        nz, ny, nx = self.local_dims

        # fill b with a sequence
        b = np.ones([nz+2, ny+2, nx+2], dtype=np.float64)
        b = b*np.arange((nx+2)*(ny+2)*(nz+2)).reshape([nz+2, ny+2, nx+2])
        b_gpu = self.da.createLocalVec()
        b_gpu.set(b)

        # a is empty
        a_gpu = self.da.createGlobalVec()
        self.da.localToGlobal(b_gpu, a_gpu)

        # test ltog:
        if self.rank == 0:
            assert_equal(a_gpu.get(), b_gpu.get()[1:-1,1:-1,1:-1,:])
    
    def test_get_ranges(self):

        nz, ny, nx = self.local_dims
        (zstart, zend), (ystart, yend), (xstart, xend) = self.da.getRanges()

        if self.rank == 13:
            assert_equal(zstart, nz)
            assert_equal(ystart, ny)
            assert_equal(xstart, nx)
            assert_equal(zend, 2*nz)
            assert_equal(yend, 2*ny)
            assert_equal(xend, 2*nx)
    
    def test_get_sizes(self):
        assert_equal(self.da.getSizes(), (12, 12, 12, 2))
    
    @classmethod
    def teardown_class(cls):
        MPI.Finalize()
