from createDA import *
from numpy.testing import *
from pycuda import autoinit

class TestGPU1d:

    @classmethod
    def setup_class(cls):
        comm = MPI.COMM_WORLD
        cls.rank = comm.Get_rank()
        cls.size = comm.Get_size()
        proc_sizes = [1, 1, 3]
        local_dims = [1, 3, 3]
        nz, ny, nx = local_dims
        cls.da = create_da(proc_sizes, local_dims)

    def test_center(self):
        a_gpu = self.da.createGlobalVec()
        a_gpu.set_value(self.rank)
        b_gpu = self.da.createLocalVec()
        b_gpu.set_value(1.0)
        self.da.globalToLocal(a_gpu, b_gpu)

        # test center:
        if self.rank == 1:
            assert(np.all(self.da.left_recv_halo.get() == 0))
            assert(np.all(self.da.left_send_halo.get() == 1))
            assert(np.all(self.da.right_send_halo.get() == 1))
            assert(np.all(self.da.right_recv_halo.get() == 2))

    def test_sides(self):
        a_gpu = self.da.createGlobalVec()
        a_gpu.set_value(self.rank)
        b_gpu = self.da.createLocalVec()
        self.da.globalToLocal(a_gpu, b_gpu)

        # test left and right:
        if self.rank == 0:
            assert_equal(self.da.right_recv_halo.get(), 1)

        if self.rank == 2:
            assert_equal(self.da.left_recv_halo.get(), 1)

    @classmethod
    def teardown_class(cls):
        MPI.Finalize()
