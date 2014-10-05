from set_boundary import *
import nose

class TestSetBoundary:
    
    def setup(self):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        npx = 3
        npy = 3
        npz = 3
        assert(size == 27)

        self.comm = comm.Create_cart([npz, npx, npy])
        self.rank = comm.Get_rank()

    def test_set_boundary_values(self):
        f_local = np.zeros([3,3,3],dtype=np.float64)
        set_boundary_values(self.comm, f_local, [1, 2, 3, 4, 5, 6])
        if self.rank == 0 or self.rank == 12 or self.rank == 24:
            assert(f_local[1,1,0] == 1)


