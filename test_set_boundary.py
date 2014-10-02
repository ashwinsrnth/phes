from set_boundary import *

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

npx = 3
npy = 3
npz = 3

assert(npx*npy*npz == size)

comm = comm.Create_cart([npz, npx, npy])
f_local = np.zeros([3,3,3],dtype=np.float64)

set_boundary_values(comm, f_local, [1, 2, 3, 4, 5, 6])

if rank == 0 or rank == 12 or rank == 24:
    assert(f_local[1,1,0] == 1)

