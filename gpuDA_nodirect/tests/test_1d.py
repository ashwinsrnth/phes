from createDA import *
from pycuda import autoinit

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
proc_sizes = [1, 1, 3]
local_dims = [1, 3, 3]
nz, ny, nx = local_dims

da = create_da(proc_sizes, local_dims)

a = np.empty(local_dims, dtype=np.float64)
a.fill(rank)
a_gpu = gpuarray.to_gpu(a)
b_gpu = gpuarray.empty([nz+2, ny+2, nx+2], dtype=np.float64)

da.global_to_local(a_gpu, b_gpu)

# test center:
if rank == 1:
    assert(np.all(da.left_recv_halo.get() == 0))
    assert(np.all(da.left_send_halo.get() == 1))
    assert(np.all(da.right_send_halo.get() == 1))
    assert(np.all(da.right_recv_halo.get() == 2))

# test left and right:
if rank == 0:
    assert(np.all(da.right_recv_halo.get() == 1))

if rank == 2:
    assert(np.all(da.left_recv_halo.get() == 1))

MPI.Finalize()
