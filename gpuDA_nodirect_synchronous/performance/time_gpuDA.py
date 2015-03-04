import sys
sys.path.append('..')
from gpuDA import *
from pycuda import autoinit

# Measures the time taken by
# the halo swap routine to complete
# and sends that time to stdout.

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

npx = 3
npy = 3
npz = 3
    
assert(npx*npy*npz == size)

# try to pick up problem size from the
# command line:

try:
    nz = int(sys.argv[1])
    ny = int(sys.argv[2])
    nx = int(sys.argv[3])

except:
    "This script must be provided with nz, ny, nx as command line parameters."
    sys.exit()

a = np.arange(nx*ny*nz, dtype=np.float64).reshape([nz, ny, nx])
a_gpu = gpuarray.to_gpu(a)
b_gpu = gpuarray.to_gpu(np.zeros([nz+2, ny+2, nx+2], dtype=np.float64))

comm = comm.Create_cart([npz, npy, npx], reorder=False)
da = GpuDA(comm, [nz, ny, nx], [npz, npy, npx], 1)

t1 = MPI.Wtime()
da.global_to_local(a_gpu, b_gpu)
t2 = MPI.Wtime()

if rank == 0:
    print t2-t1

MPI.Finalize()
