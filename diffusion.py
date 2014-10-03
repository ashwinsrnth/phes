from mpi4py import MPI
import pycuda.driver as cuda
from pycuda import autoinit
import pycuda.gpuarray as gpuarray
import numpy as np

import sys
sys.path.append('gpuDA')

from gpuDA import GpuDA
from set_boundary import set_boundary_values

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

npz = 1
npy = 1
npx = 1

assert(size == npx*npy*npz)

mod = cuda.module_from_file('diffusion_kernel.cubin')
func = mod.get_function('temperature_update16x16')

nx = 17
ny = 17
nz = 17
alpha = 1.0
dt = 0.0001

dx = 1./((npx*nx)-1)
dy = 1./((npy*ny)-1)
dz = 1./((npz*nz)-1)

# create communicator:
comm = comm.Create_cart([npz, npy, npx])

# prepare kernel:
func.prepare([np.intp, np.intp, np.float64, np.float64,
                np.intc, np.intc, np.intc,
                    np.float64, np.float64, np.float64])

# communication information for distributed array:
local_dims = [nz, ny, nx]
proc_sizes = [npz, npy, npx]
da = GpuDA(comm, local_dims, proc_sizes, 1)

# create arrays:
T2_local = np.zeros([nz+2, ny+2, nx+2], dtype=np.float64)
T1_local = np.zeros([nz+2, ny+2, nx+2], dtype=np.float64)
T1_global = np.zeros([nz, ny, nx], dtype=np.float64)

# set initial conditions:
set_boundary_values(comm, T1_local, (500., 100., 100., 100., 100., 100.))

# transfer to gpu:
T2_local_gpu = gpuarray.to_gpu(T2_local)
T1_local_gpu = gpuarray.to_gpu(T1_local)
T1_global_gpu = gpuarray.to_gpu(T1_global)

da.local_to_global(T1_local_gpu, T1_global_gpu)

np.set_printoptions(precision=2)
for step in range(1000):

    da.global_to_local(T1_global_gpu, T1_local_gpu)
    func.prepared_call((2, 2, 2), (16, 16, 1),
                        T1_local_gpu.gpudata, T2_local_gpu.gpudata,
                        alpha, dt,
                        nx+2, ny+2, nz+2,
                        dx, dy, dz)

    da.local_to_global(T2_local_gpu, T1_global_gpu)

if rank == 0:
    print T2_local_gpu.get()[1:-1,1:-1,1:-1]
    print T1_global_gpu
MPI.Finalize()
