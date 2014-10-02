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

npz = 3
npy = 3
npx = 3

assert(size == npx*npy*npz)

mod = cuda.module_from_file('diffusion_kernel.cubin')
func = mod.get_function('temperature_update16x16')

nx = 32
ny = 32
nz = 32
alpha = 0.1
dt = 0.0001

dx = 1./(nx-1)
dy = 1./(ny-1)
dz = 1./(nz-1)

# create communicator:
comm = comm.Create_cart([npz, npy, npx])

# create arrays:
T1_global = np.ones([nz, ny, nx], dtype=np.float64)
T1_global_gpu = gpuarray.to_gpu(T1_global)

T2_local = np.zeros([nz+2, ny+2, nx+2], dtype=np.float64)
T2_local_gpu = gpuarray.to_gpu(T2_local)

T1_local = np.zeros([nz+2, ny+2, nx+2], dtype=np.float64)
T1_local_gpu = gpuarray.to_gpu(T1_local)

# prepare kernel:
func.prepare([np.intp, np.intp, np.float64, np.float64,
                np.intc, np.intc, np.intc,
                    np.float64, np.float64, np.float64])

# create distributed array:
local_dims = [nz, ny, nx]
proc_sizes = [npz, npy, npx]

da = GpuDA(comm, local_dims, proc_sizes, 1)

for step in range(100):
    
    da.global_to_local(T1_global_gpu, T2_local_gpu)

    func.prepared_call((1,1,1), (16,16,1),
                        T1_local_gpu.gpudata, T2_local_gpu.gpudata,
                        alpha, dt,
                        nx, ny, nz,
                        dx, dy, dz)

    da.local_to_global(T2_local_gpu, T1_global_gpu)
    print step

if rank == 13:
    print T1_global_gpu.get()[0,0,0]

MPI.Finalize()
