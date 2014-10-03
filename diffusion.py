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

# local sizes:
nx = 398
ny = 398
nz = 398

# global lengths:
lx = 1.0
ly = 1.0
lz = 1.0

# material properties:
alpha = 1e-5
dt = 0.01

dx = lx/((npx*nx)-1)
dy = ly/((npy*ny)-1)
dz = lz/((npz*nz)-1)

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
set_boundary_values(comm, T1_local, (0., 0., 0., 0., 0., 500.))

# transfer to gpu:
T2_local_gpu = gpuarray.to_gpu(T2_local)
T1_local_gpu = gpuarray.to_gpu(T1_local)
T1_global_gpu = gpuarray.to_gpu(T1_global)

da.local_to_global(T1_local_gpu, T1_global_gpu)

np.set_printoptions(precision=2)

comm_time = np.zeros(1000)
comp_time = np.zeros(1000)


t_start = MPI.Wtime()
for step in range(1000):

    t1 = MPI.Wtime()
    da.global_to_local(T1_global_gpu, T1_local_gpu)
    t2 = MPI.Wtime()

    func.prepared_call(((nx+2)/16, (ny+2)/16, 1), (16, 16, 1),
                        T1_local_gpu.gpudata, T2_local_gpu.gpudata,
                        alpha, dt,
                        nx+2, ny+2, nz+2,
                        dx, dy, dz)

    da.local_to_global(T2_local_gpu, T1_global_gpu)
    t3 = MPI.Wtime()

    comp_time[step] = t3-t2
    comm_time[step] = t2-t1

t_end = MPI.Wtime()

if rank == 0:
    print 'Computation: ', comp_time.mean()*1000
    print 'Communication: ', comm_time.mean()*1000
    print 'Total: ', t_end-t_start

MPI.Finalize()
