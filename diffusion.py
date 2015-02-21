from petsc4py import PETSc as petsc
from mpi4py import MPI
import pycuda.driver as cuda
from pycuda import autoinit
import pycuda.gpuarray as gpuarray
import numpy as np
import time
import sys
sys.path.append('gpuDA')
sys.path.append('petsc-pycuda')
from gpuDA import GpuDA
import GPUArray
from set_boundary import set_boundary_values

da_backend = 'gpuDA'

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
nx = 32
ny = 32
nz = 32

# global lengths:
lx = 0.3
ly = 0.3
lz = 0.3

# material properties:
alpha = 1e-5

dx = lx/((npx*nx)-1)
dy = ly/((npy*ny)-1)
dz = lz/((npz*nz)-1)

# compute dt for stability:
dt = 0.1*(dx**2)/(alpha)
nsteps = 1000

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

T2_local_gpu = da.createLocalVec()
T1_local_gpu = da.createLocalVec()
T1_global_gpu = da.createGlobalVec()

T2_local_gpu.fill(0)
T1_local_gpu.fill(0)
T1_global_gpu.fill(0)

T1_local_cpu = T1_local_gpu.get()

# set initial conditions:
set_boundary_values(da, T1_local_cpu, (100., 100., 100., 500., 100., 100.))
T1_local_gpu.set(T1_local_cpu)

da.local_to_global(T1_local_gpu, T1_global_gpu)

comm_time = np.zeros(nsteps)
comp_time = np.zeros(nsteps)

t_start = time.time()

start = cuda.Event()
end = cuda.Event()

# simulation loop:
for step in range(nsteps):

    da.global_to_local(T1_global_gpu, T1_local_gpu)
    
    start.record()
    func.prepared_call(((nx+2)/16, (ny+2)/16, 1), (16, 16, 1),
                        T1_local_gpu.gpudata, T2_local_gpu.gpudata,
                        alpha, dt,
                        nx+2, ny+2, nz+2,
                        dx, dy, dz)
    end.record()
    end.synchronize()
    comm.Barrier()      
    comp_time[step] = start.time_till(end) * 1e-3
    da.local_to_global(T2_local_gpu, T1_global_gpu)

t_end = time.time()

if rank == 13:
    print 'Computation: ', comp_time.sum()
    print 'Total: ', t_end-t_start

# copy all the data to rank-0:
T1_global_cpu = T1_global_gpu.get()
T1_local_cpu = T1_local_gpu.get()

T1_full = None
if rank == 0:
    T1_full = np.zeros([npz*nz, npy*ny, npx*nx], dtype=np.float64)

# create the MPI data-type required for the copy:
proc_z, proc_y, proc_x = comm.Get_topo()[2]
start_z, start_y, start_x = proc_z*nz, proc_y*ny, proc_x*nx
subarray_aux = MPI.DOUBLE.Create_subarray([npz*nz, npy*ny, npx*nx], 
            [nz, ny, nx], [start_z, start_y, start_x])
subarray = subarray_aux.Create_resized(0, 1*8)
subarray.Commit()

# everyone sends their displacement to rank-0:
start_index = np.array(start_z*(npx*nx*npy*ny) + start_y*(npx*nx) + start_x, dtype=np.int)
sendbuf = [start_index, MPI.INT]
displs = np.zeros(size, dtype=np.int)
recvbuf = [displs, MPI.INT]
comm.Gather(sendbuf, recvbuf, root=0)
comm.Barrier()

# perform the gather:
comm.Gatherv([T1_global_cpu, MPI.DOUBLE],
             [T1_full,  np.ones(size, dtype=np.int), displs, subarray], root=0)

subarray.Free()

# plot the data at rank-0
if rank == 0:
    from matplotlib.pyplot import contourf, savefig, pcolor, colorbar
    pcolor(T1_full[npz*nz/2, :, :], cmap='jet')
    colorbar()
    savefig('heat-solution.png')


