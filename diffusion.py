from mpi4py import MPI
import pycuda.driver as cuda
from pycuda import autoinit
import pycuda.gpuarray as gpuarray
import numpy as np
import time
import sys
from set_boundary import set_boundary_values
import argparse

def diffusion_kernel(T1_local, T2_local,
                      alpha, dt,
                      NX, NY, NZ,
                      dx, dy, dz):

    T2_local[1:-1,1:-1,1:-1] = T1_local[1:-1,1:-1,1:-1] + alpha*dt*(
            (T1_local[0:-2, 1:-1, 1:-1] - 2*T1_local[1:-1, 1:-1, 1:-1] + T1_local[2:, 1:-1, 1:-1])/dz**2. +
            (T1_local[1:-1, 0:-2, 1:-1] - 2*T1_local[1:-1, 1:-1, 1:-1] + T1_local[1:-1, 2:, 1:-1])/dy**2. +
            (T1_local[1:-1, 1:-1, 0:-2] - 2*T1_local[1:-1, 1:-1, 1:-1] + T1_local[1:-1, 1:-1, 2:])/dx**2.)




# Parse command line arguments:
parser = argparse.ArgumentParser()

parser.add_argument('--local_size', dest='local_size', type=int, nargs=3, metavar=('NX', 'NY', 'NZ'),
                    help='The problem size local to each process')
parser.add_argument('--nsteps', dest='nsteps', type=int, default=10,
                    help='Number of time steps')
parser.add_argument('--use_gpu', dest='use_gpu',
                    help='Use the gpu', action='store_true')
parser.add_argument('--gpudirect', dest='gpudirect',
                    help='Use GPUDirect for gpu-gpu communication (ignored if not using GPU)', action='store_true')
parser.add_argument('--asynchronous', dest='asynchronous',
                    help='Use asynchronous MPI calls', action='store_true')

args = parser.parse_args()

# Get the right GpuDA:
if args.use_gpu:
    if args.gpudirect and args.asynchronous:
        from gpuDA.gpuDA import GpuDA
    elif args.asynchronous:
        from gpuDA_nodirect.gpuDA import GpuDA
    elif args.gpudirect:
        from gpuDA_synchronous.gpuDA import GpuDA
    else:
        from gpuDA_nodirect_synchronous.gpuDA import GpuDA
    device = 'gpu'
else:
    from gpuDA_cpu.gpuDA import GpuDA
    device = 'cpu'

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

npz = 3
npy = 3
npx = 3

assert(size == npx*npy*npz)

nz, ny, nx = args.local_size
nsteps = args.nsteps

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
dt = 0.1 *(dx**2)/(alpha)


# create communicator:
comm = comm.Create_cart([npz, npy, npx], reorder=False)

if args.use_gpu:
    # prepare kernel:
    mod = cuda.module_from_file('diffusion_kernel.cubin')
    func = mod.get_function('temperature_update16x16')
    func.prepare([np.intp, np.intp, np.float64, np.float64,
                    np.intc, np.intc, np.intc,
                        np.float64, np.float64, np.float64])

# communication information for distributed array:
local_dims = [nz, ny, nx]
proc_sizes = [npz, npy, npx]
da = GpuDA(comm, local_dims, proc_sizes, 1)

# create arrays:
T2_local = np.ones([nz+2, ny+2, nx+2], dtype=np.float64)
T1_local = np.zeros([nz+2, ny+2, nx+2], dtype=np.float64)
T1_global = np.ones([nz, ny, nx], dtype=np.float64)

# set initial conditions:
set_boundary_values(comm, T1_local, (500., -100., 500., -100., 0., 0.))

if args.use_gpu:
    # transfer to gpu:
    T2_local_gpu = gpuarray.to_gpu(T2_local)
    T1_local_gpu = gpuarray.to_gpu(T1_local)
    T1_global_gpu = gpuarray.to_gpu(T1_global)

    da.local_to_global(T1_local_gpu, T1_global_gpu)

    gtol_time = np.zeros(nsteps)
    comp_time = np.zeros(nsteps)

    start = cuda.Event()
    end = cuda.Event()

    t_start = time.time()
    # simulation loop:
    for step in range(nsteps):
        t1 = time.time()
        da.global_to_local(T1_global_gpu, T1_local_gpu)
        comm.Barrier()
        t2 = time.time()
        gtol_time[step] = t2-t1

        start.record()
        func.prepared_call(((nx+2)/16, (ny+2)/16, 1), (16, 16, 1),
                            T1_local_gpu.gpudata, T2_local_gpu.gpudata,
                            alpha, dt,
                            nx+2, ny+2, nz+2,
                            dx, dy, dz)
        end.record()
        end.synchronize()
        comp_time[step] = start.time_till(end)*1e-3

        da.local_to_global(T2_local_gpu, T1_global_gpu)

    t_end = time.time()

else:
    da.local_to_global(T1_local, T1_global)

    gtol_time = np.zeros(nsteps)
    comp_time = np.zeros(nsteps)

    t_start = time.time()
    # simulation loop:
    for step in range(nsteps):
        t1 = time.time()
        da.global_to_local(T1_global, T1_local)
        t2 = time.time()
        gtol_time[step] = t2-t1

        t1 = time.time()
        diffusion_kernel(T1_local, T2_local,
                          alpha, dt,
                          nx+2, ny+2, nz+2,
                          dx, dy, dz)
        t2 = time.time()

        comp_time[step] = t2-t1

        da.local_to_global(T2_local, T1_global)

    t_end = time.time()

if rank == 13:
    print 'Total: ', t_end-t_start
    print 'Comp: ', comp_time.sum()
    print 'Comm: ', gtol_time.sum()

'''
# copy all the data to rank-0:
T1_global = T1_global_gpu.get()
T1_local = T1_local_gpu.get()

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
comm.Gatherv([T1_global, MPI.DOUBLE],
             [T1_full,  np.ones(size, dtype=np.int), displs, subarray], root=0)

subarray.Free()

# plot the data at rank-0
if rank == 0:
    from matplotlib.pyplot import contourf, savefig, pcolor, colorbar
    pcolor(T1_full[npz*nz/2, :, :], cmap='jet')
    colorbar()
    savefig('heat-solution.png')
'''
MPI.Finalize()
