from mpi4py import MPI
import numpy as np

def set_boundary_values(comm, f, vals):

    # Set the boundary values for
    # a distributed array representing
    # values on a cartesian grid. This
    # code assumes a stencil width of 1.
    #
    # Parameters:
    # ----------
    #
    # comm:     An instance of MPI.Cartcomm
    # f:        Local portion of distributed array (with ghost points)
    # vals:     6-tuple of scalar boundary values (xlow, xhigh, ylow, yhigh, zlow, zhigh)
    
    npz, npy, npx = comm.Get_topo()[0]
    mz, my, mx = comm.Get_topo()[2]

    # left and right (x)
    if mx == 0:
        f[:,:,0] = vals[0]

    if mx == npx-1:
        f[:,:,-1] = vals[1]

    # bottom and top (y)
    if my == 0:
        f[:,0,:] = vals[2]

    if my == npy-1:
        f[:,-1,:] = vals[3]

    # front and back (z)
    if mz == 0:
        f[0,:,:] = vals[4]

    if mz == npz-1:
        f[-1,:,:] = vals[5]
