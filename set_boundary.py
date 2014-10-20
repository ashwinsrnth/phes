from mpi4py import MPI
import numpy as np

def set_boundary_values(da, f, vals):

    # Set the boundary values for
    # a distributed array representing
    # values on a cartesian grid. This
    # code assumes a stencil width of 1.
    #
    # Parameters:
    # ----------
    #
    # f:        Local portion of distributed array (with ghost points)
    # vals:     6-tuple of scalar boundary values (xlow, xhigh, ylow, yhigh, zlow, zhigh)


    nz, ny, nx = da.getSizes()
    (zstart, zend), (ystart, yend), (xstart, xend) = da.getRanges()

    # left and right (x)
    if xstart == 0:
        f[:,:,0] = vals[0]

    if xend == nx:
        f[:,:,-1] = vals[1]

    # bottom and top (y)
    if ystart == 0:
        f[:,0,:] = vals[2]

    if yend == ny:
        f[:,-1,:] = vals[3]

    # front and back (z)
    if zstart == 0:
        f[0,:,:] = vals[4]

    if zend == nz:
        f[-1,:,:] = vals[5]
