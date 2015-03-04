from theano import shared
import theano.tensor
import numpy as np
import pygpu

class MyShared(theano.tensor.sharedvar.SharedVariable):

    def getGPU(self):
        return self.vec.container.data

    @property
    def itemsize(self):
        return self.vec.container.data.dtype.itemsize

    @property
    def shape(self):
        return self.vec.container.data.shape

    @property
    def nbytes(self):
        return self.vec.container.data.size*self.vec.container.data.dtype.itemsize

    @property
    def strides(self):
        return self.vec.container.data.strides

    @property
    def gpudata(self):
        try:
            return pygpu.gpuarray.get_raw_ptr(self.vec.container.data.gpudata)
        except:
            raise AttributeError('can''t get gpudata (are you using the GPU?)')
