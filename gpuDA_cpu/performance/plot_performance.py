import numpy as np
from matplotlib.pyplot import *


def effective_bandwidth(array_size, time_taken):
    # compute the effective bandwidth
    # of transferring an array of 
    # `array_size` in `time_taken` seconds

    # number of bits transported:
    nbits = array_size*64.0
    nGbits = nbits/(1024.*1024*1024)

    return nGbits/time_taken


for fname in ['x_timings.txt', 'y_timings.txt', 'z_timings.txt']:

    prob_sizes = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    times = np.loadtxt(fname)

    bandwidths = []
    for time, size in zip(times, prob_sizes):
        bandwidths.append(effective_bandwidth(size*size, time))

    plot(prob_sizes, bandwidths, label=fname[0])

legend()
xlabel('Halo size in one dimension')
ylabel('Effective bandwidth')

savefig('effective_bandwidths.png')
