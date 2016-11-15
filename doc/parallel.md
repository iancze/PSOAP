# Parallel

The sampling for each chunk will be parallelized across multiple cores. Since we will want to run this on a cluster, we need a way to do efficient sampling.

This probably means that there is a master process which does the MCMC sampling, and worker processes that send and receive parameter values, and then report back the lnprob. Presumably, we will eventually want to do sampling in the sub-parameters, right? The sub-parameters will be different length scales and amplitudes for the GP. I guess we will need to do some sort of sub-sampling.

Possible lower level parameters (orbital parameters fixed):
- amplitude, length scale difference for each chunk
- velocity jitter for each chunk, epoch
