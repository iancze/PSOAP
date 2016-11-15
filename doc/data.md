# Data formats

PSOAP relies upon chunks of data. When working with real data, there are a few things to keep in mind.

First, it may so happen that certain pixels may need to be masked, for example due to cosmic ray hits.

This means that actual data chunks will probably have an un-equal number of pixels per epoch. This is OK.

We also need to optimize the fluxes for each chunk.

So, all chunks will be generated, but when going to the inference routines, the masks will be applied to the data.

I think how we parallelize this will affect the data storage format.

Each chunk should be able to be read in on a process.

Really, we probably want the chunks to just be fed velocities, not orbital parameters, right?

Anyway, this can be straightened out later. First get the data format squared away.

## Generate Masks
Ability to create masks for ranges of dates, and ranges of wavelengths

# Processing for setup

psoap_generate_chunks.py
psoap_generate_mask.py
psoap_optimize_cal.py

then, to infer parameters,

psoap_sample_parallel.py
