#!/bin/bash

fname=LP661-13
echo $fname

rm $fname.hdf5
rm files.txt

# Regenerate the files list
cd spectra
ls *fits > ../files.txt
cd ..

# Remove any converted stuff
rm -rf jsons_BCV
rm -rf plots_BCV


# Make the jsons_BCV and its plots folder
mkdir jsons_BCV
mkdir plots_BCV

for f in `cat files.txt`
do
	g=`basename $f .fits`
	TRESio_astropy.py spectra_raw/$f spectra/$f jsons_BCV/$g.json --BCV
done

# Actually make the plots
python plot_all_orders.py

# Convert the dates in the header into an actual array
python fname_to_date.py

python StuffHDF5.py $fname
