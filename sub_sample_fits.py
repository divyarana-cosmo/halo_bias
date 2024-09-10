import fitsio
import numpy as np

# Open the original FITS file
input_fits_file = 'input_file.fits'
output_fits_file = 'output_subsampled.fits'

# Read the data from the FITS file (assuming it's a table)
with fitsio.FITS(input_fits_file) as fits:
    # Read the table from the first HDU (Header/Data Unit)
    data = fits[1].read()

# Calculate the number of rows to sample (10% of the total data)
n_total = data.shape[0]
n_sample = int(n_total * 0.1)

# Randomly choose indices to keep
np.random.seed(42)  # For reproducibility, optional
sampled_indices = np.random.choice(n_total, n_sample, replace=False)

# Get the subsampled data
subsampled_data = data[sampled_indices]

# Write the subsampled data to a new FITS file
with fitsio.FITS(output_fits_file, 'rw', clobber=True) as fits:
    fits.write(subsampled_data)

print(f'Subsampled data written to {output_fits_file}')

