import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
import matplotlib
from matplotlib.gridspec import GridSpec


dpi = 666  # 200-300 as per guidelines
maxpix = 3000  # max pixels of plot
width = maxpix / dpi  # max allowed with
matplotlib.rcParams.update({'axes.labelsize': 'x-large', 'axes.titlesize': 'xx-large',  # the size of labels and title
                 'xtick.labelsize': 'xx-large', 'ytick.labelsize': 'x-large',  # the size of the axes ticks
                 'legend.fontsize': 'large', 'legend.frameon': False,  # legend font size, no frame
                 'legend.facecolor': 'none', 'legend.handletextpad': 0.25,
                 # legend no background colour, separation from label to point
                 'font.serif': ['Computer Modern', 'Helvetica', 'Arial',  # default fonts to try and use
                                'Tahoma', 'Lucida Grande', 'DejaVu Sans'],
                 'font.family': 'serif',  # use serif fonts
                 'mathtext.fontset': 'cm', 'mathtext.default': 'regular',  # if in math mode, use these
                 'figure.figsize': [width, 0.7 * width], 'figure.dpi': dpi,
                 # the figure size in inches and dots per inch
                 'lines.linewidth': .75,  # width of plotted lines
                 'xtick.top': True, 'ytick.right': True,  # ticks on right and top of plot
                 'xtick.minor.visible': True, 'ytick.minor.visible': True,  # show minor ticks
                 'text.usetex': True, 'xtick.labelsize':'x-large',
                 'ytick.labelsize':'x-large'})  # process text with LaTeX instead of matplotlib math mode






# Open the fits file with memmap to conserve memory space
filename = '/beegfs/car/njm/OUTPUT/PRIMVS.fits'
with fits.open(filename, memmap=True) as hdulist:
    data = Table.read(hdulist[1], format='fits')

    header = hdulist[1].header


# Create a boolean mask based on the 'best_fap' feature
apmask = data['best_fap'] > 0.8
pmask =  data['best_fap'] < 0.1

# Now you can select three features/columns from the data table to plot their histograms
Pcyclefeat = data['time_range'][pmask] / data['true_period'][pmask]  # Replace 'feature1_column_name' with the actual column name
Psnrfeat = data['true_amplitude'][pmask] / data['magerr_avg'][pmask]  # Replace 'feature2_column_name' with the actual column name
Pnfeat = data['mag_n'][pmask]  # Replace 'feature3_column_name' with the actual column name

APsnrfeat = data['true_amplitude'][apmask] / data['magerr_avg'][apmask]  # Replace 'feature2_column_name' with the actual column name
APnfeat = data['mag_n'][apmask]  # Replace 'feature3_column_name' with the actual column name




# Define the number of bins for the histograms
num_bins = 40
# Define linearly spaced bins
cycle_bins = np.logspace(np.log10(1),np.log10(10000), num_bins)#Pcyclefeat.max(), num_bins)
snr_bins = np.linspace(1, 110, num_bins)#Psnrfeat.max(), num_bins)
n_bins = np.linspace(40, 300, num_bins)#Pnfeat.max(), num_bins)

# Assuming you have the data in the variables Pnfeat, APnfeat, Psnrfeat, APsnrfeat, Pcyclefeat

def calculate_histogram_percentage(data, bins):
    hist_percentage = []
    total_data = len(data)
    for i in range(len(bins)-1):
        bin_start = bins[i]
        #print(bin_start, np.min(data), np.max(data))
        if i == len(bins):
            bin_end = np.max(data)
        else:
            bin_end = bins[i + 1]
        data_in_bin = [x for x in data if bin_start <= x < bin_end]
        bin_percentage = len(data_in_bin) / total_data
        print(len(data_in_bin) , total_data)
        hist_percentage.append(bin_percentage)
    return hist_percentage

# Calculate histograms and percentages
hist_Pn_percentage = calculate_histogram_percentage(Pnfeat, n_bins)
hist_APn_percentage = calculate_histogram_percentage(APnfeat, n_bins)
hist_Psnr_percentage = calculate_histogram_percentage(Psnrfeat, snr_bins)
hist_APsnr_percentage = calculate_histogram_percentage(APsnrfeat, snr_bins)
hist_Pcycle_percentage = calculate_histogram_percentage(Pcyclefeat, cycle_bins)




# Create the figure and subplots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(7, 4), sharey=True)

# Plot the leftmost subplot
ax1.bar(n_bins[:-1], hist_Pn_percentage, width=np.diff(n_bins), color='b', alpha=0.4, label='Periodic')
ax1.bar(n_bins[:-1], hist_APn_percentage, width=np.diff(n_bins), color='r', alpha=0.4, label='Aperiodic')
ax1.set_ylabel(r'Log(Percentage)')
ax1.set_yticklabels([])
ax1.set_xticks([50,150,250])
ax1.legend()
ax1.set_xlabel(r'N')
ax1.set_yscale('log')

# Plot the middle subplot
ax2.bar(snr_bins[:-1], hist_Psnr_percentage, width=np.diff(snr_bins), color='b', alpha=0.4, label='Psnrfeat')
ax2.bar(snr_bins[:-1], hist_APsnr_percentage, width=np.diff(snr_bins), color='r', alpha=0.4, label='APsnrfeat')
ax2.set_yticklabels([])
ax2.set_xticks([2,50,100])
ax2.set_xlabel(r'$A/\bar{\sigma}$')
ax2.set_yscale('log')

# Plot the rightmost subplot
ax3.bar(cycle_bins[:-1], hist_Pcycle_percentage, width=np.diff(cycle_bins), color='b', alpha=0.7, label='Pcyclefeat')
ax3.set_xlabel(r'Cycles')
ax3.set_yticklabels([])

ax3.set_xticks([10,100,1000])
ax3.set_xscale('log')
ax3.set_yscale('log')

# Remove the vertical space between subplots
plt.subplots_adjust(wspace=0)

# Ensure that the bars in each histogram add up to 1
plt.yscale('log')

# Save the plot
plt.savefig('/home/njm/training_data_hist.pdf', bbox_inches='tight')

# Show the plot (optional)
plt.show()

