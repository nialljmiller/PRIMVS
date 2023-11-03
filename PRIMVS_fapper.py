import pandas as pd
from astropy.table import Table, vstack, hstack
import glob
import random
import csv
import os
import tqdm as tqdm
import numpy as np
import itertools
# List of column names
col_names = ['name','mag_n','mag_avg','magerr_avg','Cody_M','stet_k','eta','eta_e','med_BRP',
		'range_cum_sum','max_slope','MAD','mean_var','percent_amp','true_amplitude','roms','p_to_p_var',
		'lag_auto','AD','std_nxs','weight_mean','weight_std','weight_skew','weight_kurt','mean','std','skew',
		'kurt','time_range','true_period','true_class','best_fap','best_method','trans_flag','ls_p','ls_y_y_0',
		'ls_peak_width_0','ls_period1','ls_y_y_1','ls_peak_width_1','ls_period2','ls_y_y_2','ls_peak_width_2',
		'ls_q001','ls_q01','ls_q1','ls_q25','ls_q50','ls_q75','ls_q99','ls_q999','ls_q9999','ls_fap','ls_bal_fap',
		'Cody_Q_ls','pdm_p','pdm_y_y_0','pdm_peak_width_0','pdm_period1','pdm_y_y_1','pdm_peak_width_1','pdm_period2',
		'pdm_y_y_2','pdm_peak_width_2','pdm_q001','pdm_q01','pdm_q1','pdm_q25','pdm_q50','pdm_q75','pdm_q99',
		'pdm_q999','pdm_q9999','pdm_fap','Cody_Q_pdm','ce_p','ce_y_y_0','ce_peak_width_0','ce_period1','ce_y_y_1',
		'ce_peak_width_1','ce_period2','ce_y_y_2','ce_peak_width_2','ce_q001','ce_q01','ce_q1','ce_q25','ce_q50',
		'ce_q75','ce_q99','ce_q999','ce_q9999','ce_fap','Cody_Q_ce','gp_lnlike','gp_b','gp_c','gp_p','gp_fap','Cody_Q_gp']

meta_col_names = ['sourceid','ra','ra_error','dec','dec_error','l','b','parallax','parallax_error','pmra','pmra_error','pmdec','pmdec_error',
		'chisq','uwe','ks_n_detections','ks_n_observations','ks_n_ambiguous','ks_n_chilt5','ks_med_mag','ks_mean_mag','ks_ivw_mean_mag',
		'ks_chilt5_ivw_mean_mag','ks_std_mag','ks_mad_mag','ks_ivw_err_mag','ks_chilt5_ivw_err_mag','z_n_observations','z_med_mag',
		'z_mean_mag','z_ivw_mean_mag','z_chilt5_ivw_mean_mag','z_std_mag','z_mad_mag','z_ivw_err_mag','z_chilt5_ivw_err_mag','y_n_detections',
		'y_n_observations','y_med_mag','y_mean_mag','y_ivw_mean_mag','y_chilt5_ivw_mean_mag','y_std_mag','y_mad_mag','y_ivw_err_mag',
		'y_chilt5_ivw_err_mag','j_n_detections','j_n_observations','j_med_mag','j_mean_mag','j_ivw_mean_mag','j_chilt5_ivw_mean_mag',
		'j_std_mag','j_mad_mag','j_ivw_err_mag','j_chilt5_ivw_err_mag','h_n_detections','h_n_observations','h_med_mag','h_mean_mag',
		'h_ivw_mean_mag','h_chilt5_ivw_mean_mag','h_std_mag','h_mad_mag','h_ivw_err_mag','h_chilt5_ivw_err_mag']		



fits_col_types = ['int', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float',		 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float',		  'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float',		   'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float',		    'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float',		    'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float',		     'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'str', 'float', 'str', 		     'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 		     'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 		     'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 		     'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 		     'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float']

csv_meta_files = glob.glob("/beegfs/car/njm/OUTPUT/results/complete/*")
csv_files = glob.glob("/beegfs/car/njm/OUTPUT/vars/complete/rescan/*.csv")
csv_files = glob.glob("/beegfs/car/njm/OUTPUT/vars/complete_lower_var/*.csv") + glob.glob("/beegfs/car/njm/OUTPUT/vars/complete/*.csv")

    
# Define the output FITS table
output_table = Table(names=meta_col_names+col_names[1:], dtype=fits_col_types)

# iterate over a list of file names
for csv_file in tqdm.tqdm(csv_files):#_sample):
   # Open the first CSV file
    with open(csv_file, 'r') as f1:
        reader1 = csv.reader(f1)
        meta_fp = "/beegfs/car/njm/OUTPUT/results/complete/"+csv_file.split('/')[-1].split('.')[0]
        meta_fp = next((os.path.join(loc, csv_file.split('/')[-1].split('.')[0]) for loc in ["/beegfs/car/njm/OUTPUT/results/"] if os.path.exists(os.path.join(loc, csv_file.split('/')[-1].split('.')[0]))), meta_fp)
        # Use dropwhile to skip over rows until we find a qualifying row
        qualifying_rows1 = itertools.dropwhile(lambda row: float(row[31]) >= 0.2, reader1)
        # Use takewhile to read in all qualifying rows
        qualifying_rows1 = itertools.takewhile(lambda row: float(row[31]) < 0.2, qualifying_rows1)
        # Convert qualifying rows to a list so we can loop through them multiple times if necessary
        qualifying_rows1 = list(qualifying_rows1)
        with open(meta_fp, 'r') as f2:
            reader2 = csv.reader(f2)
            meta_list = [row for row in reader2]
            meta_dict = {row[0]: row for row in meta_list}
            # Loop over the rows in the first CSV file
            for i, row1 in enumerate(qualifying_rows1):
                if float(row1[31]) < 0.2:
                    matching_row = meta_dict.get(row1[0])
                    if matching_row is not None:
                        # Combine the two rows
                        combined_row = matching_row + row1[1:]
                        # Add the combined row to the output table
                        output_table.add_row(combined_row)
                        break


# Write the table to a FITS file
try:
    os.system('rm /beegfs/car/njm/OUTPUT/PRIMVS_old.fits')  
    os.rename('/beegfs/car/njm/OUTPUT/PRIMVS.fits', '/beegfs/car/njm/OUTPUT/PRIMVS_old.fits')
except:
    pass
output_table.write('/beegfs/car/njm/OUTPUT/PRIMVS.fits')

    
    
