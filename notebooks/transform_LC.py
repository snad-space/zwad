import os
import pandas as pd
import json
import joblib
import numpy as np

def transform_ztf_pickle(data_dir: str, output_fname: str):
    """Load individual ZTF light curves and save sample to pickle file.
    
    Parameters
    ----------
    data_dir: str
        Path to ZTF light curves. Each objects is consider 
        given in 1 json file.
    output_fname: str
        Output file name where sample will be saved.
        If False, return the entire sample as a list of lists.
    """
    
    # store light curves    
    light_curves = []

    # read filenames: 1 file for each light curve
    flist = os.listdir(data_dir)

    failed = []
    
    for fname in flist:
        try:
            with open(data_dir + fname) as json_file:    
                    data_raw = json.load(json_file)               
  
                    # get object id
                    objid = fname[:-5]
        
                    # number of observations
                    nobs = len(data_raw[objid]['lc'])

                    # format useful data 
                    mjd = [data_raw[objid]['lc'][i]['mjd'] for i in range(nobs)]
                    mag = [data_raw[objid]['lc'][i]['mag'] for i in range(nobs)]
                    magerr = [data_raw[objid]['lc'][i]['magerr'] for i in range(nobs)]
                    clrcoeff = [data_raw[objid]['lc'][i]['clrcoeff'] for i in range(nobs)]
                    catflags = [data_raw[objid]['lc'][i]['catflags'] for i in range(nobs)]

            # build a data frame
            data_lc = pd.DataFrame()
            data_lc['mjd'] = mjd
            data_lc['mag'] = mag
            data_lc['magerr'] = magerr
            data_lc['clrcoeff'] = clrcoeff
            data_lc['catflags'] = catflags
            data_lc['mag_mean'] = np.mean(data_lc['mag'].values)
            data_lc['mag_mean_err'] = np.mean(data_lc['mag_mean_err'].values)
            data_lc['mag_cent'] = data_lc['mag'].values - data_lc['mag_mean'].values[0]
            
            # drop duplicates (if any)
            data_lc.drop_duplicates(subset=['mjd'], keep='first', inplace=True)
        
            # subtract previous obs time
            time_diff = [0]
            for i in range(1, data_lc['mjd'].shape[0]):
                time_diff.append(data_lc['mjd'].values[i] - data_lc['mjd'].values[i - 1])
    
            data_lc['time_diff'] = time_diff
        
            light_curves.append(data_lc[['time_diff', 'mag', 'magerr']].values)
    
            if output_fname:
                joblib.dump(light_curves, output_fname, compress=3)
        
        except:
                failed.append(fname)
                pass
            
    return np.array(light_curves), failed