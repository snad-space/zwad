import os
import pandas as pd
import requests
import numpy as np
import joblib 

def transform_ztf(oids_file: str, output_fname=None):
    """Load individual ZTF light curves and save sample to pickle file.
    
    Parameters
    ----------
    oids_file: str
        Path to ZTF light curves. Each objects is consider 
        given in 1 json file.
    output_fname: str (optional)
        Output file name where sample will be saved.
        If None, return the entire sample as a list of lists.
        Default is None.
    """
    
    # store light curves    
    light_curves = []
    failed = []

    # read filenames: 1 file for each light curve
    with open(oids_file) as f:
        flist = f.read().split()
    
    for objid in flist:
        
        print(list(flist).index(objid))
        
        try:
            data_raw = requests.get('http://db.ztf.snad.space/api/v2/oid/full/json?oid=' + str(objid)).json()
    
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
            
            # drop duplicates (if any)
            data_lc.drop_duplicates(subset=['mjd'], keep='first', inplace=True)
        
            # subtract previous obs time
            time_diff = [0]
            for i in range(1, data_lc['mjd'].shape[0]):
                time_diff.append(data_lc['mjd'].values[i] - data_lc['mjd'].values[i - 1])
    
            data_lc['time_diff'] = time_diff
        
            light_curves.append(data_lc[['time_diff', 'mag', 'magerr']].values)
                
        except:
            print('failed: ', objid)
            failed.append(objid)
            
    np.save(output_fname, np.array(light_curves), allow_pickle=True)
    np.save('failed', np.array(failed))