import os
import sys
import tqdm
from pathlib import Path
if str(Path.cwd().parent) not in sys.path:
    sys.path.append(str(Path.cwd().parent))
sys.path.append(os.path.join(str(Path.cwd().parent),"qucats_paper", "codes", "settings"))
print(sys.path)
import glob
import pandas as pd
from astropy.table import Table

def concat_fields_stripe(list_files, save_path, columns_to_save = None, condition = None,  replace_file=True):
    
    if os.path.exists(save_path):
        if replace_file==True:
            try:
                os.system(f"""rm {save_path}""")
                print("replacing file")
            except:
                pass
        else: 
            print("File already exists. Please, change the name of the file or set replace_file=True")
        
    for i, file in tqdm.tqdm(enumerate(list_files)):
        dat = Table.read(file, format="fits")
        names = [name for name in dat.colnames if len(dat[name].shape) <= 1]
        try:
            dat["ID_1"] = dat["ID_1"].str.encode('utf-8')
        except:
            pass
        dat = dat[names].to_pandas()
            
        if condition:
            n = len(dat)
            dat=dat[condition]
            print(n-len(dat), " objects removed.")
            
        if i == 0:
            header = True
        else:
            header = False

        if columns_to_save:
            dat[columns_to_save].to_csv(save_path, mode='a', index=False, header=header)
        else:
            dat.to_csv(save_path, mode='a', index=False, header=header)

if __name__ == '__main__':
    print("Initializing...")
    list_files = [f for f in glob.glob("/home/mariko/Research/Projects/qucats_paper/data/dr4/*fits")]
    print("Concatenating %s fields" % len(list_files))
    concat_fields_stripe(list_files, "/home/mariko/Research/Projects/qucats_paper/data/dr4/SPLUS_STRIPE82_DR4.csv")
    