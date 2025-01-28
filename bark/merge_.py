import os
import pandas as pd
import shutil
import subprocess
import sys

def merge(system):
    dfs = []
    filenames=[]
    for filename in os.listdir(os.getcwd()):
        if 'csv' in filename and system in filename:
            dfs.append(pd.read_csv(filename,index_col=0))
            filenames.append(filename)
    
    df0=dfs[0]
    dfs.pop(0)
    
    for df in dfs:
        df0=pd.merge(df0, df)
    
    print(df0)
    for filename in filenames:
        subprocess.run(['mv',f'{filename}',f'baks/{filename}'])


    df0.to_csv(f'{system}_mrg.csv')

if __name__ == '__main__':
    merge(sys.argv[1])
        


