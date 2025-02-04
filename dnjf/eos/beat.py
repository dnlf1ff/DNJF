import pandas as pd
import pickle
from plot import scatter_b0, scatter_del_E

import sys

from util import *
# from log import *

def open_jar(df):
    with open(df, 'rb') as f:
        pkl = pickle.load(f)
    return pkl

def clean_df(df, b0 = False):
    rmv = []
    for col in df.columns:
        if 'fit' in col or 'rel' in col or 'vol' in col or 'pe' in col:
            rmv.append(col)
    rmv.append('mp_id')
    if b0:
        for col in df.columns:
            if 'del_E' in col:
                rmv.append(col)
    else:
        for col in df.columns:
            if 'b0' in col:
                rmv.append(col)
    df2=df.drop(columns=rmv,inplace=False)
    df2.reset_index(drop=True,inplace=True)
    return df2

def get_pickles(system):
    df1=os.path.join(os.environ['JAR'], f'{system}_fccbcc_fit.pkl')
    df2=os.path.join(os.environ['JAR'], f'{system}_fcchcp_fit.pkl')
    df3=os.path.join(os.environ['JAR'], f'{system}_hcpbcc_fit.pkl')
    pkl1 = open_jar(df1)
    pkl2 = open_jar(df2)
    pkl3 = open_jar(df3)
    return pkl1, pkl2, pkl3

def get_b0_df(system):
    fb, fh, hb = get_pickles(system)
    df = pd.concat([fb, fh, hb])
    df = clean_df(df, b0=True)
    df = df.drop_duplicates('bravais', keep='first')
    return df

def get_b0_tot(systems):
    dfs = []
    for system in systems:
        df = get_b0_df(system)
        dfs.append(df)
    tot_jar = pd.concat(dfs)
    tot_jar.rename(columns={'del_b0-dft':'b0-dft'},inplace=True)
    return tot_jar

def get_del_df(system):
    fb, fh, hb = get_pickles(system)
    fb['bravais'] = 'FCC-BCC'
    fh['bravais'] = 'FCC-HCP'
    hb['bravais'] = 'HCP-BCC'
    fb.drop(1, inplace=True)
    fh.drop(1, inplace=True)
    hb.drop(1, inplace=True)
    df = pd.concat([fb, fh, hb])
    df = clean_df(df)
    return df

def convert_eV(df):
    eVs = []
    for col in df.columns:
        if 'del_E' in col:
            eVs.append(col)
    for eV in eVs:
        df[eV] = df[eV].apply(lambda x: x*1000)
    return df

def get_del_tot(systems):
    dfs = []
    for system in systems:
        df = get_del_df(system)
        dfs.append(df)
    tot_jar = pd.concat(dfs)
    return tot_jar

def main(pbe):
    set_env(task='eos',pbe=pbe) 
    
    systems = ['Au','Ag','Al','Ca','Co','Cu','Fe','Hf','Ir','K','Li','Mg','Mo','Na','Nb','Ni','Os','Pd','Pt','Re','Rb','Rh','Sr','Ti','Ta','V','W','Zn','Zr','Cs'] #Y, Cd
    mlps = ['chgTot','chgTot_l3i3','chgTot_l3i5','chgTot_l4i3','m3g_n','m3g_r6','m3g_r55','omat_epoch1','omat_epoch2','omat_epoch3','omat_epoch4','omat_ft_r5','r5pp','omat_i5pp_epoch1','omat_i5pp_epoch2','omat_i5pp_epoch3','omat_i5pp_epoch4','omat_i5_epoch1','omat_i5_epoch2','omat_i5_epoch3','omat_i5_epoch4','omat_i3pp','mace','matsim']
    tot_b0 = get_b0_tot(systems)
    tot_del = get_del_tot(systems)
    tot_del = convert_eV(tot_del)
    
    save_dict(tot_b0, os.path.join(os.environ['JAR'], 'tot_b0.pkl'))
    save_dict(tot_del, os.path.join(os.environ['JAR'], 'tot_del.pkl'))
    
    for mlp in mlps:
        scatter_b0(tot_b0, mlp)
        scatter_del_E(tot_del, mlp) 
        
if __name__ == '__main__':
    main(pbe=sys.argv[1])
