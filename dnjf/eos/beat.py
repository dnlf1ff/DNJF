import pandas as pd
import pickle
from plot import scatter_b0, scatter_del_E

import sys, os 
from loguru import logger
from util import save_dict, set_env, load_dict, tot_sys_mlps
# from log import *
from plot2 import scatter

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
    df1=f'{system}_fccbcc_fit'
    df2=f'{system}_fcchcp_fit'
    df3=f'{system}_hcpbcc_fit'
    pkl1 = load_dict(df1)
    pkl2 = load_dict(df2)
    pkl3 = load_dict(df3)
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

def replot():
    systems, mlps = tot_sys_mlps('tot')
#    tot_b0 = load_dict('tot_b0.pkl')
    tot_del = load_dict('tot_del.pkl')
    
    for mlp in mlps:
        scatter_del_E(tot_del, mlp)

def main():
    systems, mlps = tot_sys_mlps('tot')
    tot_b0 = get_b0_tot(systems)
    tot_del = get_del_tot(systems)
    tot_del = convert_eV(tot_del)
    
    save_dict(tot_b0, 'tot_b0.pkl')
    save_dict(tot_del, 'tot_del.pkl')
    
    for mlp in mlps:
        scatter_b0(tot_b0, mlp)
        scatter_del_E(tot_del, mlp) 

def parities():
    systems, mlps = tot_sys_mlps('tot')
    for system in systems:
        out = load_dict(system)
        for mlp in mlps:


#TODO: logger    
if __name__ == '__main__':
    set_env('eos',sys.argv[1])
    replot()
