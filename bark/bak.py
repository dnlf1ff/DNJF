import pandas as pd
from mob_utils import *

def run_first():
    df = pd.read_csv('../df/metals_clean.csv',index_col=0).reset_index().drop(['index'], axis=1)
    df['task_id'] = df['task_id'].apply(lambda x: eval(x))
    df = df.explode('task_id')
    set_env('eos')

    df.to_csv(os.path.join(os.environ['OUT'], 'metals.csv'))

def get_mpr_id(system):
    df = pd.read_csv(os.path.join(os.environ['PRESETS'],'metals.csv'), index_col=0)
    mpr_df = pd.read_csv(os.path.join(os.environ['PRESETS'],'mptrj-23.csv'), index_col=0)
    df = get_system_df(df, system)
    sane_ids = []
    for i, row in df.iterrows():
        mp_id = row['material_id']
        task_id = row['task_id']
        mpr_df_ = get_system_df(mpr_df, system=None, mp_id=mp_id)
        for mpr_id in mpr_df_['task_id']:
            if mpr_id == task_id:
                sane_ids.append(mpr_id)
    mask = df['task_id'].apply(lambda x: x in sane_ids)
    df = df[mask]
    mpr_mask = mpr_df['task_id'].apply(lambda x: x in sane_ids)
    mpr_df = mpr_df[mpr_mask]
    mpr_df = mpr_df.drop_duplicates(subset='task_id', keep='last')
    # pd.concat([df, mpr_df], axis=1)
    df_ = pd.merge(df, mpr_df, on='task_id', how='inner')
    df_ = df_[['formula_pretty','mp_id','task_id','space_group','bravais_lattice','vol','e','e_corr','e_rlx','n_sites']]
    df_.to_csv(os.path.join(os.environ['OUT'],f'{system}.csv'))
    return df_

if __name__ == '__main__':
    set_env('eos')
