import os
from ase.io import read
import pandas as pd
import sys
df = pd.DataFrame()
os.chdir('mptrj-24')

task_ids = []
e_per_atom_relaxeds = []
e_per_atoms = []
correcteds = []
n_atoms = []
atoms = read(f'{sys.argv[1]}.extxyz', index=':')
for i, atom in enumerate(atoms):
    task_ids.append(atom.info['task_id'])
    correcteds.append(atom.info['corrected_total_energy'])
    e_per_atoms.append(atom.info['energy_per_atom'])
    e_per_atom_relaxeds.append(atom.info['e_per_atom_relaxed'])
    n_atoms.append(len(atom))
df['mp_id'] = mp_ids
df['task_id'] = task_ids
df['e_corr'] = correcteds
df['e'] = e_per_atoms
df['e_rlx'] = e_per_atom_relaxeds
df['n_sites'] = n_atoms

df.to_csv(f'/home/dnjf/Research../eos/output/comp/{system}_24.csv')
