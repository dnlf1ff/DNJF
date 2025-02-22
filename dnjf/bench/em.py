import pickle
import pandas as pd
import os
from util import load_dict, tot_sys_mlps, set_env, save_csv, save_dict, load_csv

set_env('bench')
systems, mlps = tot_sys_mlps()
labels = ['Force','Energy','Stress','Volume']
units = ['meV/A','meV','meV/A^3','A^3']
metrics = ['MAV','MAE','RMSE']

errors = load_dict('errors')


def TLQKF():
    #force:eV/A energy:eV/atom stress:kBar volume:A^3
    #TM23: force: meV/A energy: meV stress: meV/A3
    # force conversion
    # energy conversion
    # stress conversion
    nions = load_dict('nions')
    for system in systems:
        out = load_dict(system)
        nion = nions[system]
        for mlp in mlps:
            out[f'force-{mlp}'] = out[f'force-{mlp}']*1000
            out[f'pe-{mlp}'] = out[f'pe-{mlp}']*1000*nion
            out[f'stress-{mlp}'] = out[f'stress-{mlp}']*1.602
        save_dict(out, system)

def get_nions():
    from ase.io import read
    nion_dict = {}
    for system in systems:
        atoms = read(os.path.join(os.environ['BARK'],'TM23',f'{system}.xyz'))
        nion_dict[system] = len(atoms)
    save_dict(nion_dict, 'nions')

def get_errors(errors, metric, mlp):
    force, energy, stress, volume = [], [], [], []
    for system in systems:
        force.append(errors[system][mlp]['force'][metric.lower()])
        energy.append(errors[system][mlp]['pe'][metric.lower()])
        stress.append(errors[system][mlp]['stress'][metric.lower()])
        volume.append(errors[system][mlp]['vol'][metric.lower()])
    return (force, energy, stress, volume)


def get_errors_all():
    errors = load_dict('errors')
    for mlp in mlps:
        df=pd.DataFrame()
        df['Element'] = systems
        for metric in metrics:
            errors_out = get_errors(errors, metric, mlp)
            for i, label in enumerate(labels):
                _label = f'{label}_{metric}'
                df[_label] = errors_out[i]
            save_csv(df, f'errors/{mlp}')


def fiddle():
    errors = load_dict('errors')
    nions = load_dict('nions')
    for mlp in mlps:
        df = load_csv(f'errors/{mlp}')
        energy_mav = df['Energy_MAV']
        new_mav = []
        for i, mav in enumerate(energy_mav):
            new_ = mav/nions[systems[i]]
            new_mav.append(new_)
        df['Energy_MAV'] = new_mav
        save_csv(df, f'errors/{mlp}')

if __name__ == '__main__':
#    get_errors_all()
    fiddle()
