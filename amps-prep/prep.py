import numpy as np
import os
import pandas as pd

L_MAX = 5
N_MAX = 55

all_modes = []

for l in range(2, L_MAX + 1):
    for m in range(-l, l + 1):
        if abs(m) < l - 2:
            continue
        for n in range(-N_MAX, N_MAX + 1):
            all_modes.append((l, m, n))

print("Num modes:", len(all_modes))

outs = np.zeros((len(all_modes), 33, 33, 33), dtype=np.complex128)

for k, (ell_select, emm_select, enn_select) in enumerate(all_modes):
    print(ell_select, emm_select, enn_select)
    if enn_select >= 0:
        mode_path = os.path.join('./regionA/', f'l{ell_select}', f'm{emm_select}', f'amplitudes_{ell_select}_{emm_select}_{enn_select}.feather')
        modetest = pd.read_feather(mode_path)
        mode = modetest.to_numpy().reshape(33, 33, 33, 2).copy()
        
    else:
        emm_select = -emm_select
        enn_select = -enn_select
        mode_path = os.path.join('./regionA/', f'l{ell_select}', f'm{emm_select}', f'amplitudes_{ell_select}_{emm_select}_{enn_select}.feather')
        modetest = pd.read_feather(mode_path)
        mode = modetest.to_numpy().reshape(33, 33, 33, 2).copy()
        mode[:,:,:,1] *= -1
        mode *= (-1) ** ell_select
    outs[k] = (mode[:,:,:,0] + 1j * mode[:,:,:,1])

np.save('./amp_interp_grid.npy', outs)
