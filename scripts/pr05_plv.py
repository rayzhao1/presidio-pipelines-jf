"""pr05_stageone_ieeg_waveletpower.py
"""

import sys
import presidio_pipelines as prespipe
import os
import numpy as np
from glob import glob
from scipy.io import savemat

if __name__ == '__main__':

    assert len(sys.argv) == 5, f'Expected 4 arguments, got {sys.argv}'
    wt_fn = sys.argv[1]
    h5_fn = sys.argv[2]
    output_dir = sys.argv[3]
    night_idx = sys.argv[4]

    #plv, time_axis, cfreqs = prespipe.ieeg.stage_one_plv.Pipeline(h5_fn, wt_fn)
    out_fn = os.path.join(output_dir, f'{os.path.basename(wt_fn)[:-4]}_plv-n{night_idx}.npz')
    mat_fn = f'{out_fn[:-3]}mat'

    np.savez(out_fn, plv=plv, time_axis=time_axis, cfreqs=cfreqs)
    savemat(mat_fn, {'plv': plv, 'time_axis': time_axis, 'cfreqs': cfreqs}, appendmat=True)

    #assert out_fn in glob(os.path.join(output_dir, '*'))
    #assert mat_fn in glob(os.path.join(output_dir, '*'))
    print("Success")
