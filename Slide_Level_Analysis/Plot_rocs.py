""" Generate ROC curve's for each % of training data """

import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

## Path to ROC dir
ROC_DIR = '/home/srinidhi/Research/Code/camelyon16/ROC/1.0/'

roc_paths = []
for file_ext in ['npy']:
    roc_paths = roc_paths + glob.glob('{}/*.{}'.format(ROC_DIR, file_ext))
    roc_paths = sorted(roc_paths)
    roc_paths = [roc_paths[2], roc_paths[6], roc_paths[0], roc_paths[4], roc_paths[3], roc_paths[7], roc_paths[1], roc_paths[5]]


plt.figure()
for file_ID in range(len(roc_paths)):

    roc_path = roc_paths[file_ID]

    # Get Roc id
    roc_id = str(os.path.split(roc_path)[-1])
    pattern = "ROC\_(.*?).npy"
    roc_id = re.search(pattern, roc_id).group(1)

    # Plot ROC curve
    data = np.load(roc_path)
    fpr = data[0].reshape(-1,)
    tpr = data[1].reshape(-1,)
    roc_auc = auc(fpr, tpr)

    # plt.figure()
    if roc_id[0] == 'S':
        roc_id = roc_id + str(' (Ours)')
        plt.plot(fpr, tpr, label=roc_id)
    else:
        plt.plot(fpr, tpr, label=roc_id)

plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
lg = plt.legend(loc='lower right', borderaxespad=1.)
lg.get_frame().set_edgecolor('k')
plt.grid(True, linestyle='-')
plt.savefig(os.path.join(ROC_DIR, 'ROC.svg'), dpi=500, bbox_inches='tight')
plt.show()
