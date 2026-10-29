# -*- coding: utf-8 -*-
"""
Buka ulang SEMUA plot (1-4) dari hasil mapping yang sudah tersimpan di folder output/.
Tidak butuh CoppeliaSim. Taruh file ini satu folder dengan mapping_drone.py dan folder output/.
Jalankan: python replot_all.py
"""
import os
import numpy as np
import mapping_drone as m

raw = np.load(os.path.join(m.OUT_DIR, "cloud_raw.npy"))
vox = np.load(os.path.join(m.OUT_DIR, "cloud_voxel.npy"))
pth = np.load(os.path.join(m.OUT_DIR, "path.npy"))

print("Titik mentah: %d | voxel: %d | titik jalur: %d" % (len(raw), len(vox), len(pth)))
m.make_plots(raw, vox, pth, show=True)