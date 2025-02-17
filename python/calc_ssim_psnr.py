from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.io import imsave
import os
#import cv2

import numpy as np

import struct

path="."

# Open binary files for reading
with open(path + "/stats_in.bin", "rb") as f_in, open(path + "/stats_out.bin", "rb") as f_out:
    # Read dimensions (assuming g_dims is stored as a fixed-size struct)
    dims_size = struct.calcsize("3i")  # Assuming g_dims contains three integers (x, y, z)
    g_dims = struct.unpack("3i", f_in.read(dims_size))
    g_dims_out = struct.unpack("3i", f_out.read(dims_size))
    
    assert g_dims == g_dims_out, "Dimension mismatch between input and output files"
    
    x_dim, y_dim, z_dim = g_dims
    
    # Read values
    values_in = []
    values_out = []
    
    for _ in range(z_dim):
        for _ in range(y_dim):
            for _ in range(x_dim):
                value0 = struct.unpack("f", f_in.read(4))[0]
                value1 = struct.unpack("f", f_out.read(4))[0]
                
                values_in.append(value0)
                values_out.append(value1)
    
# Convert list to numpy array
values_in_np = np.array(values_in, dtype=np.float32)
values_out_np = np.array(values_out, dtype=np.float32)

# Reshape to 3D matrix (z_dim, y_dim, x_dim)
values_in_np = values_in_np.reshape((z_dim, y_dim, x_dim))
values_out_np = values_out_np.reshape((z_dim, y_dim, x_dim))

ssim_value, _ = ssim(values_in_np, values_out_np, full=True, data_range=1.0)
print("SSIM:", ssim_value)

psnr_value = psnr(values_in_np, values_out_np, data_range=1.0)
print("PSNR:", psnr_value, "dB")

# Save each slice as a PNG image using skimage
# for z in range(z_dim):
#     print(f"slice_{z:03d}.png")
#     imsave(os.path.join(path, f"slice_{z:03d}.png"), abs(values_out_np[z] - values_in_np[z]))


diff_im = values_out_np[(values_out_np < 0) | (values_out_np > 1)] #abs(values_out_np - values_in_np)
# Normalize values to [0,255] and convert to uint8 for PNG compatibility
diff_im = ((diff_im - np.min(diff_im)) / (np.max(diff_im) - np.min(diff_im)) * 255).astype(np.uint8)

# Save each slice as a PNG image using skimage
for z in range(z_dim):
    print(f"slice_{z:03d}.png", diff_im[z].min(), diff_im[z].max())
    imsave(os.path.join(path, f"slice_{z:03d}.png"), diff_im[z])