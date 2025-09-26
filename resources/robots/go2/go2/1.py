import numpy as np, imageio
h = np.random.rand(10,10).astype(np.float32)          # 0~1
imageio.imwrite('rough_10x10.png', (h*255).astype(np.uint8))