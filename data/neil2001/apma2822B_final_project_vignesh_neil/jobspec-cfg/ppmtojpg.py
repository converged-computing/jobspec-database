import re
import numpy as np
from PIL import Image
from pathlib import Path

# Open image file, slurp the lot
contents = Path('out.ppm').read_text()
 
# Make a list of anything that looks like numbers using a regex...
# ... taking first as height, second as width and remainder as pixels
Pidentifier, h, w, *pixels = re.findall(r'[0-9]+', contents)
h = int(h)
w = int(w)
# Now make pixels into Numpy array of uint8 and reshape to correct height, width and depth
na = np.array(pixels[w*h*-3:], dtype=np.uint8).reshape((w,h,3))
 
# Now make the Numpy array into a PIL Image and save
Image.fromarray(na).save("out.jpg")