# pixray

![Alt text](https://user-images.githubusercontent.com/945979/132954388-1986e4c6-6996-48fd-9e91-91ec97963781.png "deep ocean monsters #pixelart")

Pixray is an image generation system. It combines previous ideas including:

 * [Perception Engines](https://github.com/dribnet/perceptionengines) which uses image augmentation and iteratively optimises images against an ensemble of classifiers
 * [CLIP guided GAN imagery](https://alexasteinbruck.medium.com/vqgan-clip-how-does-it-work-210a5dca5e52) from [Ryan Murdoch](https://twitter.com/advadnoun) and [Katherine Crowson](https://github.com/crowsonkb) as well as modifictions such as [CLIPDraw](https://twitter.com/kvfrans/status/1409933704856674304) from Kevin Frans
 * Useful ways of navigating latent space from [Sampling Generative Networks](https://github.com/dribnet/plat)
 * (more to come)

pixray it itself a python library and command line utility, but is also friendly to running on line in Google Colab notebooks.

There is currently [some documentation on options](https://dazhizhong.gitbook.io/pixray-docs/docs). Also checkout [THE DEMO NOTEBOOKS](https://github.com/pixray/pixray_notebooks) or join in the [discussion on discord](https://discord.gg/x2g9TWrNKe).

## Usage

Be sure to `git clone --recursive` to also get submodules.

You can install `pip install -r requirements.txt` and then `pip install basicsr` manually in a fresh python 3.8 environment (eg: using conda). After that you can use the included `pixray.py` command line utility:

    python pixray.py --drawer=pixel --prompt=sunrise --outdir sunrise01

pixray can also be run from within your own python code, like this

```python
import pixray
pixray.run("an extremely hairy panda bear", "vdiff", custom_loss="aesthetic", outdir="outputs/hairout")
```

Examples of pixray colab notebooks can be found [in this separate repo](https://github.com/pixray/pixray_notebooks).

running in a Docker using [Cog](https://github.com/replicate/cog) is also possible. First, [install Docker and Cog](https://github.com/replicate/cog#install), then you can use `cog run` to run Pixray inside Docker. For example: 

    cog run python pixray.py --drawer=pixel --prompt=sunrise --outdir sunrise01


## Colab Installation
1- Install 
```python

#@title Setup

#@markdown Please execute this cell by pressing the _Play_ button 
#@markdown on the left. You should only need to run this part once.

#@markdown **Note**: This installs the software on the Colab 
#@markdown notebook in the cloud and not on your computer.

#@markdown When complete you will need to do Runtime -> Restart Runtime from the menu

# Add a gpu check
nvidia_output = !nvidia-smi --query-gpu=memory.total --format=noheader,nounits,csv
gpu_memory = int(nvidia_output[0])
if gpu_memory < 14000:
  print(f"--> GPU check: ONLY {gpu_memory} MiB available: WARNING, some things might not work <--")
else:
  print(f"GPU check: {gpu_memory} MiB available: this should be fine")

print("Installing...")
from IPython.utils import io
with io.capture_output() as captured:
  !pip install braceexpand
  !pip install ftfy regex tqdm omegaconf pytorch-lightning
  !pip install kornia==0.6.2
  !pip install imageio-ffmpeg
  !pip install einops
  !pip install torch-optimizer
  !pip install easydict
  !pip install git+https://github.com/pvigier/perlin-numpy

  # ClipDraw deps
  !pip install svgwrite
  !pip install svgpathtools
  !pip install cssutils
  !pip install numba
  !pip install torch-tools
  !pip install visdom
  !pip install colorthief
  !pip install ftfy regex tqdm
  !pip install git+https://github.com/openai/CLIP.git
  !pip install timm==0.6.12
  !pip install git+https://github.com/bfirsh/taming-transformers.git@7a6e64ee
  !pip install resmem
  !pip install git+https://github.com/pixray/aphantasia@7e6b3bb
  !pip insatll lpips
  !pip insatll sentence_transformers
  !pip insatll opencv-python
  !pip install pytorch-wavelet
  !pip insatll PyWavelets
  !pip insatll git+https://github.com/fbcotter/pytorch_wavelets
  !pip install basicsr
  !rm -Rf pixray
  !git clone --recursive https://github.com/pixray/pixray
  !pip install -r /content/pixray/requirements.txt
  !pip uninstall -y tensorflow 
  !git clone https://github.com/pixray/diffvg
  %cd diffvg
  !git submodule update --init --recursive
  
  
  !python setup.py install
  %cd ..
  !pip freeze | grep torch
  import sys
  sys.path.append("diffvg/build/lib.linux-x86_64-cpython-310")
import os
if not os.path.isfile("first_init_complete"):
  # put stuff in here that should only happen once
  !mkdir -p models
  os.mknod("first_init_complete")
  print("Please choose Runtime -> Restart Runtime from the menu to continue!")
else:
  print("Setup Complete! Good luck with your drawing")


```
2- Add system path 
```python
import sys
sys.path.append("pixray")

```

