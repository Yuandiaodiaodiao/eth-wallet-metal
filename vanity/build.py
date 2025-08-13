from .gpu_vanity import MetalVanity
import os
here = os.path.dirname(os.path.abspath(__file__))
eng = MetalVanity(here)
eng.build_g16_table(here)