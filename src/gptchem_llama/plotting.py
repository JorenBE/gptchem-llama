import sys
import matplotlib.pyplot as plt
from scipy.constants import golden

sys.path.append("/home/joren/chemGPT/gptchem-llama/src/gptchem_llama/plotutils")

from plotutils import *

plt.style.use("/home/joren/chemGPT/gptchem-llama/src/gptchem_llama/plotutils/kevin.mplstyle")

print('Matplotlib parameters: activated')

ONE_COL_WIDTH_INCH = 5
TWO_COL_WIDTH_INCH = 7.2

ONE_COL_GOLDEN_RATIO_HEIGHT_INCH = ONE_COL_WIDTH_INCH / golden
TWO_COL_GOLDEN_RATIO_HEIGHT_INCH = TWO_COL_WIDTH_INCH / golden