# first load some standard libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from causalgraphicalmodels import CausalGraphicalModel

system_with_hidden_confounder = CausalGraphicalModel(
    nodes=["x", "y", "z"],
    edges=[("x", "z"), ("z", "y")],
    latent_edges=[("x", "y")]
)