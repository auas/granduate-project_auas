import numpy as np
import theano.tensor as tt
import pymc3 as pm

import seaborn as sns
import matplotlib.pyplot as plt


with pm.Model() as model:
    mu = pm.Normal('mu', mu=0, sd=1)
    obs = pm.Normal('obs', mu=mu, sd=1, observed=np.random.randn(100))
print(model.basic_RVs)
print(model.free_RVs)

print(mu.logp({"mu":0.}))

print(mu.logp({"mu":0.1}))

print(mu.logp({"mu":0.2}))

print(mu.logp({"mu":0.3}))