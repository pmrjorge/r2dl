import jax
from jax import numpy as jnp

x = jnp.arange(12)

print(x)
print(x.size)
print(x.shape)

X = x.reshape(3, 4)
print(X)

