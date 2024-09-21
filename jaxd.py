#@save
from dataclasses import field
from functools import partial
from types import FunctionType
from typing import Any
import flax
import jax
import numpy as np
import optax
import tensorflow as tf
import tensorflow_datasets as tfds
from flax import linen as nn
from flax.training import train_state
from jax import grad
from jax import numpy as jnp
from jax import vmap
