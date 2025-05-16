import mithril as ml
from mithril.models import *
from mithril.common import *

m = Convolution2D(kernel_size=None, padding="same", use_bias=False)
m.set_shapes(input=[1, 3, 28, 28])
backend = ml.JaxBackend()
pm = ml.compile(m, backend=backend, jit=False, file_path="model.py", inference=True, use_short_namings=False, safe_names=False)

input = backend.ones([1, 3, 28, 28])
weight = backend.ones([1, 3, 11, 11])
pm.evaluate({"weight": weight}, {"input": input})
...
# model = PolynomialRegression(5, dimension=3)
# backend = ml.JaxBackend()
# model.set_shapes(input=[4,4])
# inp = backend.ones([4, 4])

# pm = ml.compile(model, backend=backend, jit=True, inference=True, use_short_namings=False, safe_names=False)
# params = pm.randomize_params()

# pm.evaluate(params, {"input": inp})