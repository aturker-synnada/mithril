# Copyright 2022 Synnada, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

from mithril import JaxBackend, TorchBackend, compile
from mithril.framework.common import Tensor
from mithril.models import Add, IOKey, MatrixMultiply, Model, ToTensor

from .test_utils import assert_results_equal


def test_tuple_argument_1():
    backend = JaxBackend()

    model = Model()
    add = Add()
    model += add.connect(
        left=IOKey("left", type=Tensor, differentiable=True),
        right=Tensor([3.0, 4, 5]),
        output="output",
    )

    pm = compile(model=model, backend=backend)

    params = {"left": backend.array(3.0)}
    output_gradients = {"output": backend.array([1.0, 2, 3])}
    outputs, grads = pm.evaluate(params=params, output_gradients=output_gradients)

    ref_outputs = {"output": backend.array([6.0, 7, 8])}
    ref_grads = {"left": backend.array(6.0)}

    assert_results_equal(outputs, ref_outputs)
    assert_results_equal(grads, ref_grads)


@pytest.mark.skip(
    reason="Auto conversions are removed, so this test is not valid anymore."
    "This tests will be converted to operate scalar tuple add operations."
    "e.g. (3.0, 2.0) + (4.0, 5.0) = (3.0, 2.0, 4.0, 5.0)"
)
def test_tuple_argument_2():
    backend = TorchBackend()

    model = Model()
    add_model = Add()
    add_model_2 = Add()
    model += add_model.connect(left="left", right="right")
    model += add_model_2.connect(
        left=(add_model.left, add_model.right),  # type: ignore
        right=(add_model.left, add_model.right),  # type: ignore
        output="output",
    )

    pm = compile(model=model, backend=backend, jit=False)
    params = {"left": backend.array(3.0), "right": backend.array(2.0)}

    outputs = pm.evaluate(params=params)

    ref_outputs = {"output": backend.array([6.0, 4.0])}
    assert_results_equal(outputs, ref_outputs)


@pytest.mark.skip(
    reason="Auto conversions are removed, so this test is not valid anymore."
    "This tests will be converted to operate scalar tuple add operations."
    "e.g. (3.0, 2.0) + (4.0, 5.0) = (3.0, 2.0, 4.0, 5.0)"
)
def test_tuple_argument_3():
    backend = JaxBackend()

    model = Model()
    add_model = Add()
    add_model_2 = Add()
    model += add_model.connect(left="left", right="right")
    model += add_model_2.connect(
        left=(add_model.left.shape, add_model.right.shape),  # type: ignore
        right=add_model.left + add_model.right,
        output="output",
    )

    pm = compile(model=model, backend=backend, jit=False)
    params = {"left": backend.array([3.0]), "right": backend.array([2.0])}
    output_gradients = {"output": backend.array([[1.0], [1.0]])}

    outputs, grads = pm.evaluate(params=params, output_gradients=output_gradients)

    ref_outputs = {"output": backend.array([[6.0], [6.0]])}
    ref_grads = {"left": backend.array([2.0]), "right": backend.array([2.0])}
    assert_results_equal(outputs, ref_outputs)
    assert_results_equal(grads, ref_grads)


@pytest.mark.skip(
    reason="Auto conversions are removed, so this test is not valid anymore."
    "This tests will be converted to operate scalar tuple add operations."
    "e.g. (3.0, 2.0) + (4.0, 5.0) = (3.0, 2.0, 4.0, 5.0)"
)
def test_tuple_argument_4():
    backend = JaxBackend()

    model = Model()
    add_model = Add()
    add_model_2 = Add()
    model += add_model.connect(left="left", right="right")
    model += add_model_2.connect(
        left=(add_model.left.shape * 2, add_model.right.shape * 2),  # type: ignore
        right=add_model.left + add_model.right,
        output="output",
    )

    pm = compile(model=model, backend=backend, jit=False)
    params = {"left": backend.array([3.0]), "right": backend.array([2.0])}

    outputs = pm.evaluate(params=params)

    ref_outputs = {"output": backend.array([[7.0], [7.0]])}
    assert_results_equal(outputs, ref_outputs)


@pytest.mark.skip(
    reason="Auto conversions are removed, so this test is not valid anymore."
    "This tests will be converted to operate scalar tuple add operations."
    "e.g. (3.0, 2.0) + (4.0, 5.0) = (3.0, 2.0, 4.0, 5.0)"
)
def test_tuple_argument_5():
    backend = JaxBackend()

    model = Model()
    add_model = Add()
    add_model_2 = Add()
    model += add_model.connect(left="left", right="right")
    model += add_model_2.connect(
        left=(
            (add_model.left.shape[0], add_model.left.shape[0]),  # type: ignore
            (add_model.left.shape[0], add_model.left.shape[0]),  # type: ignore
        ),
        right=add_model.left + add_model.right,
        output="output",
    )

    pm = compile(model=model, backend=backend, jit=False)
    params = {"left": backend.array([3.0]), "right": backend.array([2.0])}

    outputs = pm.evaluate(params=params)

    ref_outputs = {"output": backend.array([[6.0, 6.0], [6.0, 6.0]])}
    assert_results_equal(outputs, ref_outputs)


@pytest.mark.skip(
    reason="Auto conversions are removed, so this test is not valid anymore."
    "This tests will be converted to operate scalar tuple add operations."
    "e.g. (3.0, 2.0) + (4.0, 5.0) = (3.0, 2.0, 4.0, 5.0)"
)
def test_list_tuple_mixed_argument_1():
    backend = JaxBackend()

    model = Model()
    add_model = Add()
    add_model_2 = Add()
    model += add_model.connect(left="left", right="right")
    model += add_model_2.connect(
        left=(
            [add_model.left.shape[0], add_model.left.shape[0]],  # type: ignore
            [add_model.left.shape[0], add_model.left.shape[0]],  # type: ignore
        ),
        right=add_model.left + add_model.right,
        output="output",
    )

    pm = compile(model=model, backend=backend, jit=False)
    params = {"left": backend.array([3.0]), "right": backend.array([2.0])}

    outputs = pm.evaluate(params=params)

    ref_outputs = {"output": backend.array([[6.0, 6.0], [6.0, 6.0]])}
    assert_results_equal(outputs, ref_outputs)


@pytest.mark.skip(
    reason="Auto conversions are removed, so this test is not valid anymore."
    "This tests will be converted to operate scalar tuple add operations."
    "e.g. (3.0, 2.0) + (4.0, 5.0) = (3.0, 2.0, 4.0, 5.0)"
)
def test_list_tuple_mixed_argument_2():
    backend = JaxBackend()

    model = Model()
    add_model = Add()

    model += add_model.connect(left="left", right="right")

    left_first_shape = add_model.left.shape[0]
    right_first_shape = add_model.right.shape[0]

    matmul_left = ([left_first_shape, 0], [2, right_first_shape])

    matmul_right = ([1, 1], [1, 1])
    model += MatrixMultiply().connect(
        left=matmul_left,  # type: ignore
        right=matmul_right,  # type: ignore
        output="output",
    )

    pm = compile(model=model, backend=backend, jit=False)
    params = {"left": backend.array([3.0]), "right": backend.array([2.0])}

    outputs = pm.evaluate(params=params)

    ref_outputs = {"output": backend.array([[1.0, 1.0], [3.0, 3.0]])}
    assert_results_equal(outputs, ref_outputs)


def test_list_tuple_mixed_argument_3():
    backend = JaxBackend()

    model = Model()
    tensor = [(1, 2), [3.0, 4], (5, 5)]

    to_tensor = ToTensor()
    model += to_tensor.connect(input=tensor, output="output")

    pm = compile(model=model, backend=backend, jit=True, inference=True)

    outputs = pm.evaluate()

    ref_outputs = {"output": backend.array([[1, 2], [3.0, 4], (5, 5)])}
    assert_results_equal(outputs, ref_outputs)


def test_list_argument_1():
    backend = JaxBackend()

    model = Model()
    add = Add()
    model += add.connect(
        left=IOKey("left", differentiable=True),
        right=Tensor([3.0, 4, 5]),
        output="output",
    )

    pm = compile(model=model, backend=backend)

    params = {"left": backend.array(3.0)}

    output_gradients = {"output": backend.array([1.0, 2, 3])}

    outputs, grads = pm.evaluate(params=params, output_gradients=output_gradients)

    ref_outputs = {"output": backend.array([6.0, 7, 8])}

    ref_grads = {"left": backend.array(6.0)}

    assert_results_equal(outputs, ref_outputs)
    assert_results_equal(grads, ref_grads)


@pytest.mark.skip(
    reason="Auto conversions are removed, so this test is not valid anymore."
    "This tests will be converted to operate scalar tuple add operations."
    "e.g. (3.0, 2.0) + (4.0, 5.0) = (3.0, 2.0, 4.0, 5.0)"
)
def test_list_argument_2():
    backend = TorchBackend()

    model = Model()
    add_model = Add()
    add_model_2 = Add()

    model += add_model.connect(left="left", right="right")
    model += add_model_2.connect(
        left=[add_model.left, add_model.right],  # type: ignore
        right=[add_model.left, add_model.right],  # type: ignore
        output="output",
    )

    pm = compile(model=model, backend=backend, jit=False)
    params = {"left": backend.array(3.0), "right": backend.array(2.0)}

    outputs = pm.evaluate(params=params)

    ref_outputs = {"output": backend.array([6.0, 4.0])}
    assert_results_equal(outputs, ref_outputs)


@pytest.mark.skip(
    reason="Auto conversions are removed, so this test is not valid anymore."
    "This tests will be converted to operate scalar tuple add operations."
    "e.g. (3.0, 2.0) + (4.0, 5.0) = (3.0, 2.0, 4.0, 5.0)"
)
def test_list_argument_3():
    backend = JaxBackend()

    model = Model()
    add_model = Add()
    add_model_2 = Add()

    model += add_model.connect(left="left", right="right")
    model += add_model_2.connect(
        left=[add_model.left.shape, add_model.right.shape],  # type: ignore
        right=add_model.left + add_model.right,
        output="output",
    )

    pm = compile(model=model, backend=backend, jit=False)
    params = {"left": backend.array([3.0]), "right": backend.array([2.0])}
    output_gradients = {"output": backend.array([[1.0], [1.0]])}

    outputs, grads = pm.evaluate(params=params, output_gradients=output_gradients)

    ref_outputs = {"output": backend.array([[6.0], [6.0]])}
    ref_grads = {"left": backend.array([2.0]), "right": backend.array([2.0])}
    assert_results_equal(outputs, ref_outputs)
    assert_results_equal(grads, ref_grads)


@pytest.mark.skip(
    reason="Auto conversions are removed, so this test is not valid anymore."
    "This tests will be converted to operate scalar tuple add operations."
    "e.g. (3.0, 2.0) + (4.0, 5.0) = (3.0, 2.0, 4.0, 5.0)"
)
def test_list_argument_4():
    backend = JaxBackend()

    model = Model()
    add_model = Add()
    add_model_2 = Add()

    model += add_model.connect(left="left", right="right")
    model += add_model_2.connect(
        left=[add_model.left.shape * 2, add_model.right.shape * 2],  # type: ignore
        right=add_model.left + add_model.right,
        output="output",
    )

    pm = compile(model=model, backend=backend, jit=False)
    params = {"left": backend.array([3.0]), "right": backend.array([2.0])}

    outputs = pm.evaluate(params=params)

    ref_outputs = {"output": backend.array([[7.0], [7.0]])}
    assert_results_equal(outputs, ref_outputs)


@pytest.mark.skip(
    reason="Auto conversions are removed, so this test is not valid anymore."
    "This tests will be converted to operate scalar tuple add operations."
    "e.g. (3.0, 2.0) + (4.0, 5.0) = (3.0, 2.0, 4.0, 5.0)"
)
def test_list_argument_5():
    backend = JaxBackend()

    model = Model()
    add_model = Add()
    add_model_2 = Add()

    model += add_model.connect(left="left", right="right")
    model += add_model_2.connect(
        left=[
            [add_model.left.shape[0], add_model.left.shape[0]],  # type: ignore
            [add_model.left.shape[0], add_model.left.shape[0]],  # type: ignore
        ],
        right=add_model.left + add_model.right,
        output="output",
    )

    pm = compile(model=model, backend=backend, jit=False)
    params = {"left": backend.array([3.0]), "right": backend.array([2.0])}

    outputs = pm.evaluate(params=params)

    ref_outputs = {"output": backend.array([[6.0, 6.0], [6.0, 6.0]])}
    assert_results_equal(outputs, ref_outputs)
