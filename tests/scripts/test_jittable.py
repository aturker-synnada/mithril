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

import typing
from collections.abc import Mapping, Sequence
from types import EllipsisType

import jax.numpy as jnp
import numpy as np
import pytest

from mithril import JaxBackend, NumpyBackend, TorchBackend, compile
from mithril.cores.python.jax.ops import (
    add,
    partial,
    reduce_mean,
    shape,
    to_tensor,
)
from mithril.framework import NOT_GIVEN, ConnectionType, ExtendInfo
from mithril.framework.common import Tensor
from mithril.framework.constraints import bcast
from mithril.framework.logical.base import BaseKey
from mithril.models import (
    TBD,
    Add,
    Buffer,
    CustomPrimitiveModel,
    Indexer,
    IOKey,
    Item,
    MatrixMultiply,
    Mean,
    Model,
    Multiply,
    PrimitiveUnion,
    Reshape,
    Shape,
    Slice,
    TensorToList,
    ToTensor,
    Unique,
)

from ..utils import with_temp_file
from .test_utils import assert_results_equal

to_tensor = partial(to_tensor, device="cpu")

############################################################################################
# In this file some of our models are tested to see if they are jittable
# in all possible cases.
############################################################################################


class MyModel(Model):
    def __init__(self, dimension: int | None = None) -> None:
        """This model implements above model.

        mult_model = MatrixMultiplication()
        sum_model = Add()
        self.extend(mult_model, input = "input", rhs = "w")
        self.extend(sum_model, input = mult_model.output, rhs = "b")
        self.extend((reshp := Reshape(shape = [sum_model.output.shape[:], 1, 1])),
        input = sum_model.output) self.extend((sum2 := Sum()), input = sum_model.
        output.shape[mult_model.output.shape[reshp.output.shape[-1]]], rhs = 3.0)
        self.extend(Multiplication(), input = sum2.output, rhs = 2.0, output =
        IOKey(name = "output"))
        """
        sum_slc = Model()
        sum_slc |= (slc := Slice(start=None, stop=None, step=None))
        sum_slc |= Indexer().connect(
            input="input", index=slc.output, output=IOKey("output")
        )
        super().__init__()
        mult_model = MatrixMultiply()
        sum_model = Add()
        sum_model.set_types(left=Tensor, right=Tensor)
        self |= mult_model.connect(left="input", right="w")  # (10, 1)
        self |= sum_model.connect(left=mult_model.output, right="b")  # (10, 1)
        self |= (sum_shp := Shape()).connect(input=sum_model.output)  # (10, 1)
        self |= sum_slc.connect(input=sum_shp.output)  # (10, 1)
        self |= (uni := PrimitiveUnion(n=3)).connect(
            input1=sum_slc.output,  # type: ignore
            input2=1,
            input3=1,
        )  # (10, 1, 1, 1)
        self |= (reshp_1 := Reshape()).connect(
            input=sum_model.output, shape=uni.output
        )  # (10, 1, 1, 1)
        self |= (reshp_shp := Shape()).connect(input=reshp_1.output)  # (10, 1, 1, 1)
        self |= (idx_1 := Indexer()).connect(index=-1, input=reshp_shp.output)  # 1
        self |= (mult_shp := Shape()).connect(input=mult_model.output)  # (10, 1)
        self |= (idx_2 := Indexer()).connect(
            index=idx_1.output, input=mult_shp.output
        )  # 1
        self |= (idx_3 := Indexer()).connect(
            index=idx_2.output, input=sum_shp.output
        )  # 1
        self |= (tens := ToTensor()).connect(input=idx_3.output)  # array(1)
        self |= (sum := Add()).connect(left=tens.output, right=Tensor(3.0))  # array(4)
        self |= Multiply().connect(
            left=sum.output, right=Tensor(2.0), output=IOKey(name="output")
        )  # array(8)

        shapes: Mapping[str, Sequence[str | tuple[str, EllipsisType] | int | None]] = {
            "input": ["N", ("Var_inter", ...), "d_in"],
            "w": ["d_in", dimension],
            "b": [dimension],
        }
        self.set_shapes(**shapes)


class MyModel2(Model):
    def __init__(self, dimension: int | None = None) -> None:
        """This model implements above model.

        mult_model = MatrixMultiplication()
        sum_model = Add()
        self.extend(mult_model, input = "input", rhs = "w")
        self.extend(sum_model, input = mult_model.output, rhs = "b")
        self.extend((reshp := Reshape(shape = [sum_model.output.shape[:], 1, 1])),
        input = sum_model.output) self.extend((sum2 := Sum()), input = sum_model.
        output.shape[mult_model.output.shape[reshp.output.shape[-1]]], rhs = 3.0)
        self.extend(Multiplication(), input = sum2.output, rhs = 2.0, output =
        IOKey(name = "output"))
        """
        super().__init__()
        mult_model = MatrixMultiply()
        sum_model = Add()
        self += mult_model.connect(left="input", right="w")  # (10, 1)
        self += sum_model.connect(
            left=mult_model.output, right=IOKey("b", type=Tensor)
        )  # (10, 1)
        self += (sum_shp := Shape()).connect(input=sum_model.output)  # (10, 1)
        self += (uni := PrimitiveUnion(n=3)).connect(
            input1=sum_shp.output, input2=1, input3=3
        )  # (10, 1, 1, 1)
        self += (idx_1 := Indexer()).connect(index=-1, input=uni.output)  # 1
        self += (tens := ToTensor()).connect(input=idx_1.output)  # array(1)
        self += Multiply().connect(
            left=tens.output, right=Tensor(2.0), output=IOKey(name="output")
        )  # array(8)

        shapes: Mapping[str, Sequence[str | tuple[str, EllipsisType] | int | None]] = {
            "input": ["N", ("Var_inter", ...), "d_in"],
            "w": ["d_in", dimension],
            "b": [dimension],
        }
        self.set_shapes(**shapes)


np_input = np.random.randn(10, 3).astype(np.float32)


def test_mymodel_numpy():
    model = MyModel(dimension=1)
    static_inputs = {"input": np_input}
    compiled_model = compile(
        model=model,
        backend=NumpyBackend(),
        constant_keys=static_inputs,
        jit=False,
        inference=True,
    )
    inputs = compiled_model.randomize_params()
    result = compiled_model.evaluate(inputs)
    ref_output = {"output": np.array(8.0)}
    assert_results_equal(result, ref_output)


@with_temp_file(".py")
def test_mymodel_jax_jit(file_path: str):
    model = MyModel(dimension=1)
    backend = JaxBackend()
    static_inputs = {"input": backend.array(np_input)}
    compiled_model = compile(
        model=model,
        backend=backend,
        constant_keys=static_inputs,
        jit=True,
        file_path=file_path,
        inference=True,
    )
    inputs = compiled_model.randomize_params()
    result = compiled_model.evaluate(inputs)
    ref_output = {"output": backend.array(8.0)}
    assert_results_equal(result, ref_output)


@with_temp_file(".py")
def test_mymodel_torch_jit(file_path: str):
    model = MyModel(dimension=1)
    backend = TorchBackend()
    static_inputs = {"input": backend.array(np_input)}
    compiled_model = compile(
        model=model,
        backend=backend,
        constant_keys=static_inputs,
        jit=True,
        file_path=file_path,
        inference=True,
    )
    inputs = compiled_model.randomize_params()
    result = compiled_model.evaluate(inputs)
    ref_output = {"output": backend.array(8.0)}
    assert_results_equal(result, ref_output)


def test_mymodel_jax_1():
    model = MyModel(dimension=1)
    static_inputs = {"input": jnp.array(np_input)}
    compiled_model = compile(
        model=model,
        backend=JaxBackend(),
        constant_keys=static_inputs,
        jit=False,
        inference=True,
    )
    inputs = compiled_model.randomize_params()
    result = compiled_model.evaluate(inputs)
    out = result["output"]
    assert isinstance(out, jnp.ndarray)
    ref_output = {"output": jnp.array(8.0)}
    assert_results_equal(result, ref_output)


@pytest.mark.skip(reason="Provide ref_output!")
def test_mymodel_jax_2():
    model = MyModel2(dimension=1)
    static_inputs = {"input": jnp.array(np_input)}
    compiled_model = compile(
        model=model,
        backend=JaxBackend(),
        constant_keys=static_inputs,
        jit=False,
        inference=True,
    )
    inputs = compiled_model.randomize_params()
    result = compiled_model.evaluate(inputs)
    out = result["output"]
    assert isinstance(out, jnp.ndarray)
    # assert_results_equal(result, ref_output)


def test_mymodel_jax():
    """This function tests if jax model is
    properly jitted.
    """
    static_inputs = {"input": jnp.array(np_input)}

    # set a dict_counter dict, if the function is properly jitted,
    # We only
    jit_counter = {"jit_counter": 0}

    def adder(left, right):
        jit_counter["jit_counter"] += 1
        return left + right

    JaxBackend.register_primitive(adder)

    class Adder(CustomPrimitiveModel):
        def __init__(self) -> None:
            super().__init__(
                formula_key="adder",
                output=BaseKey(shape=[("Var_out", ...)], type=Tensor),
                left=BaseKey(shape=[("Var_1", ...)], type=Tensor),
                right=BaseKey(shape=[("Var_2", ...)], type=Tensor),
            )
            self.add_constraint(fn=bcast, keys=["output", "left", "right"])

        def connect(  # type: ignore[override]
            self,
            left: ConnectionType = NOT_GIVEN,
            right: ConnectionType = NOT_GIVEN,
            output: ConnectionType = NOT_GIVEN,
        ) -> ExtendInfo:
            kwargs = {"left": left, "right": right, "output": output}
            return ExtendInfo(self, kwargs)

    model = MyModel(dimension=1)
    model |= Adder().connect(
        left="output", right=IOKey("r1", differentiable=True), output=IOKey(name="o1")
    )
    compiled_model = compile(
        model=model, backend=JaxBackend(), constant_keys=static_inputs, jit=True
    )
    inputs = compiled_model.randomize_params()
    compiled_model.evaluate(inputs)
    compiled_model.evaluate(inputs)
    compiled_model.evaluate(inputs)
    assert jit_counter["jit_counter"] == 1


def test_logical_model_jittable_1():
    """Tests for jittablity in Logical domain. Since this model
    requires TensorToList operation before ToTensor, it breaks the
    jit.
    """
    model = Model()
    model |= (add1 := Add()).connect(left="l1", right="l2", output=IOKey(name="out1"))
    model |= (add2 := Add()).connect(left="l3", right="l4")
    model.merge_connections(add1.left, add2.left, name="input")
    model |= Item().connect(add1.left)
    with pytest.raises(Exception) as error_info:
        compile(model=model, backend=JaxBackend(), jit=True)

    assert str(error_info.value) == (
        "Operator 'item' is not JIT compatible. "
        "Please set jit=False in compile() function."
    )


def test_logical_model_jittable_2():
    """Tests for jittablity in Logical domain. Since this model
    sets enforce_jit to False, no error will be thrown.
    """
    model = Model()
    model |= (add1 := Add()).connect(left="l1", right="l2", output=IOKey(name="out1"))
    model |= (add2 := Add()).connect(left="l3", right="l4")
    model.merge_connections(add1.left, add2.left, name="input")
    model |= Item().connect(input=add1.left)
    compiled_model = compile(
        model=model, backend=JaxBackend(), jit=False, inference=True
    )

    assert not compiled_model.jit


def test_physical_model_jit_1():
    """Tests for jittablity in Physical domain. Since compilation is done
    with jit = False, no errors will be raised when model is not jittable.
    """
    model = Model()
    add1 = Add()
    add2 = Add()
    model |= add1.connect(
        left=IOKey("l1", differentiable=True),
        right=IOKey("l2", differentiable=True),
        output=IOKey(name="out1"),
    )
    model |= add2.connect(
        left=IOKey("l3", differentiable=True), right=IOKey("l4", differentiable=True)
    )
    model.enforce_jit = False
    model.merge_connections(add1.left, add2.left, name="input")
    model |= Item().connect(input="input")

    backend = JaxBackend()
    compiled_model = compile(model=model, backend=backend, jit=False)
    inputs = compiled_model.randomize_params()
    output_gradients = {"out1": backend.ones_like(inputs["input"])}
    compiled_model.evaluate(inputs, output_gradients=output_gradients)


def test_physical_model_jit_2():
    """Tests for jittablity in Physical domain. Since compilation is done
    with jit = True, exception will be raised because model is not jittable.
    """
    model = Model()
    model |= (add1 := Add()).connect(left="l1", right="l2", output=IOKey(name="out1"))
    model |= (add2 := Add()).connect(left="l3", right="l4")
    model.merge_connections(add1.left, add2.left, name="input")
    model |= Item().connect(input="input")

    backend = JaxBackend()

    with pytest.raises(Exception) as error_info:
        compile(model=model, backend=backend, jit=True)

    assert str(error_info.value) == (
        "Operator 'item' is not JIT compatible. "
        "Please set jit=False in compile() function."
    )


def test_jit_1():
    jit_counter = {"jit_counter": 0}

    def adder(left, right):
        jit_counter["jit_counter"] += 1
        return left + right

    JaxBackend.register_primitive(adder)

    class Adder(CustomPrimitiveModel):
        def __init__(self) -> None:
            super().__init__(
                formula_key="adder",
                output=BaseKey(shape=[("Var_out", ...)], type=Tensor),
                left=BaseKey(shape=[("Var_1", ...)], type=Tensor),
                right=BaseKey(shape=[("Var_2", ...)], type=Tensor),
            )
            self.add_constraint(fn=bcast, keys=["output", "left", "right"])

    add_model = Add()
    model = Model()
    model |= add_model.connect(left="left", right="right")
    model |= TensorToList().connect(add_model.output)

    with pytest.raises(Exception) as err_info:
        compile(model=model, backend=JaxBackend(), jit=True)

    assert str(err_info.value) == (
        "Operator 'tensor_to_list' is not JIT compatible. "
        "Please set jit=False in compile() function."
    )


def test_jit_2():
    backend = JaxBackend()
    model = Model()
    model |= (add_model := Add()).connect(
        left=IOKey("left", differentiable=True),
        right=IOKey("right", differentiable=True),
    )
    in1 = add_model.output
    out1 = in1.shape
    out2 = out1.tensor().sum()
    mean_model = Mean(axis=TBD)
    model |= (to_list := Item()).connect(input=out2)
    model |= mean_model.connect(
        input=IOKey("input", differentiable=True),
        axis=to_list.output,
        output=IOKey(name="output"),
    )
    pm = compile(model=model, backend=backend, jit=False)
    params = {
        "left": backend.randn(1, 1),
        "right": backend.randn(1, 1),
        "input": backend.randn(1, 1, 1, 1, 1, 1, 1, 1, 1),
    }
    pm.evaluate(params=params)
    # TODO: Make required assertions!!!


def test_jit_3():
    backend = JaxBackend()
    model = Model()
    model |= Mean(axis=TBD).connect(
        input="input", output=IOKey(name="output"), axis="axis"
    )
    pm = compile(model=model, backend=backend, jit=False, inference=True)

    inputs = {"input": backend.randn(1, 2, 3, 2, 3, 2, 3, 2), "axis": 3}

    pm.evaluate(data=inputs)  # type: ignore


def test_jit_4():
    backend = JaxBackend()
    model = Model()
    model |= Mean(axis=TBD).connect(
        input="input", output=IOKey(name="output"), axis="axis"
    )
    pm = compile(
        model=model,
        backend=backend,
        jit=True,
        constant_keys={"axis": 3},
        inference=True,
    )

    inputs = {"input": backend.randn(1, 2, 3, 2, 3, 2, 3, 2)}

    pm.evaluate(data=inputs)


def test_jit_5():
    backend = JaxBackend()
    import jax

    @jax.jit
    @typing.no_type_check
    def evaluate(params):
        input = params["input"]
        keepdim_1 = False
        left = params["left"]
        right = params["right"]
        _Add_0_output = add(left, right)
        _Shape_1_output = shape(_Add_0_output)
        sum_shape = sum(_Shape_1_output)
        idx = (sum_shape**2) * 2 + 3
        output = reduce_mean(input, axis=sum(_Shape_1_output), keepdim=keepdim_1)
        for _ in range(idx):
            output = output + 1
        return {"output": output}

    params = {
        "left": backend.randn(1, 1),
        "right": backend.randn(1, 1),
        "input": backend.randn(1, 1, 1, 1, 1, 1, 1, 1, 1),
    }
    evaluate(params)
    evaluate(params)
    evaluate(params)


@pytest.mark.skip(reason="Bool indexing for Tensors is not implemented")
def test_jit_compile_tensor_bool_slicing():
    input = IOKey(type=Tensor[float])
    index = IOKey(type=Tensor[bool])

    indexed = input[index]

    model = Model()
    model |= Buffer().connect(input=indexed)

    with pytest.raises(RuntimeError) as err_jax:
        compile(model=model, backend=JaxBackend(), jit=True, inference=True)

    assert str(err_jax.value) == (
        "Operator 'slice' is not JIT compatible. "
        "Please set jit=False in compile() function."
    )


def test_jit_compile_to_list():
    input = IOKey(name="input", type=Tensor[float])

    model = Model()
    model |= TensorToList().connect(input=input)

    # Jax should raise error
    with pytest.raises(RuntimeError) as err_jax:
        compile(model=model, backend=JaxBackend(), jit=True, inference=True)

    with pytest.raises(RuntimeError) as err_torch:
        compile(model=model, backend=TorchBackend(), jit=True, inference=True)

    assert (
        str(err_jax.value)
        == str(err_torch.value)
        == (
            "Operator 'tensor_to_list' is not JIT compatible. "
            "Please set jit=False in compile() function."
        )
    )


def test_jit_compile_item():
    input = IOKey(name="input", type=Tensor[float])

    model = Model()
    model |= Item().connect(input=input)

    # Jax should raise error
    with pytest.raises(RuntimeError) as err_jax:
        compile(model=model, backend=JaxBackend(), jit=True, inference=True)

    with pytest.raises(RuntimeError) as err_torch:
        compile(model=model, backend=TorchBackend(), jit=True, inference=True)

    assert (
        str(err_jax.value)
        == str(err_torch.value)
        == (
            "Operator 'item' is not JIT compatible. Please set "
            "jit=False in compile() function."
        )
    )


def test_jit_compile_unique():
    model = Model()
    model |= Unique().connect(input="input", output=IOKey(name="output"))

    # Jax should raise error
    with pytest.raises(RuntimeError) as err_jax:
        compile(model=model, backend=JaxBackend(), jit=True, inference=True)

    # Torch should not raise error
    compile(model=model, backend=TorchBackend(), jit=True, inference=True)

    assert str(err_jax.value) == (
        "Operator 'unique' is not JIT compatible. Please set "
        "jit=False in compile() function."
    )
