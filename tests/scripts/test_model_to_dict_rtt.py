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

import re

import pytest

import mithril
from mithril import JaxBackend, TorchBackend
from mithril.framework.common import TBD, Tensor
from mithril.framework.constraints import squeeze_constraints
from mithril.framework.logical.base import BaseKey
from mithril.models import (
    L2,
    MLP,
    Add,
    Buffer,
    Convolution2D,
    CrossEntropy,
    IOKey,
    Layer,
    Linear,
    Mean,
    Model,
    Operator,
    Relu,
    Sigmoid,
    Sqrt,
    SquaredError,
    ToTensor,
    TrainModel,
)
from mithril.models.primitives import PrimitiveModel
from mithril.types import Constant, Dtype
from mithril.utils import dict_conversions

from .helper import assert_evaluations_equal, assert_models_equal

# TODO: assigned_constraint utility of dict to models are not tested.
# add tests for assigned_constraint utility of dict to models


def test_linear_expose():
    model = Model()
    model += Linear(dimension=42).connect(
        input="input", weight="weight", output=IOKey(name="output")
    )
    model.expose_keys("output")
    model_dict_created = dict_conversions.model_to_dict(model)
    model_recreated = dict_conversions.dict_to_model(model_dict_created)
    model_dict_recreated = dict_conversions.model_to_dict(model_recreated)

    assert model_dict_created == model_dict_recreated
    assert_models_equal(model, model_recreated)

    backend = JaxBackend(dtype=mithril.float64)
    assert_evaluations_equal(
        model, model_recreated, backend, static_keys={"input": backend.ones([4, 256])}
    )


def test_linear_expose_set_shapes():
    model = Model()
    lin_1 = Linear()
    lin_2 = Linear()
    model |= lin_1.connect(input="input", weight="weight")
    model |= lin_2.connect(
        input=lin_1.output, weight="weight1", output=IOKey(name="output2")
    )
    model.set_shapes({lin_1.bias: [42]})
    model.set_shapes({lin_2.bias: [21]})
    model_dict_created = dict_conversions.model_to_dict(model)
    model_recreated = dict_conversions.dict_to_model(model_dict_created)
    model_dict_recreated = dict_conversions.model_to_dict(model_recreated)

    assert model_dict_created == model_dict_recreated
    assert model.shapes == model_recreated.shapes
    assert_models_equal(model, model_recreated)

    backend = JaxBackend(dtype=mithril.float64)
    assert_evaluations_equal(
        model, model_recreated, backend, static_keys={"input": backend.ones([4, 256])}
    )


def test_linear_expose_set_shapes_extend_from_inputs():
    model = Model()
    lin_1 = Linear()
    lin_2 = Linear()
    model |= lin_2.connect(weight="weight1", output=IOKey(name="output2"))
    model |= lin_1.connect(input="input", weight="weight", output=lin_2.input)
    model.set_shapes({lin_1.bias: [42]})
    model.set_shapes({lin_2.bias: [21]})
    model_dict_created = dict_conversions.model_to_dict(model)
    model_recreated = dict_conversions.dict_to_model(model_dict_created)
    model_dict_recreated = dict_conversions.model_to_dict(model_recreated)

    assert model_dict_created == model_dict_recreated
    assert model.shapes == model_recreated.shapes
    assert_models_equal(model, model_recreated)

    backend = JaxBackend(dtype=mithril.float64)
    assert_evaluations_equal(
        model, model_recreated, backend, static_keys={"input": backend.ones([4, 256])}
    )


def test_linear_set_diff():
    model = Model()
    linear = Linear(dimension=42)
    model += linear.connect(input="input", weight="weight", output=IOKey(name="output"))
    linear.set_differentiability(weight=False)

    model_dict_created = dict_conversions.model_to_dict(model)
    model_recreated = dict_conversions.dict_to_model(model_dict_created)
    model_dict_recreated = dict_conversions.model_to_dict(model_recreated)

    assert model_dict_created == model_dict_recreated
    assert_models_equal(model, model_recreated)

    backend = JaxBackend(dtype=mithril.float64)
    assert_evaluations_equal(
        model,
        model_recreated,
        backend,
        static_keys={
            "input": backend.ones([4, 256]),
            "weight": backend.ones([42, 256]),
        },
    )


def test_linear_expose_2():
    model = Model()
    model |= Linear(dimension=42).connect(
        input="input", weight="weight", output=IOKey(name="output")
    )
    model_dict_created = dict_conversions.model_to_dict(model)
    model_recreated = dict_conversions.dict_to_model(model_dict_created)
    model_dict_recreated = dict_conversions.model_to_dict(model_recreated)

    assert model_dict_created == model_dict_recreated
    assert_models_equal(model, model_recreated)

    backend = JaxBackend(dtype=mithril.float64)
    assert_evaluations_equal(
        model, model_recreated, backend, static_keys={"input": backend.ones([4, 256])}
    )


def test_linear_not_expose():
    model = Model()
    model |= Linear(dimension=42).connect(input="input")
    model_dict_created = dict_conversions.model_to_dict(model)
    model_recreated = dict_conversions.dict_to_model(model_dict_created)
    model_dict_recreated = dict_conversions.model_to_dict(model_recreated)

    assert model_dict_created == model_dict_recreated
    assert_models_equal(model, model_recreated)

    backend = JaxBackend(dtype=mithril.float64)
    assert_evaluations_equal(
        model, model_recreated, backend, static_keys={"input": backend.ones([4, 256])}
    )


def test_set_cins_couts():
    model = Model()
    linear_1 = Linear(dimension=42)
    model |= linear_1.connect(
        input="input", weight="weight", output=IOKey(name="output")
    )
    model.set_cin("weight", linear_1.bias)
    outer_model = Model()
    linear_2 = Linear(dimension=42)
    outer_model |= model.connect(input="input_1", output="output_1")
    outer_model |= linear_2.connect(input="input_2", output="output_2")
    outer_model.expose_keys("output_1", "output_2")
    outer_model.set_cout("output_2")

    model_dict_created = dict_conversions.model_to_dict(outer_model)
    model_recreated = dict_conversions.dict_to_model(model_dict_created)
    model_dict_recreated = dict_conversions.model_to_dict(model_recreated)

    assert model_dict_created == model_dict_recreated
    assert (
        model_dict_created["assigned_cins"]  # type: ignore
        == model_dict_recreated["assigned_cins"]  # type: ignore
        == []
    )
    assert (
        model_dict_created["assigned_couts"]  # type: ignore
        == model_dict_recreated["assigned_couts"]  # type: ignore
        == [("m_1", 3)]
    )
    assert (
        model_dict_created["submodels"]["m_0"]["assigned_cins"]  # type: ignore
        == model_dict_recreated["submodels"]["m_0"]["assigned_cins"]  # type: ignore
        == [("self", 2), "weight"]
    )
    assert (
        model_dict_created["submodels"]["m_0"]["assigned_couts"]  # type: ignore
        == model_dict_recreated["submodels"]["m_0"]["assigned_couts"]  # type: ignore
        == []
    )

    assert_models_equal(outer_model, model_recreated)

    backend = JaxBackend(dtype=mithril.float64)
    assert_evaluations_equal(
        outer_model,
        model_recreated,
        backend,
        static_keys={
            "input_1": backend.ones([4, 256]),
            "input_2": backend.ones([4, 256]),
        },
    )


def test_constant_key():
    model = Model()
    model | Add().connect(left="input", right=Tensor(3), output=IOKey(name="output"))
    model2 = Model()
    model2 | model.connect(input="input")

    model_dict_created = dict_conversions.model_to_dict(model2)
    model_recreated = dict_conversions.dict_to_model(model_dict_created)
    model_dict_recreated = dict_conversions.model_to_dict(model_recreated)

    assert model_dict_created == model_dict_recreated
    assert_models_equal(model2, model_recreated)

    backend = JaxBackend(dtype=mithril.float64)
    assert_evaluations_equal(
        model2,
        model_recreated,
        backend,
        static_keys={"input": backend.ones([4, 256])},
        inference=True,
    )


def test_constant_key_2():
    model = Model()
    model |= (add := Add()).connect(
        left=IOKey("input", type=Tensor, differentiable=True),
        right=IOKey(value=Tensor(3)),
        output=IOKey(name="output"),
    )
    model |= Add().connect(
        left=IOKey("input2", type=Tensor),
        right=add.right,
        output=IOKey(name="output2"),
    )
    model2 = Model()
    model2 |= model.connect(
        input2="input", output=IOKey(name="output"), output2=IOKey(name="output2")
    )

    model_dict_created = dict_conversions.model_to_dict(model2)
    model_recreated = dict_conversions.dict_to_model(model_dict_created)
    model_dict_recreated = dict_conversions.model_to_dict(model_recreated)

    assert model_dict_created == model_dict_recreated
    assert_models_equal(model2, model_recreated)

    backend = JaxBackend(dtype=mithril.float64)
    assert_evaluations_equal(
        model2, model_recreated, backend, static_keys={"input": backend.ones([4, 256])}
    )


def test_linear_directly():
    model = Linear(dimension=42)
    model_dict_created = dict_conversions.model_to_dict(model)
    model_recreated = dict_conversions.dict_to_model(model_dict_created)
    model_dict_recreated = dict_conversions.model_to_dict(model_recreated)

    assert model_dict_created == model_dict_recreated
    assert_models_equal(model, model_recreated)

    backend = JaxBackend(dtype=mithril.float64)
    assert_evaluations_equal(
        model, model_recreated, backend, static_keys={"input": backend.ones([4, 256])}
    )


def test_mlp_directly():
    model = MLP(dimensions=[11, 76], activations=[Sigmoid(), Relu()])

    model_dict_created = dict_conversions.model_to_dict(model)
    model_recreated = dict_conversions.dict_to_model(model_dict_created)
    model_dict_recreated = dict_conversions.model_to_dict(model_recreated)

    assert model_dict_created == model_dict_recreated
    assert_models_equal(model, model_recreated)

    backend = JaxBackend(dtype=mithril.float64)
    assert_evaluations_equal(
        model, model_recreated, backend, static_keys={"input": backend.ones([4, 256])}
    )


def test_composite_1():
    model = Model()
    model |= Linear(dimension=10).connect(
        input="input", weight="weight", output=IOKey(name="output")
    )
    model |= Linear(dimension=71).connect(
        input="output", weight="weight1", output=IOKey(name="output2")
    )

    model_dict_created = dict_conversions.model_to_dict(model)
    model_recreated = dict_conversions.dict_to_model(model_dict_created)
    model_dict_recreated = dict_conversions.model_to_dict(model_recreated)
    assert model_dict_created == model_dict_recreated
    assert_models_equal(model, model_recreated)

    backend = JaxBackend(dtype=mithril.float64)
    assert_evaluations_equal(
        model, model_recreated, backend, static_keys={"input": backend.ones([4, 256])}
    )


def test_composite_2():
    model = Model()
    model |= Linear(dimension=10).connect(
        input="input", weight="weight", output=IOKey(name="output")
    )
    model |= Linear(dimension=71).connect(
        input=model.output,  # type: ignore
        weight="weight1",
        output=IOKey(name="output2"),
    )

    model_dict_created = dict_conversions.model_to_dict(model)
    model_recreated = dict_conversions.dict_to_model(model_dict_created)
    model_dict_recreated = dict_conversions.model_to_dict(model_recreated)
    assert model_dict_created == model_dict_recreated
    assert_models_equal(model, model_recreated)

    backend = JaxBackend(dtype=mithril.float64)
    assert_evaluations_equal(
        model, model_recreated, backend, static_keys={"input": backend.ones([4, 256])}
    )


def test_composite_2_1():
    model = Model()
    model |= (l1 := Linear(dimension=10)).connect(
        input="input", weight="weight", output=IOKey(name="output")
    )
    model |= Linear(dimension=71).connect(
        input=l1.output, weight="weight1", output=IOKey(name="output2")
    )

    model_dict_created = dict_conversions.model_to_dict(model)
    model_recreated = dict_conversions.dict_to_model(model_dict_created)
    model_dict_recreated = dict_conversions.model_to_dict(model_recreated)
    assert model_dict_created == model_dict_recreated
    assert_models_equal(model, model_recreated)

    backend = JaxBackend(dtype=mithril.float64)
    assert_evaluations_equal(
        model, model_recreated, backend, static_keys={"input": backend.ones([4, 256])}
    )


def test_composite_2_2():
    model = Model()
    model |= (l1 := Linear(dimension=10)).connect(input="input", weight="weight")
    model |= Linear(dimension=71).connect(
        input=l1.output, weight="weight1", output=IOKey(name="output2")
    )

    model_dict_created = dict_conversions.model_to_dict(model)
    model_recreated = dict_conversions.dict_to_model(model_dict_created)
    model_dict_recreated = dict_conversions.model_to_dict(model_recreated)
    assert model_dict_created == model_dict_recreated
    assert_models_equal(model, model_recreated)

    backend = JaxBackend(dtype=mithril.float64)
    assert_evaluations_equal(
        model, model_recreated, backend, static_keys={"input": backend.ones([4, 256])}
    )


def test_composite_2_3():
    model = Model()
    model |= (l1 := Linear()).connect(input="input", weight="weight")
    model |= Linear().connect(
        input=l1.output, weight=l1.weight, bias=l1.bias, output=IOKey(name="output2")
    )

    model_dict_created = dict_conversions.model_to_dict(model)
    model_recreated = dict_conversions.dict_to_model(model_dict_created)
    model_dict_recreated = dict_conversions.model_to_dict(model_recreated)
    assert model_dict_created == model_dict_recreated
    assert_models_equal(model, model_recreated)

    backend = JaxBackend(dtype=mithril.float64)
    assert_evaluations_equal(
        model, model_recreated, backend, static_keys={"input": backend.ones([4, 256])}
    )


def test_composite_3():
    model = Model()
    model |= (l1 := Linear(dimension=10)).connect(
        input="input", weight="weight", output=IOKey(name="output")
    )
    model |= Linear(dimension=71).connect(
        input=l1.output, weight="weight1", output=IOKey(name="output2")
    )
    model |= Linear(dimension=71).connect(
        input="input2", weight="weight1", output=IOKey(name="output3")
    )

    model_dict_created = dict_conversions.model_to_dict(model)
    model_recreated = dict_conversions.dict_to_model(model_dict_created)
    model_dict_recreated = dict_conversions.model_to_dict(model_recreated)
    assert model_dict_created == model_dict_recreated
    assert_models_equal(model, model_recreated)

    backend = JaxBackend(dtype=mithril.float64)
    assert_evaluations_equal(
        model,
        model_recreated,
        backend,
        static_keys={"input": backend.ones([4, 256]), "input2": backend.ones([4, 10])},
    )


def test_composite_4():
    model = Model()
    model |= (l1 := Linear(dimension=10)).connect(
        input="input", weight="weight", output=IOKey(name="output")
    )
    model |= Linear(dimension=71).connect(
        input=l1.output, weight="weight1", output=IOKey(name="output2")
    )
    model |= Linear(dimension=71).connect(
        input=l1.output, weight="weight1", output=IOKey(name="output3")
    )

    model_dict_created = dict_conversions.model_to_dict(model)
    model_recreated = dict_conversions.dict_to_model(model_dict_created)
    model_dict_recreated = dict_conversions.model_to_dict(model_recreated)
    assert model_dict_created == model_dict_recreated
    assert_models_equal(model, model_recreated)

    backend = JaxBackend(dtype=mithril.float64)
    assert_evaluations_equal(
        model, model_recreated, backend, static_keys={"input": backend.ones([4, 256])}
    )


def test_composite_5():
    model = Model()
    model |= Linear(dimension=10).connect(
        input="input", weight="weight", output=IOKey(name="output")
    )
    model |= Linear(dimension=71).connect(
        input=model.cout, weight="weight1", output=IOKey(name="output2")
    )
    model |= Linear(dimension=71).connect(
        input=model.cout, weight="weight2", output=IOKey(name="output3")
    )

    model_dict_created = dict_conversions.model_to_dict(model)
    model_recreated = dict_conversions.dict_to_model(model_dict_created)
    model_dict_recreated = dict_conversions.model_to_dict(model_recreated)
    assert model_dict_created == model_dict_recreated
    assert_models_equal(model, model_recreated)

    backend = JaxBackend(dtype=mithril.float64)
    assert_evaluations_equal(
        model,
        model_recreated,
        backend,
        static_keys={"input": backend.ones([4, 256])},
    )


def test_composite_6():
    model = Model()
    model |= Linear(dimension=10).connect(
        input="input", weight="weight", output=IOKey(name="output")
    )
    model |= Linear(dimension=71).connect(
        input=model.cout, weight="weight1", output=IOKey(name="output2")
    )
    model |= Layer(dimension=71, activation=Sigmoid()).connect(
        input="output2", weight="weight2", output=IOKey(name="output3")
    )

    model_dict_created = dict_conversions.model_to_dict(model)
    model_recreated = dict_conversions.dict_to_model(model_dict_created)
    model_dict_recreated = dict_conversions.model_to_dict(model_recreated)
    assert model_dict_created == model_dict_recreated
    assert_models_equal(model, model_recreated)

    backend = JaxBackend(dtype=mithril.float64)
    assert_evaluations_equal(
        model,
        model_recreated,
        backend,
        static_keys={"input": backend.ones([4, 256])},
    )


def test_composite_7():
    model = Model()
    model |= (l1 := Linear(dimension=10)).connect(
        input="my_input", weight="weight", output=IOKey(name="output")
    )
    model |= Linear(dimension=71).connect(
        input="input2", weight="weight1", output=l1.input
    )

    model_dict_created = dict_conversions.model_to_dict(model)
    model_recreated = dict_conversions.dict_to_model(model_dict_created)
    model_dict_recreated = dict_conversions.model_to_dict(model_recreated)
    assert model_dict_created == model_dict_recreated
    assert_models_equal(model, model_recreated)

    backend = JaxBackend(dtype=mithril.float64)
    assert_evaluations_equal(
        model, model_recreated, backend, static_keys={"input2": backend.ones([4, 256])}
    )


def test_composite_8():
    model = Model()
    model |= (l1 := Linear(dimension=10)).connect(
        weight="weight", output=IOKey(name="output")
    )
    model |= Linear(dimension=71).connect(
        input="input2", weight="weight1", output=l1.input
    )

    model_dict_created = dict_conversions.model_to_dict(model)
    model_recreated = dict_conversions.dict_to_model(model_dict_created)
    model_dict_recreated = dict_conversions.model_to_dict(model_recreated)
    assert model_dict_created == model_dict_recreated
    assert_models_equal(model, model_recreated)

    backend = JaxBackend(dtype=mithril.float64)
    assert_evaluations_equal(
        model,
        model_recreated,
        backend,
        static_keys={"input2": backend.ones([4, 256])},
    )


def test_composite_9():
    model = Model()
    model |= (l1 := Linear(dimension=10)).connect(
        weight="weight", output=IOKey(name="output")
    )
    model |= (l2 := Linear(dimension=10)).connect(
        weight="weight1", output=IOKey(name="output2")
    )
    model.merge_connections(l1.input, l2.input)
    model |= Linear(dimension=71).connect(
        input="input", weight="weight2", output=l2.input
    )

    model_dict_created = dict_conversions.model_to_dict(model)
    model_recreated = dict_conversions.dict_to_model(model_dict_created)
    model_dict_recreated = dict_conversions.model_to_dict(model_recreated)

    assert model_dict_created == model_dict_recreated
    assert_models_equal(model, model_recreated)

    backend = JaxBackend(dtype=mithril.float64)
    assert_evaluations_equal(
        model, model_recreated, backend, static_keys={"input": backend.ones([4, 256])}
    )


def test_composite_10():
    model = Model()
    model |= Linear(dimension=10).connect(
        input="input2", weight="weight", output=IOKey(name="output")
    )
    model |= Linear(dimension=10).connect(
        input="input1", weight="weight1", output=IOKey(name="output2")
    )
    model.merge_connections("input1", "input2", name="my_input")
    model |= Linear(dimension=71).connect(
        input="input", weight="weight2", output="my_input"
    )

    model_dict_created = dict_conversions.model_to_dict(model)
    model_recreated = dict_conversions.dict_to_model(model_dict_created)
    model_dict_recreated = dict_conversions.model_to_dict(model_recreated)

    assert model_dict_created == model_dict_recreated
    assert_models_equal(model, model_recreated)

    backend = JaxBackend(dtype=mithril.float64)
    assert_evaluations_equal(
        model, model_recreated, backend, static_keys={"input": backend.ones([4, 256])}
    )


def test_composite_10_expose_false():
    model = Model()
    model |= Linear(dimension=10).connect(
        input="input2", weight="weight", output=IOKey(name="output")
    )
    model |= Linear(dimension=10).connect(
        input="input1", weight="weight1", output=IOKey(name="output2")
    )
    model.merge_connections("input1", "input2", name="my_input")
    model |= Linear(dimension=71).connect(
        input="input", weight="weight2", output="my_input"
    )

    model_dict_created = dict_conversions.model_to_dict(model)
    model_recreated = dict_conversions.dict_to_model(model_dict_created)
    model_dict_recreated = dict_conversions.model_to_dict(model_recreated)

    assert model_dict_created == model_dict_recreated
    assert_models_equal(model, model_recreated)

    backend = JaxBackend(dtype=mithril.float64)
    assert_evaluations_equal(
        model, model_recreated, backend, static_keys={"input": backend.ones([4, 256])}
    )


def test_composite_11():
    mlp_model = MLP(activations=[Relu(), Relu(), Relu()], dimensions=[12, 24, None])
    model = TrainModel(mlp_model)
    model.add_loss(
        SquaredError(),
        input=mlp_model.cout,
        target=Tensor([[2.2, 4.2], [2.2, 4.2]]),
        reduce_steps=[Mean()],
    )

    context_dict = dict_conversions.model_to_dict(model)
    context_recreated = dict_conversions.dict_to_model(context_dict)
    context_dict_recreated = dict_conversions.model_to_dict(context_recreated)

    assert context_dict == context_dict_recreated
    assert_models_equal(model, context_recreated)


def test_composite_12():
    # Case where submodel output keys only named
    model = Model()
    model |= Linear(dimension=10).connect(
        input="input2", weight="weight", output="output"
    )
    model |= Linear(dimension=10).connect(
        input="input1", weight="weight1", output="output2"
    )
    model.merge_connections("input1", "input2", name="my_input")
    model |= Linear(dimension=71).connect(
        input="input", weight="weight2", output="my_input"
    )
    model.set_cout("output2")

    model_dict_created = dict_conversions.model_to_dict(model)
    model_recreated = dict_conversions.dict_to_model(model_dict_created)
    model_dict_recreated = dict_conversions.model_to_dict(model_recreated)

    assert model_dict_created == model_dict_recreated

    backend = JaxBackend(dtype=mithril.float64)
    assert_evaluations_equal(
        model, model_recreated, backend, static_keys={"input": backend.ones([4, 256])}
    )


def test_composite_13():
    # Case where submodel output keys IOKey but not exposed
    model = Model()
    model |= Linear(dimension=10).connect(
        input="input2",
        weight="weight",
        output="output",
    )
    model |= Linear(dimension=10).connect(
        input="input1",
        weight="weight1",
        output="output2",
    )
    model.merge_connections("input1", "input2", name="my_input")
    model |= Linear(dimension=71).connect(
        input="input",
        weight="weight2",
        output="my_input",
    )
    model.set_cout("output2")

    model_dict_created = dict_conversions.model_to_dict(model)
    model_recreated = dict_conversions.dict_to_model(model_dict_created)
    model_dict_recreated = dict_conversions.model_to_dict(model_recreated)

    assert model_dict_created == model_dict_recreated

    backend = JaxBackend(dtype=mithril.float64)
    assert_evaluations_equal(
        model, model_recreated, backend, static_keys={"input": backend.ones([4, 256])}
    )


def test_basic_extend_from_input():
    model = Model()
    model |= Linear(dimension=10).connect(
        input="lin", weight="weight", output=IOKey(name="output")
    )
    model |= Linear(dimension=71).connect(input="input", weight="weight1", output="lin")

    model_dict_created = dict_conversions.model_to_dict(model)
    model_recreated = dict_conversions.dict_to_model(model_dict_created)
    model_dict_recreated = dict_conversions.model_to_dict(model_recreated)

    assert model_dict_created == model_dict_recreated

    backend = JaxBackend(dtype=mithril.float64)
    assert_evaluations_equal(
        model, model_recreated, backend, static_keys={"input": backend.ones([4, 256])}
    )


def test_auto_iadd_1():
    model = Model()
    model |= Sigmoid().connect(input="input", output="output")
    model |= Sigmoid().connect(output="output2")
    model.expose_keys("output")
    model_dict_created = dict_conversions.model_to_dict(model)
    model_recreated = dict_conversions.dict_to_model(model_dict_created)
    model_dict_recreated = dict_conversions.model_to_dict(model_recreated)

    assert model_dict_created == model_dict_recreated
    assert_models_equal(model, model_recreated)

    backend = JaxBackend(dtype=mithril.float64)
    assert_evaluations_equal(
        model,
        model_recreated,
        backend,
        static_keys={"input": backend.ones([4, 256])},
        inference=True,
    )


def test_auto_iadd_2():
    model = Model()
    model |= Sigmoid().connect(input="input", output="output")
    model |= Sigmoid().connect(output="output2")
    model.expose_keys("output")
    model_dict_created = dict_conversions.model_to_dict(model)
    model_recreated = dict_conversions.dict_to_model(model_dict_created)
    model_dict_recreated = dict_conversions.model_to_dict(model_recreated)

    assert model_dict_created == model_dict_recreated
    assert_models_equal(model, model_recreated)

    backend = JaxBackend(dtype=mithril.float64)
    assert_evaluations_equal(
        model,
        model_recreated,
        backend,
        static_keys={"input": backend.ones([4, 256])},
        inference=True,
    )


def test_convolution():
    model = Model()
    model |= Convolution2D(kernel_size=3, out_channels=20).connect(
        input="input", output=IOKey(name="output")
    )

    model_dict_created = dict_conversions.model_to_dict(model)
    model_recreated = dict_conversions.dict_to_model(model_dict_created)
    model_dict_recreated = dict_conversions.model_to_dict(model_recreated)
    assert model_dict_created == model_dict_recreated
    assert_models_equal(model, model_recreated)

    backend = JaxBackend(dtype=mithril.float64)
    assert_evaluations_equal(
        model,
        model_recreated,
        backend,
        static_keys={"input": backend.ones([4, 3, 32, 32])},
    )


def test_tbd():
    model = Model()
    model += Convolution2D(kernel_size=3, out_channels=20, stride=TBD).connect(
        input="input", output=IOKey(name="output"), stride=(1, 1)
    )

    model_dict_created = dict_conversions.model_to_dict(model)
    model_recreated = dict_conversions.dict_to_model(model_dict_created)
    model_dict_recreated = dict_conversions.model_to_dict(model_recreated)
    assert model_dict_created == model_dict_recreated
    assert_models_equal(model, model_recreated)

    backend = JaxBackend(dtype=mithril.float64)
    assert_evaluations_equal(
        model,
        model_recreated,
        backend,
        static_keys={"input": backend.ones([4, 3, 32, 32])},
    )


def test_train_context_1():
    model = Model()
    layer1 = Linear(dimension=16)
    layer2 = Linear(dimension=10)

    model |= layer1.connect(input="input", weight="weight0", bias="bias0")
    model |= layer2.connect(
        input=layer1.output, weight="weight1", bias="bias1", output="output"
    )
    model.expose_keys("output")

    context = TrainModel(model)
    context.add_loss(CrossEntropy(), [Mean()], target="target", input=model.cout)
    context_dict = dict_conversions.model_to_dict(context)
    context_recreated = dict_conversions.dict_to_model(context_dict)
    context_dict_recreated = dict_conversions.model_to_dict(context_recreated)

    assert context_dict == context_dict_recreated
    assert_models_equal(context, context_recreated)

    backend = TorchBackend(dtype=mithril.float64)
    assert_evaluations_equal(
        context,
        context_recreated,
        backend,
        static_keys={
            "input": backend.ones([4, 32]),
            "target": backend.ones([4], dtype=mithril.int64),
        },
    )


def test_train_context_2():
    model = Model()
    layer1 = Linear(dimension=16)
    layer2 = Linear(dimension=10)

    model |= layer1.connect(weight="weight0", bias="bias0", input="input")
    model |= layer2.connect(
        input=layer1.output, weight="weight1", bias="bias1", output="output"
    )
    model.expose_keys("output")

    context = TrainModel(model)
    context.add_loss(CrossEntropy(), [Mean()], target="target", input=model.cout)
    context.add_regularization(
        model=L2(), coef=Tensor(1e-1), input=re.compile("weight\\d")
    )
    context_dict = dict_conversions.model_to_dict(context)
    context_recreated = dict_conversions.dict_to_model(context_dict)
    context_dict_recreated = dict_conversions.model_to_dict(context_recreated)
    assert context_dict == context_dict_recreated
    assert_models_equal(context, context_recreated)

    backend = TorchBackend(dtype=mithril.float64)
    assert_evaluations_equal(
        context,
        context_recreated,
        backend,
        static_keys={
            "input": backend.ones([4, 32]),
            "target": backend.ones([4], dtype=mithril.types.Dtype.int64),
        },
    )


def test_set_values_constant_1():
    # Set value using IOKey
    model = Model()
    model |= Linear(10).connect(
        weight="weight0",
        bias="bias0",
        input="input",
        output="output",
    )
    model |= Linear(1).connect(
        weight="weight1",
        bias=IOKey(value=Tensor([123.0]), name="bias1"),
        input="input2",
        output="output2",
    )
    model.expose_keys("output2")

    model_dict_created = dict_conversions.model_to_dict(model)
    model_recreated = dict_conversions.dict_to_model(model_dict_created)
    model_dict_recreated = dict_conversions.model_to_dict(model_recreated)

    assert model_dict_created == model_dict_recreated
    assert_models_equal(model, model_recreated)

    backend = JaxBackend(dtype=mithril.float64)
    assert_evaluations_equal(
        model,
        model_recreated,
        backend,
        static_keys={"input": backend.ones([10, 4]), "input2": backend.ones([10, 4])},
    )


def test_set_values_constant_2():
    # Set value using set_values api
    model = Model()
    model |= Linear(10).connect(
        weight="weight0",
        bias="bias0",
        input="input",
        output="output",
    )
    model |= Linear(1).connect(
        weight="weight1",
        bias="bias1",
        input="input2",
        output="output2",
    )
    model.expose_keys("output2")
    model.set_values(bias1=Tensor([123.0]))

    model_dict_created = dict_conversions.model_to_dict(model)
    model_recreated = dict_conversions.dict_to_model(model_dict_created)
    model_dict_recreated = dict_conversions.model_to_dict(model_recreated)

    assert model_dict_created == model_dict_recreated
    assert_models_equal(model, model_recreated)

    backend = JaxBackend(dtype=mithril.float64)
    assert_evaluations_equal(
        model,
        model_recreated,
        backend,
        static_keys={"input": backend.ones([10, 4]), "input2": backend.ones([10, 4])},
    )


def test_set_values_tbd_1():
    model = Model()
    model |= Linear(10).connect(
        weight="weight0",
        bias="bias0",
        input="input",
        output=IOKey(name="output"),
    )
    model |= Linear(1).connect(
        weight="weight1", bias=IOKey(value=TBD, name="bias1"), input="input2"
    )

    model_dict_created = dict_conversions.model_to_dict(model)
    model_recreated = dict_conversions.dict_to_model(model_dict_created)
    model_dict_recreated = dict_conversions.model_to_dict(model_recreated)

    assert model_dict_created == model_dict_recreated
    assert_models_equal(model, model_recreated)


def test_set_values_ellipsis_2():
    model = Model()
    model |= Linear(10).connect(
        weight="weight0",
        bias="bias0",
        input="input",
        output=IOKey(name="output"),
    )
    lin2 = Linear(1)
    model |= lin2.connect(weight="weight1", bias="bias1", input="input2")
    lin2.set_differentiability(bias=False)

    model_dict_created = dict_conversions.model_to_dict(model)
    model_recreated = dict_conversions.dict_to_model(model_dict_created)
    model_dict_recreated = dict_conversions.model_to_dict(model_recreated)

    assert model_dict_created == model_dict_recreated
    assert_models_equal(model, model_recreated)


@pytest.mark.skip(reason="Waiting for the fix in the conversion bug")
def test_make_shape_constraint():
    model = Model()

    def my_adder(input, rhs):
        return input + rhs

    TorchBackend.register_primitive(my_adder)  # After serialization is this available?

    class MyAdder(PrimitiveModel):
        def __init__(self, threshold=3) -> None:
            threshold *= 2
            super().__init__(
                formula_key="my_adder",
                output=BaseKey(shape=[("Var_out", ...)], type=Tensor),
                input=BaseKey(shape=[("Var_1", ...)], type=Tensor),
                rhs=BaseKey(type=int, value=threshold),
            )
            self.add_constraint(
                fn=squeeze_constraints, keys=[Operator.output_key, "input"]
            )

    model += MyAdder().connect(input="input")
    # model.extend(MyAdder(), input = "input")
    model.set_shapes(input=[1, 128, 1, 8, 16])

    model_dict_created = dict_conversions.model_to_dict(model)
    model_recreated = dict_conversions.dict_to_model(model_dict_created)
    model_dict_recreated = dict_conversions.model_to_dict(model_recreated)

    # TODO: Handle TensorType and Scalar conversions!!!
    assert model_dict_created == model_dict_recreated
    assert model.shapes == model_recreated.shapes
    assert_models_equal(model, model_recreated)
    TorchBackend.registered_primitives.pop("my_adder")


def test_valued_scalar_in_init():
    model = Model()
    model |= Buffer().connect(input="buff_input", output=IOKey(name="buff_out"))
    model |= Mean().connect(input="mean_input", output=IOKey(name="mean_out"))
    outer_model = Model()
    outer_model |= model.connect()

    model_dict_created = dict_conversions.model_to_dict(outer_model)
    model_recreated = dict_conversions.dict_to_model(model_dict_created)
    model_dict_recreated = dict_conversions.model_to_dict(model_recreated)

    assert model_dict_created == model_dict_recreated
    assert_models_equal(outer_model, model_recreated)


def test_valued_scalar_in_extend():
    model = Model()
    model |= Buffer().connect(input="buff_input", output=IOKey(name="buff_out"))
    model |= Mean(axis=TBD).connect(
        input="mean_input", axis=1, output=IOKey(name="mean_out")
    )
    outer_model = Model()
    outer_model |= model.connect()

    model_dict_created = dict_conversions.model_to_dict(outer_model)
    model_recreated = dict_conversions.dict_to_model(model_dict_created)
    model_dict_recreated = dict_conversions.model_to_dict(model_recreated)

    assert model_dict_created == model_dict_recreated
    assert_models_equal(outer_model, model_recreated)


def test_valued_scalar_iokey():
    model = Model()
    model |= Buffer().connect(input="buff_input", output="buff_out")
    model |= Mean(axis=TBD).connect(input="mean_input", axis="axis", output="mean_out")
    model.expose_keys("buff_out", "mean_out")
    outer_model = Model()
    outer_model |= model.connect(axis=IOKey(name="axis", value=1))

    model_dict_created = dict_conversions.model_to_dict(outer_model)
    model_recreated = dict_conversions.dict_to_model(model_dict_created)
    model_dict_recreated = dict_conversions.model_to_dict(model_recreated)

    assert model_dict_created == model_dict_recreated
    assert_models_equal(outer_model, model_recreated)


def test_non_valued_scalar():
    model = Model()
    model |= Buffer().connect(input="buff_input", output=IOKey(name="buff_out"))
    model |= Mean(axis=TBD).connect(input="mean_input", output=IOKey(name="mean_out"))
    outer_model = Model()
    outer_model |= model.connect()

    model_dict_created = dict_conversions.model_to_dict(outer_model)
    model_recreated = dict_conversions.dict_to_model(model_dict_created)
    model_dict_recreated = dict_conversions.model_to_dict(model_recreated)

    assert model_dict_created == model_dict_recreated
    assert_models_equal(outer_model, model_recreated)


def test_assigned_shapes():
    model = Model()
    model |= Buffer().connect(input="buff_input", output=IOKey(name="buff_out"))
    model |= Mean(axis=TBD).connect(input="mean_input", output=IOKey(name="mean_out"))
    model.set_shapes(buff_input=[1, 2, ("V", ...)], mean_input=[("V", ...), 3, 4])

    model_dict_created = dict_conversions.model_to_dict(model)
    model_recreated = dict_conversions.dict_to_model(model_dict_created)
    model_dict_recreated = dict_conversions.model_to_dict(model_recreated)

    assert model_dict_created == model_dict_recreated
    assert_models_equal(model, model_recreated)

    assert (
        model_dict_created.get("assigned_shapes")
        == model_dict_recreated.get("assigned_shapes")
        == [
            [(("m_0", 0), [1, 2, "V,..."]), (("m_1", 0), ["V,...", 3, 4])],
        ]
    )


def test_assigned_types_1():
    model = Model()
    model |= Buffer().connect(input="buff_input", output=IOKey(name="buff_out"))
    model |= Mean(axis=TBD).connect(input="mean_input", output=IOKey(name="mean_out"))
    model.set_types(mean_input=Tensor[int | float])

    model_dict_created = dict_conversions.model_to_dict(model)
    model_recreated = dict_conversions.dict_to_model(model_dict_created)
    model_dict_recreated = dict_conversions.model_to_dict(model_recreated)

    assert model_dict_created == model_dict_recreated
    assert_models_equal(model, model_recreated)

    assert (
        model_dict_created.get("assigned_types")
        == model_dict_recreated.get("assigned_types")
        == [
            (("m_1", 0), {"Tensor": ["int", "float"]}),
        ]
    )


def test_assigned_types_2():
    model = Model()
    model |= Buffer().connect(input="buff_input", output=IOKey(name="buff_out"))
    model |= Mean(axis=TBD).connect(input="mean_input", output=IOKey(name="mean_out"))
    model.set_types(mean_input=Tensor[int | float])

    outer_model = Model()
    outer_model |= model

    model_dict_created = dict_conversions.model_to_dict(outer_model)
    model_recreated = dict_conversions.dict_to_model(model_dict_created)
    model_dict_recreated = dict_conversions.model_to_dict(model_recreated)

    assert model_dict_created == model_dict_recreated
    assert_models_equal(outer_model, model_recreated)

    assert (
        model_dict_created.get("assigned_types")
        == model_dict_recreated.get("assigned_types")
        == []
    )

    assert (
        model_dict_created["submodels"]["m_0"].get("assigned_types")  # type: ignore
        == model_dict_recreated["submodels"]["m_0"].get("assigned_types")  # type: ignore
        == [
            ("mean_input", {"Tensor": ["int", "float"]}),
        ]
    )


def test_assigned_types_multiple_times():
    model = Model()
    model |= Buffer().connect(input="buff_input", output=IOKey(name="buff_out"))
    mean_model = Mean(axis=TBD)
    model |= mean_model.connect(input="mean_input", output=IOKey(name="mean_out"))
    model.set_types(mean_input=Tensor[int | float])
    model.set_types({mean_model.input: Tensor[int | float]})

    outer_model = Model()
    outer_model |= model

    # Assert only one assignment made even thought set multiple
    # times.
    assert len(model.assigned_types) == 1

    model_dict_created = dict_conversions.model_to_dict(outer_model)
    model_recreated = dict_conversions.dict_to_model(model_dict_created)
    model_dict_recreated = dict_conversions.model_to_dict(model_recreated)

    assert model_dict_created == model_dict_recreated
    assert_models_equal(outer_model, model_recreated)

    assert (
        model_dict_created.get("assigned_types")
        == model_dict_recreated.get("assigned_types")
        == []
    )

    assert (
        model_dict_created["submodels"]["m_0"].get("assigned_types")  # type: ignore
        == model_dict_recreated["submodels"]["m_0"].get("assigned_types")  # type: ignore
        == [
            ("mean_input", {"Tensor": ["int", "float"]}),
        ]
    )


def test_assigned_types_multiple_times_different_types():
    model = Model()
    buff_model = Buffer()
    model |= buff_model.connect(input="buff_input", output=IOKey(name="buff_out"))
    mean_model = Mean(axis=TBD)
    model |= mean_model.connect(input="mean_input", output=IOKey(name="mean_out"))
    # Set types for buff_model 2 times with different types.
    # Note that the last assignment will be used.
    model.set_types(buff_input=Tensor[int | float] | int | float)
    model.set_types({buff_model.input: int})

    outer_model = Model()
    outer_model |= model

    # Assert only one assignment made even thought set multiple
    # times.
    assert len(model.assigned_types) == 1

    model_dict_created = dict_conversions.model_to_dict(outer_model)
    model_recreated = dict_conversions.dict_to_model(model_dict_created)
    model_dict_recreated = dict_conversions.model_to_dict(model_recreated)

    assert model_dict_created == model_dict_recreated
    assert_models_equal(outer_model, model_recreated)

    assert (
        model_dict_created.get("assigned_types")
        == model_dict_recreated.get("assigned_types")
        == []
    )

    assert (
        model_dict_created["submodels"]["m_0"].get("assigned_types")  # type: ignore
        == model_dict_recreated["submodels"]["m_0"].get("assigned_types")  # type: ignore
        == [
            ("buff_input", "int"),
        ]
    )


def test_assigned_types_from_outermost_model():
    model = Model()
    buff_model = Buffer()
    model |= buff_model.connect(input="buff_input", output=IOKey(name="buff_out"))
    model |= Mean(axis=TBD).connect(input="mean_input", output=IOKey(name="mean_out"))

    outer_model = Model()
    outer_model |= model
    outer_model |= Buffer().connect(input="buff_input_2")
    outer_model.set_types(buff_input_2=Tensor)
    outer_model.merge_connections(buff_model.input, "buff_input_2")

    model_dict_created = dict_conversions.model_to_dict(outer_model)
    model_recreated = dict_conversions.dict_to_model(model_dict_created)
    model_dict_recreated = dict_conversions.model_to_dict(model_recreated)

    assert model_dict_created == model_dict_recreated
    assert_models_equal(outer_model, model_recreated)

    assert (
        model_dict_created.get("assigned_types")
        == model_dict_recreated.get("assigned_types")
        == [(("m_0", 0), {"Tensor": ["int", "float", "bool"]})]
    )


def test_assigned_constant_enum_value():
    model = Model()
    model |= Sqrt(robust=True, threshold=TBD).connect(
        threshold=Tensor(Constant.MIN_POSITIVE_SUBNORMAL)
    )

    model_dict_created = dict_conversions.model_to_dict(model)
    model_recreated = dict_conversions.dict_to_model(model_dict_created)
    model_dict_recreated = dict_conversions.model_to_dict(model_recreated)

    assert model_dict_created == model_dict_recreated

    assert (
        model_dict_created["connections"]["m_0"]["threshold"]  # type: ignore
        == model_dict_recreated["connections"]["m_0"]["threshold"]  # type: ignore
        == {"tensor": "Constant.MIN_POSITIVE_SUBNORMAL"}
    )

    assert_models_equal(model, model_recreated)


def test_assigned_dtype_enum_value():
    model = Model()
    model |= ToTensor(dtype=TBD).connect(dtype=Dtype.float16)

    model_dict_created = dict_conversions.model_to_dict(model)
    model_recreated = dict_conversions.dict_to_model(model_dict_created)
    model_dict_recreated = dict_conversions.model_to_dict(model_recreated)

    assert model_dict_created == model_dict_recreated

    assert (
        model_dict_created["connections"]["m_0"]["dtype"]  # type: ignore
        == model_dict_recreated["connections"]["m_0"]["dtype"]  # type: ignore
        == "5"
    )

    assert_models_equal(model, model_recreated)


def test_assigned_int_value():
    model = Model()
    model |= Mean(axis=TBD).connect(axis=3)

    model_dict_created = dict_conversions.model_to_dict(model)
    model_recreated = dict_conversions.dict_to_model(model_dict_created)
    model_dict_recreated = dict_conversions.model_to_dict(model_recreated)

    assert model_dict_created == model_dict_recreated

    assert (
        model_dict_created["connections"]["m_0"]["axis"]  # type: ignore
        == model_dict_recreated["connections"]["m_0"]["axis"]  # type: ignore
        == 3
    )

    assert_models_equal(model, model_recreated)
