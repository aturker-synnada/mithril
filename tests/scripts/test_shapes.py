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

from collections.abc import Mapping, Sequence
from copy import deepcopy
from itertools import combinations, product
from types import EllipsisType, NoneType

import numpy as np
import pytest

import mithril
import mithril.framework
from mithril import NumpyBackend, TorchBackend, compile
from mithril.framework.common import (
    AND,
    DNF,
    NOT_GIVEN,
    Equivalences,
    PossibleValues,
    ShapeNode,
    ShapeRepr,
    Tensor,
    ToBeDetermined,
    Uniadic,
    UniadicRecord,
    Updates,
    Variadic,
)
from mithril.framework.constraints import reverse_constraints
from mithril.framework.logical.base import BaseKey
from mithril.framework.logical.primitive import OperatorModel, PrimitiveModel
from mithril.models import (
    AUC,
    MLP,
    TBD,
    Absolute,
    Accuracy,
    Activation,
    Add,
    Arange,
    AUCCore,
    BaseModel,
    BinaryCrossEntropy,
    BroadcastTo,
    Buffer,
    Cast,
    Cholesky,
    Concat,
    Connection,
    ConnectionType,
    ConstraintSolver,
    Convolution1D,
    Convolution2D,
    Cosine,
    CrossEntropy,
    CustomPrimitiveModel,
    Divide,
    Eigvalsh,
    Exponential,
    ExtendInfo,
    Flatten,
    Floor,
    Gelu,
    GPRAlpha,
    GPRVOuter,
    IOKey,
    IsNan,
    Layer,
    LeakyRelu,
    Linear,
    Log,
    LogicalNot,
    MatrixMultiply,
    MaxPool1D,
    MaxPool2D,
    Mean,
    Model,
    Multiply,
    NanToNum,
    Negate,
    NormModifier,
    Operator,
    Pad,
    PermuteTensor,
    PositionalEncoding,
    PrimitiveUnion,
    Relu,
    Reshape,
    ScaledDotProduct,
    Shape,
    Sigmoid,
    Sign,
    Sine,
    Size,
    Softmax,
    Softplus,
    Sqrt,
    Square,
    SquaredError,
    Squeeze,
    StableReciprocal,
    Sum,
    SwapAxes,
    Tanh,
    ToList,
    ToTuple,
    TrainModel,
    Transpose,
    Trapezoid,
    TsnePJoint,
    Unique,
    ZerosLike,
    primitives,
)

from .test_utils import (
    assert_shape_results,
    check_shapes_semantically,
    check_single_shape_semantically,
    get_all_nodes,
    get_all_reprs,
    get_all_symbols,
)


def assert_shapes(
    model: Model,
    logical_ref: Mapping[str, Sequence[Sequence[int | str] | int | str] | None],
    physical_ref: Mapping[str, Sequence[Sequence[int | str] | int | str] | None]
    | None = None,
    *,
    shapes: Mapping[str, Sequence[int | None]] | None = None,
    static_inputs: dict[str, np.ndarray] | None = None,
    inference: bool = True,
    check_all_shapes=True,
):
    # All different nodes should have different shapes
    assert_all_nodes_unique(model)

    # All different shaperepr objects should have different shapes
    assert_all_reprs_unique(model)

    # all uniadics with same integer value should be the same objects
    assert_all_integer_uniadics_unique(model)

    if physical_ref is not None:
        comp_model = mithril.compile(
            model=model,
            backend=NumpyBackend(),
            shapes=shapes,
            constant_keys=static_inputs,
            safe_shapes=True,
            inference=inference,
            safe_names=False,
        )

    m_shapes = model.shapes if not check_all_shapes else model.get_shapes(verbose=True)
    assert isinstance(m_shapes, dict)
    m_shapes.pop("final_cost", None)
    check_shapes_semantically(m_shapes, logical_ref)  # type: ignore
    if physical_ref is None:
        return
    # If shapes are given in shapes, simply set model's shapes using it.
    # Otherwise get corresponding shapes from static_inputs
    if shapes is not None:
        assert isinstance(shapes, dict)
        model.set_shapes(**shapes)

    if static_inputs is not None:
        input_shapes = {key: value.shape for key, value in static_inputs.items()}
        model.set_shapes(**input_shapes)

    comp_shapes = {
        key: value
        for key, value in comp_model.shapes.items()
        if "_cache" not in key and key != "final_cost"
    }
    check_shapes_semantically(comp_shapes, physical_ref)  # type: ignore


def repr_sort(repr):
    return len(repr.prefix)


def get_deterministic_shape(node: ShapeNode):
    uni: dict[UniadicRecord, str] = {}
    var: dict[Variadic, str] = {}
    if len(reprs := node.reprs) != 1:
        sorted_reprs = sorted(reprs, key=repr_sort, reverse=True)
        return [repr.get_shapes(uni, var) for repr in sorted_reprs]
    else:
        return node.get_shapes(uni, var)


# def get_all_nodes(model: BaseModel):
#     return {con.metadata.shape for con in model.conns.all.values()}


def assert_all_nodes_unique(model: BaseModel):
    """Asserts if all nodes in a given model is unique.
    Extract all unique nodes found in a model, After all unique nodes
    are extracted, compare each node two by two. It is expected that all shapes found in
    all unique nodes are also unique. If not, Raises assertion error
    """
    all_nodes = get_all_nodes(model)

    uni_cache: dict[UniadicRecord, str] = {}
    var_cache: dict[Variadic, str] = {}

    for node1, node2 in combinations(all_nodes, 2):
        node1_shapes = node1.get_shapes(uni_cache, var_cache, verbose=True)
        node2_shapes = node2.get_shapes(uni_cache, var_cache, verbose=True)

        if len(node1.reprs) == 1:
            node1_shapes = [node1_shapes]

        if len(node2.reprs) == 1:
            node2_shapes = [node2_shapes]

        for shape_1, shape_2 in product(node1_shapes, node2_shapes):
            assert shape_1 != shape_2


def assert_all_reprs_unique(model: BaseModel):
    """Asserts if all ShapeRepr objects in a given model is unique.
    Extract all unique ShapeReprs found in a model, After all unique ShapeReprs
    are extracted, compare each ShapeRepr. It is expected that shape of
    each different ShapeRepr object is unique. If not, Raises assertion error
    """
    all_reprs = get_all_reprs(model)

    uni_cache: dict[UniadicRecord, str] = {}
    var_cache: dict[Variadic, str] = {}

    for repr1, repr2 in combinations(all_reprs, 2):
        repr1_shapes = repr1.get_shapes(uni_cache, var_cache)
        repr2_shapes = repr2.get_shapes(uni_cache, var_cache)

        if repr1_shapes:
            assert repr1_shapes != repr2_shapes


def assert_all_integer_uniadics_unique(model: BaseModel):
    """Checks if all integer uniadics have the same uniadic record
    Extracting all symbols in the given model, checks if all uniadics
    with the same integers have same uniadic records. Raises assertion
    error otherwise.
    """

    all_symbols = get_all_symbols(model)
    integer_symbol_dict: dict[int, UniadicRecord | None] = {}
    for symbol in all_symbols:
        if isinstance(symbol, Uniadic) and (val := symbol.value) is not None:
            record = integer_symbol_dict.get(val)
            if record is not None:
                assert symbol.metadata is record
            else:
                integer_symbol_dict[val] = record


def assert_match_shapes(
    repr1: ShapeRepr, repr2: ShapeRepr, repr1_ref_shapes: list, repr2_ref_shapes: list
):
    uni_cache: dict[UniadicRecord, str] = {}
    var_cache: dict[Variadic, str] = {}

    repr1.match(repr2)

    ref_shapes = {"repr1": repr1_ref_shapes, "repr2": repr2_ref_shapes}

    shapes = {
        "repr1": repr1.node.get_shapes(
            verbose=True, u_keys=uni_cache, v_keys=var_cache
        ),
        "repr2": repr2.node.get_shapes(
            verbose=True, u_keys=uni_cache, v_keys=var_cache
        ),
    }

    check_shapes_semantically(ref_shapes, shapes)


def test_shapes_1():
    # TODO: What is the purpose of this test?
    model = Model()
    model |= (add1 := Add()).connect(left="left", right="right")
    model |= Add().connect(
        left=add1.output, right=add1.output, output=IOKey(name="output")
    )
    model.set_shapes(left=[3, 4, 5, 1], right=[1, 7])
    logical_ref = {
        "$_Add_0_output": [3, 4, 5, 7],
        "left": [3, 4, 5, 1],
        "right": [1, 7],
        "output": [3, 4, 5, 7],
    }
    physical_ref = {
        "output_0": [3, 4, 5, 7],
        "left": [3, 4, 5, 1],
        "right": [1, 7],
        "output": [3, 4, 5, 7],
    }
    assert_shapes(model, logical_ref, physical_ref)


def test_shapes_2():
    model = Model()
    model += Convolution2D(kernel_size=3, out_channels=64, padding=1).connect(
        input="input"
    )
    model += Convolution2D(kernel_size=3, out_channels=64, padding=1)
    model += Convolution2D(kernel_size=3, out_channels=64, padding=1)
    model += Convolution2D(kernel_size=3, out_channels=64, padding=1)
    model |= Convolution2D(kernel_size=3, out_channels=64, padding=1).connect(
        input=model.cout, output=IOKey(name="output")
    )

    shapes = {"input": [8, 3, 64, 64]}
    logical_ref: Mapping[str, list | None] = {
        "input": ["u1", "u2", "u3", "u4"],
        "$weight_0": [64, "u2", 3, 3],
        "$bias_0": [1, 64, 1, 1],
        "$_Convolution2D_0_output": ["u1", 64, "u5", "u6"],
        "$weight_1": [64, 64, 3, 3],
        "$bias_1": [1, 64, 1, 1],
        "$_Convolution2D_1_output": ["u1", 64, "u7", "u8"],
        "$weight_2": [64, 64, 3, 3],
        "$bias_2": [1, 64, 1, 1],
        "$_Convolution2D_2_output": ["u1", 64, "u9", "u10"],
        "$weight_3": [64, 64, 3, 3],
        "$bias_3": [1, 64, 1, 1],
        "$_Convolution2D_3_output": ["u1", 64, "u11", "u12"],
        "$weight_4": [64, 64, 3, 3],
        "$bias_4": [1, 64, 1, 1],
        "output": ["u1", 64, "u13", "u14"],
        "$_Convolution2D_0_padding": None,
        "$_Convolution2D_0_stride": None,
        "$_Convolution2D_0_dilation": None,
        "$_Convolution2D_0_groups": None,
        "$_Convolution2D_1_padding": None,
        "$_Convolution2D_1_stride": None,
        "$_Convolution2D_1_dilation": None,
        "$_Convolution2D_1_groups": None,
        "$_Convolution2D_2_padding": None,
        "$_Convolution2D_2_stride": None,
        "$_Convolution2D_2_dilation": None,
        "$_Convolution2D_2_groups": None,
        "$_Convolution2D_3_padding": None,
        "$_Convolution2D_3_stride": None,
        "$_Convolution2D_3_dilation": None,
        "$_Convolution2D_3_groups": None,
        "$_Convolution2D_4_padding": None,
        "$_Convolution2D_4_stride": None,
        "$_Convolution2D_4_dilation": None,
        "$_Convolution2D_4_groups": None,
    }
    physical_ref = {
        "weight_0": [64, 3, 3, 3],
        "output_4": None,
        "output_5": None,
        "output_6": None,
        "input": [8, 3, 64, 64],
        "bias_0": [1, 64, 1, 1],
        "output_7": [8, 64, 64, 64],
        "weight_1": [64, 64, 3, 3],
        "output_12": None,
        "output_13": None,
        "output_14": None,
        "bias_1": [1, 64, 1, 1],
        "output_15": [8, 64, 64, 64],
        "weight_2": [64, 64, 3, 3],
        "output_20": None,
        "output_21": None,
        "output_22": None,
        "bias_2": [1, 64, 1, 1],
        "output_23": [8, 64, 64, 64],
        "weight_3": [64, 64, 3, 3],
        "output_28": None,
        "output_29": None,
        "output_30": None,
        "bias_3": [1, 64, 1, 1],
        "output_31": [8, 64, 64, 64],
        "weight_4": [64, 64, 3, 3],
        "output_36": None,
        "output_37": None,
        "output_38": None,
        "bias_4": [1, 64, 1, 1],
        "output": [8, 64, 64, 64],
        "groups_0": None,
        "groups_1": None,
        "groups_2": None,
        "groups_3": None,
        "groups_4": None,
    }
    assert_shapes(model, logical_ref, physical_ref, shapes=shapes)


@pytest.mark.skip("Compiled model shapes may have missing keys. Investigate'")
def test_shapes_conv__():
    # TODO: Investigate why only stride exists in compiled
    # model shapes.
    from mithril import JaxBackend

    model = Model()
    model += Convolution2D(kernel_size=3, out_channels=64).connect(input="input")
    comp_model = mithril.compile(model, JaxBackend(), jit=False)
    assert comp_model.shapes


def test_shapes_3():
    submodel = Model()
    submodel |= Convolution2D(kernel_size=3, out_channels=64, padding=1)
    submodel += Convolution2D(kernel_size=3, out_channels=64, padding=0)
    submodel += Convolution2D(kernel_size=3, out_channels=64, padding=3, stride=2)

    model = Model()
    model += Convolution2D(kernel_size=3, out_channels=64, padding=1).connect(
        input="input"
    )  # 62x62, #33x33
    model += submodel  # 31x31, 18x18
    model += deepcopy(model)  # 16x16, 10x10
    model += deepcopy(model)  # 8x8, 6x6
    model |= Convolution2D(kernel_size=3, out_channels=64, padding=1).connect(
        input=model.cout, output=IOKey(name="output")
    )

    shapes = {"input": [8, 3, 64, 64]}
    logical_ref: Mapping[str, list | None] = {
        "$_Convolution2D_0_output": ["u1", 64, "u2", "u3"],
        "$_Model_1_output": ["u1", 64, "u4", "u5"],
        "$_Model_2_output": ["u1", 64, "u6", "u7"],
        "$_Model_3_output": ["u1", 64, "u8", "u9"],
        "$weight_0": [64, "u10", 3, 3],
        "input": ["u1", "u10", "u11", "u12"],
        "$bias_0": [1, 64, 1, 1],
        "$weight_1": [64, 64, 3, 3],
        "$bias_1": [1, 64, 1, 1],
        "$weight_2": [64, 64, 3, 3],
        "$bias_2": [1, 64, 1, 1],
        "$weight_3": [64, 64, 3, 3],
        "$bias_3": [1, 64, 1, 1],
        "$weight_4": [64, 64, 3, 3],
        "$bias_4": [1, 64, 1, 1],
        "$weight_5": [64, 64, 3, 3],
        "$bias_5": [1, 64, 1, 1],
        "$weight_6": [64, 64, 3, 3],
        "$bias_6": [1, 64, 1, 1],
        "$weight_7": [64, 64, 3, 3],
        "$bias_7": [1, 64, 1, 1],
        "$weight_8": [64, 64, 3, 3],
        "$bias_8": [1, 64, 1, 1],
        "$weight_9": [64, 64, 3, 3],
        "$bias_9": [1, 64, 1, 1],
        "$weight_10": [64, 64, 3, 3],
        "$bias_10": [1, 64, 1, 1],
        "$weight_11": [64, 64, 3, 3],
        "$bias_11": [1, 64, 1, 1],
        "$weight_12": [64, 64, 3, 3],
        "$bias_12": [1, 64, 1, 1],
        "$weight_13": [64, 64, 3, 3],
        "$bias_13": [1, 64, 1, 1],
        "$weight_14": [64, 64, 3, 3],
        "$bias_14": [1, 64, 1, 1],
        "$weight_15": [64, 64, 3, 3],
        "$bias_15": [1, 64, 1, 1],
        "$weight_16": [64, 64, 3, 3],
        "$bias_16": [1, 64, 1, 1],
        "output": ["u1", 64, "u13", "u14"],
        "$_Convolution2D_0_padding": None,
        "$_Convolution2D_0_stride": None,
        "$_Convolution2D_0_dilation": None,
        "$_Convolution2D_0_groups": None,
        "$_Convolution2D_4_padding": None,
        "$_Convolution2D_4_stride": None,
        "$_Convolution2D_4_dilation": None,
        "$_Convolution2D_4_groups": None,
    }

    physical_ref = {
        "weight_0": [64, 3, 3, 3],
        "output_4": None,
        "output_5": None,
        "output_6": None,
        "input": [8, 3, 64, 64],
        "bias_0": [1, 64, 1, 1],
        "output_7": [8, 64, 64, 64],
        "weight_1": [64, 64, 3, 3],
        "output_12": None,
        "output_13": None,
        "output_14": None,
        "bias_1": [1, 64, 1, 1],
        "output_15": [8, 64, 64, 64],
        "weight_2": [64, 64, 3, 3],
        "output_20": None,
        "output_21": None,
        "output_22": None,
        "bias_2": [1, 64, 1, 1],
        "output_23": [8, 64, 62, 62],
        "weight_3": [64, 64, 3, 3],
        "output_28": None,
        "output_29": None,
        "output_30": None,
        "bias_3": [1, 64, 1, 1],
        "output_31": [8, 64, 33, 33],
        "weight_4": [64, 64, 3, 3],
        "output_36": None,
        "output_37": None,
        "output_38": None,
        "bias_4": [1, 64, 1, 1],
        "output_39": [8, 64, 33, 33],
        "weight_5": [64, 64, 3, 3],
        "output_44": None,
        "output_45": None,
        "output_46": None,
        "bias_5": [1, 64, 1, 1],
        "output_47": [8, 64, 33, 33],
        "weight_6": [64, 64, 3, 3],
        "output_52": None,
        "output_53": None,
        "output_54": None,
        "bias_6": [1, 64, 1, 1],
        "output_55": [8, 64, 31, 31],
        "weight_7": [64, 64, 3, 3],
        "output_60": None,
        "output_61": None,
        "output_62": None,
        "bias_7": [1, 64, 1, 1],
        "output_63": [8, 64, 18, 18],
        "weight_8": [64, 64, 3, 3],
        "output_68": None,
        "output_69": None,
        "output_70": None,
        "bias_8": [1, 64, 1, 1],
        "output_71": [8, 64, 18, 18],
        "weight_9": [64, 64, 3, 3],
        "output_76": None,
        "output_77": None,
        "output_78": None,
        "bias_9": [1, 64, 1, 1],
        "output_79": [8, 64, 18, 18],
        "weight_10": [64, 64, 3, 3],
        "output_84": None,
        "output_85": None,
        "output_86": None,
        "bias_10": [1, 64, 1, 1],
        "output_87": [8, 64, 16, 16],
        "weight_11": [64, 64, 3, 3],
        "output_92": None,
        "output_93": None,
        "output_94": None,
        "bias_11": [1, 64, 1, 1],
        "output_95": [8, 64, 10, 10],
        "weight_12": [64, 64, 3, 3],
        "output_100": None,
        "output_101": None,
        "output_102": None,
        "bias_12": [1, 64, 1, 1],
        "output_103": [8, 64, 10, 10],
        "weight_13": [64, 64, 3, 3],
        "output_108": None,
        "output_109": None,
        "output_110": None,
        "bias_13": [1, 64, 1, 1],
        "output_111": [8, 64, 10, 10],
        "weight_14": [64, 64, 3, 3],
        "output_116": None,
        "output_117": None,
        "output_118": None,
        "bias_14": [1, 64, 1, 1],
        "output_119": [8, 64, 8, 8],
        "weight_15": [64, 64, 3, 3],
        "output_124": None,
        "output_125": None,
        "output_126": None,
        "bias_15": [1, 64, 1, 1],
        "output_127": [8, 64, 6, 6],
        "weight_16": [64, 64, 3, 3],
        "output_132": None,
        "output_133": None,
        "output_134": None,
        "bias_16": [1, 64, 1, 1],
        "output": [8, 64, 6, 6],
        "groups_0": None,
        "groups_1": None,
        "groups_2": None,
        "groups_3": None,
        "groups_4": None,
        "groups_5": None,
        "groups_6": None,
        "groups_7": None,
        "groups_8": None,
        "groups_9": None,
        "groups_10": None,
        "groups_11": None,
        "groups_12": None,
        "groups_13": None,
        "groups_14": None,
        "groups_15": None,
        "groups_16": None,
    }

    assert_shapes(model, logical_ref, physical_ref, shapes=shapes)


def test_shapes_4():
    # Extend to input
    model = Model()
    model |= (l1 := Linear(dimension=10)).connect(weight="weight", output="output")
    model |= (l2 := Linear(dimension=10)).connect(weight="weight1", output="output2")
    model.merge_connections(l1.input, l2.input)
    model |= Linear(dimension=71).connect(
        input="input", weight="weight2", output=l1.input
    )
    model.expose_keys("output", "output2")
    shapes = {"input": [4, 256]}
    logical_ref: Mapping[str, list | None] = {
        "$_Linear_0_output": [["(V1, ...)", "u1", 71], ["u2", "(V2, ...)", 71]],
        "weight": [10, 71],
        "$bias_1": [10],
        "weight1": [10, 71],
        "$bias_2": [10],
        "weight2": [71, "u3"],
        "input": [["(V1, ...)", "u1", "u3"], ["u2", "(V2, ...)", "u3"]],
        "$bias_0": [71],
        "output": [["(V1, ...)", "u1", 10], ["u2", "(V2, ...)", 10]],
        "output2": [["(V1, ...)", "u1", 10], ["u2", "(V2, ...)", 10]],
    }
    physical_ref = {
        "weight": [10, 71],
        "axes_0": None,
        "output_0": [71, 10],
        "weight1": [10, 71],
        "axes_1": None,
        "output_1": [71, 10],
        "weight2": [71, 256],
        "axes_2": None,
        "output_2": [256, 71],
        "input": [4, 256],
        "output_3": [4, 71],
        "bias_2": [71],
        "output_4": [4, 71],
        "output_5": [4, 10],
        "bias_0": [10],
        "output": [4, 10],
        "output_6": [4, 10],
        "bias_1": [10],
        "output2": [4, 10],
    }
    assert_shapes(
        model, logical_ref, physical_ref, shapes=shapes, check_all_shapes=True
    )


def test_linear_1_set_shapes():
    model = Linear()
    model.set_shapes(input=[100, 4])
    shapes = {"target": [100, 1]}
    ctx = TrainModel(model)
    loss_model = SquaredError()
    loss_model.set_shapes(**loss_model.safe_shapes)
    loss_model.set_shapes(**loss_model.submodel.safe_shapes)
    ctx.add_loss(
        loss_model=loss_model, reduce_steps=[Mean()], input="output", target="target"
    )
    logical_ref: Mapping[str, list | None] = {
        "$_SquaredError_1_output": [100, "u1"],
        "$_Mean_2_output": [],
        "weight": ["u1", 4],
        "input": [100, 4],
        "bias": ["u1"],
        "target": [100, "u1"],
        "output": [100, "u1"],
        "$_Mean_2_axis": None,
        "$_Mean_2_keepdim": None,
    }
    physical_ref = {
        "weight": [1, 4],
        "axes": None,
        "output_0": [4, 1],
        "input": [100, 4],
        "output_1": [100, 1],
        "bias": [1],
        "output": [100, 1],
        "target": [100, 1],
        "output_2": [100, 1],
        "axis": None,
        "keepdim": None,
        "output_3": [],
    }
    assert_shapes(ctx, logical_ref, physical_ref, shapes=shapes)


def test_linear_1_static_shapes():
    model = Linear()
    shapes = {"input": [100, 4], "target": [100, 1]}
    ctx = TrainModel(model)
    loss_model = SquaredError()
    loss_model.set_shapes(**loss_model.submodel.safe_shapes)
    ctx.add_loss(
        loss_model=loss_model, reduce_steps=[Mean()], input="output", target="target"
    )
    logical_ref: Mapping[str, list | None] = {
        "$_SquaredError_1_output": [
            ["(V1, ...)", "u1", "u2"],
            ["u3", "(V2, ...)", "u2"],
        ],
        "$_Mean_2_output": [],
        "weight": ["u2", "u4"],
        "input": [["(V1, ...)", "u1", "u4"], ["u3", "(V2, ...)", "u4"]],
        "bias": ["u2"],
        "target": [["(V1, ...)", "u1", "u2"], ["u3", "(V2, ...)", "u2"]],
        "output": [["(V1, ...)", "u1", "u2"], ["u3", "(V2, ...)", "u2"]],
        "$_Mean_2_axis": None,
        "$_Mean_2_keepdim": None,
    }
    physical_ref = {
        "weight": [1, 4],
        "axes": None,
        "output_0": [4, 1],
        "input": [100, 4],
        "output_1": [100, 1],
        "bias": [1],
        "output": [100, 1],
        "target": [100, 1],
        "output_2": [100, 1],
        "axis": None,
        "keepdim": None,
        "output_3": [],
    }
    assert_shapes(ctx, logical_ref, physical_ref, shapes=shapes, check_all_shapes=True)


def test_linear_1_static_inputs():
    model = Linear()
    static_inputs = {
        "input": np.random.randn(100, 4),
        "target": np.random.randn(100, 1),
    }
    ctx = TrainModel(model)
    loss_model = SquaredError()
    loss_model.set_shapes(**loss_model.submodel.safe_shapes)
    ctx.add_loss(
        loss_model=loss_model, reduce_steps=[Mean()], input="output", target="target"
    )
    # ctx.set_shapes(input = [100, 4], target = [100, 1])
    logical_ref: Mapping[str, list | None] = {
        "$_SquaredError_1_output": [
            ["(V1, ...)", "u1", "u2"],
            ["u3", "(V2, ...)", "u2"],
        ],
        "$_Mean_2_output": [],
        "weight": ["u2", "u4"],
        "input": [["(V1, ...)", "u1", "u4"], ["u3", "(V2, ...)", "u4"]],
        "bias": ["u2"],
        "target": [["(V1, ...)", "u1", "u2"], ["u3", "(V2, ...)", "u2"]],
        "output": [["(V1, ...)", "u1", "u2"], ["u3", "(V2, ...)", "u2"]],
        "$_Mean_2_axis": None,
        "$_Mean_2_keepdim": None,
    }
    physical_ref = {
        "weight": [1, 4],
        "axes": None,
        "output_0": [4, 1],
        "input": [100, 4],
        "output_1": [100, 1],
        "bias": [1],
        "output": [100, 1],
        "target": [100, 1],
        "output_2": [100, 1],
        "axis": None,
        "keepdim": None,
        "output_3": [],
    }
    assert_shapes(
        ctx,
        logical_ref,
        physical_ref,
        static_inputs=static_inputs,
        check_all_shapes=True,
    )


def test_simple_composite_1_set_shapes():
    model = Model()
    mult = Multiply()
    mult.set_shapes(right=[2, 2])
    model += mult.connect(
        left=IOKey(value=Tensor([[2.0]]), name="left"),
        right="input2",
        output=IOKey(name="output"),
    )
    logical_ref = {
        "input2": [2, 2],
        "output": [2, 2],
        "left": [1, 1],
        # 'Multiply_0_left': [1, 1]
    }
    physical_ref = {
        "left": [1, 1],
        "input2": [2, 2],
        "output": [2, 2],
    }

    assert_shapes(model, logical_ref, physical_ref)


def test_simple_composite_1_extend_inputs():
    model = Model()
    mult = Multiply()
    right_input: Tensor[float] = Tensor(np.random.randn(2, 2).tolist())
    model += mult.connect(
        left=IOKey(value=Tensor([[2.0]]), name="left"),
        right=IOKey(value=right_input, name="right"),
        output=IOKey(name="output"),
    )

    logical_ref = {
        "right": [2, 2],
        "output": [2, 2],
        "left": [1, 1],
    }
    physical_ref = {
        "output": [2, 2],
    }

    assert_shapes(model, logical_ref, physical_ref)


def test_simple_composite_1_set_shapes_2():
    model = Model()
    mult = Multiply()
    model += mult.connect(
        left=IOKey(value=Tensor([[2.0]]), name="left"),
        right="input2",
        output=IOKey(name="output"),
    )
    mult.set_shapes(right=[2, 2])

    logical_ref = {
        "input2": [2, 2],
        "output": [2, 2],
        "left": [1, 1],
    }
    physical_ref = {
        "left": [1, 1],
        "input2": [2, 2],
        "output": [2, 2],
    }

    assert_shapes(model, logical_ref, physical_ref)


def test_simple_composite_1_static_shapes():
    model = Model()
    model += Multiply().connect(
        left=IOKey(value=Tensor(0.5), name="left"),
        right=IOKey("input2", type=Tensor),
        output=IOKey(name="output"),
    )
    shapes = {"input2": [2, 2]}

    logical_ref = {
        "input2": ["(V1, ...)"],
        "output": ["(V1, ...)"],
        "left": [],
    }
    physical_ref = {"left": [], "input2": [2, 2], "output": [2, 2]}

    assert_shapes(model, logical_ref, physical_ref, shapes=shapes)


def test_simple_composite_1_static_inputs():
    model = Model()
    model += Add().connect(
        left=IOKey(value=Tensor(0.5), name="left"),
        right=IOKey("input2", type=Tensor),
        output=IOKey(name="output"),
    )
    static_inputs = {"input2": np.random.randn(2, 2)}
    logical_ref = {
        "input2": ["(V1, ...)"],
        "output": ["(V1, ...)"],
        "left": [],
    }
    physical_ref = {"output": [2, 2]}

    assert_shapes(model, logical_ref, physical_ref, static_inputs=static_inputs)


def test_simple_composite_2_set_shapes():
    model = Model()
    mult = Multiply()
    mult.set_shapes(right=[2, 2])
    model |= mult.connect(left=IOKey(value=Tensor(2.0), name="left"), right="in1")
    model |= Divide().connect(
        numerator=IOKey(value=Tensor(2.0), name="numerator"),
        denominator=mult.output,
        output=IOKey(name="output"),
    )

    logical_ref = {
        "left": [],
        "in1": [2, 2],
        "numerator": [],
        "$_Multiply_0_output": [2, 2],
        "output": [2, 2],
    }
    physical_ref = {
        "left": [],
        "in1": [2, 2],
        "output_0": [2, 2],
        "numerator": [],
        "output": [2, 2],
    }

    assert_shapes(model, logical_ref, physical_ref)


def test_simple_composite_2_set_shapes_2():
    model = Model()
    mult = Multiply()
    model |= mult.connect(left=IOKey(value=Tensor(2.0), name="left"), right="in1")
    model |= Divide().connect(
        numerator=IOKey(value=Tensor(2.0), name="numerator"),
        denominator=mult.output,
        output=IOKey(name="output"),
    )
    mult.set_shapes(right=[2, 2])

    logical_ref = {
        "left": [],
        "in1": [2, 2],
        "numerator": [],
        "$_Multiply_0_output": [2, 2],
        "output": [2, 2],
    }
    physical_ref = {
        "left": [],
        "in1": [2, 2],
        "output_0": [2, 2],
        "output": [2, 2],
        "numerator": [],
    }

    assert_shapes(model, logical_ref, physical_ref)


def test_simple_composite_2_extend_inputs():
    model = Model()
    mult = Multiply()
    Multiply_0_right: Tensor[float] = Tensor(np.random.randn(2, 2).tolist())
    model |= mult.connect(
        left=IOKey(value=Tensor(2.0), name="left"),
        right=IOKey(value=Multiply_0_right, name="in1"),
    )
    model |= Divide().connect(
        numerator=IOKey(value=Tensor(2.0), name="numerator"),
        denominator=mult.output,
        output=IOKey(name="output"),
    )
    mult.set_shapes(right=[2, 2])

    logical_ref = {
        "left": [],
        "in1": [2, 2],
        "numerator": [],
        "$_Multiply_0_output": [2, 2],
        "output": [2, 2],
    }
    physical_ref = {
        "output": [2, 2],
    }

    assert_shapes(model, logical_ref, physical_ref)


def test_simple_composite_2_static_shapes():
    model = Model()
    mult = Multiply()
    model |= mult.connect(
        left=IOKey(value=Tensor(2.0), name="left"),
        right=IOKey("in1", type=Tensor),
    )
    model |= Divide().connect(
        numerator=IOKey(value=Tensor(2.0), name="numerator"),
        denominator=mult.output,
        output=IOKey(name="output"),
    )
    shapes = {"in1": [2, 2]}

    logical_ref = {
        "left": [],
        "in1": ["(V1, ...)"],
        "numerator": [],
        "$_Multiply_0_output": ["(V1, ...)"],
        "output": ["(V1, ...)"],
    }
    physical_ref = {
        "left": [],
        "in1": [2, 2],
        "output_0": [2, 2],
        "output": [2, 2],
        "numerator": [],
    }

    assert_shapes(model, logical_ref, physical_ref, shapes=shapes)


def test_simple_composite_2_static_inputs():
    model = Model()
    mult = Multiply()
    model |= mult.connect(
        left=IOKey(value=Tensor(2.0), name="left"),
        right=IOKey("in1", type=Tensor),
    )
    model |= Divide().connect(
        numerator=IOKey(value=Tensor(2.0), name="numerator"),
        denominator=mult.output,
        output=IOKey(name="output"),
    )
    static_inputs = {"in1": np.random.randn(2, 2)}

    logical_ref = {
        "left": [],
        "in1": ["(V1, ...)"],
        "numerator": [],
        "$_Multiply_0_output": ["(V1, ...)"],
        "output": ["(V1, ...)"],
    }
    physical_ref = {"output": [2, 2]}

    assert_shapes(model, logical_ref, physical_ref, static_inputs=static_inputs)


def test_composite_1_set_shapes_1():
    composite = Model()
    m1 = Multiply()
    m1.set_shapes(left=[1, 1, 1, 1, 1, 1, 1, 37, 43], right=[134, 47, 1, 1, 1])
    composite |= m1.connect(left="input1", right="input2")
    composite |= (m2 := Multiply()).connect(left="input2", right=m1.output)
    composite |= Add().connect(
        left=m2.output, right=m2.output, output=IOKey(name="output")
    )
    logical_ref = {
        "input1": [1, 1, 1, 1, 1, 1, 1, 37, 43],
        "input2": [134, 47, 1, 1, 1],
        "$_Multiply_0_output": [1, 1, 1, 1, 134, 47, 1, 37, 43],
        "$_Multiply_1_output": [1, 1, 1, 1, 134, 47, 1, 37, 43],
        "output": [1, 1, 1, 1, 134, 47, 1, 37, 43],
    }
    physical_ref = {
        "input1": [1, 1, 1, 1, 1, 1, 1, 37, 43],
        "input2": [134, 47, 1, 1, 1],
        "output_0": [1, 1, 1, 1, 134, 47, 1, 37, 43],
        "output_1": [1, 1, 1, 1, 134, 47, 1, 37, 43],
        "output": [1, 1, 1, 1, 134, 47, 1, 37, 43],
    }

    assert_shapes(composite, logical_ref, physical_ref)


def test_composite_1_set_shapes_1_2():
    composite = Model()
    m1 = Multiply()
    composite |= m1.connect(left="input1", right="input2")
    composite |= (m2 := Multiply()).connect(left="input2", right=m1.output)
    composite |= Add().connect(
        left=m2.output, right=m2.output, output=IOKey(name="output")
    )
    m1.set_shapes(left=[1, 1, 1, 1, 1, 1, 1, 37, 43], right=[134, 47, 1, 1, 1])
    logical_ref = {
        "input1": [1, 1, 1, 1, 1, 1, 1, 37, 43],
        "input2": [134, 47, 1, 1, 1],
        "$_Multiply_0_output": [1, 1, 1, 1, 134, 47, 1, 37, 43],
        "$_Multiply_1_output": [1, 1, 1, 1, 134, 47, 1, 37, 43],
        "output": [1, 1, 1, 1, 134, 47, 1, 37, 43],
    }
    physical_ref = {
        "input1": [1, 1, 1, 1, 1, 1, 1, 37, 43],
        "input2": [134, 47, 1, 1, 1],
        "output_0": [1, 1, 1, 1, 134, 47, 1, 37, 43],
        "output_1": [1, 1, 1, 1, 134, 47, 1, 37, 43],
        "output": [1, 1, 1, 1, 134, 47, 1, 37, 43],
    }

    assert_shapes(composite, logical_ref, physical_ref)


def test_composite_1_set_shapes_2():
    composite = Model()
    m1 = Multiply()
    m1.set_shapes(left=[1, 1, 1, 1, 1, 1, 1, 37, 43])
    composite |= m1.connect(left="input1", right="input2")
    m2 = Multiply()
    m2.set_shapes(left=[134, 47, 1, 1, 1])
    composite |= m2.connect(left="input2", right=m1.output)
    composite |= Add().connect(
        left=m2.output, right=m2.output, output=IOKey(name="output")
    )
    logical_ref = {
        "input1": [1, 1, 1, 1, 1, 1, 1, 37, 43],
        "input2": [134, 47, 1, 1, 1],
        "$_Multiply_0_output": [1, 1, 1, 1, 134, 47, 1, 37, 43],
        "$_Multiply_1_output": [1, 1, 1, 1, 134, 47, 1, 37, 43],
        "output": [1, 1, 1, 1, 134, 47, 1, 37, 43],
    }
    physical_ref = {
        "input1": [1, 1, 1, 1, 1, 1, 1, 37, 43],
        "input2": [134, 47, 1, 1, 1],
        "output_0": [1, 1, 1, 1, 134, 47, 1, 37, 43],
        "output_1": [1, 1, 1, 1, 134, 47, 1, 37, 43],
        "output": [1, 1, 1, 1, 134, 47, 1, 37, 43],
    }

    assert_shapes(composite, logical_ref, physical_ref)


def test_composite_1_set_shapes_2_2():
    composite = Model()
    m1 = Multiply()
    composite |= m1.connect(left="input1", right="input2")
    m2 = Multiply()
    composite |= m2.connect(left="input2", right=m1.output)
    composite |= Add().connect(
        left=m2.output, right=m2.output, output=IOKey(name="output")
    )
    m1.set_shapes(left=[1, 1, 1, 1, 1, 1, 1, 37, 43])
    m2.set_shapes(left=[134, 47, 1, 1, 1])
    logical_ref = {
        "input1": [1, 1, 1, 1, 1, 1, 1, 37, 43],
        "input2": [134, 47, 1, 1, 1],
        "$_Multiply_0_output": [1, 1, 1, 1, 134, 47, 1, 37, 43],
        "$_Multiply_1_output": [1, 1, 1, 1, 134, 47, 1, 37, 43],
        "output": [1, 1, 1, 1, 134, 47, 1, 37, 43],
    }
    physical_ref = {
        "input1": [1, 1, 1, 1, 1, 1, 1, 37, 43],
        "input2": [134, 47, 1, 1, 1],
        "output_0": [1, 1, 1, 1, 134, 47, 1, 37, 43],
        "output_1": [1, 1, 1, 1, 134, 47, 1, 37, 43],
        "output": [1, 1, 1, 1, 134, 47, 1, 37, 43],
    }

    assert_shapes(composite, logical_ref, physical_ref)


@pytest.mark.skip(reason="Known Bugs")
def test_composite_1_set_shapes_3():
    """This test requires backtracking in order to infer all source of
    unknown values comes from input2. Since we solve constraints
    locally for now, it is impossible to infer final values.
    """
    composite = Model()
    m1 = Multiply()
    m1.set_shapes(left=[1, 1, 1, 1, 1, 1, 1, 37, 43])
    composite |= m1.connect(left="input1", right="input2")
    composite |= (m2 := Multiply()).connect(left="input2", right=m1.output)
    add = Add()
    add.set_shapes(output=[1, 1, 1, 1, 134, 47, 1, 37, 43])
    composite |= add.connect(
        left=m2.output, right=m2.output, output=IOKey(name="output")
    )
    logical_ref: dict[str, list] = {
        "input1": [1, 1, 1, 1, 1, 1, 1, 37, 43],
        # "input2": [134, 47, 1, 1, 1],
        "input2": ["(V1, ...)", 134, 47, 1, "u1", "u2"],
        "$_Multiply_0_output": [1, 1, 1, 1, 134, 47, 1, 37, 43],
        "$_Multiply_1_output": [1, 1, 1, 1, 134, 47, 1, 37, 43],
        "output": [1, 1, 1, 1, 134, 47, 1, 37, 43],
    }
    physical_ref = {
        "input1": [1, 1, 1, 1, 1, 1, 1, 37, 43],
        "input2": [134, 47, 1, 1, 1],
        "_Multiply_0_output": [1, 1, 1, 1, 134, 47, 1, 37, 43],
        "_Multiply_1_output": [1, 1, 1, 1, 134, 47, 1, 37, 43],
        "output": [1, 1, 1, 1, 134, 47, 1, 37, 43],
    }
    assert_shapes(composite, logical_ref, physical_ref)


@pytest.mark.skip(reason="Extraction from possibilities is not implemented yet.")
def test_extraction_from_possibilities():
    m1 = Multiply()
    m1.set_shapes(
        left=[1, 1, 1, 1, 1, 1, 1, 37, 43],
        output=[1, 1, 1, 1, 134, 47, 1, 37, 43],
    )
    logical_ref: Mapping[str, list | None] = {
        "left": [1, 1, 1, 1, 1, 1, 1, 37, 43],
        "right": ["(V1, ...)", 134, 47, 1, "u1", "u2"],
        "output": [1, 1, 1, 1, 134, 47, 1, 37, 43],
    }
    assert_shapes(m1, logical_ref)


@pytest.mark.skip(reason="Known Bugs")
def test_composite_1_set_shapes_3_2():
    """This test requires backtracking in order to infer all source of
    unknown values comes from input2. Since we solve constraints
    locally for now, it is impossible to infer final values.
    """
    composite = Model()
    m1 = Multiply()
    composite |= m1.connect(left="input1", right="input2")
    composite |= (m2 := Multiply()).connect(left="input2", right=m1.output)
    add = Add()
    composite |= add.connect(
        left=m2.output, right=m2.output, output=IOKey(name="output")
    )
    add.set_shapes(output=[1, 1, 1, 1, 134, 47, 1, 37, 43])
    m1.set_shapes(left=[1, 1, 1, 1, 1, 1, 1, 37, 43])
    logical_ref = {
        "input1": [1, 1, 1, 1, 1, 1, 1, 37, 43],
        "input2": [134, 47, 1, 1, 1],
        "$_Multiply_0_output": [1, 1, 1, 1, 134, 47, 1, 37, 43],
        "$_Multiply_1_output": [1, 1, 1, 1, 134, 47, 1, 37, 43],
        "output": [1, 1, 1, 1, 134, 47, 1, 37, 43],
    }
    physical_ref = {
        "input1": [1, 1, 1, 1, 1, 1, 1, 37, 43],
        "input2": [134, 47, 1, 1, 1],
        "_Multiply_0_output": [1, 1, 1, 1, 134, 47, 1, 37, 43],
        "_Multiply_1_output": [1, 1, 1, 1, 134, 47, 1, 37, 43],
        "output": [1, 1, 1, 1, 134, 47, 1, 37, 43],
    }
    assert_shapes(composite, logical_ref, physical_ref)


@pytest.mark.skip(reason="Known Bugs")
def test_composite_1_set_shapes_4():
    """This test requires backtracking in order to infer all source of
    unknown values comes from input2. Since we solve constraints
    locally for now, it is impossible to infer final values.
    """
    composite = Model()
    m1 = Multiply()
    m1.set_shapes(
        left=[1, 1, 1, 1, 1, 1, 1, 37, 43],
        output=[1, 1, 1, 1, 134, 47, 1, 37, 43],
    )
    composite |= m1.connect(left="input1", right="input2")
    composite |= (m2 := Multiply()).connect(left="input2", right=m1.output)
    composite |= Add().connect(
        left=m2.output, right=m2.output, output=IOKey(name="output")
    )
    logical_ref = {
        "input1": [1, 1, 1, 1, 1, 1, 1, 37, 43],
        "input2": [134, 47, 1, 1, 1],
        "$_Multiply_0_output": [1, 1, 1, 1, 134, 47, 1, 37, 43],
        "$_Multiply_1_output": [1, 1, 1, 1, 134, 47, 1, 37, 43],
        "output": [1, 1, 1, 1, 134, 47, 1, 37, 43],
    }
    physical_ref = {
        "input1": [1, 1, 1, 1, 1, 1, 1, 37, 43],
        "input2": [134, 47, 1, 1, 1],
        "_Multiply_0_output": [1, 1, 1, 1, 134, 47, 1, 37, 43],
        "_Multiply_1_output": [1, 1, 1, 1, 134, 47, 1, 37, 43],
        "output": [1, 1, 1, 1, 134, 47, 1, 37, 43],
    }
    assert_shapes(composite, logical_ref, physical_ref)


@pytest.mark.skip(reason="Known Bugs")
def test_composite_1_set_shapes_4_2():
    """This test requires backtracking in order to infer all source of
    unknown values comes from input2. Since we solve constraints
    locally for now, it is impossible to infer final values.
    """
    composite = Model()
    m1 = Multiply()
    composite |= m1.connect(left="input1", right="input2")
    composite |= (m2 := Multiply()).connect(left="input2", right=m1.output)
    composite |= Add().connect(
        left=m2.output, right=m2.output, output=IOKey(name="output")
    )
    m1.set_shapes(
        left=[1, 1, 1, 1, 1, 1, 1, 37, 43],
        output=[1, 1, 1, 1, 134, 47, 1, 37, 43],
    )
    logical_ref = {
        "input1": [1, 1, 1, 1, 1, 1, 1, 37, 43],
        "input2": [134, 47, 1, 1, 1],
        "$_Multiply_0_output": [1, 1, 1, 1, 134, 47, 1, 37, 43],
        "$_Multiply_1_output": [1, 1, 1, 1, 134, 47, 1, 37, 43],
        "output": [1, 1, 1, 1, 134, 47, 1, 37, 43],
    }
    physical_ref = {
        "input1": [1, 1, 1, 1, 1, 1, 1, 37, 43],
        "input2": [134, 47, 1, 1, 1],
        "_Multiply_0_output": [1, 1, 1, 1, 134, 47, 1, 37, 43],
        "_Multiply_1_output": [1, 1, 1, 1, 134, 47, 1, 37, 43],
        "output": [1, 1, 1, 1, 134, 47, 1, 37, 43],
    }
    assert_shapes(composite, logical_ref, physical_ref)


def test_composite_1_set_shapes_5():
    model = Model()
    m1 = Multiply()
    m1.set_types(left=Tensor, right=Tensor)
    m1.set_shapes(right=[134, 47, 1, 1, 1])
    model |= m1.connect(left="input1", right="input2")
    model |= (m2 := Multiply()).connect(left="input2", right=m1.output)
    add = Add()
    add.set_shapes(output=[1, 1, 1, 1, 134, 47, 1, 37, 43])
    model |= add.connect(left=m2.output, right=m2.output, output=IOKey(name="output"))
    logical_ref: Mapping[str, list | None] = {
        "input1": [1, 1, 1, 1, "u1", "u2", 1, 37, 43],
        "input2": [134, 47, 1, 1, 1],
        "$_Multiply_0_output": [1, 1, 1, 1, 134, 47, 1, 37, 43],
        "$_Multiply_1_output": [1, 1, 1, 1, 134, 47, 1, 37, 43],
        "output": [1, 1, 1, 1, 134, 47, 1, 37, 43],
    }
    physical_ref: Mapping[str, list | None] = {
        "input1": [1, 1, 1, 1, None, None, 1, 37, 43],
        "input2": [134, 47, 1, 1, 1],
        "output_0": [1, 1, 1, 1, 134, 47, 1, 37, 43],
        "output_1": [1, 1, 1, 1, 134, 47, 1, 37, 43],
        "output": [1, 1, 1, 1, 134, 47, 1, 37, 43],
    }
    assert_shapes(model, logical_ref, physical_ref)


def test_composite_1_set_shapes_5_dfs():
    composite = Model()
    add = Add()
    add.set_shapes(output=[1, 1, 1, 1, 134, 47, 1, 37, 43])
    composite |= add.connect(left="input1", right="input1", output=IOKey(name="output"))
    assert_all_nodes_unique(composite)


def test_composite_1_set_shapes_6_dfs():
    composite = Model()
    add = Add()
    composite += add.connect(left="input1", right="input1", output=IOKey(name="output"))
    composite.set_shapes(output=[1, 1, 1, 1, 134, 47, 1, 37, 43])
    assert_all_nodes_unique(composite)


def test_composite_1_set_shapes_7_dfs():
    composite = Model()
    add = Add()
    composite += add.connect(left="input1", right="input2", output=IOKey(name="output"))


def test_composite_1_set_shapes_5_2():
    composite = Model()
    m1 = Multiply()
    m1.set_types(left=Tensor, right=Tensor)
    composite |= m1.connect(left="input1", right="input2")
    composite |= (m2 := Multiply()).connect(left="input2", right=m1.output)
    add = Add()
    composite |= add.connect(
        left=m2.output, right=m2.output, output=IOKey(name="output")
    )
    m1.set_shapes(right=[134, 47, 1, 1, 1])
    add.set_shapes(output=[1, 1, 1, 1, 134, 47, 1, 37, 43])
    logical_ref: Mapping[str, list | None] = {
        "input1": [1, 1, 1, 1, "u1", "u2", 1, 37, 43],
        "input2": [134, 47, 1, 1, 1],
        "$_Multiply_0_output": [1, 1, 1, 1, 134, 47, 1, 37, 43],
        "$_Multiply_1_output": [1, 1, 1, 1, 134, 47, 1, 37, 43],
        "output": [1, 1, 1, 1, 134, 47, 1, 37, 43],
    }
    physical_ref: Mapping[str, list | None] = {
        "input1": [1, 1, 1, 1, None, None, 1, 37, 43],
        "input2": [134, 47, 1, 1, 1],
        "output_0": [1, 1, 1, 1, 134, 47, 1, 37, 43],
        "output_1": [1, 1, 1, 1, 134, 47, 1, 37, 43],
        "output": [1, 1, 1, 1, 134, 47, 1, 37, 43],
    }
    assert_shapes(composite, logical_ref, physical_ref)


def get_composite_1():
    # Create common composite_1 model for corresponding tests.
    composite_1 = Model()
    composite_1 |= (m1 := Multiply()).connect(
        left=IOKey("input1", type=Tensor),
        right=IOKey("input2", type=Tensor),
    )
    composite_1 |= (m2 := Multiply()).connect(left="input2", right=m1.output)
    composite_1 |= Add().connect(
        left=m2.output, right=m2.output, output=IOKey(name="output")
    )
    return composite_1


def test_composite_1_static_shapes_1():
    model = deepcopy(get_composite_1())
    shapes = {"input1": [1, 1, 1, 1, 1, 1, 1, 37, 43], "input2": [134, 47, 1, 1, 1]}
    logical_ref = {
        "input1": ["(V1, ...)"],
        "input2": ["(V2, ...)"],
        "$_Multiply_0_output": ["(V3, ...)"],
        "$_Multiply_1_output": ["(V4, ...)"],
        "output": ["(V4, ...)"],
    }
    physical_ref = {
        "input1": [1, 1, 1, 1, 1, 1, 1, 37, 43],
        "input2": [134, 47, 1, 1, 1],
        "output_0": [1, 1, 1, 1, 134, 47, 1, 37, 43],
        "output_1": [1, 1, 1, 1, 134, 47, 1, 37, 43],
        "output": [1, 1, 1, 1, 134, 47, 1, 37, 43],
    }

    assert_shapes(model, logical_ref, physical_ref, shapes=shapes)


def test_composite_1_extend_inputs_1():
    composite = Model()
    m1 = Multiply()
    Multiply_0_left: Tensor[float] = Tensor(
        np.random.randn(1, 1, 1, 1, 1, 1, 1, 37, 43).tolist()
    )
    Multiply_0_right: Tensor[float] = Tensor(np.random.randn(134, 47, 1, 1, 1).tolist())
    composite |= m1.connect(
        left=IOKey(value=Multiply_0_left, name="left"),
        right=IOKey(value=Multiply_0_right, name="right"),
    )
    composite |= (m2 := Multiply()).connect(left=m1.right, right=m1.output)
    composite |= Add().connect(
        left=m2.output, right=m2.output, output=IOKey(name="output")
    )
    key_mappings = composite.generate_keys()

    m1_out_metadata = composite.conns.get_con_by_metadata(m1.output.metadata)
    assert m1_out_metadata is not None
    m1_out_key = key_mappings[m1_out_metadata.key]

    m2_out_metadata = composite.conns.get_con_by_metadata(m2.output.metadata)
    assert m2_out_metadata is not None
    m2_out_key = key_mappings[m2_out_metadata.key]

    logical_ref = {
        "left": [1, 1, 1, 1, 1, 1, 1, 37, 43],
        "right": [134, 47, 1, 1, 1],
        m1_out_key: [1, 1, 1, 1, 134, 47, 1, 37, 43],
        m2_out_key: [1, 1, 1, 1, 134, 47, 1, 37, 43],
        "output": [1, 1, 1, 1, 134, 47, 1, 37, 43],
    }
    physical_ref = {
        "output": [1, 1, 1, 1, 134, 47, 1, 37, 43],
    }
    assert_shapes(composite, logical_ref, physical_ref)


@pytest.mark.skip(reason="Known Bugs")
def test_composite_1_static_shapes_2():
    """This test requires backtracking in order to infer all source of
    unknown values comes from input2. Since we solve constraints
    locally for now, it is impossible to infer final values.
    """
    model = deepcopy(get_composite_1())
    shapes = {
        "input1": [1, 1, 1, 1, 1, 1, 1, 37, 43],
        "output": [1, 1, 1, 1, 134, 47, 1, 37, 43],
    }
    logical_ref = {
        "input1": ["(V1, ...)"],
        "input2": ["(V2, ...)"],
        "$_Multiply_0_output": ["(V3, ...)"],
        "$_Multiply_1_output": ["(V4, ...)"],
        "output": ["(V4, ...)"],
    }
    physical_ref: dict[str, list | None] = {
        "input1": [1, 1, 1, 1, 1, 1, 1, 37, 43],
        "input2": ["...", 134, 47, 1, None, None],
        "_Multiply_0_output": [1, 1, 1, 1, 134, 47, 1, 37, 43],
        "_Multiply_1_output": [1, 1, 1, 1, 134, 47, 1, 37, 43],
        "output": [1, 1, 1, 1, 134, 47, 1, 37, 43],
    }
    assert_shapes(model, logical_ref, physical_ref, shapes=shapes)


def test_composite_1_static_shapes_3():
    model = deepcopy(get_composite_1())
    shapes = {"input2": [134, 47, 1, 1, 1], "output": [1, 1, 1, 1, 134, 47, 1, 37, 43]}
    logical_ref = {
        "input1": ["(V1, ...)"],
        "input2": ["(V2, ...)"],
        "$_Multiply_0_output": ["(V3, ...)"],
        "$_Multiply_1_output": ["(V4, ...)"],
        "output": ["(V4, ...)"],
    }
    physical_ref: Mapping[str, list | None] = {
        "input1": [1, 1, 1, 1, None, None, 1, 37, 43],
        "input2": [134, 47, 1, 1, 1],
        "output_0": [1, 1, 1, 1, 134, 47, 1, 37, 43],
        "output_1": [1, 1, 1, 1, 134, 47, 1, 37, 43],
        "output": [1, 1, 1, 1, 134, 47, 1, 37, 43],
    }
    assert_shapes(model, logical_ref, physical_ref, shapes=shapes)


def test_composite_1_static_inputs_1():
    model = deepcopy(get_composite_1())
    static_inputs = {
        "input1": np.random.randn(1, 1, 1, 1, 1, 1, 1, 37, 43),
        "input2": np.random.randn(134, 47, 1, 1, 1),
    }
    logical_ref = {
        "input1": ["(V1, ...)"],
        "input2": ["(V2, ...)"],
        "$_Multiply_0_output": ["(V3, ...)"],
        "$_Multiply_1_output": ["(V4, ...)"],
        "output": ["(V4, ...)"],
    }
    physical_ref = {
        "output": [1, 1, 1, 1, 134, 47, 1, 37, 43],
    }

    assert_shapes(model, logical_ref, physical_ref, static_inputs=static_inputs)


@pytest.mark.skip(reason="Known Bugs")
def test_composite_1_static_inputs_2():
    """This test requires backtracking in order to infer all source of
    unknown values comes from input2. Since we solve constraints
    locally for now, it is impossible to infer final values.
    """
    model = deepcopy(get_composite_1())
    static_inputs = {"input1": np.random.randn(1, 1, 1, 1, 1, 1, 1, 37, 43)}
    shapes = {"output": [1, 1, 1, 1, 134, 47, 1, 37, 43]}
    logical_ref = {
        "input1": ["(V1, ...)"],
        "input2": ["(V2, ...)"],
        "$_Multiply_0_output": ["(V3, ...)"],
        "$_Multiply_1_output": ["(V4, ...)"],
        "output": ["(V4, ...)"],
    }
    physical_ref: Mapping[str, list | None] = {
        "input1": [1, 1, 1, 1, 1, 1, 1, 37, 43],
        "input2": ["...", 134, 47, 1, None, None],
        "_Multiply_0_output": [1, 1, 1, 1, 134, 47, 1, 37, 43],
        "_Multiply_1_output": [1, 1, 1, 1, 134, 47, 1, 37, 43],
        "output": [1, 1, 1, 1, 134, 47, 1, 37, 43],
    }
    assert_shapes(
        model, logical_ref, physical_ref, shapes=shapes, static_inputs=static_inputs
    )


@pytest.mark.skip(reason="Known Bugs")
def test_composite_1_static_inputs_3():
    """This test requires backtracking in order to infer all source of
    unknown values comes from input2. Since we solve constraints
    locally for now, it is impossible to infer final values.
    """
    model = deepcopy(get_composite_1())
    static_inputs = {"input2": np.random.randn(134, 47, 1, 1, 1)}
    shapes = {"output": [1, 1, 1, 1, 134, 47, 1, 37, 43]}
    logical_ref = {
        "input1": ["(V1, ...)"],
        "input2": ["(V2, ...)"],
        "$_Multiply_0_output": ["(V3, ...)"],
        "$_Multiply_1_output": ["(V4, ...)"],
        "output": ["(V4, ...)"],
    }
    physical_ref: Mapping[str, list | None] = {
        "input1": [1, 1, 1, 1, None, None, 1, 37, 43],
        "input2": [134, 47, 1, 1, 1],
        "_Multiply_0_output": [1, 1, 1, 1, 134, 47, 1, 37, 43],
        "_Multiply_1_output": [1, 1, 1, 1, 134, 47, 1, 37, 43],
        "output": [1, 1, 1, 1, 134, 47, 1, 37, 43],
    }
    assert_shapes(
        model, logical_ref, physical_ref, shapes=shapes, static_inputs=static_inputs
    )


def test_composite_2_set_shapes_1():
    composite = Model()
    m1 = Model()
    m2 = Model()
    m3 = Model()

    mult1 = Multiply()
    mult1.set_shapes(left=[4, 5, 7, 1, 1], right=[1, 1, 7, 3, 4])
    m1 |= mult1.connect(left="input1", right="input2")
    m1 |= (mult2 := Multiply()).connect(left="input2", right=mult1.output)
    m1 |= Add().connect(
        left=mult2.output, right=mult2.output, output=IOKey(name="output")
    )

    m2 |= (mult3 := Multiply()).connect(left="input1", right="input2")
    m2 |= (mult4 := Multiply()).connect(left="input2", right=mult3.output)
    m2 |= Add().connect(
        left=mult4.output, right=mult4.output, output=IOKey(name="output")
    )

    m3 |= (add1 := Add()).connect(left="input1", right="input2")
    m3 |= (mult5 := Multiply()).connect(left="input2", right=add1.output)
    m3 |= Add().connect(
        left=mult5.output, right=mult5.output, output=IOKey(name="output")
    )

    composite |= m1.connect(input1="input1", input2="input2")
    composite |= m2.connect(input1=m1.output, input2="input2")  # type: ignore
    composite |= m3.connect(
        input1=m2.output,  # type: ignore
        input2=m2.output,  # type: ignore
        output=IOKey(name="output"),
    )

    logical_ref = {
        "input1": [4, 5, 7, 1, 1],
        "input2": [1, 1, 7, 3, 4],
        "$_Model_0_output": [4, 5, 7, 3, 4],
        "$_Model_1_output": [4, 5, 7, 3, 4],
        "output": [4, 5, 7, 3, 4],
    }
    physical_ref = {
        "input1": [4, 5, 7, 1, 1],
        "input2": [1, 1, 7, 3, 4],
        "output_0": [4, 5, 7, 3, 4],
        "output_1": [4, 5, 7, 3, 4],
        "output_2": [4, 5, 7, 3, 4],
        "output_3": [4, 5, 7, 3, 4],
        "output_4": [4, 5, 7, 3, 4],
        "output_5": [4, 5, 7, 3, 4],
        "output_6": [4, 5, 7, 3, 4],
        "output_7": [4, 5, 7, 3, 4],
        "output": [4, 5, 7, 3, 4],
    }
    assert_shapes(composite, logical_ref, physical_ref)


def test_composite_2_set_shapes_2():
    composite = Model()
    m1 = Model()
    m2 = Model()
    m3 = Model()

    mult1 = Multiply()
    mult1.set_shapes(left=[4, 5, 7, 1, 1])
    m1 |= mult1.connect(left="input1", right="input2")
    m1 |= (mult2 := Multiply()).connect(left="input2", right=mult1.output)
    m1 |= Add().connect(
        left=mult2.output, right=mult2.output, output=IOKey(name="output")
    )

    mult3 = Multiply()
    mult3.set_shapes(right=[1, 1, 7, 3, 4])
    m2 |= mult3.connect(left="input1", right="input2")
    m2 |= (mult4 := Multiply()).connect(left="input2", right=mult3.output)
    m2 |= Add().connect(
        left=mult4.output, right=mult4.output, output=IOKey(name="output")
    )

    m3 |= (add1 := Add()).connect(left="input1", right="input2")
    m3 |= (mult5 := Multiply()).connect(left="input2", right=add1.output)
    m3 |= Add().connect(
        left=mult5.output, right=mult5.output, output=IOKey(name="output")
    )

    composite |= m1.connect(input1="input1", input2="input2")
    composite |= m2.connect(input1=m1.output, input2="input2")  # type: ignore
    composite |= m3.connect(
        input1=m2.output,  # type: ignore
        input2=m2.output,  # type: ignore
        output=IOKey(name="output"),
    )

    logical_ref = {
        "input1": [4, 5, 7, 1, 1],
        "input2": [1, 1, 7, 3, 4],
        "$_Model_0_output": [4, 5, 7, 3, 4],
        "$_Model_1_output": [4, 5, 7, 3, 4],
        "output": [4, 5, 7, 3, 4],
    }
    physical_ref = {
        "input1": [4, 5, 7, 1, 1],
        "input2": [1, 1, 7, 3, 4],
        "output_0": [4, 5, 7, 3, 4],
        "output_1": [4, 5, 7, 3, 4],
        "output_2": [4, 5, 7, 3, 4],
        "output_3": [4, 5, 7, 3, 4],
        "output_4": [4, 5, 7, 3, 4],
        "output_5": [4, 5, 7, 3, 4],
        "output_6": [4, 5, 7, 3, 4],
        "output_7": [4, 5, 7, 3, 4],
        "output": [4, 5, 7, 3, 4],
    }
    assert_shapes(composite, logical_ref, physical_ref)


def test_composite_2_set_shapes_3():
    """This test requires backtracking in order to infer all source of
    unknown values comes from input2. Since we solve constraints
    locally for now, it is impossible to infer final values.
    """
    composite = Model()
    m1 = Model()
    m2 = Model()
    m3 = Model()

    mult1 = Multiply()
    mult1.set_types(left=Tensor, right=Tensor)
    mult1.set_shapes(left=[4, 5, 7, 1, 1])
    m1 |= mult1.connect(left="input1", right="input2")
    m1 |= (mult2 := Multiply()).connect(left="input2", right=mult1.output)
    m1 |= Add().connect(
        left=mult2.output, right=mult2.output, output=IOKey(name="output")
    )

    mult3 = Multiply()
    mult3.set_types(left=Tensor, right=Tensor)
    mult3.set_shapes(left=[4, 5, 7, 3, 4])
    m2 |= mult3.connect(left="input1", right="input2")
    m2 |= (mult4 := Multiply()).connect(left="input2", right=mult3.output)
    m2 |= Add().connect(
        left=mult4.output, right=mult4.output, output=IOKey(name="output")
    )

    m3 |= (add1 := Add()).connect(
        left=IOKey("input1", type=Tensor),
        right=IOKey("input2", type=Tensor),
    )
    m3 |= (mult5 := Multiply()).connect(left="input2", right=add1.output)
    m3 |= Add().connect(
        left=mult5.output, right=mult5.output, output=IOKey(name="output")
    )

    composite |= m1.connect(input1="input1", input2="input2")
    composite |= m2.connect(input1=m1.output, input2="input2")  # type: ignore
    composite |= m3.connect(
        input1=m2.output,  # type: ignore
        input2=m2.output,  # type: ignore
        output=IOKey(name="output"),
    )

    logical_ref: Mapping[str, list | None] = {
        "input1": [4, 5, 7, 1, 1],
        "input2": ["(V1, ...)", 3, 4],
        "$_Model_0_output": [4, 5, 7, 3, 4],
        "$_Model_1_output": [4, 5, 7, 3, 4],
        "output": [4, 5, 7, 3, 4],
    }
    physical_ref: Mapping[str, list | None] = {
        "input1": [4, 5, 7, 1, 1],
        "input2": ["...", 3, 4],
        "output_0": [4, 5, 7, 3, 4],
        "output_1": [4, 5, 7, 3, 4],
        "output_2": [4, 5, 7, 3, 4],
        "output_3": [4, 5, 7, 3, 4],
        "output_4": [4, 5, 7, 3, 4],
        "output_5": [4, 5, 7, 3, 4],
        "output_6": [4, 5, 7, 3, 4],
        "output_7": [4, 5, 7, 3, 4],
        "output": [4, 5, 7, 3, 4],
    }
    assert_shapes(composite, logical_ref, physical_ref)


def test_composite_2_set_shapes_3_1():
    """This test requires backtracking in order to infer all source of
    unknown values comes from input2. Since we solve constraints
    locally for now, it is impossible to infer final values.
    """
    composite = Model()
    m1 = Model()
    m2 = Model()

    mult1 = Multiply()
    mult1.set_types(left=Tensor, right=Tensor)
    mult1.set_shapes(left=[4, 5, 7, 1, 1])
    m1 |= mult1.connect(left="input1", right="input2")
    m1 |= Multiply().connect(
        left="input2", right=mult1.output, output=IOKey(name="output")
    )

    mult3 = Multiply()
    mult3.set_types(left=Tensor, right=Tensor)
    mult3.set_shapes(left=[4, 5, 7, 3, 4])
    m2 |= mult3.connect(left="input1", right="input2")
    m2 |= Multiply().connect(
        left="input2", right=mult3.output, output=IOKey(name="output")
    )

    composite |= m1.connect(input1="input1", input2="input2")
    composite |= m2.connect(
        input1=m1.output,  # type: ignore
        input2="input2",
        output=IOKey(name="output"),
    )

    logical_ref: Mapping[str, list | None] = {
        "input1": [4, 5, 7, 1, 1],
        "input2": ["(V1, ...)", 3, 4],
        "$_Model_0_output": [4, 5, 7, 3, 4],
        "output": [4, 5, 7, 3, 4],
    }
    physical_ref: Mapping[str, list | None] = {
        "input1": [4, 5, 7, 1, 1],
        "input2": ["...", 3, 4],
        "output_0": [4, 5, 7, 3, 4],
        "output_1": [4, 5, 7, 3, 4],
        "output_2": [4, 5, 7, 3, 4],
        "output": [4, 5, 7, 3, 4],
    }

    assert_shapes(composite, logical_ref, physical_ref)


def test_composite_2_set_shapes_3_2():
    """This test requires backtracking in order to infer all source of
    unknown values comes from input2. Since we solve constraints
    locally for now, it is impossible to infer final values.
    """
    for _ in range(20):
        composite = Model()
        m1 = Model()

        mult1 = Multiply()
        mult1.set_types(left=Tensor, right=Tensor)
        mult1.set_shapes(left=[4, 5, 7, 1, 1])
        m1 |= mult1.connect(left="input1", right="input2")
        m1 |= Multiply().connect(
            left="input2", right=mult1.output, output=IOKey(name="output")
        )

        mult3 = Multiply()
        mult3.set_types(left=Tensor)
        mult3.set_shapes(left=[4, 5, 7, 3, 4])

        composite |= m1.connect(input1="input1", input2="input2")
        composite |= mult3.connect(
            left=m1.output,  # type: ignore
            right="input2",
            output=IOKey(name="output"),
        )

    logical_ref: Mapping[str, list | None] = {
        "input1": [4, 5, 7, 1, 1],
        "input2": ["(V1, ...)", 3, 4],
        "$_Model_0_output": [4, 5, 7, 3, 4],
        "output": [4, 5, 7, 3, 4],
    }
    physical_ref: Mapping[str, list | None] = {
        "input1": [4, 5, 7, 1, 1],
        "input2": ["...", 3, 4],
        "output_0": [4, 5, 7, 3, 4],
        "output_1": [4, 5, 7, 3, 4],
        "output": [4, 5, 7, 3, 4],
    }

    assert_shapes(composite, logical_ref, physical_ref)


def get_composite_2():
    # Create common composite_2 model for corresponding tests.
    composite_2 = Model()
    m1 = Model()
    m2 = Model()
    m3 = Model()
    mult1 = Multiply()
    mult1.set_types(left=Tensor, right=Tensor)
    m1 |= mult1.connect(left="input1", right="input2")
    m1 |= (mult2 := Multiply()).connect(left="input2", right=mult1.output)
    m1 |= Add().connect(
        left=mult2.output, right=mult2.output, output=IOKey(name="output")
    )
    mult3 = Multiply()
    mult3.set_types(left=Tensor, right=Tensor)
    m2 |= mult3.connect(left="input1", right="input2")
    m2 |= (mult4 := Multiply()).connect(left="input2", right=mult3.output)
    m2 |= Add().connect(
        left=mult4.output, right=mult4.output, output=IOKey(name="output")
    )
    m3 |= (add1 := Add()).connect(left="input1", right="input2")
    m3 |= (mult5 := Multiply()).connect(left="input2", right=add1.output)
    m3 |= Add().connect(
        left=mult5.output, right=mult5.output, output=IOKey(name="output")
    )
    composite_2 |= m1.connect(input1="input1", input2="input2")
    composite_2 |= m2.connect(input1=m1.output, input2="input2")  # type: ignore
    composite_2 |= m3.connect(
        input1=m2.output,  # type: ignore
        input2=m2.output,  # type: ignore
        output=IOKey(name="output"),
    )
    return composite_2


def test_composite_2_static_shapes_1():
    model = deepcopy(get_composite_2())
    shapes = {"input1": [4, 5, 7, 1, 1], "input2": [1, 1, 7, 3, 4]}
    logical_ref = {
        "input1": ["(V1, ...)"],
        "input2": ["(V2, ...)"],
        "$_Model_0_output": ["(V3, ...)"],
        "$_Model_1_output": ["(V4, ...)"],
        "output": ["(V4, ...)"],
    }
    physical_ref = {
        "input1": [4, 5, 7, 1, 1],
        "input2": [1, 1, 7, 3, 4],
        "output_0": [4, 5, 7, 3, 4],
        "output_1": [4, 5, 7, 3, 4],
        "output_2": [4, 5, 7, 3, 4],
        "output_3": [4, 5, 7, 3, 4],
        "output_4": [4, 5, 7, 3, 4],
        "output_5": [4, 5, 7, 3, 4],
        "output_6": [4, 5, 7, 3, 4],
        "output_7": [4, 5, 7, 3, 4],
        "output": [4, 5, 7, 3, 4],
    }
    assert_shapes(model, logical_ref, physical_ref, shapes=shapes)


@pytest.mark.skip(reason="Known Bugs")
def test_composite_2_static_shapes_2():
    model = deepcopy(get_composite_2())
    shapes = {"input1": [4, 5, 7, 1, 1], "output": [4, 5, 7, 3, 4]}
    logical_ref = {
        "input1": ["(V1, ...)"],
        "input2": ["(V2, ...)"],
        "$_Model_0_output": ["(V3, ...)"],
        "$_Model_1_output": ["(V4, ...)"],
        "output": ["(V4, ...)"],
    }
    physical_ref: Mapping[str, list | None] = {
        "input1": [4, 5, 7, 1, 1],
        "input2": ["...", 3, 4],
        "_Model_0_output": [4, 5, 7, 3, 4],
        "_Model_1_output": [4, 5, 7, 3, 4],
        "output": [4, 5, 7, 3, 4],
    }
    assert_shapes(model, logical_ref, physical_ref, shapes=shapes)


def test_variadic_contradiction():
    ce = CrossEntropy()
    with pytest.raises(ValueError):
        ce.set_shapes(output=[8, ("V1", ...)], input=[8, 4, ("V1", ...), 64, 128])


def test_cross_entropy_shapes_1():
    model = Model()
    ce = CrossEntropy()
    ce.set_shapes(input=[8, 10], target=[8])
    model |= ce.connect(
        input="input", target="target", categorical=True, output=IOKey(name="output")
    )
    logical_ref = {
        "input": [8, 10],
        "target": [8],
        "$_CrossEntropy_0_categorical": None,
        "$_CrossEntropy_0_threshold": [],
        "$_CrossEntropy_0_robust": None,
        "output": [8],
        "$_CrossEntropy_0_weights": None,
    }
    physical_ref = {
        "input": [8, 10],
        "target": [8],
        "weights": None,
        "categorical": None,
        "threshold": [],
        "robust": None,
        "output": [8],
    }

    assert_shapes(model, logical_ref, physical_ref)


def test_cross_entropy_shapes_2():
    model = Model()
    ce = CrossEntropy(categorical=TBD)
    ce.set_shapes(input=[8, 10])
    model |= ce.connect(
        input="input", target="target", categorical=False, output=IOKey(name="output")
    )

    logical_ref = {
        "input": [8, 10],
        "target": [8, 10],
        "$categorical": None,
        "$_CrossEntropy_0_threshold": [],
        "$_CrossEntropy_0_robust": None,
        "output": [8],
        "$_CrossEntropy_0_weights": None,
    }
    physical_ref = {
        "input": [8, 10],
        "target": [8, 10],
        "weights": None,
        "categorical": None,
        "threshold": [],
        "robust": None,
        "output": [8],
    }

    assert_shapes(model, logical_ref, physical_ref)


def test_cross_entropy_shapes_3():
    model = Model()
    ce = CrossEntropy(categorical=TBD)
    ce.set_shapes(input=[8, 16, 32, 64], target=[8, 32, 64])
    model += ce.connect(
        input="input", target="target", categorical=True, output=IOKey(name="output")
    )
    logical_ref = {
        "input": [8, 16, 32, 64],
        "target": [8, 32, 64],
        "$categorical": None,
        "$_CrossEntropy_0_threshold": [],
        "$_CrossEntropy_0_robust": None,
        "output": [8, 32, 64],
        "$_CrossEntropy_0_weights": None,
    }
    physical_ref = {
        "input": [8, 16, 32, 64],
        "target": [8, 32, 64],
        "weights": None,
        "categorical": None,
        "threshold": [],
        "robust": None,
        "output": [8, 32, 64],
    }

    assert_shapes(model, logical_ref, physical_ref)


def test_cross_entropy_shapes_5():
    model = Model()
    ce = CrossEntropy(categorical=TBD)
    ce.set_shapes(input=[8, 16, ("V1", ...), 64], target=[8, 32, 64])
    model += ce.connect(
        input="input", target="target", categorical=True, output=IOKey(name="output")
    )
    logical_ref = {
        "input": [8, 16, 32, 64],
        "target": [8, 32, 64],
        "$categorical": None,
        "$_CrossEntropy_0_threshold": [],
        "$_CrossEntropy_0_robust": None,
        "output": [8, 32, 64],
        "$_CrossEntropy_0_weights": None,
    }
    physical_ref = {
        "input": [8, 16, 32, 64],
        "target": [8, 32, 64],
        "weights": None,
        "categorical": None,
        "threshold": [],
        "robust": None,
        "output": [8, 32, 64],
    }

    assert_shapes(model, logical_ref, physical_ref)


def test_cross_entropy_shapes_6():
    model = Model()
    ce = CrossEntropy(categorical=TBD)
    ce.set_shapes(input=[8, 16, ("V1", ...), 64], output=[8, 32, 64])
    model += ce.connect(
        input="input", target="target", categorical=True, output=IOKey(name="output")
    )
    logical_ref = {
        "input": [8, 16, 32, 64],
        "target": [8, 32, 64],
        "$categorical": None,
        "$_CrossEntropy_0_threshold": [],
        "$_CrossEntropy_0_robust": None,
        "output": [8, 32, 64],
        "$_CrossEntropy_0_weights": None,
    }
    physical_ref = {
        "input": [8, 16, 32, 64],
        "target": [8, 32, 64],
        "weights": None,
        "categorical": None,
        "threshold": [],
        "robust": None,
        "output": [8, 32, 64],
    }

    assert_shapes(model, logical_ref, physical_ref)


def test_cross_entropy_shapes_7():
    model = Model()
    ce = CrossEntropy(categorical=TBD)
    ce.set_shapes(input=[("V1", ...), 64], target=[8, 16, 32, 64])
    model += ce.connect(
        input="input", target="target", categorical=True, output=IOKey(name="output")
    )

    logical_ref: Mapping = {
        "input": [8, "u1", 16, 32, 64],
        "target": [8, 16, 32, 64],
        "$categorical": None,
        "$_CrossEntropy_0_threshold": [],
        "$_CrossEntropy_0_robust": None,
        "output": [8, 16, 32, 64],
        "$_CrossEntropy_0_weights": None,
    }
    physical_ref: Mapping = {
        "input": [8, None, 16, 32, 64],
        "target": [8, 16, 32, 64],
        "weights": None,
        "categorical": None,
        "threshold": [],
        "robust": None,
        "output": [8, 16, 32, 64],
    }

    assert_shapes(model, logical_ref, physical_ref)


def test_cross_entropy_shapes_8():
    model = Model()
    ce = CrossEntropy(categorical=TBD)
    ce.set_shapes(input=[("V1", ...), 64], target=[8, 16, 32, 64])
    model += ce.connect(
        input="input", target="target", categorical=False, output=IOKey(name="output")
    )

    logical_ref = {
        "input": [8, 16, 32, 64],
        "target": [8, 16, 32, 64],
        "$categorical": None,
        "$_CrossEntropy_0_threshold": [],
        "$_CrossEntropy_0_robust": None,
        "output": [8, 32, 64],
        "$_CrossEntropy_0_weights": None,
    }
    physical_ref = {
        "input": [8, 16, 32, 64],
        "target": [8, 16, 32, 64],
        "weights": None,
        "categorical": None,
        "threshold": [],
        "robust": None,
        "output": [8, 32, 64],
    }

    assert_shapes(model, logical_ref, physical_ref)


def test_cross_entropy_shapes_9():
    model = Model()
    ce = CrossEntropy(categorical=TBD)
    ce.set_shapes(input=[8, 16, ("V1", ...), 64])
    model += ce.connect(
        input="input", target="target", categorical=True, output=IOKey(name="output")
    )
    logical_ref: Mapping = {
        "input": [8, 16, "(V1, ...)", 64],
        "target": [8, "(V1, ...)", 64],
        "$categorical": None,
        "$_CrossEntropy_0_threshold": [],
        "$_CrossEntropy_0_robust": None,
        "output": [8, "(V1, ...)", 64],
        "$_CrossEntropy_0_weights": None,
    }
    physical_ref: Mapping = {
        "input": [8, 16, "...", 64],
        "target": [8, "...", 64],
        "weights": None,
        "categorical": None,
        "threshold": [],
        "robust": None,
        "output": [8, "...", 64],
    }

    assert_shapes(model, logical_ref, physical_ref)


def test_cross_entropy_shapes_10():
    model = Model()
    ce = CrossEntropy()
    ce.set_shapes(input=[8, 16, ("V1", ...), 64, 128])
    model += ce.connect(input="input", target="target", output=IOKey(name="output"))

    logical_ref: Mapping = {
        "input": [8, 16, "(V1, ...)", 64, 128],
        "target": [8, "(V1, ...)", 64, 128],
        "$_CrossEntropy_0_categorical": None,
        "$_CrossEntropy_0_threshold": [],
        "$_CrossEntropy_0_robust": None,
        "output": [8, "(V1, ...)", 64, 128],
        "$_CrossEntropy_0_weights": None,
    }
    physical_ref: Mapping = {
        "input": [8, 16, "...", 64, 128],
        "target": [8, "...", 64, 128],
        "weights": None,
        "categorical": None,
        "threshold": [],
        "robust": None,
        "output": [8, "...", 64, 128],
    }

    assert_shapes(model, logical_ref, physical_ref)


def test_cross_entropy_shapes_11():
    model = Model()
    ce = CrossEntropy()
    ce.set_shapes(input=[8, 4, ("V2", ...), 64, 128], output=[8, ("V1", ...)])
    model += ce.connect(input="input", target="target", output=IOKey(name="output"))

    logical_ref: Mapping = {
        "input": [8, 4, "(V1, ...)", 64, 128],
        "target": [8, "(V1, ...)", 64, 128],
        "$_CrossEntropy_0_categorical": None,
        "$_CrossEntropy_0_threshold": [],
        "$_CrossEntropy_0_robust": None,
        "output": [8, "(V1, ...)", 64, 128],
        "$_CrossEntropy_0_weights": None,
    }
    physical_ref: Mapping = {
        "input": [8, 4, "...", 64, 128],
        "target": [8, "...", 64, 128],
        "weights": None,
        "categorical": None,
        "threshold": [],
        "robust": None,
        "output": [8, "...", 64, 128],
    }

    assert_shapes(model, logical_ref, physical_ref)


def test_composite_2_static_inputs_1():
    model = deepcopy(get_composite_2())
    logical_ref = {
        "input1": ["(V1, ...)"],
        "input2": ["(V2, ...)"],
        "$_Model_0_output": ["(V3, ...)"],
        "$_Model_1_output": ["(V4, ...)"],
        "output": ["(V4, ...)"],
    }
    physical_ref = {
        "output": [4, 5, 7, 3, 4],
    }

    inputs = {
        "input1": np.random.randn(4, 5, 7, 1, 1),
        "input2": np.random.randn(1, 1, 7, 3, 4),
    }
    assert_shapes(model, logical_ref, physical_ref, static_inputs=inputs)


@pytest.mark.skip(reason="Known Bugs")
def test_composite_2_static_inputs_2():
    model = deepcopy(get_composite_2())
    logical_ref = {
        "input1": ["(V1, ...)"],
        "input2": ["(V2, ...)"],
        "Model_0_output": ["(V3, ...)"],
        "Model_1_output": ["(V4, ...)"],
        "output": ["(V5, ...)"],
    }
    physical_ref: dict[str, list] = {
        "input1": [4, 5, 7, 1, 1],
        "input2": ["...", 3, 4],
        "Model_0_output": [4, 5, 7, 3, 4],
        "Model_1_output": [4, 5, 7, 3, 4],
        "output": [4, 5, 7, 3, 4],
    }
    inputs = {
        "input1": np.random.randn(4, 5, 7, 1, 1),
        "output": np.random.randn(4, 5, 7, 3, 4),
    }
    assert_shapes(model, logical_ref, physical_ref, static_inputs=inputs)


def get_composite_3():
    composite_3 = Model()
    m1 = Model()
    m1 |= Add().connect(
        left=IOKey("input1", type=Tensor),
        right=IOKey("input2", type=Tensor),
        output=IOKey(name="output"),
    )
    m2 = Model()
    m2 |= m1.connect(input1="input1", input2="input2")
    m2 |= Add().connect(left="input1", right=m1.output, output=IOKey(name="output"))  # type: ignore
    m3 = Model()
    m3 |= m2.connect(input1="input1", input2="input2")
    m3 |= Add().connect(left="input1", right=m2.output, output=IOKey(name="output"))  # type: ignore
    m4 = Model()
    m4 |= m3.connect(input1="input1", input2="input2")
    m4 |= Add().connect(left="input1", right=m3.output, output=IOKey(name="output"))  # type: ignore
    composite_3 |= m4.connect(input1="input1", input2="input2")
    composite_3 |= Add().connect(
        left="input1",
        right=m4.output,  # type: ignore
        output=IOKey(name="output"),
    )
    return composite_3


def test_composite_3_set_shapes_1():
    composite_3 = Model()
    m1 = Model()
    add1 = Add()
    add1.set_shapes(left=[3, 4, 5, 6, 1], right=[1, 1, 1, 1, 7])
    m1 |= add1.connect(left="input1", right="input2", output=IOKey(name="output"))
    m2 = Model()
    m2 |= m1.connect(input1="input1", input2="input2")
    m2 |= Add().connect(left="input1", right=m1.output, output=IOKey(name="output"))  # type: ignore
    m3 = Model()
    m3 |= m2.connect(input1="input1", input2="input2")
    m3 |= Add().connect(left="input1", right=m2.output, output=IOKey(name="output"))  # type: ignore
    m4 = Model()
    m4 |= m3.connect(input1="input1", input2="input2")
    m4 |= Add().connect(left="input1", right=m3.output, output=IOKey(name="output"))  # type: ignore
    composite_3 |= m4.connect(input1="input1", input2="input2")
    composite_3 |= Add().connect(
        left="input1",
        right=m4.output,  # type: ignore
        output=IOKey(name="output"),
    )
    logical_ref = {
        "input1": [3, 4, 5, 6, 1],
        "input2": [1, 1, 1, 1, 7],
        "$_Model_0_output": [3, 4, 5, 6, 7],
        "output": [3, 4, 5, 6, 7],
    }
    physical_ref = {
        "input1": [3, 4, 5, 6, 1],
        "input2": [1, 1, 1, 1, 7],
        "output_0": [3, 4, 5, 6, 7],
        "output_1": [3, 4, 5, 6, 7],
        "output_2": [3, 4, 5, 6, 7],
        "output_3": [3, 4, 5, 6, 7],
        "output": [3, 4, 5, 6, 7],
    }

    assert_shapes(composite_3, logical_ref, physical_ref)


def test_composite_3_extend_shapes_1():
    composite_3 = Model()
    m1 = Model()
    add1 = Add()
    add_1_left: Tensor[float] = Tensor(np.random.randn(3, 4, 5, 6, 1).tolist())
    add_1_right: Tensor[float] = Tensor(np.random.randn(1, 1, 1, 1, 7).tolist())
    m1 += add1.connect(
        left=IOKey(value=add_1_left, name="left"),
        right=IOKey(value=add_1_right, name="right"),
        output="output",
    )
    m1.expose_keys("output")
    m2 = Model()
    m2 |= m1.connect(left=IOKey(name="left"), right=IOKey(name="right"))
    m2 |= Add().connect(left=m1.left, right=m1.output, output="output")  # type: ignore
    m2.expose_keys("output", "left", "right")
    m3 = Model()
    m3 |= m2.connect(right=IOKey(name="right"))
    m3 |= Add().connect(
        left=IOKey(name="left"),  # type: ignore
        right=m2.output,  # type: ignore
        output="output",
    )  # type: ignore
    m3.merge_connections(m3.left, m1.left)  # type: ignore
    m3.expose_keys("output", "left", "right")
    m4 = Model()
    m4 |= m3.connect(left=IOKey(name="left"), right=IOKey(name="right"))
    m4 |= (add4 := Add()).connect(
        left=m1.left,  # type: ignore
        right=m3.output,  # type: ignore
        output="output",
    )
    m4.expose_keys("output", "left", "right")
    composite_3 |= m4.connect()
    composite_3 |= (add5 := Add()).connect(
        left=m1.left,  # type: ignore
        right=m4.output,  # type: ignore
        output="output",
    )
    composite_3.expose_keys("output")

    key_mappings = composite_3.generate_keys()

    composite_3_left_metadata = composite_3.conns.get_con_by_metadata(m1.left.metadata)  # type: ignore
    assert composite_3_left_metadata is not None
    composite_3_left_key = key_mappings[composite_3_left_metadata.key]

    composite_3_right_metadata = composite_3.conns.get_con_by_metadata(
        m1.right.metadata  # type: ignore
    )
    assert composite_3_right_metadata is not None
    composite_3_right_key = key_mappings[composite_3_right_metadata.key]

    add5_left_metadata = composite_3.conns.get_con_by_metadata(add5.left.metadata)
    assert add5_left_metadata is not None
    add5_left_key = key_mappings[add5_left_metadata.key]

    m4_out_metadata = composite_3.conns.get_con_by_metadata(add4.output.metadata)
    assert m4_out_metadata is not None
    m4_out_key = key_mappings[m4_out_metadata.key]

    logical_ref = {
        composite_3_left_key: [3, 4, 5, 6, 1],
        composite_3_right_key: [1, 1, 1, 1, 7],
        add5_left_key: [3, 4, 5, 6, 1],
        m4_out_key: [3, 4, 5, 6, 7],
        "output": [3, 4, 5, 6, 7],
    }
    physical_ref = {
        "output": [3, 4, 5, 6, 7],
    }

    assert_shapes(composite_3, logical_ref, physical_ref)


def test_composite_3_set_shapes_1_2():
    composite_3 = Model()
    m1 = Model()
    add1 = Add()
    m1 |= add1.connect(left="input1", right="input2", output=IOKey(name="output"))
    m2 = Model()
    m2 |= m1.connect(input1="input1", input2="input2")
    m2 |= Add().connect(left="input1", right=m1.output, output=IOKey(name="output"))  # type: ignore
    m3 = Model()
    m3 |= m2.connect(input1="input1", input2="input2")
    m3 |= Add().connect(left="input1", right=m2.output, output=IOKey(name="output"))  # type: ignore
    m4 = Model()
    m4 |= m3.connect(input1="input1", input2="input2")
    m4 |= Add().connect(left="input1", right=m3.output, output=IOKey(name="output"))  # type: ignore
    composite_3 |= m4.connect(input1="input1", input2="input2")
    composite_3 |= Add().connect(
        left="input1",
        right=m4.output,  # type: ignore
        output=IOKey(name="output"),
    )
    add1.set_shapes(left=[3, 4, 5, 6, 1], right=[1, 1, 1, 1, 7])
    logical_ref = {
        "input1": [3, 4, 5, 6, 1],
        "input2": [1, 1, 1, 1, 7],
        "$_Model_0_output": [3, 4, 5, 6, 7],
        "output": [3, 4, 5, 6, 7],
    }
    physical_ref = {
        "input1": [3, 4, 5, 6, 1],
        "input2": [1, 1, 1, 1, 7],
        "output_0": [3, 4, 5, 6, 7],
        "output_1": [3, 4, 5, 6, 7],
        "output_2": [3, 4, 5, 6, 7],
        "output_3": [3, 4, 5, 6, 7],
        "output": [3, 4, 5, 6, 7],
    }

    assert_shapes(composite_3, logical_ref, physical_ref)


def test_composite_3_set_shapes_2_2():
    composite_3 = Model()
    m1 = Model()
    add1 = Add()
    m1 |= add1.connect(left="input1", right="input2", output=IOKey(name="output"))
    m2 = Model()
    m2 |= m1.connect(input1="input1", input2="input2")
    m2 |= Add().connect(left="input1", right=m1.output, output=IOKey(name="output"))  # type: ignore
    m3 = Model()
    m3 |= m2.connect(input1="input1", input2="input2")
    add3 = Add()
    m3 |= add3.connect(left="input1", right=m2.output, output=IOKey(name="output"))  # type: ignore
    m4 = Model()
    m4 |= m3.connect(input1="input1", input2="input2")
    m4 |= Add().connect(left="input1", right=m3.output, output=IOKey(name="output"))  # type: ignore
    composite_3 |= m4.connect(input1="input1", input2="input2")
    add1.set_shapes(right=[1, 1, 1, 1, 7])
    composite_3 |= Add().connect(
        left="input1",
        right=m4.output,  # type: ignore
        output=IOKey(name="output"),
    )
    add3.set_shapes(left=[3, 4, 5, 6, 1])

    logical_ref = {
        "input1": [3, 4, 5, 6, 1],
        "input2": [1, 1, 1, 1, 7],
        "$_Model_0_output": [3, 4, 5, 6, 7],
        "output": [3, 4, 5, 6, 7],
    }
    physical_ref = {
        "input1": [3, 4, 5, 6, 1],
        "input2": [1, 1, 1, 1, 7],
        "output_0": [3, 4, 5, 6, 7],
        "output_1": [3, 4, 5, 6, 7],
        "output_2": [3, 4, 5, 6, 7],
        "output_3": [3, 4, 5, 6, 7],
        "output": [3, 4, 5, 6, 7],
    }

    assert_shapes(composite_3, logical_ref, physical_ref)


def test_composite_3_set_shapes_2_3():
    composite_3 = Model()
    m1 = Model()
    add1 = Add()
    m1 |= add1.connect(left="input1", right="input2", output=IOKey(name="output"))
    m2 = Model()
    m2 |= m1.connect(input1="input1", input2="input2")
    m2 |= Add().connect(left="input1", right=m1.output, output=IOKey(name="output"))  # type: ignore
    m3 = Model()
    m3 |= m2.connect(input1="input1", input2="input2")
    add3 = Add()
    m3 |= add3.connect(left="input1", right=m2.output, output=IOKey(name="output"))  # type: ignore
    m4 = Model()
    m4 |= m3.connect(input1="input1", input2="input2")
    m4 |= Add().connect(left="input1", right=m3.output, output=IOKey(name="output"))  # type: ignore
    composite_3 |= m4.connect(input1="input1", input2="input2")
    add3.set_shapes(left=[3, 4, 5, 6, 1])
    composite_3 |= Add().connect(
        left="input1",
        right=m4.output,  # type: ignore
        output=IOKey(name="output"),
    )
    add1.set_shapes(right=[1, 1, 1, 1, 7])

    logical_ref = {
        "input1": [3, 4, 5, 6, 1],
        "input2": [1, 1, 1, 1, 7],
        "$_Model_0_output": [3, 4, 5, 6, 7],
        "output": [3, 4, 5, 6, 7],
    }
    physical_ref = {
        "input1": [3, 4, 5, 6, 1],
        "input2": [1, 1, 1, 1, 7],
        "output_0": [3, 4, 5, 6, 7],
        "output_1": [3, 4, 5, 6, 7],
        "output_2": [3, 4, 5, 6, 7],
        "output_3": [3, 4, 5, 6, 7],
        "output": [3, 4, 5, 6, 7],
    }

    assert_shapes(composite_3, logical_ref, physical_ref)


def test_composite_3_set_shapes_2():
    composite_3 = Model()
    m1 = Model()
    add1 = Add()
    add1.set_shapes(right=[1, 1, 1, 1, 7])
    m1 |= add1.connect(left="input1", right="input2", output=IOKey(name="output"))
    m2 = Model()
    m2 |= m1.connect(input1="input1", input2="input2")
    m2 |= Add().connect(left="input1", right=m1.output, output=IOKey(name="output"))  # type: ignore
    m3 = Model()
    m3 |= m2.connect(input1="input1", input2="input2")
    add3 = Add()
    add3.set_shapes(left=[3, 4, 5, 6, 1])
    m3 |= add3.connect(left="input1", right=m2.output, output=IOKey(name="output"))  # type: ignore
    m4 = Model()
    m4 |= m3.connect(input1="input1", input2="input2")
    m4 |= Add().connect(left="input1", right=m3.output, output=IOKey(name="output"))  # type: ignore
    composite_3 |= m4.connect(input1="input1", input2="input2")
    composite_3 |= Add().connect(
        left="input1",
        right=m4.output,  # type: ignore
        output=IOKey(name="output"),
    )

    logical_ref = {
        "input1": [3, 4, 5, 6, 1],
        "input2": [1, 1, 1, 1, 7],
        "$_Model_0_output": [3, 4, 5, 6, 7],
        "output": [3, 4, 5, 6, 7],
    }

    physical_ref = {
        "input1": [3, 4, 5, 6, 1],
        "input2": [1, 1, 1, 1, 7],
        "output_0": [3, 4, 5, 6, 7],
        "output_1": [3, 4, 5, 6, 7],
        "output_2": [3, 4, 5, 6, 7],
        "output_3": [3, 4, 5, 6, 7],
        "output": [3, 4, 5, 6, 7],
    }
    assert_shapes(composite_3, logical_ref, physical_ref)


def test_composite_3_static_shapes_1():
    model = deepcopy(get_composite_3())
    logical_ref = {
        "input1": ["(V1, ...)"],
        "input2": ["(V2, ...)"],
        "$_Model_0_output": ["(V3, ...)"],
        "output": ["(V4, ...)"],
    }
    physical_ref = {
        "input1": [3, 4, 5, 6, 1],
        "input2": [1, 1, 1, 1, 7],
        "output_0": [3, 4, 5, 6, 7],
        "output_1": [3, 4, 5, 6, 7],
        "output_2": [3, 4, 5, 6, 7],
        "output_3": [3, 4, 5, 6, 7],
        "output": [3, 4, 5, 6, 7],
    }
    shapes = {"input1": [3, 4, 5, 6, 1], "input2": [1, 1, 1, 1, 7]}
    assert_shapes(model, logical_ref, physical_ref, shapes=shapes)


# @pytest.mark.skip("Known Bug")
def test_composite_3_static_shapes_2():
    model = deepcopy(get_composite_3())
    logical_ref = {
        "input1": ["(V1, ...)"],
        "input2": ["(V2, ...)"],
        "$_Model_0_output": ["(V3, ...)"],
        "output": ["(V4, ...)"],
    }
    physical_ref: dict[str, list] = {
        "input1": [3, 4, 5, 6, 1],
        "input2": ["...", 7],
        "output_0": [3, 4, 5, 6, 7],
        "output_1": [3, 4, 5, 6, 7],
        "output_2": [3, 4, 5, 6, 7],
        "output_3": [3, 4, 5, 6, 7],
        "output": [3, 4, 5, 6, 7],
    }
    shapes = {"input1": [3, 4, 5, 6, 1], "output": [3, 4, 5, 6, 7]}
    assert_shapes(model, logical_ref, physical_ref, shapes=shapes)


def test_composite_3_static_inputs_2():
    model = deepcopy(get_composite_3())
    logical_ref = {
        "input1": ["(V1, ...)"],
        "input2": ["(V2, ...)"],
        "$_Model_0_output": ["(V3, ...)"],
        "output": ["(V4, ...)"],
    }
    physical_ref = {
        "output": [3, 4, 5, 6, 7],
    }
    inputs = {
        "input1": np.random.randn(3, 4, 5, 6, 1),
        "input2": np.random.randn(1, 1, 1, 1, 7),
    }
    assert_shapes(model, logical_ref, physical_ref, static_inputs=inputs)


def test_mlp_1_static_shapes():
    model = MLP(activations=[Softplus(), Buffer(), Buffer()], dimensions=[5, 10, 1])
    ctx = TrainModel(model)
    loss_model = SquaredError()
    loss_model.set_shapes(**loss_model.submodel.safe_shapes)
    ctx.add_loss(loss_model, input=model.output, target="target", reduce_steps=[Mean()])
    static_input_shapes = {"input": [100, 4], "target": [100, 1]}
    logical_ref: dict[str, list | None] = {
        "$_SquaredError_1_output": [["(V1, ...)", "u1", 1], ["u2", "(V2, ...)", 1]],
        "$_Mean_2_output": [],
        "weight0": [5, "u3"],
        "input": [["(V1, ...)", "u1", "u3"], ["u2", "(V2, ...)", "u3"]],
        "bias0": [5],
        "weight1": [10, 5],
        "bias1": [10],
        "weight2": [1, 10],
        "bias2": [1],
        "target": [["(V1, ...)", "u1", 1], ["u2", "(V2, ...)", 1]],
        "output": [["(V1, ...)", "u1", 1], ["u2", "(V2, ...)", 1]],
        "$_Mean_2_axis": None,
        "$_Mean_2_keepdim": None,
    }

    physical_ref = {
        "weight0": [5, 4],
        "axes_0": None,
        "output_0": [4, 5],
        "input": [100, 4],
        "output_1": [100, 5],
        "bias0": [5],
        "output_2": [100, 5],
        "output_3": [100, 5],
        "weight1": [10, 5],
        "axes_1": None,
        "output_4": [5, 10],
        "output_5": [100, 10],
        "bias1": [10],
        "output_6": [100, 10],
        "weight2": [1, 10],
        "axes_2": None,
        "output_8": [10, 1],
        "output_9": [100, 1],
        "bias2": [1],
        "output_10": [100, 1],
        "target": [100, 1],
        "output_11": [100, 1],
        "axis": None,
        "keepdim": None,
        "output_12": [],
    }

    assert_shapes(
        ctx,
        logical_ref,
        physical_ref,
        shapes=static_input_shapes,
        check_all_shapes=True,
    )


def test_mlp_1_set_shapes():
    model = MLP(activations=[Softplus(), Buffer(), Buffer()], dimensions=[5, 10, 1])
    ctx = TrainModel(model)
    loss_model = SquaredError()
    loss_model.set_shapes(**loss_model.submodel.safe_shapes)
    ctx.add_loss(loss_model, input=model.output, target="target", reduce_steps=[Mean()])
    ctx.set_shapes(input=[100, 4], target=[100, 1])

    logical_ref = {
        "$_SquaredError_1_output": [100, 1],
        "$_Mean_2_output": [],
        "weight0": [5, 4],
        "input": [100, 4],
        "bias0": [5],
        "weight1": [10, 5],
        "bias1": [10],
        "weight2": [1, 10],
        "bias2": [1],
        "target": [100, 1],
        "output": [100, 1],
        "$_Mean_2_axis": None,
        "$_Mean_2_keepdim": None,
    }
    physical_ref = {
        "weight0": [5, 4],
        "axes_0": None,
        "output_0": [4, 5],
        "input": [100, 4],
        "output_1": [100, 5],
        "bias0": [5],
        "output_2": [100, 5],
        "output_3": [100, 5],
        "weight1": [10, 5],
        "axes_1": None,
        "output_4": [5, 10],
        "output_5": [100, 10],
        "bias1": [10],
        "output_6": [100, 10],
        "weight2": [1, 10],
        "axes_2": None,
        "output_8": [10, 1],
        "output_9": [100, 1],
        "bias2": [1],
        "output_10": [100, 1],
        "target": [100, 1],
        "output_11": [100, 1],
        "axis": None,
        "keepdim": None,
        "output_12": [],
    }
    assert_shapes(ctx, logical_ref, physical_ref)


def test_mlp_1_static_inputs():
    model = MLP(activations=[Softplus(), Buffer(), Buffer()], dimensions=[5, 10, 1])
    ctx = TrainModel(model)
    loss_model = SquaredError()
    loss_model.set_shapes(**loss_model.submodel.safe_shapes)

    ctx.add_loss(loss_model, input=model.output, target="target", reduce_steps=[Mean()])
    static_inputs = {
        "input": np.random.randn(100, 4),
        "target": np.random.randn(100, 1),
    }
    logical_ref: dict[str, list | None] = {
        "$_SquaredError_1_output": [["(V1, ...)", "u1", 1], ["u2", "(V2, ...)", 1]],
        "$_Mean_2_output": [],
        "weight0": [5, "u3"],
        "input": [["(V1, ...)", "u1", "u3"], ["u2", "(V2, ...)", "u3"]],
        "bias0": [5],
        "weight1": [10, 5],
        "bias1": [10],
        "weight2": [1, 10],
        "bias2": [1],
        "target": [["(V1, ...)", "u1", 1], ["u2", "(V2, ...)", 1]],
        "output": [["(V1, ...)", "u1", 1], ["u2", "(V2, ...)", 1]],
        "$_Mean_2_axis": None,
        "$_Mean_2_keepdim": None,
    }

    physical_ref = {
        "weight0": [5, 4],
        "axes_0": None,
        "output_0": [4, 5],
        "input": [100, 4],
        "output_1": [100, 5],
        "bias0": [5],
        "output_2": [100, 5],
        "output_3": [100, 5],
        "weight1": [10, 5],
        "axes_1": None,
        "output_4": [5, 10],
        "output_5": [100, 10],
        "bias1": [10],
        "output_6": [100, 10],
        "weight2": [1, 10],
        "axes_2": None,
        "output_8": [10, 1],
        "output_9": [100, 1],
        "bias2": [1],
        "output_10": [100, 1],
        "target": [100, 1],
        "output_11": [100, 1],
        "axis": None,
        "keepdim": None,
        "output_12": [],
    }
    assert_shapes(
        ctx,
        logical_ref,
        physical_ref,
        static_inputs=static_inputs,
        check_all_shapes=True,
    )


def test_mlp_reshape_model():
    mlp = Model()
    mlp += (_mlp := MLP([Softplus(), Softplus(), Softplus()], [100, 50, None]))
    _mlp.set_shapes(input=[100, 200])
    mlp += Reshape(shape=(100, 74, 1))
    logical_ref = {
        "$_MLP_0_output": [100, 74],
        "$_Reshape_1_output": [100, 74, 1],
        "$weight0": [100, 200],
        "$input": [100, 200],
        "$bias0": [100],
        "$weight1": [50, 100],
        "$bias1": [50],
        "$weight2": [74, 50],
        "$bias2": [74],
        "$_Reshape_1_shape": None,
    }

    physical_ref = {
        "weight0": [100, 200],
        "axes_0": None,
        "output_0": [200, 100],
        "input": [100, 200],
        "output_1": [100, 100],
        "bias0": [100],
        "output_2": [100, 100],
        "output_3": [100, 100],
        "weight1": [50, 100],
        "axes_1": None,
        "output_4": [100, 50],
        "output_5": [100, 50],
        "bias1": [50],
        "output_6": [100, 50],
        "output_7": [100, 50],
        "weight2": [74, 50],
        "axes_2": None,
        "output_8": [50, 74],
        "output_9": [100, 74],
        "bias2": [74],
        "output_10": [100, 74],
        "output_11": [100, 74],
        "_shape": None,
        "output": [100, 74, 1],
    }
    assert_shapes(
        mlp, logical_ref, physical_ref, static_inputs={}, check_all_shapes=True
    )


def test_flatten_1():
    model = Flatten(start_dim=1, end_dim=-1)
    static_shapes = {"input": [10, 5, 20, 30]}

    logical_ref = {
        "output": ["u1", "u2"],
        "input": ["u1", "u3", "(V1, ...)"],
        "start_dim": None,
        "end_dim": None,
    }
    physical_ref: dict[str, list | None] = {
        "input": [10, 5, 20, 30],
        "output": [10, 3000],
        "start_dim": None,
        "end_dim": None,
    }
    assert_shapes(model, logical_ref, physical_ref, shapes=static_shapes)


def test_flatten_2():
    model = Flatten(start_dim=1, end_dim=-1)
    static_shapes = {"input": [None, 5, 20, 30]}

    logical_ref = {
        "output": ["u1", "u2"],
        "input": ["u1", "u3", "(V1, ...)"],
        "start_dim": None,
        "end_dim": None,
    }
    physical_ref: dict[str, list | None] = {
        "input": [None, 5, 20, 30],
        "output": [None, 3000],
        "start_dim": None,
        "end_dim": None,
    }
    assert_shapes(model, logical_ref, physical_ref, shapes=static_shapes)


def test_flatten_3():
    model = Flatten(start_dim=1, end_dim=-1)
    static_shapes = {"input": [8, None, 20, 30]}

    logical_ref = {
        "output": ["u1", "u2"],
        "input": ["u1", "u3", "(V1, ...)"],
        "start_dim": None,
        "end_dim": None,
    }
    physical_ref: dict[str, list | None] = {
        "input": [8, None, 20, 30],
        "output": [8, None],
        "start_dim": None,
        "end_dim": None,
    }
    assert_shapes(model, logical_ref, physical_ref, shapes=static_shapes)


def test_flatten_4():
    model = Flatten(start_dim=1, end_dim=-1)
    static_shapes = {"input": [8, 3, None, None]}

    logical_ref = {
        "output": ["u1", "u2"],
        "input": ["u1", "u3", "(V1, ...)"],
        "start_dim": None,
        "end_dim": None,
    }
    physical_ref: dict[str, list | None] = {
        "input": [8, 3, None, None],
        "output": [8, None],
        "start_dim": None,
        "end_dim": None,
    }
    assert_shapes(model, logical_ref, physical_ref, shapes=static_shapes)


def test_flatten_5():
    model = Flatten(start_dim=3, end_dim=-1)
    static_shapes = {"input": [8, 3, 2, 4, 5]}

    logical_ref = {
        "output": ["u1", "u2", "u3", "u4"],
        "input": ["u1", "u2", "u3", "u5", "(V1, ...)"],
        "start_dim": None,
        "end_dim": None,
    }
    physical_ref = {
        "input": [8, 3, 2, 4, 5],
        "output": [8, 3, 2, 20],
        "start_dim": None,
        "end_dim": None,
    }
    assert_shapes(model, logical_ref, physical_ref, shapes=static_shapes)


def test_flatten_6():
    model = Flatten(start_dim=3, end_dim=-1)
    static_shapes = {"input": [8, 3, None, 4, 5]}

    logical_ref = {
        "output": ["u1", "u2", "u3", "u4"],
        "input": ["u1", "u2", "u3", "u5", "(V1, ...)"],
        "start_dim": None,
        "end_dim": None,
    }
    physical_ref: dict[str, list | None] = {
        "output": [8, 3, None, 20],
        "input": [8, 3, None, 4, 5],
        "start_dim": None,
        "end_dim": None,
    }
    assert_shapes(model, logical_ref, physical_ref, shapes=static_shapes)


def test_flatten_7():
    model = Flatten()
    static_shapes = {"input": [8, 3, 2]}

    logical_ref = {
        "output": ["u1"],
        "input": ["u2", "(V1, ...)"],
        "start_dim": None,
        "end_dim": None,
    }
    physical_ref = {
        "input": [8, 3, 2],
        "output": [48],
        "start_dim": None,
        "end_dim": None,
    }
    assert_shapes(model, logical_ref, physical_ref, shapes=static_shapes)


def test_shape_1():
    """
    This is test is for testing overlap algorthm of the shapes, In this test, two
    buffer models are connected serially, shape of first model is set to
    [("V1", ...), "a", "b"] as seen in the test. And shape of second model is set
    to ["c", "d", ("V1", ...)], When these models connected serially, it is expected
    that shapes will set to ["u1", "u2", ("Var1", ...), "u3", "u4] with overlap
    limit = 2. When shape of model's input set to [5, 6, 7]. It is expected to overlap
    amount become 1. This means that shapes collapse 1 amount and therefore "u2" and
    "u3" point to same representations. This test should work without any errors.
    """
    model = Model()
    buff1 = Buffer()
    buff2 = Buffer()
    shapes_1: dict[str, list] = {"input": [("V1", ...), "a", "b"]}
    shapes_2: dict[str, list] = {"input": ["c", "d", ("V1", ...)]}
    buff1.set_shapes(**shapes_1)
    buff2.set_shapes(**shapes_2)
    model |= buff1.connect(input="input")
    model |= buff2.connect(input=buff1.output, output=IOKey(name="output"))
    model.set_shapes(input=[5, 6, 7])
    logical_ref = {
        "input": [5, 6, 7],
        "$_Buffer_0_output": [5, 6, 7],
        "output": [5, 6, 7],
    }
    physical_ref = {"input": [5, 6, 7]}
    assert_shapes(model, logical_ref, physical_ref)


class Model1(PrimitiveModel):
    input: Connection
    output: Connection

    def __init__(self) -> None:
        super().__init__(
            formula_key="relu",
            input=BaseKey(shape=[("Var1", ...)], type=Tensor),
            output=BaseKey(shape=[("Var1", ...), "u1", "u2"], type=Tensor),
        )

    def connect(  # type: ignore[override]
        self, input: ConnectionType = NOT_GIVEN, output: ConnectionType = NOT_GIVEN
    ):
        return ExtendInfo(self, {"input": input, "output": output})


class Model2(PrimitiveModel):
    input: Connection
    output: Connection

    def __init__(self) -> None:
        super().__init__(
            formula_key="relu",
            input=BaseKey(shape=[("Var1", ...), "u1"], type=Tensor),
            output=BaseKey(shape=["u1", ("Var1", ...)], type=Tensor),
        )

    def connect(  # type: ignore[override]
        self, input: ConnectionType = NOT_GIVEN, output: ConnectionType = NOT_GIVEN
    ):
        return ExtendInfo(self, {"input": input, "output": output})


class Model3(PrimitiveModel):
    input: Connection
    output: Connection
    axis: Connection

    def __init__(self) -> None:
        uni1 = Uniadic()
        uni2 = Uniadic()
        uni3 = Uniadic()
        var = Variadic()
        input_shapes = [
            ShapeRepr([uni1, uni2, uni3]).node,
            ShapeRepr([uni3, uni2, uni1]).node,
        ]
        output_shape = ShapeRepr(prefix=[uni1], root=var, suffix=[uni3]).node
        super().__init__(
            formula_key="concat",
            input=BaseKey(value=[Tensor(shape=shp) for shp in input_shapes]),
            output=BaseKey(value=Tensor(shape=output_shape)),
            axis=BaseKey(type=int),
        )
        # super().__init__(
        #     formula_key="concat",
        #     input1=BaseKey(shape=["u1", "u2", "u3"], type=Tensor),
        #     input2=BaseKey(shape=["u3", "u2", "u1"], type=Tensor),
        #     output=BaseKey(shape=["u1", ("Var1", ...), "u3"], type=Tensor),
        #     axis=BaseKey(type=int),
        # )

    def connect(  # type: ignore[override]
        self,
        input: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ):
        return ExtendInfo(self, {"input": input, "output": output})


class Model4(PrimitiveModel):
    input: Connection
    output: Connection

    def __init__(self) -> None:
        super().__init__(
            formula_key="relu",
            input=BaseKey(shape=[("Var1", ...)], type=Tensor),
            output=BaseKey(shape=[("Var1", ...), 1], type=Tensor),
        )

    def connect(  # type: ignore[override]
        self, input: ConnectionType = NOT_GIVEN, output: ConnectionType = NOT_GIVEN
    ):
        return ExtendInfo(self, {"input": input, "output": output})


class Model5(PrimitiveModel):
    input: Connection
    output: Connection
    axis: Connection

    def __init__(self, axis=None) -> None:
        super().__init__(
            formula_key="relu",
            input=BaseKey(shape=[("Var1", ...)], type=Tensor),
            output=BaseKey(shape=[("Var2", ...)], type=Tensor),
            axis=BaseKey(type=NoneType | list[int], value=axis),
        )

    def connect(  # type: ignore[override]
        self,
        input: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
        axis: ConnectionType = NOT_GIVEN,
    ):
        return ExtendInfo(self, {"input": input, "output": output, "axis": axis})


class Model6(PrimitiveModel):
    input: Connection
    output: Connection

    def __init__(self) -> None:
        super().__init__(
            formula_key="relu",
            input=BaseKey(shape=["u1", ("Var1", ...)], type=Tensor),
            output=BaseKey(shape=[("Var1", ...), "u1"], type=Tensor),
        )

    def connect(  # type: ignore[override]
        self, input: ConnectionType = NOT_GIVEN, output: ConnectionType = NOT_GIVEN
    ):
        return ExtendInfo(self, {"input": input, "output": output})


class Model7(PrimitiveModel):
    input: Connection
    output: Connection

    def __init__(self) -> None:
        super().__init__(
            formula_key="relu",
            input=BaseKey(shape=["u1", ("Var1", ...)], type=Tensor),
            output=BaseKey(shape=["u1", ("Var1", ...)], type=Tensor),
        )

    def connect(  # type: ignore[override]
        self, input: ConnectionType = NOT_GIVEN, output: ConnectionType = NOT_GIVEN
    ):
        return ExtendInfo(self, {"input": input, "output": output})


class Model8(PrimitiveModel):
    input: Connection
    output: Connection

    def __init__(self) -> None:
        super().__init__(
            formula_key="relu",
            input=BaseKey(shape=[("Var1", ...), "u1"], type=Tensor),
            output=BaseKey(shape=[("Var1", ...), "u1"], type=Tensor),
        )

    def connect(  # type: ignore[override]
        self, input: ConnectionType = NOT_GIVEN, output: ConnectionType = NOT_GIVEN
    ):
        return ExtendInfo(self, {"input": input, "output": output})


class Model9(PrimitiveModel):
    input: Connection
    output: Connection

    def __init__(self) -> None:
        super().__init__(
            formula_key="relu",
            input=BaseKey(shape=["u1", ("Var1", ...)], type=Tensor),
            output=BaseKey(shape=["u2", "u1", ("Var1", ...)], type=Tensor),
        )

    def connect(  # type: ignore[override]
        self, input: ConnectionType = NOT_GIVEN, output: ConnectionType = NOT_GIVEN
    ):
        return ExtendInfo(self, {"input": input, "output": output})


def test_shape_2():
    """
    For this test, a dummy primitive model (Model1) is created, it takes one variadic
    as input and returns same variadic with two newly added uniadics. When three of
    these models are connected serially, it is expected that the # of uniadics will
    increase by two in every output.
    """
    model = Model()
    model1 = Model1()
    model1.set_shapes(input=[("Var1", ...)], output=[("Var1", ...), "u1", "u2"])
    model2 = deepcopy(model1)
    model3 = deepcopy(model2)
    model |= model1.connect(input="input")
    model |= model2.connect(input=model1.output)
    model |= model3.connect(input=model2.output, output=IOKey(name="output"))
    logical_ref = {
        "input": ["(V1, ...)"],
        "$_Model1_0_output": ["(V1, ...)", "u1", "u2"],
        "$_Model1_1_output": ["(V1, ...)", "u1", "u2", "u3", "u4"],
        "output": ["(V1, ...)", "u1", "u2", "u3", "u4", "u5", "u6"],
    }
    physical_ref: dict[str, list | None] = {
        "input": ["..."],
        "output_0": ["...", None, None],
        "output_1": ["...", None, None, None, None],
        "output": ["...", None, None, None, None, None, None],
    }
    assert_shapes(model, logical_ref, physical_ref)


def test_shape_3():
    """
    This test is for testing of higher dimension cyclic models. Examiming the model
    in higher dimension, it looks like a cyclic extension occurred. However, even
    though there is a cyclic extension in higher dimension, if inner model structure
    of all all models examined carefully, it can be seen that this model does not
    violate "acyclic" assumption. In lower level, there are just three buffers model
    connected serially, in that setting, if one of the connections' shape is set,
    it is expected to all shapes is set to that shape.
    """
    model = Model()
    two_buff_model = Model()
    two_buff_model |= Buffer().connect(input="input1", output=IOKey(name="output1"))
    two_buff_model |= Buffer().connect(input="input2", output=IOKey(name="output2"))
    model |= two_buff_model.connect(input1="input1", output2=IOKey(name="output2"))
    buff1 = Buffer()
    model |= buff1.connect(input=two_buff_model.output1, output=two_buff_model.input2)  # type: ignore
    model.generate_keys()
    buff1.set_shapes(input=[3, 4, 5, 6])
    logical_ref = {
        "input1": [3, 4, 5, 6],
        "$_Buffer_0_output": [3, 4, 5, 6],
        "$_Model_1_output1": [3, 4, 5, 6],
        "output2": [3, 4, 5, 6],
    }
    physical_ref = {"input1": [3, 4, 5, 6]}
    assert_shapes(model, logical_ref, physical_ref)


def test_shape_4():
    """
    For this test, a dummy Primitive model of Model2 is introduced (see all dummy
    models above test_shape_2). What Model2 does is basically a circular shift.
    Takes [("Var1", ...), "u1"] as input and gives ["u1", ("Var1", ...)] as output.
    Three of these models are connected serially. This test is for testing whether
    shape algorithm determines its symbolic shapes correctly.
    """

    model = Model()
    model1 = Model2()
    model2 = deepcopy(model1)
    model3 = deepcopy(model2)
    model |= model1.connect(input="input")
    model |= model2.connect(input=model1.output)
    model |= model3.connect(input=model2.output, output=IOKey(name="output"))

    logical_ref: dict[str, list | None] = {
        "input": ["(V1, ...)", "u1"],
        "$_Model2_0_output": [["u1", "(V1, ...)"], ["(V2, ...)", "u2"]],
        "$_Model2_1_output": [["u2", "(V2, ...)"], ["(V3, ...)", "u3"]],
        "output": ["u3", "(V3, ...)"],
    }
    physical_ref: dict[str, list | None] = {
        "input": ["...", None],
        "output_0": [None, "..."],
        "output_1": [None, "..."],
        "output": [None, "..."],
    }
    assert_shapes(model, logical_ref, physical_ref)


def test_shape_5():
    """
    This test uses same dummy model as test_shape_4. This test uses two Model2 models.
    At first, both of inputs of these models' shapes set to [5, 10]. If these models
    are to be connected serially, an error should be raised. Becuase output of first
    model's shape should be [10, 5] due to IO shape relation. And we are trying to
    connect [5, 10] shape connection with [10, 5] shape connection. Value mismatch.
    """
    model = Model()
    model1 = Model2()
    model1.set_shapes(input=[5, 10])
    model2 = deepcopy(model1)
    model += model1.connect(input="input")

    # TODO: Assert test and change Exception
    with pytest.raises(Exception):  # noqa
        model += model2.connect(input=model1.output)


def test_shape_6():
    """
    This test has the same setting with test_shape_4. Three cascaded Model2 models.
    input of the first model is set to [3, 4, 5, 6, 7, 8]. Shape of output of these
    models should circular shift by 1 at every output of these cascaded models.
    """
    model = Model()
    model1 = Model2()
    model2 = deepcopy(model1)
    model3 = deepcopy(model2)
    model |= model1.connect(input="input")
    model |= model2.connect(input=model1.output)
    model |= model3.connect(input=model2.output, output=IOKey(name="output"))
    model.set_shapes(input=[3, 4, 5, 6, 7, 8])
    logical_ref = {
        "input": [3, 4, 5, 6, 7, 8],
        "$_Model2_0_output": [8, 3, 4, 5, 6, 7],
        "$_Model2_1_output": [7, 8, 3, 4, 5, 6],
        "output": [6, 7, 8, 3, 4, 5],
    }
    physical_ref = {
        "input": [3, 4, 5, 6, 7, 8],
        "output_0": [8, 3, 4, 5, 6, 7],
        "output_1": [7, 8, 3, 4, 5, 6],
        "output": [6, 7, 8, 3, 4, 5],
    }
    assert_shapes(model, logical_ref, physical_ref)


# @pytest.mark.skip("Known bug")
def test_shape_7():
    """
    This test is also uses same setting with test_shape_4. Three cascaded Model2
    models. This test is similar to test_shape_6. Different from test_shape_6,
    shape input of the first model is set to [3, 4]. Therefore, When circular
    shifting by 1, algorithm should also take overlap amount into account.
    """
    model = Model()
    model1 = Model2()
    model2 = deepcopy(model1)
    model3 = deepcopy(model2)
    model |= model1.connect(input="input")
    model |= model2.connect(input=model1.output)
    model |= model3.connect(input=model2.output, output=IOKey(name="output"))
    model.set_shapes(input=[3, 4])
    logical_ref = {
        "input": [3, 4],
        "$_Model2_0_output": [4, 3],
        "$_Model2_1_output": [3, 4],
        "output": [4, 3],
    }
    physical_ref = {
        "input": [3, 4],
        "output_0": [4, 3],
        "output_1": [3, 4],
        "output": [4, 3],
    }
    assert_shapes(model, logical_ref, physical_ref)


def test_shape_8():
    """
    This test is similar to test_shape_7, only difference is set_shape() function
    is called before model is builded.
    """
    model = Model()
    model1 = Model2()
    model2 = deepcopy(model1)
    model3 = deepcopy(model2)
    model1.set_shapes(input=[3, 4])
    model |= model1.connect(input="input")
    model |= model2.connect(input=model1.output)
    model |= model3.connect(input=model2.output, output=IOKey(name="output"))
    logical_ref = {
        "input": [3, 4],
        "$_Model2_0_output": [4, 3],
        "$_Model2_1_output": [3, 4],
        "output": [4, 3],
    }
    physical_ref = {
        "input": [3, 4],
        "output_0": [4, 3],
        "output_1": [3, 4],
        "output": [4, 3],
    }
    assert_shapes(model, logical_ref, physical_ref)


def test_shape_9():
    """
    For this test, Model3 is created (see all created models above test_shape_2).
    New Model3 has a complex input - output shape relation. This test includes
    three cascaded Model3 models. This test is written for testing whether shape
    algorithm can solve models with complex IO realtions.
    """
    model = Model()
    model_1 = Model3()
    model_2 = Model3()
    model_3 = Model3()
    model |= model_1.connect(input=[IOKey("input"), IOKey()])  # type: ignore
    model |= model_2.connect(input=[model_1.output, IOKey()])  # type: ignore
    model |= model_3.connect(
        input=[model_2.output, IOKey()],  # type: ignore
        output=IOKey(name="output"),
    )
    logical_ref = {
        "input": ["u1", "u2", "u3"],
        "$input2_0": ["u3", "u2", "u1"],
        "$_Model3_1_output": ["u1", "u4", "u3"],
        "$_ToList_0_output": None,
        "$_ToList_2_output": None,
        "$_ToList_4_output": None,
        "$axis_0": None,
        "$axis_1": None,
        "$axis_2": None,
        "$input2_1": ["u3", "u4", "u1"],
        "$_Model3_3_output": ["u1", "u5", "u3"],
        "$input2_2": ["u3", "u5", "u1"],
        "output": ["u1", "(V1, ...)", "u3"],
    }
    physical_ref: dict[str, list | None] = {
        "input": [None, None, None],
        "input2_0": [None, None, None],
        "output_0": None,
        "output_2": None,
        "output_4": None,
        "axis_0": None,
        "output_1": [None, None, None],
        "input2_1": [None, None, None],
        "axis_1": None,
        "output_3": [None, None, None],
        "input2_2": [None, None, None],
        "axis_2": None,
        "output": [None, "...", None],
    }
    assert_shapes(model, logical_ref, physical_ref)


# @pytest.mark.skip("Known bug")
def test_shape_10():
    """
    This test is same with test_shape_7. Only difference is shape of the input is set
    to [4] instead of [3, 4]
    """
    model = Model()
    model1 = Model2()
    model2 = deepcopy(model1)
    model3 = deepcopy(model2)
    model |= model1.connect(input="input")
    model |= model2.connect(input=model1.output)
    model |= model3.connect(input=model2.output, output=IOKey(name="output"))
    model.set_shapes(input=[4])
    logical_ref = {
        "input": [4],
        "$_Model2_0_output": [4],
        "$_Model2_1_output": [4],
        "output": [4],
    }
    physical_ref = {"input": [4], "output_0": [4], "output_1": [4], "output": [4]}
    assert_shapes(model, logical_ref, physical_ref)


def test_shape_11():
    """
    This test is also tests Variadic overlap amonut collapse alorigthm, At first
    step of this test, two buffer model of respective shapes ["u1", "u2",
    ("Var1", ...)], and [("Var1", ...), "u1", "u2"]) connected serially. Therefore,
    shapes of these buffer model is expected to ["u1", "u2", (Var1, ...), "u3", "u4"]
    with max_overlap_amount = 2. at the second part, a mean model with axis = -1 is
    connected so that final symbolic shape will become ["u1", "u2", (Var1, ...), "u3"]
    with max_overlap_amount = 2. In final step, Dummy primitive model Model4 is
    connected to output of the mean model (see all models above test_shape_2).
    what does  Model4 do is basically adding new axis to the last shape. When this
    model is connected to the output of the reduce model, New symbolic shape will be
    ["u1", "u2", (Var1, ...), "u3", 1] with max_overlap_limit = 2. However when input
    of the first buffer model initialized with [3, 4], overlap amount will become 2 and
    therefore alogrithm will try to match "u2", whose value is set to 4, with 1. Hence,
    it will give a value mismatch error. However, this test should not give any error
    as there is nothing wrong in user-wise
    """

    model = Model()
    buff1 = Buffer()
    buff2 = Buffer()
    reduce_model = Mean(axis=-1)
    newaxis_model = Model4()
    shapes_1: dict[str, list] = {"input": ["u1", "u2", ("Var1", ...)]}
    shapes_2: dict[str, list] = {"input": [("Var1", ...), "u1", "u2"]}
    buff1.set_shapes(**shapes_1)
    buff2.set_shapes(**shapes_2)
    model |= buff1.connect(input="input")
    model |= buff2.connect(input=buff1.output)
    model |= reduce_model.connect(input=buff2.output)
    model |= newaxis_model.connect(
        input=reduce_model.output, output=IOKey(name="output")
    )
    model.set_shapes(input=[3, 4])
    logical_ref = {
        "input": [3, 4],
        "$_Buffer_0_output": [3, 4],
        "$_Buffer_1_output": [3, 4],
        "$_Mean_2_axis": None,
        "$_Mean_2_keepdim": None,
        "$_Mean_2_output": [3],
        "output": [3, 1],
    }
    physical_ref = {
        "input": [3, 4],
        "axis": None,
        "keepdim": None,
        "output_2": [3],
        "output": [3, 1],
    }
    assert_shapes(model, logical_ref, physical_ref)


def test_shape_12():
    model = Model()
    add1 = Add()
    add1.set_cin("left")
    model += add1
    add2 = Add()
    add2.set_cin("left")
    shapes: dict[str, list] = {"left": ["a", "b", "c"], "right": [1, 1, 1]}
    add2.set_shapes(**shapes)
    add1.set_shapes(left=[1, 2, 9], right=[1, 2, 1])
    model |= add2.connect(left=add1.left)

    logical_ref = {
        "$input": [1, 2, 9],
        "$_Add_0_output": [1, 2, 9],
        "$_Add_1_output": [1, 2, 9],
        "$right_0": [1, 2, 1],
        "$right_1": [1, 1, 1],
    }
    check_shapes_semantically(model.get_shapes(verbose=True), logical_ref)


def test_broadcast_to():
    model = Model()
    bcast_to = BroadcastTo(shape=(3, 4, 5))
    model += bcast_to.connect(input="input", output=IOKey(name="output"))
    with pytest.raises(ValueError) as err_info:
        model.set_shapes(input=[7, 8, 9])
    assert str(err_info.value) == "Shape mismatch in broadcast_to model"


def test_broadcast_to_2():
    model = Model()
    bcast_to = BroadcastTo(shape=TBD)
    model += bcast_to.connect(
        input="input", output=IOKey(name="output"), shape=(3, 4, 5)
    )
    model.set_shapes(input=[3, 4, 5])
    logical_ref = {"input": [3, 4, 5], "$shape": None, "output": [3, 4, 5]}
    physical_ref = {"input": [3, 4, 5], "shape": None, "output": [3, 4, 5]}
    comp_model = mithril.compile(model=model, backend=TorchBackend(), inference=True)
    check_shapes_semantically(model.get_shapes(verbose=True), logical_ref)
    check_shapes_semantically(comp_model.get_shapes(verbose=True), physical_ref)


def test_broadcast_to_4():
    model = Model()
    bcast_to = BroadcastTo(shape=TBD)
    model += bcast_to.connect(
        input="input", output=IOKey(name="output"), shape=(3, 4, 5)
    )
    with pytest.raises(ValueError) as err_info:
        model.set_shapes(input=[1, 1, 3, 4, 5])
    assert str(err_info.value) == "Cannot broadcast to lower dimension"


def test_broadcast_to_5():
    model = Model()
    bcast_to = BroadcastTo(shape=TBD)
    model += bcast_to.connect(
        input="input", output=IOKey(name="output"), shape=(1, 1, 3, 4, 5)
    )
    with pytest.raises(ValueError) as err_info:
        model.set_shapes(input=[5, 4, 5])
    assert str(err_info.value) == "Shape mismatch in broadcast_to model"


def test_transpose_1():
    model = Model()
    buff1 = Buffer()
    transpose_model = Transpose()
    transpose_model.set_shapes(input=["u1", "u2", "u3"])
    model |= buff1.connect(input="input1", output=IOKey(name="my_input"))
    model |= transpose_model.connect(input="my_input", output=IOKey(name="output"))
    model.set_shapes(input1=[3, 4, 5])
    logical_ref = {
        "input1": [3, 4, 5],
        "$_Transpose_1_axes": None,
        "my_input": [3, 4, 5],
        "output": [5, 4, 3],
    }
    physical_ref = {"input1": [3, 4, 5], "axes": None, "output": [5, 4, 3]}

    assert_shapes(model, logical_ref, physical_ref)


def test_logical_constraint_1():
    model = Model()
    t_model_1 = Model5()
    t_model_2 = Model5()
    t_model_3 = Model5()
    t_model_4 = Model5()
    t_model_5 = Model5()
    model |= t_model_1.connect(input="input", axis="axis")
    model |= t_model_2
    model |= t_model_3
    model |= t_model_4
    model |= t_model_5.connect(input=t_model_4.output, output=IOKey(name="output"))
    model.add_constraint(fn=reverse_constraints, keys=["input", "output", "axis"])
    model.set_shapes(input=[1, 2, 3, 4, 5, 6])
    assert model.get_shapes(verbose=True)["input"] == [1, 2, 3, 4, 5, 6]
    assert model.get_shapes(verbose=True)["output"] == [6, 5, 4, 3, 2, 1]


def test_logical_constraint_2():
    model = Model()
    add_1 = Add()
    add_1.set_types(left=Tensor, right=Tensor)
    add_2 = Add()
    add_3 = Add()
    t_model = Transpose()
    model |= add_1.connect(left="in1", right="in2")
    model |= add_2.connect(left=add_1.output, right=IOKey("in3", type=Tensor))
    model |= add_3.connect(
        left=add_2.output,
        right=IOKey("in4", type=Tensor),
        output=IOKey(name="output"),
    )
    model |= t_model.connect(input="in1", output=IOKey(name="output1"), axes="axes")
    model.add_constraint(fn=reverse_constraints, keys=["in1", "in2", "axes"])
    model.add_constraint(fn=reverse_constraints, keys=["in2", "in3", "axes"])
    model.add_constraint(fn=reverse_constraints, keys=["in3", "in4", "axes"])
    model.set_shapes(in1=[6, 6, 1, 1, 1, 1])
    logical_ref = {
        "in1": [6, 6, 1, 1, 1, 1],
        "in2": [1, 1, 1, 1, 6, 6],
        "in3": [6, 6, 1, 1, 1, 1],
        "in4": [1, 1, 1, 1, 6, 6],
        "$_Add_0_output": [6, 6, 1, 1, 6, 6],
        "$_Add_1_output": [6, 6, 1, 1, 6, 6],
        "output": [6, 6, 1, 1, 6, 6],
        "axes": None,
        "output1": [1, 1, 1, 1, 6, 6],
    }
    physical_ref = {
        "in1": [6, 6, 1, 1, 1, 1],
        "in2": [1, 1, 1, 1, 6, 6],
        "output_0": [6, 6, 1, 1, 6, 6],
        "in3": [6, 6, 1, 1, 1, 1],
        "output_1": [6, 6, 1, 1, 6, 6],
        "in4": [1, 1, 1, 1, 6, 6],
        "output": [6, 6, 1, 1, 6, 6],
        "axes": None,
        "output1": [1, 1, 1, 1, 6, 6],
    }
    assert_shapes(model, logical_ref, physical_ref)


def test_reduce_init_1():
    reduce_model = Mean()
    ref_shapes = {
        "output": [],
        "input": ["(V1, ...)"],
        "axis": None,
        "keepdim": None,
    }
    check_shapes_semantically(reduce_model.get_shapes(verbose=True), ref_shapes)


def test_reduce_init_2():
    reduce_model = Mean(axis=(2, 3))
    ref_shapes = {
        "output": ["u1", "u2", "(V1, ...)"],
        "input": ["u1", "u2", "u3", "u4", "(V1, ...)"],
        "axis": None,
        "keepdim": None,
    }
    check_shapes_semantically(reduce_model.get_shapes(verbose=True), ref_shapes)


def test_reduce_init_3():
    reduce_model = Mean(axis=(2, 3), keepdim=True)
    assert_shapes(
        reduce_model,
        logical_ref={
            "output": ["u1", "u2", 1, 1, "(V1, ...)"],
            "input": ["u1", "u2", "u3", "u4", "(V1, ...)"],
            "axis": None,
            "keepdim": None,
        },
    )


def test_reduce_init_4():
    reduce_model = Mean(axis=(1, 3), keepdim=True)
    assert_shapes(
        reduce_model,
        logical_ref={
            "output": ["u1", 1, "u2", 1, "(V1, ...)"],
            "input": ["u1", "u3", "u2", "u4", "(V1, ...)"],
            "axis": None,
            "keepdim": None,
        },
    )


def test_reduce_init_5():
    reduce_model = Mean(axis=(1, 3))
    assert_shapes(
        reduce_model,
        logical_ref={
            "output": ["u1", "u2", "(V1, ...)"],
            "input": ["u1", "u3", "u2", "u4", "(V1, ...)"],
            "axis": None,
            "keepdim": None,
        },
    )


def test_reduce_init_6():
    reduce_model = Mean(axis=(-1, -2))
    assert_shapes(
        reduce_model,
        logical_ref={
            "output": ["(V1, ...)"],
            "input": ["(V1, ...)", "u1", "u2"],
            "axis": None,
            "keepdim": None,
        },
    )


def test_reduce_init_7():
    reduce_model = Mean(axis=(-1, -3))
    assert_shapes(
        reduce_model,
        logical_ref={
            "output": ["(V1, ...)", "u1"],
            "input": ["(V1, ...)", "u2", "u1", "u3"],
            "axis": None,
            "keepdim": None,
        },
    )


def test_reduce_init_8():
    reduce_model = Mean(axis=(-1, -3), keepdim=True)
    assert_shapes(
        reduce_model,
        logical_ref={
            "output": ["(V1, ...)", 1, "u1", 1],
            "input": ["(V1, ...)", "u2", "u1", "u3"],
            "axis": None,
            "keepdim": None,
        },
    )


def test_reduce_init_9():
    reduce_model = Mean(axis=(1, -1))
    assert_shapes(
        reduce_model,
        logical_ref={
            "output": ["u1", "(V1, ...)"],
            "input": ["u1", "u2", "(V1, ...)", "u3"],
            "axis": None,
            "keepdim": None,
        },
    )


def test_reduce_init_10():
    reduce_model = Mean(axis=(1, -2))
    assert_shapes(
        reduce_model,
        logical_ref={
            "output": ["(V1, ...)"],
            "input": ["u1", "u2", "(V2, ...)"],
            "axis": None,
            "keepdim": None,
        },
    )


def test_reduce_init_11():
    reduce_model = Mean(axis=(1, -2), keepdim=True)
    assert_shapes(
        reduce_model,
        logical_ref={
            "output": ["u3", 1, "(V1, ...)"],
            "input": ["u1", "u2", "(V2, ...)"],
            "axis": None,
            "keepdim": None,
        },
    )


def test_reduce_init_12():
    reduce_model = Mean(axis=(1, -3), keepdim=True)
    assert_shapes(
        reduce_model,
        logical_ref={
            "output": ["u4", 1, "(V1, ...)", "u3"],
            "input": ["u1", "u2", "(V2, ...)", "u3"],
            "axis": None,
            "keepdim": None,
        },
    )


def test_reduce_init_13():
    reduce_model = Mean(axis=-3, keepdim=False)
    assert_shapes(
        reduce_model,
        logical_ref={
            "output": ["(V1, ...)", "u1", "u2"],
            "input": ["(V1, ...)", "u3", "u1", "u2"],
            "axis": None,
            "keepdim": None,
        },
    )


def test_reduce_init_14():
    reduce_model = Mean(axis=(1, 2, 3, 6, -1, -2, -3), keepdim=False)
    assert_shapes(
        reduce_model,
        logical_ref={
            "output": ["u1", "u2", "u3", "(V1, ...)"],
            "input": [
                "u1",
                "u4",
                "u5",
                "u6",
                "u2",
                "u3",
                "u7",
                "(V1, ...)",
                "u8",
                "u9",
                "u10",
            ],
            "axis": None,
            "keepdim": None,
        },
    )


def test_flatten_init_1():
    flat_model = Flatten(start_dim=1, end_dim=-1)
    assert_shapes(
        flat_model,
        logical_ref={
            "output": ["u1", "u2"],
            "input": ["u1", "u3", "(V1, ...)"],
            "start_dim": None,
            "end_dim": None,
        },
    )


def test_flatten_init_2():
    flat_model = Flatten(start_dim=2, end_dim=4)
    assert_shapes(
        flat_model,
        logical_ref={
            "output": ["u1", "u2", "u3", "(V1, ...)"],
            "input": ["u1", "u2", "u4", "u5", "u6", "(V1, ...)"],
            "start_dim": None,
            "end_dim": None,
        },
    )


def test_flatten_init_3():
    flat_model = Flatten(start_dim=2, end_dim=TBD)
    assert_shapes(
        flat_model,
        logical_ref={
            "output": ["u1", "u2", "u3", "(V1, ...)"],
            "input": ["u1", "u2", "u4", "(V2, ...)"],
            "start_dim": None,
            "end_dim": None,
        },
    )


def test_flatten_init_4():
    flat_model = Flatten()
    assert_shapes(
        flat_model,
        logical_ref={
            "output": ["u1"],
            "input": ["u2", "(V1, ...)"],
            "start_dim": None,
            "end_dim": None,
        },
    )


def test_flatten_init_5():
    flat_model = Flatten(start_dim=-4, end_dim=-2)
    assert_shapes(
        flat_model,
        logical_ref={
            "output": ["(V1, ...)", "u1", "u2"],
            "input": ["(V1, ...)", "u3", "u4", "u5", "u2"],
            "start_dim": None,
            "end_dim": None,
        },
    )


def test_flatten_init_6():
    flat_model = Flatten(start_dim=-4, end_dim=TBD)
    assert_shapes(
        flat_model,
        logical_ref={
            "output": ["u5", "(V1, ...)"],
            "input": ["(V2, ...)", "u1", "u2", "u3", "u4"],
            "start_dim": None,
            "end_dim": None,
        },
    )


def test_concat_init_1():
    flat_model = Concat()
    tensor1: Tensor[float] = Tensor()
    tensor2: Tensor[float] = Tensor()
    flat_model.set_values(input=[tensor1, tensor2])
    data = {k: v.metadata for k, v in flat_model.conns.all.items()}
    ref = {
        "output": ["u1", "(V1, ...)"],
        "input": (["u2", "(V1, ...)"], ["u3", "(V1, ...)"]),
        "axis": [],
    }
    assert_shape_results(data, ref, {}, Updates(), set())


def test_concat_init_2():
    flat_model = Concat(axis=3)
    tensor1: Tensor[float] = Tensor()
    tensor2: Tensor[float] = Tensor()
    flat_model.set_values(input=[tensor1, tensor2])
    data = {k: v.metadata for k, v in flat_model.conns.all.items()}
    ref = {
        "output": ["u1", "u2", "u3", "u4", "(V1, ...)"],
        "input": (
            ["u1", "u2", "u3", "u5", "(V1, ...)"],
            ["u1", "u2", "u3", "u6", "(V1, ...)"],
        ),
        "axis": [],
    }
    assert_shape_results(data, ref, {}, Updates(), set())


def test_concat_init_3():
    flat_model = Concat(axis=3)
    tensor1: Tensor[float] = Tensor()
    tensor2: Tensor[float] = Tensor()
    tensor3: Tensor[float] = Tensor()
    tensor4: Tensor[float] = Tensor()
    tensor5: Tensor[float] = Tensor()
    tensor6: Tensor[float] = Tensor()
    flat_model.set_values(input=[tensor1, tensor2, tensor3, tensor4, tensor5, tensor6])
    data = {k: v.metadata for k, v in flat_model.conns.all.items()}
    ref = {
        "output": ["u1", "u2", "u3", "u4", "(V1, ...)"],
        "input": (
            ["u1", "u2", "u3", "u5", "(V1, ...)"],
            ["u1", "u2", "u3", "u6", "(V1, ...)"],
            ["u1", "u2", "u3", "u7", "(V1, ...)"],
            ["u1", "u2", "u3", "u8", "(V1, ...)"],
            ["u1", "u2", "u3", "u9", "(V1, ...)"],
            ["u1", "u2", "u3", "u10", "(V1, ...)"],
        ),
        "axis": [],
    }
    assert_shape_results(data, ref, {}, Updates(), set())
    # assert_shapes(
    #     flat_model,
    #     logical_ref={
    #         "output": ["u1", "u2", "u3", "u4", "(V1, ...)"],
    #         "input1": ["u1", "u2", "u3", "u5", "(V1, ...)"],
    #         "input2": ["u1", "u2", "u3", "u6", "(V1, ...)"],
    #         "input3": ["u1", "u2", "u3", "u7", "(V1, ...)"],
    #         "input4": ["u1", "u2", "u3", "u8", "(V1, ...)"],
    #         "input5": ["u1", "u2", "u3", "u9", "(V1, ...)"],
    #         "input6": ["u1", "u2", "u3", "u10", "(V1, ...)"],
    #         "axis": None,
    #     },
    # )


def test_concat_init_4():
    flat_model = Concat(axis=-2)
    tensor1: Tensor[float] = Tensor()
    tensor2: Tensor[float] = Tensor()
    tensor3: Tensor[float] = Tensor()
    tensor4: Tensor[float] = Tensor()
    flat_model.set_values(input=[tensor1, tensor2, tensor3, tensor4])
    data = {k: v.metadata for k, v in flat_model.conns.all.items()}
    ref = {
        "output": ["(V1, ...)", "u1", "u2"],
        "input": (
            ["(V1, ...)", "u3", "u2"],
            ["(V1, ...)", "u4", "u2"],
            ["(V1, ...)", "u5", "u2"],
            ["(V1, ...)", "u6", "u2"],
        ),
        "axis": [],
    }
    assert_shape_results(data, ref, {}, Updates(), set())
    # assert_shapes(
    #     flat_model,
    #     logical_ref={
    #         "output": ["(V1, ...)", "u1", "u2"],
    #         "input1": ["(V1, ...)", "u3", "u2"],
    #         "input2": ["(V1, ...)", "u4", "u2"],
    #         "input3": ["(V1, ...)", "u5", "u2"],
    #         "input4": ["(V1, ...)", "u6", "u2"],
    #         "axis": None,
    #     },
    # )


def test_concat_init_5():
    flat_model = Concat(axis=None)
    tensor1: Tensor[float] = Tensor()
    tensor2: Tensor[float] = Tensor()
    tensor3: Tensor[float] = Tensor()
    tensor4: Tensor[float] = Tensor()
    flat_model.set_values(input=[tensor1, tensor2, tensor3, tensor4])
    data = {k: v.metadata for k, v in flat_model.conns.all.items()}
    ref = {
        "output": ["u1"],
        "input": (
            ["(V1, ...)"],
            ["(V2, ...)"],
            ["(V3, ...)"],
            ["(V4, ...)"],
        ),
        "axis": [],
    }
    assert_shape_results(data, ref, {}, Updates(), set())


def test_swapaxes_init_1():
    swap_axes_model = SwapAxes(axis1=3, axis2=1)

    logical_ref = {
        "output": ["u1", "u2", "u3", "u4", "(V1, ...)"],
        "input": ["u1", "u4", "u3", "u2", "(V1, ...)"],
        "axis1": None,
        "axis2": None,
    }
    assert_shapes(swap_axes_model, logical_ref)


def test_swapaxes_init_2():
    swap_axes_model = SwapAxes(axis1=TBD, axis2=1)

    logical_ref = {
        "output": ["(V2, ...)"],
        "input": ["u1", "u2", "(V1, ...)"],
        "axis1": None,
        "axis2": None,
    }
    assert_shapes(swap_axes_model, logical_ref)


def test_swapaxes_init_3():
    swap_axes_model = SwapAxes(axis1=TBD, axis2=-3)

    logical_ref = {
        "output": ["(V2, ...)"],
        "input": ["(V1, ...)", "u3", "u2", "u1"],
        "axis1": None,
        "axis2": None,
    }
    assert_shapes(swap_axes_model, logical_ref)


def test_swapaxes_init_4():
    swap_axes_model = SwapAxes(axis1=TBD, axis2=0)

    logical_ref = {
        "output": ["(V2, ...)"],
        "input": ["u1", "(V1, ...)"],
        "axis1": None,
        "axis2": None,
    }
    assert_shapes(swap_axes_model, logical_ref)


def test_variadic_naming_1():
    model = Linear()
    ref_shapes: dict[str, list | None] = {
        "$_Transpose_0_output": ["u1", "u2"],
        "$_MatrixMultiply_1_output": [
            ["(V1, ...)", "u3", "u2"],
            ["u4", "(V2, ...)", "u2"],
        ],
        "$_Transpose_0_axes": None,
        "weight": ["u2", "u1"],
        "input": [["(V1, ...)", "u3", "u1"], ["u4", "(V2, ...)", "u1"]],
        "bias": ["u2"],
        "output": [["(V1, ...)", "u3", "u2"], ["u4", "(V2, ...)", "u2"]],
    }

    assert_shapes(model, ref_shapes)


# Test models for Variadic naming tests.
class MyVariadic1(PrimitiveModel):
    input: Connection
    output: Connection

    def __init__(self) -> None:
        super().__init__(
            formula_key="buffer",
            input=BaseKey(shape=[("Var1", ...), "a"], type=Tensor),
            output=BaseKey(shape=[("Var1", ...), "a"], type=Tensor),
        )

    def connect(  # type: ignore[override]
        self, input: ConnectionType = NOT_GIVEN, output: ConnectionType = NOT_GIVEN
    ):
        return ExtendInfo(self, {"input": input, "output": output})


class MyVariadic2(PrimitiveModel):
    input: Connection
    output: Connection

    def __init__(self) -> None:
        super().__init__(
            formula_key="buffer",
            input=BaseKey(shape=[("Var1", ...), "a", "b"], type=Tensor),
            output=BaseKey(shape=[("Var1", ...), "a", "b"], type=Tensor),
        )

    def connect(  # type: ignore[override]
        self, input: ConnectionType = NOT_GIVEN, output: ConnectionType = NOT_GIVEN
    ):
        return ExtendInfo(self, {"input": input, "output": output})


class MyVariadic3(PrimitiveModel):
    input: Connection
    output: Connection

    def __init__(self) -> None:
        super().__init__(
            formula_key="buffer",
            input=BaseKey(shape=["a", "b", ("Var1", ...)], type=Tensor),
            output=BaseKey(shape=["a", "b", ("Var1", ...)], type=Tensor),
        )

    def connect(  # type: ignore[override]
        self, input: ConnectionType = NOT_GIVEN, output: ConnectionType = NOT_GIVEN
    ):
        return ExtendInfo(self, {"input": input, "output": output})


class MyVariadic4(PrimitiveModel):
    input: Connection
    output: Connection

    def __init__(self) -> None:
        super().__init__(
            formula_key="buffer",
            input=BaseKey(shape=["a", ("Var1", ...)], type=Tensor),
            output=BaseKey(shape=["a", ("Var1", ...)], type=Tensor),
        )

    def connect(  # type: ignore[override]
        self, input: ConnectionType = NOT_GIVEN, output: ConnectionType = NOT_GIVEN
    ):
        return ExtendInfo(self, {"input": input, "output": output})


class MyVariadic5(PrimitiveModel):
    input: Connection
    output: Connection

    def __init__(self) -> None:
        super().__init__(
            formula_key="buffer",
            input=BaseKey(shape=[("Var1", ...), "a", "b"], type=Tensor),
            output=BaseKey(shape=[("Var1", ...), "a"], type=Tensor),
        )

    def connect(  # type: ignore[override]
        self, input: ConnectionType = NOT_GIVEN, output: ConnectionType = NOT_GIVEN
    ):
        return ExtendInfo(self, {"input": input, "output": output})


class MyVariadic6(PrimitiveModel):
    input: Connection
    output: Connection

    def __init__(self) -> None:
        super().__init__(
            formula_key="buffer",
            input=BaseKey(shape=[("Var1", ...), "a"], type=Tensor),
            output=BaseKey(shape=["a", "a"], type=Tensor),
        )

    def connect(  # type: ignore[override]
        self, input: ConnectionType = NOT_GIVEN, output: ConnectionType = NOT_GIVEN
    ):
        return ExtendInfo(self, {"input": input, "output": output})


class MyVariadic7(PrimitiveModel):
    input: Connection
    output: Connection

    def __init__(self) -> None:
        super().__init__(
            formula_key="buffer",
            input=BaseKey(shape=[("Var1", ...), "u1", "u2"], type=Tensor),
            output=BaseKey(shape=["u3", ("Var2", ...), "u4"], type=Tensor),
        )

    def connect(  # type: ignore[override]
        self, input: ConnectionType = NOT_GIVEN, output: ConnectionType = NOT_GIVEN
    ):
        return ExtendInfo(self, {"input": input, "output": output})


class MyVariadic8(PrimitiveModel):
    input1: Connection
    input2: Connection
    input3: Connection
    input4: Connection
    input5: Connection
    input6: Connection
    output: Connection

    def __init__(self) -> None:
        super().__init__(
            formula_key="buffer",
            input1=BaseKey(shape=["u1", "u2", "u3", ("Var1", ...)], type=Tensor),
            input2=BaseKey(shape=["u4", "u5", ("Var2", ...), "u6"], type=Tensor),
            input3=BaseKey(shape=["u7", ("Var3", ...), "u8", "u9"], type=Tensor),
            input4=BaseKey(
                shape=[("Var4", ...), "u10", "u11", "u12"],
                type=Tensor,
            ),
            input5=BaseKey(
                shape=[("Var5", ...), "u13", "u14", "u15", "u16"],
                type=Tensor,
            ),
            input6=BaseKey(
                shape=["u17", "u18", ("Var6", ...), "u19", "u20"],
                type=Tensor,
            ),
            output=BaseKey(
                shape=["u13", ("Var1", ...), "u14", "u15", "u16"],
                type=Tensor,
            ),
        )

    def connect(  # type: ignore[override]
        self,
        input1: ConnectionType = NOT_GIVEN,
        input2: ConnectionType = NOT_GIVEN,
        input3: ConnectionType = NOT_GIVEN,
        input4: ConnectionType = NOT_GIVEN,
        input5: ConnectionType = NOT_GIVEN,
        input6: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ):
        return ExtendInfo(
            self,
            {
                "input1": input1,
                "input2": input2,
                "input3": input3,
                "input4": input4,
                "input5": input5,
                "input6": input6,
                "output": output,
            },
        )


class MyVariadic9(PrimitiveModel):
    input1: Connection
    input2: Connection
    input3: Connection
    output: Connection

    def __init__(self) -> None:
        super().__init__(
            formula_key="buffer",
            input1=BaseKey(shape=["u1", ("Var1", ...)], type=Tensor),
            input2=BaseKey(shape=[("Var2", ...), "u2"], type=Tensor),
            input3=BaseKey(shape=["u3", ("Var3", ...), "u4"], type=Tensor),
            output=BaseKey(shape=["u5", "u5"], type=Tensor),
        )

    def connect(  # type: ignore[override]
        self,
        input1: ConnectionType = NOT_GIVEN,
        input2: ConnectionType = NOT_GIVEN,
        input3: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ):
        return ExtendInfo(
            self,
            {"input1": input1, "input2": input2, "input3": input3, "output": output},
        )


class MyVariadic10(PrimitiveModel):
    input1: Connection
    input2: Connection
    input3: Connection
    input4: Connection
    input5: Connection
    output: Connection

    def __init__(self) -> None:
        super().__init__(
            formula_key="buffer",
            input1=BaseKey(shape=["u1", "u2", ("Var1", ...)], type=Tensor),
            input2=BaseKey(shape=["u3", ("Var2", ...), "u4"], type=Tensor),
            input3=BaseKey(shape=[("Var3", ...), "u5", "u6"], type=Tensor),
            input4=BaseKey(
                shape=["u7", "u8", ("Var4", ...), "u9", "u10"],
                type=Tensor,
            ),
            input5=BaseKey(
                shape=["u11", ("Var4", ...), "u12", "u13"],
                type=Tensor,
            ),
            output=BaseKey(shape=["u5", "u5"], type=Tensor),
        )

    def connect(  # type: ignore[override]
        self,
        input1: ConnectionType = NOT_GIVEN,
        input2: ConnectionType = NOT_GIVEN,
        input3: ConnectionType = NOT_GIVEN,
        input4: ConnectionType = NOT_GIVEN,
        input5: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ):
        return ExtendInfo(
            self,
            {
                "input1": input1,
                "input2": input2,
                "input3": input3,
                "input4": input4,
                "input5": input5,
                "output": output,
            },
        )


class MyVariadic11(PrimitiveModel):
    input: Connection
    output: Connection

    def __init__(self) -> None:
        super().__init__(
            formula_key="buffer",
            input=BaseKey(shape=["a", ("Var1", ...)], type=Tensor),
            output=BaseKey(shape=["a", ("Var1", ...), "b"], type=Tensor),
        )

    def connect(  # type: ignore[override]
        self, input: ConnectionType = NOT_GIVEN, output: ConnectionType = NOT_GIVEN
    ):
        return ExtendInfo(self, {"input": input, "output": output})


class MyVariadic12(PrimitiveModel):
    input: Connection
    output: Connection

    def __init__(self) -> None:
        super().__init__(
            formula_key="buffer",
            input=BaseKey(shape=["a", "b", ("Var1", ...)], type=Tensor),
            output=BaseKey(shape=["a", "b", "c", ("Var1", ...)], type=Tensor),
        )

    def connect(  # type: ignore[override]
        self, input: ConnectionType = NOT_GIVEN, output: ConnectionType = NOT_GIVEN
    ):
        return ExtendInfo(self, {"input": input, "output": output})


def test_multiple_shape_reprs_1():
    for _ in range(100):
        model = Model()
        m1, m2, m3 = tuple(MyVariadic9() for _ in range(3))
        model |= m1.connect(input1="input1")
        check_single_shape_semantically(
            model.get_shapes(verbose=True)["input1"], ["u2", "(V1, ...)"]
        )

        model += m2.connect(input2="input1")
        check_single_shape_semantically(
            model.get_shapes(verbose=True)["input1"],
            [["u3", "(V1, ...)"], ["(V2, ...)", "u4"]],
        )

        model += m3.connect(input3="input1")
        check_single_shape_semantically(
            model.get_shapes(verbose=True)["input1"], ["u4", "(V1, ...)", "u5"]
        )


def test_multiple_shape_reprs_2():
    model = Model()
    m1, m2, m3, m4 = tuple(MyVariadic10() for _ in range(4))
    model += m1.connect(input1="input1")
    assert model.get_shapes(verbose=True)["input1"] == ["u1", "u2", "(V1, ...)"]
    model += m2.connect(input2="input1")
    input_1_con = model.conns.get_connection("input1")
    assert input_1_con is not None
    assert input_1_con.metadata.shape is not None
    assert get_deterministic_shape(input_1_con.metadata.shape) == [
        ["u1", "u2", "(V1, ...)"],
        ["u1", "(V2, ...)", "u3"],
    ]
    model += m3.connect(input3="input1")

    input_1_con = model.conns.get_connection("input1")
    assert input_1_con is not None
    assert input_1_con.metadata.shape is not None
    assert get_deterministic_shape(input_1_con.metadata.shape) == [
        ["u1", "u2", "(V1, ...)"],
        ["u1", "(V2, ...)", "u3"],
        ["(V3, ...)", "u4", "u3"],
    ]
    model += m4.connect(input4="input1")

    input_1_con = model.conns.get_connection("input1")
    assert input_1_con is not None
    assert input_1_con.metadata.shape is not None
    assert get_deterministic_shape(input_1_con.metadata.shape) == [
        "u1",
        "u2",
        "(V1, ...)",
        "u3",
        "u4",
    ]


def test_multiple_shape_reprs_3():
    for _ in range(100):
        model = Model()
        m1, m2, m3, m4, m5 = tuple(MyVariadic10() for _ in range(5))
        model += m1.connect(input1="input1")
        assert model.get_shapes(verbose=True)["input1"] == ["u1", "u2", "(V1, ...)"]
        model += m2.connect(input2="input1")

        input_1_con = model.conns.get_connection("input1")
        assert input_1_con is not None
        assert input_1_con.metadata.shape is not None
        assert get_deterministic_shape(input_1_con.metadata.shape) == [
            ["u1", "u2", "(V1, ...)"],
            ["u1", "(V2, ...)", "u3"],
        ]
        model += m3.connect(input3="input1")

        input_1_con = model.conns.get_connection("input1")
        assert input_1_con is not None
        assert input_1_con.metadata.shape is not None
        assert get_deterministic_shape(input_1_con.metadata.shape) == [
            ["u1", "u2", "(V1, ...)"],
            ["u1", "(V2, ...)", "u3"],
            ["(V3, ...)", "u4", "u3"],
        ]
        model += m4.connect(input5="input1")

        input_1_con = model.conns.get_connection("input1")
        assert input_1_con is not None
        assert input_1_con.metadata.shape is not None
        assert get_deterministic_shape(input_1_con.metadata.shape) == [
            ["u1", "u2", "(V1, ...)", "u3"],
            ["u1", "(V2, ...)", "u4", "u3"],
        ]
        model += m5.connect(input4="input1")

        input_1_con = model.conns.get_connection("input1")
        assert input_1_con is not None
        assert input_1_con.metadata.shape is not None
        assert get_deterministic_shape(input_1_con.metadata.shape) == [
            "u1",
            "u2",
            "(V1, ...)",
            "u3",
            "u4",
        ]
        model.set_shapes(
            input1=["u1", "u2", "u3", "u4", ("Var1", ...), "u5", "u6", "u7", "u8"]
        )
        input_1_con = model.conns.get_connection("input1")
        assert input_1_con is not None
        assert input_1_con.metadata.shape is not None
        assert get_deterministic_shape(input_1_con.metadata.shape) == [
            "u1",
            "u2",
            "u3",
            "u4",
            "(V1, ...)",
            "u5",
            "u6",
            "u7",
            "u8",
        ]
        model.set_shapes(input1=[1, 2, 3, 4, 5, 6, 7, 8])
        assert model.get_shapes(verbose=True)["input1"] == [1, 2, 3, 4, 5, 6, 7, 8]


def test_multiple_shape_reprs_4():
    model = Model()
    t_1 = Model2()
    t_2 = Model2()
    t_3 = Model2()
    model += t_1.connect(input="input", output=IOKey(name="output"))
    model += t_2
    model += t_3
    model.set_shapes(output=[3, "a"])
    ref_shapes: dict[str, list] = {
        "input": ["u1", 3],
        "output": [3, "u1"],
        "$_Model2_1_output": ["u1", 3],
        "$_Model2_2_output": [3, "u1"],
    }
    assert_shapes(model, ref_shapes)


def test_total_repr_count():
    model = Model()
    var2 = MyVariadic4()
    shapes: dict[str, list] = {"input": [1, ("Var", ...)]}
    var2.set_shapes(**shapes)
    model |= (var1 := MyVariadic1()).connect(output=IOKey(name="output"))
    model |= var2.connect(input=var1.output)

    edge = var2.input.metadata

    assert edge.is_tensor
    assert edge.shape is not None
    assert len(edge.shape.reprs) == 2


def test_total_repr_count_linear_1():
    model = Linear()
    edge = model.input.metadata
    assert edge.is_tensor
    assert edge.shape is not None
    shp_repr = next(iter(edge.shape.reprs))

    assert shp_repr.root is not None
    assert len(shp_repr.root.reprs) == 2
    assert len(edge.shape.reprs) == 2


def test_variadic_naming_2():
    for _ in range(100):
        model = Model()
        model |= MyVariadic1().connect(input="input1")
        model |= (var2 := MyVariadic2()).connect(input="input2")
        model |= MyVariadic1().connect(input=var2.output)
        model |= MyVariadic3().connect(input=var2.output)
        model |= MyVariadic4().connect(input=var2.output)
        model |= MyVariadic1().connect(input="input3")
        shape_1: dict[str, list] = {
            "input1": [("Var1", ...), "a"],
            "input3": [("Var1", ...), "b"],
        }
        shape_2: dict[str, list] = {
            "input1": ["a", ("Var1", ...)],
            "input2": ["a", ("Var1", ...)],
        }
        shape_3: dict[str, list] = {"input3": ["b", ("Var2", ...)]}
        model.set_shapes(**shape_1)
        model.set_shapes(**shape_2)
        model.set_shapes(**shape_3)

        ref_shapes: dict[str, list] = {
            "$_MyVariadic1_0_output": [
                ["u1", "u2", "(V1, ...)"],
                ["u1", "(V2, ...)", "u3"],
                ["(V3, ...)", "u4", "u3"],
            ],
            "$_MyVariadic2_1_output": [
                ["u1", "u2", "(V1, ...)"],
                ["u1", "(V2, ...)", "u3"],
                ["(V3, ...)", "u4", "u3"],
            ],
            "$_MyVariadic1_2_output": [
                ["u1", "u2", "(V1, ...)"],
                ["u1", "(V2, ...)", "u3"],
                ["(V3, ...)", "u4", "u3"],
            ],
            "$_MyVariadic3_3_output": [
                ["u1", "u2", "(V1, ...)"],
                ["u1", "(V2, ...)", "u3"],
                ["(V3, ...)", "u4", "u3"],
            ],
            "$_MyVariadic4_4_output": [
                ["u1", "u2", "(V1, ...)"],
                ["u1", "(V2, ...)", "u3"],
                ["(V3, ...)", "u4", "u3"],
            ],
            "$_MyVariadic1_5_output": ["u1", "(V2, ...)", "u5"],
            "input1": [
                ["u1", "u2", "(V1, ...)"],
                ["u1", "(V2, ...)", "u3"],
                ["(V3, ...)", "u4", "u3"],
            ],
            "input2": [
                ["u1", "u2", "(V1, ...)"],
                ["u1", "(V2, ...)", "u3"],
                ["(V3, ...)", "u4", "u3"],
            ],
            "input3": ["u1", "(V2, ...)", "u5"],
        }
        assert_shapes(model, ref_shapes)


def test_variadic_naming_3():
    model = Model()
    model |= (var1 := MyVariadic1()).connect(input="input")
    model |= MyVariadic4().connect(input=var1.output, output=IOKey(name="output"))
    ref_shapes = {
        "$_MyVariadic1_0_output": [["u1", "(V1, ...)"], ["(V2, ...)", "u2"]],
        "input": [["u1", "(V1, ...)"], ["(V2, ...)", "u2"]],
        "output": [["u1", "(V1, ...)"], ["(V2, ...)", "u2"]],
    }
    assert_shapes(model, ref_shapes)


def test_variadic_naming_4():
    model = Model()
    model |= (var2 := MyVariadic4()).connect(input="a", output=IOKey(name="output"))
    model |= MyVariadic1().connect(input="input", output=var2.input)
    ref_shapes = {
        "a": [["u1", "(V1, ...)"], ["(V2, ...)", "u2"]],
        "input": [["u1", "(V1, ...)"], ["(V2, ...)", "u2"]],
        "output": [["u1", "(V1, ...)"], ["(V2, ...)", "u2"]],
    }
    assert_shapes(model, ref_shapes)


def test_variadic_naming_5():
    model = Model()
    model |= (var1 := MyVariadic5()).connect(input="input")
    model |= MyVariadic4().connect(input=var1.output, output=IOKey(name="output"))

    ref_shapes = {
        "$_MyVariadic5_0_output": [["(V1, ...)", "a"], ["c", "(V2, ...)"]],
        "input": [["(V1, ...)", "a", "b"], ["c", "(V2, ...)", "b"]],
        "output": [["(V1, ...)", "a"], ["c", "(V2, ...)"]],
    }
    assert_shapes(model, ref_shapes)


def test_variadic_naming_6():
    model = Model()
    model |= (var2 := MyVariadic4()).connect(input="a", output=IOKey(name="output"))
    model |= MyVariadic5().connect(input="input", output=var2.input)

    ref_shapes = {
        "a": [["(V1, ...)", "a"], ["c", "(V2, ...)"]],
        "input": [["(V1, ...)", "a", "b"], ["c", "(V2, ...)", "b"]],
        "output": [["(V1, ...)", "a"], ["c", "(V2, ...)"]],
    }
    assert_shapes(model, ref_shapes)


def test_variadic_naming_7():
    model = Model()
    model |= MyVariadic1().connect(
        input="input",
    )
    model += MyVariadic4()
    model += MyVariadic6()
    model.set_shapes(input=[("Var", ...), 1])
    ref_shapes: dict[str, list] = {
        "$_MyVariadic1_0_output": [["(V1, ...)", 1], ["u2", "(V2, ...)"]],
        "$_MyVariadic4_1_output": [["(V1, ...)", 1], ["u2", "(V2, ...)"]],
        "$_MyVariadic6_2_output": [1, 1],
        "input": [["u2", "(V2, ...)"], ["(V1, ...)", 1]],
    }
    assert_shapes(model, ref_shapes)


def test_unresolved_merge_1():
    # [V1, a] - [V1, a]
    #           [b, V2] - [b, V2]
    #                     [V3, c] - [c, c]
    model = Model()
    model += MyVariadic1().connect(
        input="input",
    )
    model += MyVariadic4()
    model += MyVariadic6()
    model.set_shapes(input=[("Var", ...), 1])

    ref_shapes: dict[str, list] = {
        "$_MyVariadic1_0_output": [["u1", "(V2, ...)"], ["(V1, ...)", 1]],
        "$_MyVariadic4_1_output": [["u1", "(V2, ...)"], ["(V1, ...)", 1]],
        "$_MyVariadic6_2_output": [1, 1],
        "input": [["u1", "(V2, ...)"], ["(V1, ...)", 1]],
    }
    assert_shapes(model, ref_shapes)


def test_unresolved_merge_2():
    model = Model()
    model += MyVariadic5()
    model += MyVariadic11()
    ref_shapes = {
        "$_MyVariadic5_0_output": [["(V1, ...)", "u1"], ["u2", "(V2, ...)"]],
        "$_MyVariadic11_1_output": [
            ["u2", "(V2, ...)", "u3"],
            ["(V1, ...)", "u1", "u3"],
        ],
        "$input": [["(V1, ...)", "u1", "u4"], ["u2", "(V2, ...)", "u4"]],
    }
    assert_shapes(model, ref_shapes)


def test_unresolved_merge_3():
    model = Model()
    sig_1 = Sigmoid()
    sig_2 = Sigmoid()
    sig_1.set_shapes(input=["a", ("V1", ...)])
    sig_1.set_shapes(input=[("V2", ...), "b"])
    model += sig_1
    model += sig_2
    ref_shapes = {
        "$_Sigmoid_0_output": [["(V1, ...)", "u1"], ["u2", "(V2, ...)"]],
        "$_Sigmoid_1_output": [["(V1, ...)", "u1"], ["u2", "(V2, ...)"]],
        "$input": [["(V1, ...)", "u1"], ["u2", "(V2, ...)"]],
    }
    assert_shapes(model, ref_shapes)


def test_unresolved_merge_4():
    model = Model()
    sig_1 = Sigmoid()
    sig_2 = Sigmoid()
    sig_1.set_shapes(input=["a", ("V1", ...), "b"])
    sig_1.set_shapes(input=["c", "d", ("V3", ...)])
    sig_2.set_shapes(input=["e", "f", ("V4", ...)])
    model += sig_1
    model += sig_2
    ref_shapes = {
        "$_Sigmoid_0_output": [["u1", "u2", "(V1, ...)"], ["u1", "(V2, ...)", "u3"]],
        "$_Sigmoid_1_output": [["u1", "u2", "(V1, ...)"], ["u1", "(V2, ...)", "u3"]],
        "$input": [["u1", "(V2, ...)", "u3"], ["u1", "u2", "(V1, ...)"]],
    }
    assert_shapes(model, ref_shapes)


def test_unresolved_merge_5():
    model = Model()
    sig_1 = Sigmoid()
    sig_2 = Sigmoid()
    shape_1: dict[str, list] = {"input": ["a", ("V1", ...)]}
    shape_2: dict[str, list] = {"input": [("V2", ...), "e"]}
    shape_3: dict[str, list] = {"input": [("V3", ...), "b"]}
    sig_1.set_shapes(**shape_1)
    sig_1.set_shapes(**shape_2)
    sig_2.set_shapes(**shape_3)
    model |= sig_1
    model |= sig_2.connect(input=sig_1.output, output=IOKey(name="output"))
    ref_shapes = {
        "$_Sigmoid_0_output": [["a", "(V1, ...)"], ["(V2, ...)", "b"]],
        "$input": [["a", "(V1, ...)"], ["(V2, ...)", "b"]],
        "output": [["a", "(V1, ...)"], ["(V2, ...)", "b"]],
    }
    assert_shapes(model, ref_shapes)


def test_unresolved_merge_6():
    model = Model()
    sig_1 = Sigmoid()
    sig_2 = Sigmoid()
    sig_1.set_shapes(output=[1, ("V1", ...)])
    sig_1.set_shapes(output=[("V1", ...), "u1"])
    sig_2.set_shapes(input=["u1", "u2", ("V1", ...)])
    sig_2.set_shapes(input=["u1", ("V1", ...), "u2"])
    sig_2.set_shapes(input=[("V1", ...), "u1", "u2"])
    model += sig_1
    model += sig_2
    ref_shapes: dict[str, list] = {
        "$_Sigmoid_0_output": [
            [1, "u1", "(V1, ...)"],
            [1, "(V2, ...)", "u2"],
            ["(V3, ...)", "u3", "u2"],
        ],
        "$_Sigmoid_1_output": [
            [1, "u1", "(V1, ...)"],
            [1, "(V2, ...)", "u2"],
            ["(V3, ...)", "u3", "u2"],
        ],
        "$input": [
            [1, "u1", "(V1, ...)"],
            [1, "(V2, ...)", "u2"],
            ["(V3, ...)", "u3", "u2"],
        ],
    }
    assert_shapes(model, ref_shapes)


def test_unresolved_merge_7():
    model = Model()
    sig_1 = Sigmoid()
    sig_2 = Sigmoid()

    sig_1.set_shapes(output=[1, ("V1", ...)])
    sig_1.set_shapes(output=[("V1", ...), 1])
    sig_2.set_shapes(input=["u1", "u2", ("V1", ...)])
    sig_2.set_shapes(input=["u1", ("V1", ...), "u2"])
    sig_2.set_shapes(input=[("V1", ...), "u1", "u2"])

    model += sig_1
    model += sig_2
    ref_shapes: dict[str, list] = {
        "$_Sigmoid_0_output": [
            [1, "u1", "(V1, ...)"],
            [1, "(V2, ...)", 1],
            ["(V3, ...)", "u3", 1],
        ],
        "$_Sigmoid_1_output": [
            [1, "u1", "(V1, ...)"],
            [1, "(V2, ...)", 1],
            ["(V3, ...)", "u3", 1],
        ],
        "$input": [[1, "u1", "(V1, ...)"], [1, "(V2, ...)", 1], ["(V3, ...)", "u3", 1]],
    }
    assert_shapes(model, ref_shapes)


def test_unresolved_merge_8():
    model = Model()
    sig_1 = Sigmoid()
    sig_2 = Sigmoid()
    sig_3 = Sigmoid()
    sig_3.set_shapes(input=["u1", "u2", ("V1", ...), "u2", "u1"])
    sig_1.set_shapes(output=[1, ("V1", ...)])
    sig_1.set_shapes(output=[("V1", ...), 1])

    sig_2.set_shapes(input=["u1", "u2", ("V1", ...)])
    sig_2.set_shapes(input=["u1", ("V1", ...), "u2"])
    sig_2.set_shapes(input=[("V1", ...), "u1", "u2"])

    model |= sig_1.connect(input="input")
    model |= sig_2.connect(input="input")
    model |= sig_3.connect(input="input")
    ref_shapes: dict[str, list] = {
        "$_Sigmoid_0_output": [1, "u1", "(V1, ...)", "u1", 1],
        "$_Sigmoid_1_output": [1, "u1", "(V1, ...)", "u1", 1],
        "$_Sigmoid_2_output": [1, "u1", "(V1, ...)", "u1", 1],
        "input": [1, "u1", "(V1, ...)", "u1", 1],
    }
    assert_shapes(model, ref_shapes)


def test_unresolved_merge_9():
    model = Model()
    sig_1 = Sigmoid()
    sig_2 = Sigmoid()
    sig_3 = Sigmoid()

    sig_3.set_shapes(input=["u1", "u2", ("V1", ...), "u3", "u4"])
    sig_1.set_shapes(input=[1, ("V1", ...)])
    sig_1.set_shapes(input=[("V1", ...), 4])

    sig_2.set_shapes(input=["u1", 2, ("V1", ...)])
    sig_2.set_shapes(input=[("V1", ...), 3, "u2"])

    model |= sig_1.connect(input="input")
    model |= sig_2.connect(input="input")
    model |= sig_3.connect(input="input")
    ref_shapes: dict[str, list] = {
        "$_Sigmoid_0_output": [1, 2, "(V1, ...)", 3, 4],
        "$_Sigmoid_1_output": [1, 2, "(V1, ...)", 3, 4],
        "$_Sigmoid_2_output": [1, 2, "(V1, ...)", 3, 4],
        "input": [1, 2, "(V1, ...)", 3, 4],
    }
    assert_shapes(model, ref_shapes)


def test_unresolved_merge_10():
    model = Model()
    m1 = MyVariadic11()
    m2 = Sigmoid()
    m2.set_shapes(input=[("V1", ...), "u1", "u2"])
    m2.set_shapes(input=["u1", ("V1", ...), "u2"])
    m2.set_shapes(input=["u1", "u2", ("V1", ...)])
    model += m2
    model += m1
    ref_shapes = {
        "$_Sigmoid_0_output": [
            ["u1", "u2", "(V1, ...)"],
            ["(V2, ...)", "u3", "u4"],
            ["u1", "(V3, ...)", "u4"],
        ],
        "$_MyVariadic11_1_output": [
            ["u1", "u2", "(V1, ...)", "u5"],
            ["(V2, ...)", "u3", "u4", "u5"],
            ["u1", "(V3, ...)", "u4", "u5"],
        ],
        "$input": [
            ["u1", "u2", "(V1, ...)"],
            ["(V2, ...)", "u3", "u4"],
            ["u1", "(V3, ...)", "u4"],
        ],
    }
    assert_shapes(model, ref_shapes)


def test_unresolved_merge_11():
    model = Model()
    m1 = MyVariadic11()
    m2 = Sigmoid()

    m2.set_shapes(input=[("V1", ...), "u1", "u2"])
    m2.set_shapes(input=["u1", ("V1", ...), "u2"])
    m2.set_shapes(input=["u1", "u2", ("V1", ...)])

    model += m2
    model += m1
    ref_shapes = {
        "$_Sigmoid_0_output": [
            ["u1", "u2", "(V1, ...)"],
            ["(V2, ...)", "u3", "u4"],
            ["u1", "(V3, ...)", "u4"],
        ],
        "$_MyVariadic11_1_output": [
            ["u1", "u2", "(V1, ...)", "u5"],
            ["(V2, ...)", "u3", "u4", "u5"],
            ["u1", "(V3, ...)", "u4", "u5"],
        ],
        "$input": [
            ["u1", "u2", "(V1, ...)"],
            ["(V2, ...)", "u3", "u4"],
            ["u1", "(V3, ...)", "u4"],
        ],
    }
    assert_shapes(model, ref_shapes)


def test_unresolved_merge_12():
    model = Model()
    m1 = Model2()
    model += deepcopy(m1)
    model += deepcopy(m1)
    model += deepcopy(m1)
    model += deepcopy(m1)
    model += deepcopy(m1)
    ref_shapes: dict[str, list] = {
        "$_Model2_0_output": [["u1", "(V1, ...)"], ["(V2, ...)", "u2"]],
        "$_Model2_1_output": [["u2", "(V2, ...)"], ["(V3, ...)", "u3"]],
        "$_Model2_2_output": [["u3", "(V3, ...)"], ["(V4, ...)", "u4"]],
        "$_Model2_3_output": [["u4", "(V4, ...)"], ["(V5, ...)", "u5"]],
        "$_Model2_4_output": ["u5", "(V5, ...)"],
        "$input": ["(V1, ...)", "u1"],
    }
    assert_shapes(model, ref_shapes)


def test_variadic_naming_8():
    model = Model()
    t_1 = Model2()
    t_2 = Model2()
    t_3 = Model2()
    model += t_1
    model += t_2
    model += t_3

    ref_shapes: dict[str, list] = {
        "$input": ["(V1, ...)", "u1"],
        "$_Model2_0_output": [["u1", "(V1, ...)"], ["(V2, ...)", "u2"]],
        "$_Model2_1_output": [["u2", "(V2, ...)"], ["(V3, ...)", "u3"]],
        "$_Model2_2_output": ["u3", "(V3, ...)"],
    }
    assert_shapes(model, ref_shapes)


def test_variadic_naming_9():
    model = Model()
    t_1 = Model2()
    t_2 = Model2()
    t_3 = Model2()
    model += t_1
    model += t_2
    model += t_3

    ref_shapes: dict[str, list] = {
        "$input": ["(V1, ...)", "u1"],
        "$_Model2_0_output": [["u1", "(V1, ...)"], ["(V2, ...)", "u2"]],
        "$_Model2_1_output": [["u2", "(V2, ...)"], ["(V3, ...)", "u3"]],
        "$_Model2_2_output": ["u3", "(V3, ...)"],
    }
    assert_shapes(model, ref_shapes)


def test_variadic_naming_10():
    model = Model()
    t_1 = Model2()
    t_2 = Model2()
    t_3 = Model2()
    model += t_1.connect(input="input", output=IOKey(name="output"))
    model += t_2
    model += t_3
    model.set_shapes(output=[3, 4])
    ref_shapes = {
        "input": [4, 3],
        "output": [3, 4],
        "$_Model2_1_output": [4, 3],
        "$_Model2_2_output": [3, 4],
    }
    assert_shapes(model, ref_shapes)


def test_variadic_naming_11():
    model = Model()
    t_1 = Model2()
    t_2 = Model2()
    t_3 = Model2()
    model += t_1.connect(input="input", output=IOKey(name="output"))
    model += t_2
    model += t_3
    model.set_shapes(output=[3, "a"])
    ref_shapes: dict[str, list] = {
        "input": ["u1", 3],
        "output": [3, "u1"],
        "$_Model2_1_output": ["u1", 3],
        "$_Model2_2_output": [3, "u1"],
    }
    assert_shapes(model, ref_shapes)


def test_variadic_naming_11_1():
    model = Model()
    t_1 = Model2()
    t_2 = Model6()
    model += t_1.connect(input="input", output=IOKey(name="output"))
    model += t_2
    ref_shapes = {
        "$_Model6_1_output": ["(V1, ...)", "u1"],
        "input": ["(V1, ...)", "u1"],
        "output": ["u1", "(V1, ...)"],
    }
    assert_shapes(model, ref_shapes)


def test_most_informative_repr_1():
    repr1 = ShapeRepr([Uniadic(), Uniadic()], Variadic(), [Uniadic()])
    repr2 = ShapeRepr(None, Variadic(), [Uniadic(), Uniadic()])

    repr2.node.merge(repr1.node)
    assert repr1.node.get_shapes() == ["u1", "u2", "(V1, ...)", "u3"]
    ref_shape = {
        "my_input": [["u3", "(V1, ...)", "u1", "u2"], ["u3", "u4", "(V2, ...)", "u2"]]
    }
    check_shapes_semantically(
        {"my_input": repr1.node.get_shapes(verbose=True)}, ref_shape
    )


def test_most_informative_repr_2():
    repr1 = ShapeRepr([Uniadic(), Uniadic()], Variadic(), [Uniadic()])
    repr2 = ShapeRepr(None, Variadic(), [Uniadic(), Uniadic(), Uniadic()])

    repr2.node.merge(repr1.node)
    assert repr1.node.get_shapes() == ["u1", "u2", "(V1, ...)", "u3"]
    ref_shape = {
        "my_input": [["(V1, ...)", "u1", "u2", "u3"], ["u4", "u5", "(V2, ...)", "u3"]]
    }
    check_shapes_semantically(
        {"my_input": repr1.node.get_shapes(verbose=True)}, ref_shape
    )


def test_most_informative_repr_3():
    repr1 = ShapeRepr([Uniadic(1), Uniadic()], Variadic(), [Uniadic()])
    repr2 = ShapeRepr(None, Variadic(), [Uniadic(), Uniadic(), Uniadic()])

    repr2.node.merge(repr1.node)
    assert repr1.node.get_shapes() == [1, "u1", "(V1, ...)", "u2"]
    ref_shape: dict[str, list] = {
        "my_input": [["(V1, ...)", "u1", "u2", "u3"], [1, "u4", "(V2, ...)", "u3"]]
    }
    check_shapes_semantically(
        {"my_input": repr1.node.get_shapes(verbose=True)}, ref_shape
    )


def test_most_informative_repr_4():
    repr1 = ShapeRepr([Uniadic(1), Uniadic()], Variadic(), [Uniadic()])
    repr2 = ShapeRepr(None, Variadic(), [Uniadic(2), Uniadic(), Uniadic(3)])

    repr2.node.merge(repr1.node)
    assert repr1.node.get_shapes() == [1, "(V1, ...)", 2, "u1", 3]
    ref_shape: dict[str, list] = {
        "my_input": [[1, "(V1, ...)", 2, "u1", 3], [1, "u2", "(V2, ...)", "u1", 3]]
    }
    check_shapes_semantically(
        {"my_input": repr1.node.get_shapes(verbose=True)}, ref_shape
    )


def test_most_informative_repr_5():
    repr1 = ShapeRepr([Uniadic(1), Uniadic(), Uniadic()], Variadic(), [Uniadic()])
    repr2 = ShapeRepr(None, Variadic(), [Uniadic(2), Uniadic(), Uniadic(3)])

    repr2.node.merge(repr1.node)
    assert repr1.node.get_shapes() == [1, "(V1, ...)", 2, "u1", 3]
    ref_shape: dict[str, list] = {
        "my_input": [[1, "(V1, ...)", 2, "u1", 3], [1, "u2", "u3", "(V2, ...)", 3]]
    }
    check_shapes_semantically(
        {"my_input": repr1.node.get_shapes(verbose=True)}, ref_shape
    )


def test_most_informative_repr_6():
    repr1 = ShapeRepr([Uniadic(1), Uniadic(2), Uniadic()], Variadic(), [Uniadic()])
    repr2 = ShapeRepr(None, Variadic(), [Uniadic(2), Uniadic(), Uniadic(3), Uniadic()])

    repr2.node.merge(repr1.node)
    assert repr1.node.get_shapes() == [1, 2, "u1", "(V1, ...)", 3, "u2"]
    ref_shapes: dict[str, list] = {
        "my_input": [
            [1, "(V1, ...)", 2, "u1", 3, "u2"],
            [1, 2, "u3", "(V2, ...)", 3, "u2"],
        ]
    }
    check_shapes_semantically(
        {"my_input": repr1.node.get_shapes(verbose=True)}, ref_shapes
    )


def test_most_informative_repr_7():
    u1 = Uniadic(2)
    ShapeRepr([u1])  # DO NOT DELETE!, it increases repr count of the symbol u1
    repr1 = ShapeRepr([Uniadic(1), Uniadic(2), Uniadic()], Variadic(), [Uniadic()])
    repr2 = ShapeRepr(None, Variadic(), [u1, Uniadic(), Uniadic(3), Uniadic()])
    # [1, 2, a, V1, b]
    # [V2, 2, c, 3, d]
    # [V2, 2, c, 3, d] - Expected because repr count of repr2's symbols' reprs are
    # more since we added u1 into another repr (i.e. repr3)

    repr2.node.merge(repr1.node)
    assert repr1.node.get_shapes() == [1, "(V1, ...)", 2, "u1", 3, "u2"]
    ref_shapes: dict[str, list] = {
        "my_input": [
            [1, "(V1, ...)", 2, "u1", 3, "u2"],
            [1, 2, "u3", "(V2, ...)", 3, "u2"],
        ]
    }
    check_shapes_semantically(
        {"my_input": repr1.node.get_shapes(verbose=True)}, ref_shapes
    )


def test_variadic_naming_11_2():
    model = Model()
    t_1 = Model2()
    t_2 = Model7()
    model += t_1.connect(input="input", output=IOKey(name="output"))
    model += t_2
    ref_shapes: dict[str, list] = {
        "$_Model7_1_output": ["u1", "(V1, ...)"],
        "input": ["(V1, ...)", "u1"],
        "output": ["u1", "(V1, ...)"],
    }
    assert_shapes(model, ref_shapes)


def test_variadic_naming_11_3():
    model = Model()
    t_1 = Model7()
    t_2 = Model8()
    model += t_1.connect(input="input", output=IOKey(name="output"))
    model += t_2
    ref_shapes = {
        "$_Model8_1_output": [["(V1, ...)", "u1"], ["u2", "(V2, ...)"]],
        "input": [["(V1, ...)", "u1"], ["u2", "(V2, ...)"]],
        "output": [["u2", "(V2, ...)"], ["(V1, ...)", "u1"]],
    }
    assert_shapes(model, ref_shapes)


def test_variadic_naming_12():
    # TODO: What is purpose of this test?
    model = Model()
    buff_1 = Buffer()
    buff_2 = Buffer()
    buff_3 = Buffer()
    model |= buff_1.connect(input="input", output=IOKey(name="output1"))
    model |= buff_2.connect(input="output1", output=IOKey(name="output2"))
    model |= buff_3.connect(input="output2", output=IOKey(name="output3"))


def test_variadic_naming_13():
    model = Model()
    model |= (mult := MatrixMultiply()).connect(
        left=IOKey("input", type=Tensor),
        right=IOKey("w", type=Tensor),
    )
    model |= Add().connect(
        left=mult.output,
        right=IOKey("b", type=Tensor),
        output=IOKey(name="output"),
    )

    model.set_shapes(
        input=["N", ("Var_inter", ...), "d_in"],
        w=["d_in", "d_out"],
        output=["N", ("Var_inter", ...), "d_out"],
        b=["d_out"],
    )
    ref_shapes: dict[str, list] = {
        "$_MatrixMultiply_0_output": [
            ["u1", "(V1, ...)", "u2"],
            ["(V2, ...)", "u3", "u2"],
        ],
        "input": [["(V2, ...)", "u3", "u4"], ["u1", "(V1, ...)", "u4"]],
        "w": ["u4", "u2"],
        "b": ["u2"],
        "output": [["u1", "(V1, ...)", "u2"], ["(V2, ...)", "u3", "u2"]],
    }
    assert_shapes(model, ref_shapes)


def test_variadic_naming_14() -> None:
    class MyModel(PrimitiveModel):
        input1: Connection
        input2: Connection
        output: Connection

        def __init__(self) -> None:
            super().__init__(
                formula_key="buffer",
                input1=BaseKey(shape=["a", ("Var1", ...), "b"], type=Tensor),
                input2=BaseKey(shape=[("Var1", ...)], type=Tensor),
                output=BaseKey(shape=["a", ("Var1", ...), "c"], type=Tensor),
            )

    model = MyModel()
    model.set_shapes(output=["a", "d", ("Var2", ...)])
    model.set_shapes(
        input1=[("V", ...), "e"],
        output=[("V1", ...), "e"],
    )
    ref_shapes: Mapping[str, Sequence[Sequence[str] | str]] = {
        "input1": [["u1", "(V1, ...)", "u2"], ["u1", "u3", "(V2, ...)"]],
        "input2": ["(V1, ...)"],
        "output": [["u1", "(V1, ...)", "u2"], ["u1", "u3", "(V2, ...)"]],
    }
    assert_shapes(model, ref_shapes)


def test_variadic_naming_15() -> None:
    class MyModel(PrimitiveModel):
        input1: Connection
        input2: Connection
        output: Connection

        def __init__(self) -> None:
            super().__init__(
                formula_key="buffer",
                input1=BaseKey(shape=["a", ("Var1", ...), "b"], type=Tensor),
                input2=BaseKey(shape=["b", "c"], type=Tensor),
                output=BaseKey(shape=["a", ("Var1", ...), "c"], type=Tensor),
            )

    model = MyModel()
    model.set_shapes(output=["a", "d", ("Var2", ...)])
    model.set_shapes(input2=["e", "e"])
    ref_shapes: Mapping[str, Sequence[Sequence[str] | str]] = {
        "input1": [["u1", "(V1, ...)", "u2"], ["u1", "u3", "(V2, ...)"]],
        "input2": ["u2", "u2"],
        "output": [["u1", "(V1, ...)", "u2"], ["u1", "u3", "(V2, ...)"]],
    }
    assert_shapes(model, ref_shapes)


def test_variadic_naming_16() -> None:
    class MyModel(PrimitiveModel):
        input: Connection
        output: Connection

        def __init__(self) -> None:
            super().__init__(
                formula_key="buffer",
                input=BaseKey(shape=["a", ("Var1", ...), "b"], type=Tensor),
                output=BaseKey(shape=["b", ("Var1", ...), "a"], type=Tensor),
            )

        def connect(  # type: ignore[override]
            self, input: ConnectionType = NOT_GIVEN, output: ConnectionType = NOT_GIVEN
        ):
            return ExtendInfo(self, {"input": input, "output": output})

    model = Model()
    test_model = MyModel()
    sig_model = Sigmoid()

    shape_1: Mapping[str, Sequence[str | tuple[str, EllipsisType]]] = {
        "output": ["b", ("Var1", ...)]
    }
    shape_2: Mapping[str, Sequence[str | tuple[str, EllipsisType]]] = {
        "output": [("Var1", ...), "a"]
    }
    sig_model.set_shapes(**shape_1)
    sig_model.set_shapes(**shape_2)
    model |= sig_model.connect(input="input")
    model |= test_model.connect(input=sig_model.output, output=IOKey(name="output"))

    ref_shapes: Mapping[str, Sequence[Sequence[str] | str]] = {
        "input": ["u1", "(V1, ...)", "u2"],
        "$_Sigmoid_0_output": ["u1", "(V1, ...)", "u2"],
        "output": ["u2", "(V1, ...)", "u1"],
    }

    assert_shapes(model, ref_shapes)


def test_variadic_naming_17() -> None:
    class MyModel(PrimitiveModel):
        input: Connection
        output: Connection

        def __init__(self) -> None:
            super().__init__(
                formula_key="buffer",
                input=BaseKey(shape=["a", ("Var1", ...), "b"], type=Tensor),
                output=BaseKey(shape=["b", ("Var1", ...), "a"], type=Tensor),
            )

        def connect(  # type: ignore[override]
            self, input: ConnectionType = NOT_GIVEN, output: ConnectionType = NOT_GIVEN
        ):
            return ExtendInfo(self, {"input": input, "output": output})

    for _ in range(100):
        model = Model()
        test_model = MyModel()
        sig_model = Sigmoid()
        shape_1: Mapping[str, Sequence[str | tuple[str, EllipsisType]]] = {
            "output": ["a", "c", "b", ("Var1", ...)]
        }
        shape_2: Mapping[str, Sequence[str | tuple[str, EllipsisType]]] = {
            "output": [("Var1", ...), "a", "b", "c"]
        }
        sig_model.set_shapes(**shape_1)
        sig_model.set_shapes(**shape_2)
        model |= sig_model.connect(input="input")
        model |= test_model.connect(input=sig_model.output, output=IOKey(name="output"))

        ref_shapes: Mapping[str, Sequence[Sequence[str] | str]] = {
            "input": [
                ["u1", "u2", "u3", "(V1, ...)"],
                ["u1", "u2", "(V3, ...)", "u6"],
                ["(V2, ...)", "u4", "u5", "u6"],
            ],
            "$_Sigmoid_0_output": [
                ["u1", "u2", "u3", "(V1, ...)"],
                ["(V2, ...)", "u4", "u5", "u6"],
                ["u1", "u2", "(V3, ...)", "u6"],
            ],
            "output": ["u6", "u2", "(V3, ...)", "u1"],
        }
        assert_shapes(model, ref_shapes)


def test_variadic_naming_18():
    add_model_1 = Add()
    add_model_1.set_types(left=Tensor, right=Tensor)
    add_model_1.set_cin("left")

    add_model_2 = Add()
    add_model_2.set_types(right=Tensor)
    add_model_2.set_cin("left")

    add_model_3 = Add()
    add_model_3.set_types(right=Tensor)
    add_model_3.set_cin("left")

    add_model_4 = Add()
    add_model_4.set_types(right=Tensor)
    add_model_4.set_cin("left")

    add_model_5 = Add()
    add_model_5.set_types(left=Tensor, right=Tensor)
    add_model_5.set_cin("left")

    model = Model()
    model |= add_model_1.connect(left="left", right="right")
    model += add_model_2
    model += add_model_3
    model += add_model_4
    model |= add_model_5.connect(output=IOKey(name="output"))
    shape_1: dict[str, list] = {"output": ["u1", "u2", "u3", ("Var1", ...)]}
    shape_2: dict[str, list] = {"output": ["u1", "u2", ("Var1", ...), "u3"]}
    shape_3: dict[str, list] = {"output": ["u1", ("Var1", ...), "u2", "u3"]}
    shape_4: dict[str, list] = {
        "left": [("Var1", ...), "u1", "u2", "u3"],
        "output": [("Var1", ...), "u1", "u2", "u3"],
    }
    model.set_shapes(**shape_1)
    model.set_shapes(**shape_2)
    model.set_shapes(**shape_3)
    model.set_shapes(**shape_4)

    ref_shapes: dict[str, list] = {
        "$_Add_0_output": ["(V1, ...)", "u1", "u2", "u3"],
        "$_Add_1_output": ["(V2, ...)", "u4", "u5", "u6"],
        "$_Add_2_output": ["(V3, ...)", "u7", "u8", "u9"],
        "$_Add_3_output": ["(V4, ...)", "u10", "u11", "u12"],
        "left": [
            ["(V5, ...)", "u13", "u14", "u15"],
            ["u16", "u17", "(V6, ...)", "u15"],
            ["u16", "u17", "u18", "(V7, ...)"],
            ["u16", "(V8, ...)", "u14", "u15"],
        ],
        "right": ["(V9, ...)"],
        "$_right_0": ["(V10, ...)"],
        "$_right_1": ["(V11, ...)"],
        "$_right_2": ["(V12, ...)"],
        "$_left": ["(V13, ...)"],
        "$_right_3": ["(V14, ...)"],
        "output": [
            ["u16", "u17", "u18", "(V7, ...)"],
            ["(V5, ...)", "u13", "u14", "u15"],
            ["u16", "(V8, ...)", "u14", "u15"],
            ["u16", "u17", "(V6, ...)", "u15"],
        ],
    }

    assert_shapes(model, ref_shapes)


def test_variadic_naming_19():
    for _ in range(100):
        add_model_1 = Add()
        add_model_1.set_types(left=Tensor, right=Tensor)
        add_model_1.set_cin("left")

        add_model_2 = Add()
        add_model_2.set_types(right=Tensor)
        add_model_2.set_cin("left")

        add_model_3 = Add()
        add_model_3.set_types(right=Tensor)
        add_model_3.set_cin("left")

        add_model_4 = Add()
        add_model_4.set_types(right=Tensor)
        add_model_4.set_cin("left")

        add_model_5 = Add()
        add_model_5.set_types(left=Tensor, right=Tensor)
        add_model_5.set_cin("left")

        model = Model()
        model += add_model_1.connect(left="left", right="right")
        model += add_model_2
        model += add_model_3
        model += add_model_4
        model |= add_model_5.connect(output=IOKey(name="output"))

        model.set_shapes(output=["u1", "u2", "u3", ("Var1", ...)])
        model.set_shapes(output=["u1", "u2", ("Var1", ...), "u3"])
        model.set_shapes(output=["u1", ("Var1", ...), "u2", "u3"])
        model.set_shapes(left=["u1", "u2", ("Var1", ...), "u3", "u4"])
        model.set_shapes(
            left=[("Var1", ...), "u1", "u2", "u3"],
            output=[("Var1", ...), "u1", "u2", "u3"],
        )

    ref_shapes: dict[str, list] = {
        "$_Add_0_output": ["u1", "u2", "(V1, ...)", "u3", "u4"],
        "$_Add_1_output": ["u5", "u6", "(V2, ...)", "u7", "u8"],
        "$_Add_2_output": ["u9", "u10", "(V3, ...)", "u11", "u12"],
        "$_Add_3_output": ["u13", "u14", "(V4, ...)", "u15", "u16"],
        "left": [
            ["u17", "u18", "u19", "(V5, ...)", "u20"],
            ["u17", "u18", "(V6, ...)", "u21", "u20"],
            ["u17", "(V7, ...)", "u22", "u21", "u20"],
        ],
        "right": ["(V8, ...)"],
        "$_right_0": ["(V9, ...)"],
        "$_right_1": ["(V10, ...)"],
        "$_right_2": ["(V11, ...)"],
        "$_left": ["(V12, ...)"],
        "$_right_3": ["(V13, ...)"],
        "output": [
            ["u17", "u18", "u19", "(V5, ...)", "u20"],
            ["u17", "u18", "(V6, ...)", "u21", "u20"],
            ["u17", "(V7, ...)", "u22", "u21", "u20"],
        ],
    }

    assert_shapes(model, ref_shapes)


def test_variadic_naming_20():
    add_model_1 = Add()
    add_model_1.set_types(left=Tensor, right=Tensor)
    add_model_1.set_cin("left")

    add_model_2 = Add()
    add_model_2.set_types(right=Tensor)
    add_model_2.set_cin("left")

    add_model_3 = Add()
    add_model_3.set_types(right=Tensor)
    add_model_3.set_cin("left")

    add_model_4 = Add()
    add_model_4.set_types(right=Tensor)
    add_model_4.set_cin("left")

    add_model_5 = Add()
    add_model_5.set_types(left=Tensor, right=Tensor)
    add_model_5.set_cin("left")

    model = Model()
    model += add_model_1.connect(left="left", right="right")
    model += add_model_2
    model += add_model_3
    model += add_model_4
    model |= add_model_5.connect(output=IOKey(name="output"))
    shape_1: dict[str, list] = {"output": ["a", ("Var1", ...), "b"]}
    shape_2: dict[str, list] = {"output": ["a", "b", ("Var1", ...)]}
    shape_3: dict[str, list] = {
        "left": [("Var1", ...), "a", "b", "c"],
        "output": [("Var1", ...), "a", "b"],
    }
    model.set_shapes(**shape_1)
    model.set_shapes(**shape_2)
    model.set_shapes(**shape_3)

    ref_shapes: dict[str, list] = {
        "$_Add_0_output": ["(V1, ...)", "u1", "u2", "u3"],
        "$_Add_1_output": ["(V2, ...)", "u4", "u5", "u6"],
        "$_Add_2_output": ["(V3, ...)", "u7", "u8", "u9"],
        "$_Add_3_output": ["(V4, ...)", "u10", "u11", "u12"],
        "left": [
            ["(V5, ...)", "u13", "u14", "u15"],
            ["u16", "u17", "(V6, ...)", "u15"],
            ["u16", "(V7, ...)", "u14", "u15"],
        ],
        "right": ["(V8, ...)"],
        "$_right_0": ["(V9, ...)"],
        "$_right_1": ["(V10, ...)"],
        "$_right_2": ["(V11, ...)"],
        "$_left": ["(V12, ...)"],
        "$_right_3": ["(V13, ...)"],
        "output": [
            ["u16", "(V7, ...)", "u14"],
            ["(V5, ...)", "u13", "u14"],
            ["u16", "u17", "(V6, ...)"],
        ],
    }

    assert_shapes(model, ref_shapes)


def test_variadic_naming_22():
    red_model = Mean(axis=TBD)
    sig_model = Sigmoid()
    shape_1: dict[str, list] = {"input": [("V1", ...), "a", "b"]}
    shape_2: dict[str, list] = {"input": ["a", ("V1", ...)]}
    shape_3: dict[str, list] = {"input": [("V1", ...), "a"]}
    red_model.set_shapes(**shape_1)
    sig_model.set_shapes(**shape_2)
    sig_model.set_shapes(**shape_3)
    model = Model()
    model |= sig_model.connect(output=IOKey(name="output"))
    model |= red_model.connect(input="input", output=sig_model.input, axis=-1)
    ref_shapes = {
        "$_Mean_0_output": [["u1", "(V1, ...)"], ["(V2, ...)", "u2"]],
        "input": [["(V2, ...)", "u2", "u3"], ["u1", "(V1, ...)", "u3"]],
        "$axis": None,
        "$_Mean_0_keepdim": None,
        "output": [["u1", "(V1, ...)"], ["(V2, ...)", "u2"]],
    }

    assert_shapes(model, ref_shapes)


def test_variadic_naming_23():
    model = Model()
    sig_model_1 = Sigmoid()
    sig_model_2 = Sigmoid()
    model |= sig_model_1.connect(input="input1", output=IOKey(name="output1"))
    model |= sig_model_2.connect(input="input2", output=IOKey(name="output2"))
    model.set_shapes(input1=[("V1", ...)], input2=[("V1", ...)])
    shape_1: dict[str, list] = {"output1": [("V1", ...), "a", "b"]}
    shape_2: dict[str, list] = {"output1": ["a", ("V1", ...), "b"]}
    model.set_shapes(**shape_1)
    model.set_shapes(**shape_2)

    ref_shapes: dict[str, list] = {
        "input1": [["(V1, ...)", "u1", "u2"], ["u3", "(V2, ...)", "u2"]],
        "input2": [["(V1, ...)", "u1", "u2"], ["u3", "(V2, ...)", "u2"]],
        "output1": [["(V1, ...)", "u1", "u2"], ["u3", "(V2, ...)", "u2"]],
        "output2": [["(V1, ...)", "u1", "u2"], ["u3", "(V2, ...)", "u2"]],
    }

    assert_shapes(model, ref_shapes)


def test_variadic_naming_24():
    model = Model()
    sig_model_1 = Sigmoid()
    sig_model_2 = Sigmoid()
    model |= sig_model_1.connect(input="input1", output=IOKey(name="output1"))
    model |= sig_model_2.connect(input="input2", output=IOKey(name="output2"))
    model.set_shapes(input1=[("V1", ...)], input2=[("V1", ...)])
    shape_1: dict[str, list] = {"output1": [("V1", ...), "a", "b"]}
    shape_2: dict[str, list] = {"output1": ["a", ("V1", ...), "b"]}
    model.set_shapes(**shape_1)
    model.set_shapes(**shape_2)

    ref_shapes: dict[str, list] = {
        "input1": [["(V1, ...)", "u1", "u2"], ["u3", "(V2, ...)", "u2"]],
        "input2": [["(V1, ...)", "u1", "u2"], ["u3", "(V2, ...)", "u2"]],
        "output1": [["(V1, ...)", "u1", "u2"], ["u3", "(V2, ...)", "u2"]],
        "output2": [["(V1, ...)", "u1", "u2"], ["u3", "(V2, ...)", "u2"]],
    }

    assert_shapes(model, ref_shapes)


def test_variadic_naming_25() -> None:
    class MyModel(PrimitiveModel):
        input: Connection
        output: Connection

        def __init__(self) -> None:
            super().__init__(
                formula_key="buffer",
                input=BaseKey(
                    shape=[("Var1", ...), "a", "b", "c"],
                    type=Tensor,
                ),
                output=BaseKey(
                    shape=["c", ("Var1", ...), "a", "b"],
                    type=Tensor,
                ),
            )

        def connect(  # type: ignore[override]
            self, input: ConnectionType = NOT_GIVEN, output: ConnectionType = NOT_GIVEN
        ):
            return ExtendInfo(self, {"input": input, "output": output})

    model = Model()
    model += MyModel().connect(input="input")
    model += MyModel()
    model += MyModel()
    model += MyModel()
    model += MyModel()
    model += MyModel()
    model += MyModel()
    model += MyModel()
    model.set_shapes(input=[3, 4, 5])
    ref_shapes = {
        "$_MyModel_0_output": [5, 3, 4],
        "$_MyModel_1_output": [4, 5, 3],
        "$_MyModel_2_output": [3, 4, 5],
        "$_MyModel_3_output": [5, 3, 4],
        "$_MyModel_4_output": [4, 5, 3],
        "$_MyModel_5_output": [3, 4, 5],
        "$_MyModel_6_output": [5, 3, 4],
        "$_MyModel_7_output": [4, 5, 3],
        "input": [3, 4, 5],
    }
    assert_shapes(model, ref_shapes)


def test_variadic_naming_26() -> None:
    class MyModel(PrimitiveModel):
        input1: Connection
        input2: Connection
        input3: Connection
        output: Connection

        def __init__(self) -> None:
            super().__init__(
                formula_key="buffer",
                input1=BaseKey(shape=["a", ("Var1", ...), "b"], type=Tensor),
                input2=BaseKey(shape=["_a", ("Var1", ...), "_b"], type=Tensor),
                input3=BaseKey(shape=["b", "c"], type=Tensor),
                output=BaseKey(shape=["a", ("Var1", ...), "c"], type=Tensor),
            )

        def connect(  # type: ignore[override]
            self,
            input1: ConnectionType = NOT_GIVEN,
            input2: ConnectionType = NOT_GIVEN,
            input3: ConnectionType = NOT_GIVEN,
            output: ConnectionType = NOT_GIVEN,
        ):
            return ExtendInfo(
                self,
                {
                    "input1": input1,
                    "input2": input2,
                    "input3": input3,
                    "output": output,
                },
            )

    model = MyModel()
    shape_1: Mapping[str, Sequence[str | int | tuple[str, EllipsisType]]] = {
        "output": [1, "d", ("Var2", ...)]
    }
    shape_2: Mapping[str, Sequence[str | int | tuple[str, EllipsisType]]] = {
        "input2": [1, ("Var3", ...), 2]
    }
    model.set_shapes(**shape_1)
    model.set_shapes(**shape_2)
    model.set_shapes(input3=[2, 2])
    ref_shapes: Mapping[str, Sequence[Sequence[int | str] | int | str]] = {
        "input1": [[1, "(V1, ...)", 2], [1, "u3", "(V2, ...)"]],
        "input2": [[1, "(V1, ...)", 2], [1, "u3", "(V2, ...)"]],
        "input3": [2, 2],
        "output": [[1, "(V1, ...)", 2], [1, "u3", "(V2, ...)"]],
    }
    assert_shapes(model, ref_shapes)


def test_variadic_naming_27() -> None:
    class MyModel(PrimitiveModel):
        input: Connection
        output: Connection

        def __init__(self) -> None:
            super().__init__(
                formula_key="buffer",
                input=BaseKey(shape=[("Var1", ...), "u1"], type=Tensor),
                output=BaseKey(shape=[("Var1", ...)], type=Tensor),
            )

        def connect(  # type: ignore[override]
            self, input: ConnectionType = NOT_GIVEN, output: ConnectionType = NOT_GIVEN
        ):
            return ExtendInfo(self, {"input": input, "output": output})

    test_model_1 = MyModel()

    model = Model()

    model |= (buffer1 := Buffer()).connect(input="input1")
    model |= Buffer().connect(input=buffer1.output, output="output1")

    model |= Buffer().connect(input="input2")
    model |= Buffer().connect(input=buffer1.output, output="output2")

    model.set_shapes(input1=[("Var1", ...)], input2=[("Var1", ...)])

    model |= test_model_1.connect(input="output1", output="output3")
    ref_shapes = {
        "$_Buffer_0_output": ["(V1, ...)", "u1"],
        "$_Buffer_2_output": ["(V1, ...)", "u1"],
        "input1": ["(V1, ...)", "u1"],
        "input2": ["(V1, ...)", "u1"],
        "output1": ["(V1, ...)", "u1"],
        "output2": ["(V1, ...)", "u1"],
        "output3": ["(V1, ...)"],
    }

    assert_shapes(model, ref_shapes)


def test_same_uniadic_1() -> None:
    class MyModel(PrimitiveModel):
        input: Connection
        output: Connection

        def __init__(self) -> None:
            super().__init__(
                formula_key="buffer",
                input=BaseKey(
                    shape=[("Var1", ...), "u1", "u2", "u3"],
                    type=Tensor,
                ),
                output=BaseKey(shape=[("Var1", ...), "u4"], type=Tensor),
            )

        def connect(  # type: ignore[override]
            self, input: ConnectionType = NOT_GIVEN, output: ConnectionType = NOT_GIVEN
        ):
            return ExtendInfo(self, {"input": input, "output": output})

    model = MyModel()
    shape_1: Mapping[str, Sequence[int | tuple[str, EllipsisType]]] = {
        "input": [("V1", ...), 1, 1, 1]
    }
    shape_2: Mapping[str, Sequence[int | tuple[str, EllipsisType]]] = {
        "output": [("V1", ...), 1]
    }

    model.set_shapes(**shape_1)
    model.set_shapes(**shape_2)
    in_data = model.input.metadata
    assert in_data.is_tensor
    assert (node := in_data.shape) is not None
    input_repr = next(iter(node.reprs))

    out_data = model.output.metadata
    assert out_data.is_tensor
    assert out_data.shape is not None
    output_repr = next(iter(out_data.shape.reprs))

    assert input_repr[-3] is input_repr[-2] is input_repr[-1] is output_repr[-1]


def test_same_uniadic_2() -> None:
    class MyModel(PrimitiveModel):
        input: Connection
        output: Connection

        def __init__(self) -> None:
            super().__init__(
                formula_key="buffer",
                input=BaseKey(
                    shape=[("Var1", ...), "u1", "u2", "u3"],
                    type=Tensor,
                ),
                output=BaseKey(shape=[("Var1", ...), "u4"], type=Tensor),
            )

        def connect(  # type: ignore[override]
            self, input: ConnectionType = NOT_GIVEN, output: ConnectionType = NOT_GIVEN
        ):
            return ExtendInfo(self, {"input": input, "output": output})

    model = MyModel()

    shape_1: Mapping[str, Sequence[int | tuple[str, EllipsisType]]] = {
        "input": [("V1", ...), 1, 1, 1]
    }
    shape_2: Mapping[str, Sequence[int | tuple[str, EllipsisType]]] = {
        "output": [("V1", ...), 1]
    }

    model.set_shapes(**shape_1)
    model.set_shapes(**shape_2)

    assert model.input.metadata.is_tensor
    assert (in_node := model.input.metadata.shape) is not None
    input_repr = next(iter(in_node.reprs))
    assert model.output.metadata.is_tensor
    assert (out_node := model.output.metadata.shape) is not None
    output_repr = next(iter(out_node.reprs))

    model.set_shapes(input=[2, 2, 1, 1, 1, 1])
    assert input_repr[1] is input_repr[0] is output_repr[0] is output_repr[1]
    assert (
        input_repr[-4]
        is input_repr[-3]
        is input_repr[-2]
        is input_repr[-1]
        is output_repr[-1]
        is output_repr[-2]
    )


def test_same_uniadic_3():
    model = Model()

    buffer1 = Buffer()
    buffer2 = Buffer()
    buffer3 = Buffer()

    model |= buffer1.connect(input="input1", output="output1")
    model |= buffer2.connect(input="input2", output="output2")
    model |= buffer3.connect(input="input3", output="output3")
    shape_1: dict[str, list] = {"input": [1, 1, ("V1", ...)]}
    shape_2: dict[str, list] = {"input": [1, ("V1", ...), 2]}
    shape_3: dict[str, list] = {"input": [1, ("V1", ...), 1]}
    buffer1.set_shapes(**shape_1)
    buffer2.set_shapes(**shape_2)
    buffer3.set_shapes(**shape_3)

    input1_repr = next(iter(model.input1.metadata.shape.reprs))  # type: ignore
    input2_repr = next(iter(model.input2.metadata.shape.reprs))  # type: ignore
    input3_repr = next(iter(model.input3.metadata.shape.reprs))  # type: ignore

    assert (
        input1_repr[0]
        is input1_repr[1]
        is input2_repr[0]
        is input3_repr[0]
        is input3_repr[-1]
    )


def test_same_uniadic_4():
    model = Model()

    buffer1 = Buffer()
    buffer2 = Buffer()
    buffer3 = Buffer()

    model |= buffer1.connect(input="input1", output="output1")
    model |= buffer2.connect(input="input2", output="output2")
    model |= buffer3.connect(input="input3", output="output3")

    shape_1: dict[str, list] = {"input": [1, 1, ("V1", ...)]}
    shape_2: dict[str, list] = {"input": [1, ("V1", ...), 2]}
    shape_3: dict[str, list] = {"input3": [1, ("V1", ...), 1]}

    buffer1.set_shapes(**shape_1)
    buffer2.set_shapes(**shape_2)
    model.set_shapes(**shape_3)

    input1_repr = next(iter(model.input1.metadata.shape.reprs))  # type: ignore
    input2_repr = next(iter(model.input2.metadata.shape.reprs))  # type: ignore
    input3_repr = next(iter(model.input3.metadata.shape.reprs))  # type: ignore

    assert (
        input1_repr[0]
        is input1_repr[1]
        is input2_repr[0]
        is input3_repr[0]
        is input3_repr[-1]
    )


def test_same_uniadic_5():
    buffer = Buffer()
    buffer.set_shapes(input=[("V1", ...), 1])
    buffer.set_shapes(input=[1, ("V1", ...)])

    assert buffer.input.metadata.is_tensor
    assert buffer.input.metadata.shape is not None
    input_reprs = buffer.input.metadata.shape.reprs

    repr1, repr2 = tuple(input_reprs)

    repr_prefix, repr_suffix = (repr1, repr2) if repr1.prefix else (repr2, repr1)

    assert repr_prefix[0] is repr_suffix[-1]


@pytest.mark.skip(
    reason="An output value is set here by another model's input "
    "with extending from input strategy. Is this OK?"
)
def test_scalar_propagation():
    # TODO: When this test was written multi-write error was raised.
    # After final decision about setting output values by input values
    # this test should be re-evaluated.
    model = Model()
    shape_model = Shape()
    reduce_model = Mean(axis=TBD)
    model += reduce_model.connect(
        input="input_reduce", axis=(2, 3), output=IOKey(name="output")
    )

    # TODO: assert test
    with pytest.raises(Exception):  # noqa B017
        model += shape_model.connect(input="shape_input", output=reduce_model.axis)


def test_scalar_propagation_2():
    model = Model()
    size_model = Size(dim=TBD)

    reduce_model = Mean(axis=TBD)
    model += reduce_model.connect(
        input="reduce_input", axis=(1, 2), output=IOKey(name="output")
    )

    # TODO: assert test
    with pytest.raises(Exception):  # noqa B017
        model += size_model.connect(output=reduce_model.axis, dim=reduce_model.axis)


def test_reshape_1():
    model = Reshape(shape=(-1,))
    static_shapes = {"input": [10, 5, 20, 30]}

    logical_ref = {
        "output": ["u1"],
        "input": ["(V1, ...)"],
        "shape": None,
    }
    physical_ref = {"input": [10, 5, 20, 30], "output": [30000], "shape": None}

    assert_shapes(model, logical_ref, physical_ref, shapes=static_shapes)


def test_reshape_2():
    model = Reshape(shape=(1, -1))
    static_shapes = {"input": [10, 5, 20, 30]}

    logical_ref: dict[str, list | None] = {
        "output": [1, "u1"],
        "input": ["(V1, ...)"],
        "shape": None,
    }
    physical_ref = {"input": [10, 5, 20, 30], "output": [1, 30000], "shape": None}

    assert_shapes(model, logical_ref, physical_ref, shapes=static_shapes)


def test_reshape_3():
    model = Reshape(shape=(1, -1, 1, 1))
    static_shapes = {"input": [10, 5, 20, 30]}

    logical_ref: dict[str, list | None] = {
        "output": [1, "u1", 1, 1],
        "input": ["(V1, ...)"],
        "shape": None,
    }
    physical_ref = {"input": [10, 5, 20, 30], "output": [1, 30000, 1, 1], "shape": None}

    assert_shapes(model, logical_ref, physical_ref, shapes=static_shapes)


def test_reshape_4():
    model = Reshape(shape=(1, -1, 1, 1))
    static_shapes = {"input": [10]}

    logical_ref: dict[str, list | None] = {
        "output": [1, "u1", 1, 1],
        "input": ["(V1, ...)"],
        "shape": None,
    }
    physical_ref = {"input": [10], "output": [1, 10, 1, 1], "shape": None}

    assert_shapes(model, logical_ref, physical_ref, shapes=static_shapes)


def test_reshape_5():
    model = Reshape(shape=(1, -1, 30, 1))
    static_shapes = {"input": [10, 5, 20, 30]}

    logical_ref: dict[str, list | None] = {
        "output": [1, "u1", 30, 1],
        "input": ["(V1, ...)"],
        "shape": None,
    }
    physical_ref = {"input": [10, 5, 20, 30], "output": [1, 1000, 30, 1], "shape": None}

    assert_shapes(model, logical_ref, physical_ref, shapes=static_shapes)


def test_cartesian_call():
    # This test shows and proves that some shapes are solvable due to
    # cartesian call of each constraint in the constraint call. In this test, one
    # broadcast algorithm runs with the inputs of left and right with multiple shape
    # reprs. shape node of left is [(3, Var1), (4, Var2)] and shape node of right is
    # [(3, Var3), (4, Var4)]. output is set to be (a, b). In this setting, constraint
    # may be unsolvable if all shapes are not tried. However, when bcast is in
    # cartesian call. All shapes should be found (3, 4)
    sig_1 = Sigmoid()
    sig_2 = Sigmoid()
    shape_1: dict[str, list] = {"input": [3, ("Var1", ...)]}
    shape_2: dict[str, list] = {"input": [("Var2", ...), 4]}
    sig_1.set_shapes(**shape_1)
    sig_2.set_shapes(**shape_2)
    model1 = Model()
    model1 |= sig_1.connect(input="input", output="output1")
    model1 |= sig_2.connect(input="input", output="output2")
    model1.expose_keys("output1", "output2")

    model2 = deepcopy(model1)
    model3 = Model()
    add_model = Add()
    add_model.set_shapes(output=["a", "b"])
    model3 |= model1.connect()
    model3 |= model2.connect()
    model3 |= add_model.connect(
        left=model1.input,  # type: ignore
        right=model2.input,  # type: ignore
        output="output",
    )
    model3.expose_keys("output")

    key_mappings = model3.generate_keys()
    model_1_out1 = key_mappings[
        model3.conns.get_con_by_metadata(model1.output1.metadata).key  # type: ignore
    ]
    model_1_out2 = key_mappings[
        model3.conns.get_con_by_metadata(model1.output2.metadata).key  # type: ignore
    ]
    model_2_out1 = key_mappings[
        model3.conns.get_con_by_metadata(model2.output1.metadata).key  # type: ignore
    ]
    model_2_out2 = key_mappings[
        model3.conns.get_con_by_metadata(model2.output2.metadata).key  # type: ignore
    ]
    model_3_input_1 = key_mappings[
        model3.conns.get_con_by_metadata(model1.input.metadata).key  # type: ignore
    ]
    model_3_input_2 = key_mappings[
        model3.conns.get_con_by_metadata(model2.input.metadata).key  # type: ignore
    ]

    logical_ref = {
        model_1_out1: [3, 4],
        model_1_out2: [3, 4],
        model_2_out1: [3, 4],
        model_2_out2: [3, 4],
        model_3_input_1: [3, 4],
        model_3_input_2: [3, 4],
        "output": [3, 4],
    }
    physical_ref = {
        "input_0": [3, 4],
        "output1_0": [3, 4],
        "input_1": [3, 4],
        "output1_1": [3, 4],
        "output": [3, 4],
    }
    assert_shapes(model3, logical_ref, physical_ref)


def test_cartesian_call_2():
    model = Model()
    mean_model_1 = Mean(axis=TBD)
    mean_model_2 = Mean(axis=TBD)

    # Form two sigmoid model, set their inputs' shapes (V2, c) and (a, b, V1)
    # respectively. connect these inputs. This is made for forming two reprs at
    # the same node
    sig_1 = Sigmoid()
    sig_2 = Sigmoid()
    shape_1: dict[str, list] = {"input": [("V2", ...), "c"]}
    shape_2: dict[str, list] = {"input": ["a", "b", ("V1", ...)]}
    sig_1.set_shapes(**shape_1)
    sig_2.set_shapes(**shape_2)
    model |= sig_1.connect(input="input", output=IOKey(name="output"))
    model |= sig_2.connect(input="input", output=IOKey(name="output2"))

    # connect two mean models to these inputs. One of them has axis of 0 and other
    # one of them has axis of -1. And let output of these mean models be output3 and
    # output4 respectively. In this setting, expected behavior for shapes is (b, V1) for
    # for output3 and (V2) for output4. This behavior is possible due to cartesian call
    # of each constraint.
    model |= mean_model_1.connect(axis=0, input="input", output=IOKey(name="output3"))
    model |= mean_model_2.connect(axis=-1, input="input", output="output4")

    all_input_shapes = model.input.metadata.shape.reprs  # type: ignore
    assert len(all_input_shapes) == 2

    ref_shapes = {
        "input": [["u1", "(V1, ...)", "u2"], ["u1", "u3", "(V2, ...)"]],
        "$axis_0": None,
        "$_Mean_2_keepdim": None,
        "$axis_1": None,
        "$_Mean_3_keepdim": None,
        "output": [["u1", "(V1, ...)", "u2"], ["u1", "u3", "(V2, ...)"]],
        "output2": [["u1", "(V1, ...)", "u2"], ["u1", "u3", "(V2, ...)"]],
        "output3": ["(V1, ...)", "u2"],
        "output4": ["u1", "(V1, ...)"],
    }
    assert_shapes(model, ref_shapes)


def test_check_shapes_semantically():
    shape_1 = {"input": ["u1", "u2"]}
    shape_2 = {"input": ["u3", "u4"]}
    check_shapes_semantically(shape_1, shape_2)


def test_check_shapes_semantically_2():
    shape_1 = {"input": ["u1", "u1"]}
    shape_2 = {"input": ["u3", "u4"]}
    with pytest.raises(AssertionError):
        check_shapes_semantically(shape_1, shape_2)


def test_check_shapes_semantically_3():
    shape_1 = {"input": ["u1", "u2", "u3"]}
    shape_2 = {"input": ["u4", "u5", "u6"]}
    check_shapes_semantically(shape_1, shape_2)


def test_check_shapes_semantically_4():
    shape_1 = {"input": ["u1", "u2", "u3"]}
    shape_2 = {"input": ["u4", "u3", "u2"]}
    check_shapes_semantically(shape_1, shape_2)


def test_check_shapes_semantically_5():
    shape_1 = {"input": ["u7", "u7", "u7"]}
    shape_2 = {"input": ["u2", "u2", "u2"]}
    check_shapes_semantically(shape_1, shape_2)


def test_check_shapes_semantically_6():
    shape_1 = {"input": ["u1", "u2", "u1"]}
    shape_2 = {"input": ["u2", "u1", "u2"]}
    check_shapes_semantically(shape_1, shape_2)


def test_check_shapes_semantically_7():
    shape_1: dict[str, list] = {"input": ["u1", "u2", "u1"], "input2": ["u2", "u3", 3]}

    shape_2: dict[str, list] = {"input": ["u2", "u1", "u2"], "input2": ["u1", "u4", 3]}

    check_shapes_semantically(shape_1, shape_2)


def test_check_shapes_semantically_8():
    shape_1: dict[str, list] = {
        "w1": [2, "u1"],
        "w2": ["u1", "u2"],
        "w3": ["u2", "u3"],
        "w4": ["u3", "u1"],
        "w5": [4, 5],
        "w6": ["u1", "u2", "u3", "u4", "u1", "u2", "u3", "u4"],
    }
    shape_2: dict[str, list] = {
        "w1": [2, "u6"],
        "w2": ["u6", "u7"],
        "w3": ["u7", "u8"],
        "w4": ["u8", "u6"],
        "w5": [4, 5],
        "w6": ["u6", "u7", "u8", "u10", "u6", "u7", "u8", "u10"],
    }

    check_shapes_semantically(shape_1, shape_2)


def test_check_shapes_semantically_9():
    shape_1: dict[str, list] = {
        "w1": [2, "u1"],
        "w2": ["u1", "u2"],
        "w3": ["u2", "u3"],
        "w4": ["u3", "u1"],
        "w5": [4, 5],
        "w6": ["u1", "u2", "u3", "u4", "u1", "u2", "u3", "u4"],
    }
    shape_2: dict[str, list] = {
        "w1": [2, "u6"],
        "w2": ["u6", "u7"],
        "w3": ["u7", "u8"],
        "w4": ["u8", "u6"],
        "w5": [4, 5],
        "w7": ["u6", "u7", "u8", "u10", "u6", "u7", "u8", "u10"],
    }
    with pytest.raises(AssertionError):
        check_shapes_semantically(shape_1, shape_2)


def test_check_shapes_semantically_10():
    shape_1: dict[str, list] = {
        "w1": [2, "u1"],
        "w2": ["u1", "u2", "u2"],
        "w3": ["u2", "u3"],
        "w4": ["u3", "u1"],
        "w5": [4, 5],
        "w6": ["u1", "u2", "u3", "u4", "u1", "u2", "u3", "u4"],
    }
    shape_2: dict[str, list] = {
        "w1": [2, "u6"],
        "w2": ["u6", "u7"],
        "w3": ["u7", "u8"],
        "w4": ["u8", "u6"],
        "w5": [4, 5],
        "w7": ["u6", "u7", "u8", "u10", "u6", "u7", "u8", "u10"],
    }
    with pytest.raises(AssertionError):
        check_shapes_semantically(shape_1, shape_2)


def test_check_shapes_semantically_11():
    shape_1 = {"w1": [["u1", "u2"], ["u3", "u4"]]}
    shape_2 = {"w1": [["u1", "u2"], ["u3", "u4"]]}
    check_shapes_semantically(shape_1, shape_2)


def test_check_shapes_semantically_12():
    shape_1 = {"w1": [["u1", "u2"], ["u3", "u4"]]}
    shape_2 = {"w1": [["u5", "u6"], ["u7", "u8"]]}
    check_shapes_semantically(shape_1, shape_2)


def test_check_shapes_semantically_13():
    shape_1 = {"w1": [["u1", "u2"], ["u1", "u4"]]}
    shape_2 = {"w1": [["u5", "u6"], ["u5", "u8"]]}
    check_shapes_semantically(shape_1, shape_2)


def test_check_shapes_semantically_14():
    shape_1 = {
        "w1": [["u1", "u2"], ["u1", "u4"]],
        "w2": [["u1", "u1", "u1"], ["u2", "u2", "u2"]],
    }
    shape_2 = {
        "w1": [["u5", "u6"], ["u5", "u8"]],
        "w2": [["u5", "u5", "u5"], ["u6", "u6", "u6"]],
    }
    check_shapes_semantically(shape_1, shape_2)


def test_check_shapes_semantically_15():
    shape_1 = {
        "w1": [["u1", "u2"], ["u1", "u4"]],
        "w2": [["u1", "u1", "u1"], ["u2", "u2", "u2"]],
    }
    shape_2 = {
        "w1": [["u5", "u6"], ["u5", "u8"]],
        "w2": [["u5", "u5", "u5"], ["u6", "u6", "u6", "u7"]],
    }
    with pytest.raises(AssertionError):
        check_shapes_semantically(shape_1, shape_2)


def test_check_shapes_semantically_16():
    shape_1: dict[str, list] = {
        "w1": [["u1", "u2"], ["u1", "u4"]],
        "w2": [["u1", "u1", "u1"], ["u2", "u2", "u2"]],
    }
    shape_2: dict[str, list] = {
        "w1": [["u5", "u6"], ["u5", "u8"]],
        "w2": [["u5", "u5", "u5"], ["u8", "u8", "u8"]],
    }
    with pytest.raises(AssertionError):
        check_shapes_semantically(shape_1, shape_2)


def test_check_shapes_semantically_17():
    shape_1: dict[str, list] = {"w1": [["u1", "u2"], ["u1", "u4"]], "w2": ["u1"]}
    shape_2: dict[str, list] = {"w1": [["u5", "u6"], ["u5", "u8"]], "w2": ["u5"]}
    check_shapes_semantically(shape_1, shape_2)


def test_check_shapes_semantically_18():
    shape_1: dict[str, list] = {"w1": [["u1", "u2"], ["u1", "u4"]], "w2": ["u1"]}
    shape_2: dict[str, list] = {"w1": [["u5", "u6"], ["u5", "u8"]], "w2": ["u9"]}
    with pytest.raises(AssertionError):
        check_shapes_semantically(shape_1, shape_2)


def test_check_shapes_semantically_19():
    shape_1: dict[str, list] = {
        "w1": [["u1", "u2"], ["u1", "u4"], ["u1", "u2"]],
        "w2": ["u1"],
    }
    shape_2: dict[str, list] = {"w1": [["u5", "u6"], ["u5", "u8"]], "w2": ["u9"]}
    with pytest.raises(AssertionError):
        check_shapes_semantically(shape_1, shape_2)


def test_check_shapes_semantically_20():
    shape_1: dict[str, list] = {
        "w1": [["u1", "u2"], ["u1", "u4"], ["u1", "u2"]],
        "w2": ["u1"],
    }
    shape_2: dict[str, list] = {
        "w1": [["u5", "u6"], ["u5", "u8"], ["sfdsad"]],
        "w2": ["u9"],
    }
    with pytest.raises(AssertionError):
        check_shapes_semantically(shape_1, shape_2)


def test_shaperepr_contains_1():
    root = Variadic()
    uni_1, uni_2 = Uniadic(), Uniadic()
    repr_1 = ShapeRepr(prefix=[uni_1], root=root, suffix=[uni_2])
    repr_2 = ShapeRepr(root=root)
    assert repr_2 in repr_1


def test_shaperepr_contains_2():
    root = Variadic()
    uni_1, uni_2, uni_3, uni_4 = Uniadic(), Uniadic(), Uniadic(), Uniadic()
    repr_1 = ShapeRepr(prefix=[uni_1, uni_2], root=root, suffix=[uni_3, uni_4])
    repr_2 = ShapeRepr(prefix=[uni_2], root=root, suffix=[uni_3])
    assert repr_2 in repr_1


def test_shaperepr_contains_3():
    root = Variadic()
    uni_1, uni_2, uni_3, uni_4 = Uniadic(), Uniadic(), Uniadic(), Uniadic()
    repr_1 = ShapeRepr(prefix=[uni_1, uni_2, uni_3, uni_4], root=root)
    repr_2 = ShapeRepr(prefix=[uni_2, uni_3, uni_4], root=root)
    assert repr_2 in repr_1


def test_shaperepr_contains_4():
    root = Variadic()
    uni_1, uni_2, uni_3, uni_4 = Uniadic(), Uniadic(), Uniadic(), Uniadic()
    repr_1 = ShapeRepr(prefix=[uni_1, uni_2, uni_3, uni_4], root=root, suffix=[uni_1])
    repr_2 = ShapeRepr(prefix=[uni_2, uni_3, uni_4], root=root)
    assert repr_2 in repr_1


def test_shaperepr_contains_5():
    root = Variadic()
    repr_1 = ShapeRepr(root=root)
    repr_2 = ShapeRepr(root=root)
    assert repr_2 in repr_1


def test_shaperepr_contains_6():
    root = Variadic()
    repr_1 = ShapeRepr(root=root)
    repr_2 = ShapeRepr(root=root, suffix=[Uniadic()])
    assert repr_2 not in repr_1


def test_shaperepr_contains_7():
    uni_1, uni_2, uni_3, uni_4, uni_5 = (
        Uniadic(),
        Uniadic(),
        Uniadic(),
        Uniadic(),
        Uniadic(),
    )
    repr_1 = ShapeRepr(prefix=[uni_1, uni_2, uni_3, uni_4, uni_5])
    repr_2 = ShapeRepr(prefix=[uni_1, uni_2, uni_3])
    assert repr_2 in repr_1


def test_shaperepr_contains_8():
    uni_1, uni_2, uni_3, uni_4, uni_5 = (
        Uniadic(),
        Uniadic(),
        Uniadic(),
        Uniadic(),
        Uniadic(),
    )
    repr_1 = ShapeRepr(prefix=[uni_1, uni_2, uni_3, uni_4, uni_5])
    repr_2 = ShapeRepr(prefix=[uni_2, uni_3, uni_4])
    assert repr_2 in repr_1


def test_shaperepr_contains_9():
    uni_1, uni_2, uni_3, uni_4, uni_5 = (
        Uniadic(),
        Uniadic(),
        Uniadic(),
        Uniadic(),
        Uniadic(),
    )
    repr_1 = ShapeRepr(prefix=[uni_1, uni_2, uni_3, uni_4, uni_5])
    repr_2 = ShapeRepr(prefix=[uni_3, uni_4, uni_5])
    assert repr_2 in repr_1


def test_shaperepr_contains_10():
    uni_1, uni_2, uni_3, uni_4, uni_5 = (
        Uniadic(),
        Uniadic(),
        Uniadic(),
        Uniadic(),
        Uniadic(),
    )
    repr_1 = ShapeRepr(prefix=[uni_1, uni_2, uni_3, uni_4, uni_5])
    repr_2 = ShapeRepr(prefix=[uni_2, uni_3, uni_4, uni_5])
    assert repr_2 in repr_1


def test_shaperepr_contains_11():
    uni_1, uni_2, uni_3, uni_4, uni_5 = (
        Uniadic(),
        Uniadic(),
        Uniadic(),
        Uniadic(),
        Uniadic(),
    )
    repr_1 = ShapeRepr(prefix=[uni_1, uni_2, uni_3, uni_4, uni_5])
    repr_2 = ShapeRepr(prefix=[uni_1, uni_2, uni_3, uni_4, uni_5])
    assert repr_2 in repr_1
    assert repr_1 in repr_2


def test_shaperepr_contains_12():
    uni_1, uni_2, uni_3, uni_4, uni_5 = (
        Uniadic(),
        Uniadic(),
        Uniadic(),
        Uniadic(),
        Uniadic(),
    )
    repr_1 = ShapeRepr(prefix=[uni_1, uni_2, uni_3, uni_4, uni_5])
    repr_2 = ShapeRepr(prefix=None)
    assert repr_2 in repr_1


def test_shaperepr_contains_13():
    repr_1 = ShapeRepr(prefix=None)
    repr_2 = ShapeRepr(prefix=None)
    assert repr_2 in repr_1
    assert repr_1 in repr_2


def test_shaperepr_contains_14():
    uni_1, uni_2 = Uniadic(), Uniadic()
    repr_1 = ShapeRepr(prefix=[uni_1, uni_2], root=Variadic())
    repr_2 = ShapeRepr(prefix=[uni_1])
    assert repr_2 in repr_1


def test_shaperepr_contains_15():
    repr_1 = ShapeRepr(root=Variadic())
    repr_2 = ShapeRepr(prefix=None)
    assert repr_2 in repr_1
    assert repr_1 not in repr_2


def test_shaperepr_contains_16():
    uni_1, uni_2 = Uniadic(), Uniadic()
    repr_1 = ShapeRepr(prefix=[uni_1], root=Variadic())
    repr_2 = ShapeRepr(prefix=[uni_2])
    assert repr_2 not in repr_1


def test_prune_match_1():
    """Prune model which input and output exposed"""
    model = Model()
    model |= Squeeze().connect(input="input", output=IOKey(name="out1"))
    model |= Squeeze().connect(input="input", output=IOKey(name="out2"))
    shape_1: dict[str, list] = {"out1": [3, 2, ("V1", ...)]}
    model.set_shapes(**shape_1)

    shape: dict[str, list] = {
        "input": ["(V1, ...)"],
        "out1": [3, 2, "(V2, ...)"],
        "out2": ["(V3, ...)"],
    }
    p_shape: dict[str, list] = {
        "input": ["..."],
        "out1": [3, 2, "..."],
    }

    assert_shapes(model, shape, physical_ref=p_shape)


def test_prune_match_2():
    """Prune model which input exposed but output not"""
    model = Model()
    s1, s2 = Squeeze(), Squeeze()
    shape_1: dict[str, list] = {"output": [3, 2, ("V1", ...)]}
    s1.set_shapes(**shape_1)

    model |= s1.connect(input="input")
    model |= s2.connect(input="input")
    model |= Gelu().connect(input=s1.output, output=IOKey(name="out1"))
    model |= Relu().connect(input=s2.output, output=IOKey(name="out2"))

    shape: dict[str, list | None] = {
        "input": ["(V1, ...)"],
        "$_Gelu_2_approximate": None,
        "$_Squeeze_0_output": [3, 2, "(V2, ...)"],
        "$_Squeeze_1_output": ["(V3, ...)"],
        "out1": [3, 2, "(V2, ...)"],
        "out2": ["(V3, ...)"],
    }

    p_shape: dict[str, list | None] = {
        "input": ["..."],
        "output_0": [3, 2, "..."],
        "approximate": None,
        "out1": [3, 2, "..."],
        "out2": [3, 2, "..."],
    }

    assert_shapes(model, shape, physical_ref=p_shape)


def test_prune_match_3():
    """Prune model which are not in the same canvas"""
    model = Model()
    s1, s2 = Squeeze(), Squeeze()
    shape_1: dict[str, list] = {"output": [3, 2, ("V1", ...)]}
    s1.set_shapes(**shape_1)

    model_sub = Model()
    model_sub |= s1.connect(input="input")
    model_sub |= Gelu().connect(input=s1.output, output=IOKey(name="out1"))

    model |= model_sub.connect(input="input", out1=IOKey(name="out1"))

    model |= s2.connect(input="input")
    model |= Relu().connect(input=s2.output, output=IOKey(name="out2"))

    shape: dict[str, list] = {
        "input": ["(V1, ...)"],
        "$_Squeeze_1_output": ["(V3, ...)"],
        "out1": [3, 2, "(V2, ...)"],
        "out2": ["(V3, ...)"],
    }

    p_shape: dict[str, list | None] = {
        "input": ["..."],
        "approximate": None,
        "output_0": [3, 2, "..."],
        "out1": [3, 2, "..."],
        "out2": [3, 2, "..."],
    }

    assert_shapes(model, shape, physical_ref=p_shape)


def test_prune_match_4():
    """Reversed case 3"""
    model = Model()
    s1, s2 = Squeeze(), Squeeze()
    shape_1: dict[str, list] = {"output": [3, 2, ("V1", ...)]}
    s1.set_shapes(**shape_1)
    model |= s2.connect(input="input")
    model |= Relu().connect(input=s2.output, output=IOKey(name="out2"))

    model_sub = Model()
    model_sub |= s1.connect(input="input")
    model_sub |= Gelu().connect(input=s1.output, output=IOKey(name="out1"))

    model |= model_sub.connect(input="input", out1=IOKey(name="out1"))

    shape: dict[str, list] = {
        "input": ["(V1, ...)"],
        "$_Squeeze_0_output": ["(V3, ...)"],
        "out1": [3, 2, "(V2, ...)"],
        "out2": ["(V3, ...)"],
    }

    p_shape: dict[str, list | None] = {
        "input": ["..."],
        "approximate": None,
        "output_0": [3, 2, "..."],
        "out2": [3, 2, "..."],
        "out1": [3, 2, "..."],
    }

    assert_shapes(model, shape, p_shape)


def test_prune_match_5():
    """Sequential prune shape test"""
    model = Model()
    s1, s2 = Squeeze(), Squeeze()
    shape_1: dict[str, list] = {"output": [3, 2, ("V1", ...)]}
    s1.set_shapes(**shape_1)

    model_sub = Model()
    model_sub |= s1.connect(input="input")
    model_sub |= Squeeze().connect(input=s1.output)
    model_sub += Squeeze()
    model_sub |= Gelu().connect(input=model_sub.cout, output="out1")
    model_sub.expose_keys("out1")

    model |= model_sub.connect(input="input", out1="out1")
    model |= s2.connect(input="input")
    model.set_cout(s2.output)
    model += Squeeze()
    model += Squeeze()
    model |= Relu().connect(input=model.cout, output="out2")
    model.expose_keys("out1", "out2")

    shape: dict[str, list] = {
        "input": ["(V1, ...)"],
        "$_Squeeze_1_output": ["(V2, ...)"],
        "$_Squeeze_2_output": ["(V3, ...)"],
        "$_Squeeze_3_output": ["(V4, ...)"],
        "out1": [3, 2, "(V5, ...)"],
        "out2": ["(V4, ...)"],
    }

    p_shape: dict[str, list | None] = {
        "input": ["..."],
        "approximate": None,
        "output_0": [3, 2, "..."],
        "output_1": [3, 2, "..."],
        "output_2": [3, 2, "..."],
        "out1": [3, 2, "..."],
        "out2": [3, 2, "..."],
    }

    assert_shapes(model, shape, physical_ref=p_shape)


def test_total_repr_count_1():
    # This test tests if shape algorithm creates only required amount of
    # ShapeRepr objects in primitive models. Ideally, when all primitive models are
    # initialized, # of shapereprs in this initialized primitive model should be equal
    # to # of unique shaped tensor inputs & outputs of the primitive model.
    import inspect

    def get_defaults(func):
        # Get the signature of the function
        signature = inspect.signature(func)
        # Extract parameters with default values
        return {
            key: value.default
            for key, value in signature.parameters.items()
            if value.default is not inspect.Parameter.empty
        }

    # def get_defaults(fn):
    #     if fn.__defaults__ is None:
    #         return {}
    #     return dict(
    #         zip(
    #             fn.__code__.co_varnames[-len(fn.__defaults__) :],
    #             fn.__defaults__,
    #             strict=False,
    #         )
    #     )

    def find_all_reprs(repr: ShapeRepr, repr_cache=None) -> set[ShapeRepr]:
        # this function find all ShapeRepr objects which is referenced by
        # the given ShapeRepr object
        if repr_cache is None:
            repr_cache = set()
        repr_cache.add(repr)
        all_symbols: set[Uniadic | Variadic] = {*repr.prefix, *repr.suffix}
        if repr.root is not None:
            all_symbols.add(repr.root)
        for symbol in all_symbols:
            for symbol_repr in symbol.reprs:
                if symbol_repr not in repr_cache:
                    find_all_reprs(symbol_repr, repr_cache)
        return repr_cache

    # identify primitives that will be skipped for different reasons
    skipped_primitves = {
        Concat,
        PrimitiveUnion,
        Convolution2D,
        Convolution1D,
        MaxPool1D,
        MaxPool2D,
        BinaryCrossEntropy,
        Activation,
        CrossEntropy,
        CustomPrimitiveModel,
        ToTuple,
        Operator,
        ToList,
        ScaledDotProduct,
        PrimitiveModel,
        OperatorModel,
    }
    ref_counts = {
        Exponential: 1,
        Sqrt: 1,
        Softplus: 1,
        Square: 1,
        Softmax: 1,
        Sign: 1,
        Pad: 2,
        PermuteTensor: 2,
        Sigmoid: 1,
        PositionalEncoding: 1,
        Cosine: 1,
        Gelu: 1,
        Sine: 1,
        NanToNum: 1,
        Relu: 1,
        StableReciprocal: 2,
        LeakyRelu: 1,
        IsNan: 1,
        Log: 1,
        Tanh: 1,
        GPRAlpha: 2,
        Cholesky: 1,
        Eigvalsh: 3,
        TsnePJoint: 2,
        GPRVOuter: 1,
        AUC: 1,
        Accuracy: 2,
        Buffer: 1,
        Negate: 1,
        LogicalNot: 1,
        Absolute: 1,
        NormModifier: 1,
        Cast: 1,
        Unique: 2,
        Trapezoid: 2,
        AUCCore: 2,
        ZerosLike: 1,
        Floor: 1,
    }
    # find all primitives that are defined in primitives.py

    u_primitives = mithril.framework.logical.primitive
    _all_primitives_dict = primitives.__dict__ | u_primitives.__dict__
    all_primitives = primitives.__all__ + u_primitives.__all__  # type: ignore
    all_primitives_dict = {
        value for key, value in _all_primitives_dict.items() if key in all_primitives
    }

    for primitive_model in all_primitives_dict:
        if primitive_model not in skipped_primitves:
            # for primitive models with scalar inputs, find their arguments.
            # give these arguments to function with ellipsis
            init_code = primitive_model.__init__.__code__
            model_init_params = set(init_code.co_varnames[: init_code.co_argcount][1:])
            default_args = get_defaults(primitive_model.__init__)
            kwargs = {
                param: default_args.get(param, TBD) for param in model_init_params
            }
            # kwargs = {param: ... for param in model_init_params}
            model: Model = primitive_model(**kwargs)
            # Set all untyped connections to Tensor type.
            model.set_types(
                {
                    conn: Tensor
                    for conn in model.conns.input_connections
                    if conn.metadata.edge_type is ToBeDetermined
                }
            )
            reprs = set()

            # find connections only with tensor data
            all_tensor_conns = {
                con for con in model.conns.all.values() if con.metadata.is_tensor
            }

            # Find all reprs that are linked to shape reprs of the tensors
            for con in all_tensor_conns:
                assert con.metadata.shape is not None
                shapes = con.metadata.shape.reprs
                for shape in shapes:
                    reprs |= find_all_reprs(shape)
            if (ref_result := ref_counts.get(model.__class__)) is not None:
                if len(reprs) != ref_result:
                    ...
                    model = primitive_model(**kwargs)
                assert len(reprs) == ref_result
            else:
                if len(reprs) != len(all_tensor_conns):
                    ...
                assert len(reprs) == len(all_tensor_conns)


@pytest.mark.skip("Creating high layer cascaded models are too slow!")
def test_set_shapes_1():
    model = Linear()
    model.set_shapes(input=[10, 10])


@pytest.mark.skip("Creating high layer cascaded models are too slow!")
def test_high_layer_cascaded_models_1():
    depth = 400
    MLP(
        activations=[Relu() for _ in range(depth)],
        dimensions=[idx + 1 for idx in range(depth)],
    )


@pytest.mark.skip("Creating high layer cascaded models are too slow!")
def test_high_layer_cascaded_models_2():
    model = Model()
    for _ in range(400):
        model += MLP(activations=[Relu()], dimensions=[None])


@pytest.mark.skip("Creating high layer cascaded models are too slow!")
def test_high_layer_cascaded_models_3():
    model = Model()
    sig_model = Sigmoid()
    shape_1: dict[str, list] = {"input": ["u1", "u2", ("Var1", ...)]}
    shape_2: dict[str, list] = {"input": ["u1", ("Var1", ...), "u2"]}
    shape_3: dict[str, list] = {"input": [("Var1", ...), "u1", "u2"]}
    sig_model.set_shapes(**shape_1)
    sig_model.set_shapes(**shape_2)
    sig_model.set_shapes(**shape_3)
    for _ in range(400):
        model += deepcopy(sig_model)


@pytest.mark.skip("Creating high layer cascaded models are too slow!")
def test_lstm_shape():
    shapes = {
        "input0": [10, 1, 2],
        "input1": [5, 1, 2],
        "initial_hidden": [10, 1, 2],
        "w_f": [2, 4],
        "w_i": [2, 4],
        "w_c": [2, 4],
        "w_o": [2, 4],
        "bias_f": [2],
        "bias_c": [2],
        "bias_i": [2],
        "bias_o": [2],
        "w_out": [3, 2],
    }

    from mithril.models import LSTMCell, ManyToOne

    for _ in range(100):
        a = ManyToOne(cell_type=LSTMCell(), max_sequence_length=2)
        cm = mithril.compile(
            a,
            shapes=shapes,
            backend=TorchBackend(),
            data_keys={"input0", "input1"},
            jit=False,
        )
        assert cm.shapes["hidden_compl_1"] == [5, 1, 2]


# @pytest.mark.skip("Call number is as high as 322396.")
def test_c_profile_mlp():
    import cProfile
    from pstats import Stats

    depth = 10
    # Profile the example_function
    pr = cProfile.Profile()
    pr.enable()
    MLP(
        activations=[Relu() for _ in range(depth)],
        dimensions=[idx for idx in range(depth)],
    )
    pr.disable()
    # Use pstats to process the profile results
    ps = Stats(pr)
    my_func = "bcast"
    # Extract data for the specific function
    for func_desc, func_data in ps.stats.items():  # type: ignore
        if my_func == func_desc[2]:  # Check if the function name matches
            calls, total_time, cumulative_time, _, _ = func_data
            assert calls <= 3604

    # class MyShapeRepr:
    #     pass

    # class MyShapeRepr2:
    #     pass

    # my_dict = {MyShapeRepr2(): idx for idx in range(100)}
    # my_set = {MyShapeRepr2() for idx in range(100)}
    # my_set2 = {MyShapeRepr2() for idx in range(100)}

    # def foo():
    #     for _ in range(1000000):
    #         a = my_set | my_dict.keys()

    # def bar():
    #     for _ in range(1000000):
    #         a = my_set | my_set2

    # cProfile.runctx('foo()', None, locals(), sort="cumtime")
    # cProfile.runctx('bar()', None, locals(), sort="cumtime")


def test_node_count_1():
    model = Model()
    for _ in range(100):
        model += (sub_model := MyVariadic4())

    # Check total existing node count
    all_nodes = set()
    for con in model.conns.all.values():
        assert con.metadata.is_tensor
        all_nodes.add(con.metadata.shape)
    assert len(all_nodes) == 1

    # Check total variadics repr count
    assert sub_model.input.metadata.is_tensor
    assert (in_node := sub_model.input.metadata.shape) is not None
    assert (in_repr := next(iter(in_node.reprs))) is not None
    assert in_repr.root is not None
    assert len(in_repr.root.reprs) == 1

    assert sub_model.output.metadata.is_tensor
    assert (out_node := sub_model.output.metadata.shape) is not None
    assert (out_repr := next(iter(out_node.reprs))) is not None
    assert out_repr.root is not None
    assert len(out_repr.root.reprs) == 1


def test_node_count_2():
    model = Model()
    model += MyVariadic4().connect(input="in1", output="out1")
    model += MyVariadic4()
    model += MyVariadic4()
    model += MyVariadic4()
    model += MyVariadic4()
    model += MyVariadic11()

    # Check total existing node count
    all_nodes = set()
    for con in model.conns.all.values():
        edge = con.metadata
        assert edge.is_tensor
        all_nodes.add(edge.shape)

    assert len(all_nodes) == 2

    # Check total variadics repr count
    assert len(next(iter(model.in1.metadata.shape.reprs)).root.reprs) == 2  # type: ignore


def test_node_count_3():
    model = Model()
    model |= MyVariadic4().connect(input="in1", output="out1")
    model += MyVariadic4()
    model += MyVariadic4()
    model += MyVariadic4()
    model += MyVariadic4()
    model += MyVariadic11()
    model |= MyVariadic4().connect(input="in2")

    shapes = set()
    for con in model.conns.all.values():
        edge = con.metadata
        assert edge.is_tensor
        shapes.add(edge.shape)

    assert len(shapes) == 3
    assert len(next(iter(model.in1.metadata.shape.reprs)).root.reprs) == 2  # type: ignore
    assert len(next(iter(model.in2.metadata.shape.reprs)).root.reprs) == 1  # type: ignore


def test_repr_count_1():
    class MyModel(PrimitiveModel):
        def __init__(self) -> None:
            super().__init__(
                formula_key="buffer",
                input=BaseKey(shape=["a", "b"], type=Tensor),
                output=BaseKey(shape=["b", "c", "d"], type=Tensor),
            )

    model = MyModel()
    model.set_shapes(input=[1, 1], output=[1, 1, 1])
    # Check total uniadic metadata
    uni_metadata = set()
    for symbol in get_all_symbols(model):
        if isinstance(symbol, Uniadic):
            uni_metadata.add(symbol.metadata)
    assert len(uni_metadata) == 1


def test_equalize_lengths_of_unmatchable_reprs_of_different_sizes_with_extend_1():
    class MyModel(PrimitiveModel):
        def __init__(self) -> None:
            super().__init__(
                formula_key="buffer",
                input=BaseKey(shape=["a", ("Var1", ...)], type=Tensor),
                output=BaseKey(
                    shape=[("Var1", ...), "c", "d", "e"],
                    type=Tensor,
                ),
            )

    model = Model()
    model += MyModel()
    model += MyModel()  # match [("Var1", ...), "c", "d", "e"] with ["k", ("Var2", ...)]
    ref_shapes: dict[str, list] = {
        "$_MyModel_0_output": [
            ["(V1, ...)", "u1", "u2", "u3"],
            ["u4", "(V2, ...)", "u2", "u3"],
        ],
        "$_MyModel_1_output": ["(V2, ...)", "u2", "u3", "u5", "u6", "u7"],
        "$input": ["u8", "(V1, ...)"],
    }
    assert_shapes(model, ref_shapes)


def test_equalize_lengths_of_unmatchable_reprs_of_different_sizes_with_extend_2():
    class MyModel(PrimitiveModel):
        def __init__(self) -> None:
            super().__init__(
                formula_key="buffer",
                input=BaseKey(
                    shape=[("Var1", ...), "c", "d", "e"],
                    type=Tensor,
                ),
                output=BaseKey(shape=["a", "b", ("Var1", ...)], type=Tensor),
            )

    model = Model()
    model += MyModel()
    model += (
        MyModel()
    )  # match ["k", "l", ("Var2", ...)] with [("Var1", ...), "c", "d", "e"]
    ref_shapes: dict[str, list] = {
        "$_MyModel_0_output": [
            ["u1", "u2", "(V1, ...)", "u3"],
            ["(V2, ...)", "u4", "u5", "u3"],
        ],
        "$_MyModel_1_output": ["u6", "u7", "(V2, ...)"],
        "$input": ["(V1, ...)", "u3", "u8", "u9", "u10"],
    }
    assert_shapes(model, ref_shapes)


def test_train_model_shapes_1():
    model = Layer(activation=Relu(), dimension=10)
    ctx_1 = TrainModel(model)
    ctx_1.add_loss(
        Buffer(), input="output", reduce_steps=[Sum(axis=0), Mean(axis=0), Sum(axis=0)]
    )
    ctx_1.set_shapes(input=[5, 4, 3])
    ref_shapes = {
        "$_Buffer_1_output": [5, 4, 10],
        "$_Sum_2_output": [4, 10],
        "$_Mean_3_output": [10],
        "$_Sum_4_output": [],
        "input": [5, 4, 3],
        "weight": [10, 3],
        "bias": [10],
        "$_Sum_2_axis": None,
        "$_Sum_2_keepdim": None,
        "$_Mean_3_axis": None,
        "$_Mean_3_keepdim": None,
        "$_Sum_4_axis": None,
        "$_Sum_4_keepdim": None,
        "output": [5, 4, 10],
    }
    assert_shapes(ctx_1, ref_shapes)


def test_equalize_lengths_of_unmatchable_reprs_of_different_sizes_1():
    repr1 = ShapeRepr([Uniadic(), Uniadic()], Variadic(), [Uniadic()])
    repr2 = ShapeRepr(None, Variadic(), [Uniadic(), Uniadic()])
    # repr1 = ["a", "b", ("Var1", ...), "c"]
    # repr2 = [("Var2", ...), "d", "e"]

    repr2.node.merge(repr1.node)
    ref_shape = {
        "my_input": [["u3", "(V1, ...)", "u1", "u2"], ["u3", "u4", "(V2, ...)", "u2"]]
    }
    check_shapes_semantically(
        {"my_input": repr1.node.get_shapes(verbose=True)}, ref_shape
    )


def test_equalize_lengths_of_unmatchable_reprs_of_different_sizes_2():
    repr1 = ShapeRepr([Uniadic(), Uniadic(), Uniadic()], Variadic(), [Uniadic()])
    repr2 = ShapeRepr([Uniadic()], Variadic(), [Uniadic(), Uniadic()])
    # repr1 = ["a", "b", "c", ("Var1", ...), "d"]
    # repr2 = ["e", ("Var2", ...), "f", "g"]

    repr2.node.merge(repr1.node)
    ref_shape = {
        "my_input": [
            ["a", "b", "(Var2, ...)", "f", "d"],
            ["a", "b", "c", "(Var1, ...)", "d"],
        ]
    }
    check_shapes_semantically(
        {"my_input": repr1.node.get_shapes(verbose=True)}, ref_shape
    )


def test_equalize_lengths_of_unmatchable_reprs_of_different_sizes_3():
    repr1 = ShapeRepr(None, Variadic(), [Uniadic(), Uniadic()])
    repr2 = ShapeRepr([Uniadic(), Uniadic()], Variadic(), [Uniadic()])
    # repr1 = [("Var1", ...), "d", "e"]
    # repr2 = ["a", "b", ("Var2", ...), "c"]

    repr2.node.merge(repr1.node)
    ref_shape = {
        "my_input": [["u3", "(V1, ...)", "u1", "u2"], ["u3", "u4", "(V2, ...)", "u2"]]
    }
    check_shapes_semantically(
        {"my_input": repr1.node.get_shapes(verbose=True)}, ref_shape
    )


def test_equalize_lengths_of_unmatchable_reprs_of_different_sizes_4():
    repr1 = ShapeRepr([Uniadic()], Variadic(), [Uniadic(), Uniadic()])
    repr2 = ShapeRepr([Uniadic(), Uniadic(), Uniadic()], Variadic(), [Uniadic()])
    # repr1 = ["e", ("Var1", ...), "f", "g"]
    # repr2 = ["a", "b", "c", ("Var2", ...), "d"]

    repr2.node.merge(repr1.node)
    ref_shape = {
        "my_input": [
            ["a", "b", "(Var2, ...)", "f", "d"],
            ["a", "b", "c", "(Var1, ...)", "d"],
        ]
    }
    check_shapes_semantically(
        {"my_input": repr1.node.get_shapes(verbose=True)}, ref_shape
    )


def test_match_1():
    """
    [a, V1] => [a, V3]
    [V2]       [a, V3]
    """

    repr1 = ShapeRepr([Uniadic()], Variadic(), None)
    repr2 = ShapeRepr(None, Variadic(), None)

    ref_shape_repr1 = ["u1", "(V1, ...)"]
    ref_shape_repr2 = ["u1", "(V1, ...)"]

    assert_match_shapes(repr1, repr2, ref_shape_repr1, ref_shape_repr2)


def test_match_2():
    """
    [a, V1, b] => [a, V3, b]
    [V2]          [a, V3, b]
    """

    repr1 = ShapeRepr([Uniadic()], Variadic(), [Uniadic()])
    repr2 = ShapeRepr(None, Variadic(), None)

    ref_shape_repr1 = ["u1", "(V1, ...)", "u2"]
    ref_shape_repr2 = ["u1", "(V1, ...)", "u2"]

    assert_match_shapes(repr1, repr2, ref_shape_repr1, ref_shape_repr2)


def test_match_3():
    """
    [a, V1] => [a, V1]
    [V2, b]    [V2, b]
    """

    repr1 = ShapeRepr([Uniadic()], Variadic(), None)
    repr2 = ShapeRepr(None, Variadic(), [Uniadic()])

    ref_shape_repr1 = ["u1", "(V1, ...)"]
    ref_shape_repr2 = ["(V2, ...)", "u2"]

    assert_match_shapes(repr1, repr2, ref_shape_repr1, ref_shape_repr2)


def test_match_4():
    """
    [a, b, V1] => [a, b, V1]
    [V2, c, d]    [V2, c, d]
    """

    repr1 = ShapeRepr([Uniadic(), Uniadic()], Variadic(), None)
    repr2 = ShapeRepr(None, Variadic(), [Uniadic(), Uniadic()])

    ref_shape_repr1 = ["u1", "u2", "(V1, ...)"]
    ref_shape_repr2 = ["(V2, ...)", "u3", "u4"]

    assert_match_shapes(repr1, repr2, ref_shape_repr1, ref_shape_repr2)


def test_match_5():
    """
    [a, b, V1] => [a, b, V1]
    [V2, c]       [a, V2, c]
    """

    repr1 = ShapeRepr([Uniadic(), Uniadic()], Variadic(), None)
    repr2 = ShapeRepr(None, Variadic(), [Uniadic()])

    ref_shape_repr1 = ["u1", "u2", "(V1, ...)"]
    ref_shape_repr2 = ["u1", "(V2, ...)", "u3"]

    assert_match_shapes(repr1, repr2, ref_shape_repr1, ref_shape_repr2)


def test_match_6():
    """
    [a, b, V1, c, d, e] => [a, b, V3, c, d, e]
    [V2, c]                [a, b, V3, c, d, e]
    """

    repr1 = ShapeRepr(
        [Uniadic(), Uniadic()], Variadic(), [Uniadic(), Uniadic(), Uniadic()]
    )
    repr2 = ShapeRepr(None, Variadic(), [Uniadic()])

    ref_shape_repr1 = ["u1", "u2", "(V1, ...)", "u3", "u4", "u5"]
    ref_shape_repr2 = ["u1", "u2", "(V1, ...)", "u3", "u4", "u5"]

    assert_match_shapes(repr1, repr2, ref_shape_repr1, ref_shape_repr2)


def test_match_7():
    """
    [1, V1] => [1, V3, 2]
    [V2, 2]    [1, V3, 2]
    """

    repr1 = ShapeRepr([Uniadic(1)], Variadic(), None)
    repr2 = ShapeRepr(None, Variadic(), [Uniadic(2)])

    ref_shape_repr1 = [1, "(V1, ...)", 2]
    ref_shape_repr2 = [1, "(V1, ...)", 2]

    assert_match_shapes(repr1, repr2, ref_shape_repr1, ref_shape_repr2)


def test_match_8():
    """
    [1, V1]    => [1, V3, 2, 3]
    [V2, 2, 3]    [1, V3, 2, 3]
    """

    repr1 = ShapeRepr([Uniadic(1)], Variadic(), None)
    repr2 = ShapeRepr(None, Variadic(), [Uniadic(2), Uniadic(3)])

    ref_shape_repr1 = [1, "(V1, ...)", 2, 3]
    ref_shape_repr2 = [1, "(V1, ...)", 2, 3]

    assert_match_shapes(repr1, repr2, ref_shape_repr1, ref_shape_repr2)


def test_match_9():
    """
    [1, V1]    => [1, V3, 2, 1]
    [V2, 2, 1]    [1, V3, 2, 1]
    """

    repr1 = ShapeRepr([Uniadic(1)], Variadic(), None)
    repr2 = ShapeRepr(None, Variadic(), [Uniadic(2), Uniadic(1)])

    ref_shape_repr1 = [1, "(V1, ...)", 2, 1]
    ref_shape_repr2 = [1, "(V1, ...)", 2, 1]

    assert_match_shapes(repr1, repr2, ref_shape_repr1, ref_shape_repr2)


def test_match_10():
    """
    [1, V1]    => [1, V1, 1]
    [V2, 1, 1]    [V2, 1, 1]
    """

    repr1 = ShapeRepr([Uniadic(1)], Variadic(), None)
    repr2 = ShapeRepr(None, Variadic(), [Uniadic(1), Uniadic(1)])

    ref_shape_repr1 = [1, "(V1, ...)", 1]
    ref_shape_repr2 = ["(V2, ...)", 1, 1]

    assert_match_shapes(repr1, repr2, ref_shape_repr1, ref_shape_repr2)


def test_shapes_rnn():
    from mithril.utils.dict_conversions import dict_to_model

    model = dict_to_model(
        {
            "name": "OneToMany",
            "args": {"max_sequence_length": 5, "cell_type": "RNNCell"},
        }
    )
    shapes = {
        "input": [20, 1, 15],
        "initial_hidden": [20, 1, 8],
        "target1": [17, 1, 15],
        "target2": [13, 1, 15],
        "target3": [9, 1, 15],
        "target4": [4, 1, 15],
    }
    compiled_model = compile(
        model=model, backend=TorchBackend(), shapes=shapes, jit=True
    )
    model_shape_dict = {
        key: tuple(value)
        for key, value in compiled_model.get_shapes(symbolic=False).items()
        if value is not None
    }
    ref_shapes = {
        "bias_h": (8,),
        "bias_o": (15,),
        "initial_hidden": (20, 1, 8),
        "input": (20, 1, 15),
        "w_hh": (8, 8),
        "w_ho": (15, 8),
        "w_ih": (8, 15),
        "output0": (20, 1, 15),
        "output1": (17, 1, 15),
        "output2": (13, 1, 15),
        "output3": (9, 1, 15),
        "output4": (4, 1, 15),
    }
    for key, value in ref_shapes.items():
        assert value == model_shape_dict[key]


def test_numeric_compatibility_inference_1():
    model = Model()
    model += Buffer().connect(input="input", output=IOKey(name="output"))
    shape_1: dict[str, list] = {"input": [1, ("Var1", ...)]}
    shape_2: dict[str, list] = {"input": [("Var1", ...), 2]}
    shape_3: dict[str, list] = {"input": [("Var1", ...), 3, 2]}
    model.set_shapes(**shape_1)
    model.set_shapes(**shape_2)
    model.set_shapes(**shape_3)
    ref_shapes: dict[str, list] = {
        "input": [1, "(V1, ...)", 3, 2],
        "output": [1, "(V1, ...)", 3, 2],
    }
    assert_shapes(model, ref_shapes)


def test_numeric_compatibility_inference_2():
    model = Model()
    model += Buffer().connect(input="input", output=IOKey(name="output"))
    shape_1: dict[str, list] = {"input": [("Var1", ...), 2, 3, 4]}
    shape_2: dict[str, list] = {"input": [1, 2, 3, ("Var1", ...)]}
    model.set_shapes(**shape_1)
    model.set_shapes(**shape_2)
    ref_shapes: dict[str, list] = {
        "input": [[1, "(V1, ...)", 2, 3, 4], [1, 2, 3, "(V2, ...)", 4]],
        "output": [[1, "(V1, ...)", 2, 3, 4], [1, 2, 3, "(V2, ...)", 4]],
    }
    assert_shapes(model, ref_shapes)


def test_numeric_compatibility_inference_3():
    model = Model()
    model += Buffer().connect(input="input", output=IOKey(name="output"))
    shape_1: dict[str, list] = {"input": [("Var1", ...), 2, 3, 4]}
    shape_2: dict[str, list] = {"input": [1, 2, 3, ("Var1", ...)]}
    model.set_shapes(**shape_1)
    model.set_shapes(**shape_2)
    ref_shapes: dict[str, list] = {
        "input": [[1, "(V1, ...)", 2, 3, 4], [1, 2, 3, "(V2, ...)", 4]],
        "output": [[1, "(V1, ...)", 2, 3, 4], [1, 2, 3, "(V2, ...)", 4]],
    }
    assert_shapes(model, ref_shapes)


def test_numeric_compatibility_inference_4():
    model = Model()
    model += Buffer().connect(input="input", output=IOKey(name="output"))

    shape_1: dict[str, list] = {"input": [1, 2, 3, ("Var1", ...)]}
    shape_2: dict[str, list] = {"input": [("Var1", ...), 2, 3, 4, 5]}

    model.set_shapes(**shape_1)
    model.set_shapes(**shape_2)
    ref_shapes: dict[str, list] = {
        "input": [[1, "(V1, ...)", 2, 3, 4, 5], [1, 2, 3, "(V2, ...)", 4, 5]],
        "output": [[1, "(V1, ...)", 2, 3, 4, 5], [1, 2, 3, "(V2, ...)", 4, 5]],
    }
    assert_shapes(model, ref_shapes)


def test_numeric_compatibility_inference_5():
    model = Model()
    model += Buffer().connect(input="input", output=IOKey(name="output"))
    shape_1: dict[str, list] = {"input": [("Var1", ...), 2, 3, 4, 5]}
    shape_2: dict[str, list] = {"input": [1, 2, 3, ("Var1", ...)]}
    model.set_shapes(**shape_1)
    model.set_shapes(**shape_2)
    ref_shapes: dict[str, list] = {
        "input": [[1, "(V1, ...)", 2, 3, 4, 5], [1, 2, 3, "(V2, ...)", 4, 5]],
        "output": [[1, "(V1, ...)", 2, 3, 4, 5], [1, 2, 3, "(V2, ...)", 4, 5]],
    }
    assert_shapes(model, ref_shapes)


def test_numeric_compatibility_inference_6():
    model = Model()
    model += Buffer().connect(input="input", output=IOKey(name="output"))
    shape_1: dict[str, list] = {"input": [("Var1", ...), 3, 4]}
    shape_2: dict[str, list] = {"input": [1, 2, ("Var1", ...)]}
    model.set_shapes(**shape_1)
    model.set_shapes(**shape_2)
    ref_shapes: dict[str, list] = {
        "input": [1, 2, "(V1, ...)", 3, 4],
        "output": [1, 2, "(V1, ...)", 3, 4],
    }
    assert_shapes(model, ref_shapes)


def test_numeric_compatibility_inference_7():
    model = Model()
    model += Buffer().connect(input="input", output=IOKey(name="output"))
    shape_1: dict[str, list] = {"input": [("Var1", ...), 2, 3]}
    shape_2: dict[str, list] = {"input": [1, 2, ("Var1", ...)]}
    model.set_shapes(**shape_1)
    model.set_shapes(**shape_2)
    ref_shapes: dict[str, list] = {
        "input": [[1, "(V1, ...)", 2, 3], [1, 2, "(V2, ...)", 3]],
        "output": [[1, "(V1, ...)", 2, 3], [1, 2, "(V2, ...)", 3]],
    }
    assert_shapes(model, ref_shapes)


def test_numeric_compatibility_inference_8():
    model = Model()
    model += Buffer().connect(input="input", output=IOKey(name="output"))
    shape_1: dict[str, list] = {"input": [("Var1", ...), "a", 1]}
    shape_2: dict[str, list] = {"input": ["b", 3, ("Var1", ...)]}
    model.set_shapes(**shape_1)
    model.set_shapes(**shape_2)
    ref_shapes: dict[str, list] = {
        "input": [["b", "(V1, ...)", "a", 1], ["b", 3, "(V2, ...)", 1]],
        "output": [["b", "(V1, ...)", "a", 1], ["b", 3, "(V2, ...)", 1]],
    }
    assert_shapes(model, ref_shapes)


def test_numeric_compatibility_inference_9():
    model = Model()
    model += Buffer().connect(input="input", output=IOKey(name="output"))
    shape_1: dict[str, list] = {"input": [("Var1", ...), 1, 1]}
    shape_2: dict[str, list] = {"input": [1, 1, ("Var1", ...)]}
    model.set_shapes(**shape_1)
    model.set_shapes(**shape_2)
    ref_shapes: dict[str, list] = {
        "input": [[1, 1, "(V1, ...)"], ["(V2, ...)", 1, 1]],
        "output": [[1, 1, "(V1, ...)"], ["(V2, ...)", 1, 1]],
    }
    assert_shapes(model, ref_shapes)


def test_numeric_compatibility_inference_10():
    model = Model()
    model += Buffer().connect(input="input", output=IOKey(name="output"))
    shape_1: dict[str, list] = {"input": [("Var1", ...), 1]}
    shape_2: dict[str, list] = {"input": [2, ("Var1", ...)]}
    shape_3: dict[str, list] = {"input": [("Var1", ...), 5, 4, 3, 1]}
    model.set_shapes(**shape_1)
    model.set_shapes(**shape_2)
    model.set_shapes(**shape_3)
    ref_shapes: dict[str, list] = {
        "input": [2, "(V1, ...)", 5, 4, 3, 1],
        "output": [2, "(V1, ...)", 5, 4, 3, 1],
    }
    assert_shapes(model, ref_shapes)


def test_node_count_4():
    class MyModel(PrimitiveModel):
        def __init__(self) -> None:
            super().__init__(
                formula_key="buffer",
                input=BaseKey(shape=["a", ("Var1", ...)], type=Tensor),
                output=BaseKey(shape=[("Var1", ...), "b"], type=Tensor),
            )

        def connect(  # type: ignore[override]
            self, input: ConnectionType = NOT_GIVEN, output: ConnectionType = NOT_GIVEN
        ):
            return ExtendInfo(self, {"input": input, "output": output})

    model = Model()
    test_model = MyModel()
    buff_model = Buffer()
    model += buff_model
    shapes: dict[str, list] = {"input": ["a", ("Var1", ...)]}
    buff_model.set_shapes(**shapes)
    model += Buffer()
    model += Buffer()
    model += Buffer()
    model += test_model
    all_nodes = get_all_nodes(model)
    assert buff_model.input.metadata.is_tensor
    ref_all_nodes = {
        test_model.output.metadata.shape,  # type: ignore
        buff_model.input.metadata.shape,
    }
    assert all_nodes == ref_all_nodes


def test_node_count_5():
    class MyModel(PrimitiveModel):
        def __init__(self) -> None:
            super().__init__(
                formula_key="buffer",
                input1=BaseKey(shape=["a", ("Var1", ...)], type=Tensor),
                input2=BaseKey(shape=[("Var1", ...), "b"], type=Tensor),
                output=BaseKey(shape=["a", ("Var1", ...)], type=Tensor),
            )

        def connect(  # type: ignore[override]
            self,
            input1: ConnectionType = NOT_GIVEN,
            input2: ConnectionType = NOT_GIVEN,
            output: ConnectionType = NOT_GIVEN,
        ):
            return ExtendInfo(
                self, {"input1": input1, "input2": input2, "output": output}
            )

    model = Model()
    test_model = MyModel()
    buff_model = Buffer()
    model |= buff_model
    shapes: dict[str, list] = {"input": ["a", ("Var1", ...)]}
    buff_model.set_shapes(**shapes)
    model += test_model
    model.merge_connections(test_model.input2, buff_model.input)  # type: ignore
    model |= Buffer().connect(input=buff_model.input, output=IOKey(name="output"))
    all_nodes = get_all_nodes(model)

    data = buff_model.input.metadata
    assert data.is_tensor
    ref_all_nodes = {data.shape}
    assert all_nodes == ref_all_nodes


def test_node_count_6():
    model = Model()
    buff_model = Buffer()
    shapes: dict[str, list] = {"input": ["a", ("Var1", ...)]}
    buff_model.set_shapes(**shapes)
    model += buff_model
    for _ in range(5):
        model += deepcopy(model)
    all_nodes = get_all_nodes(model)

    data = buff_model.input.metadata
    assert data.is_tensor

    ref_all_nodes = {data.shape}
    assert all_nodes == ref_all_nodes


def test_node_count_7():
    model = Model()
    buff_model = Buffer()
    shapes: dict[str, list] = {"input": ["a", ("Var1", ...)]}
    buff_model.set_shapes(**shapes)
    model += buff_model
    for _ in range(5):
        model += deepcopy(model)
    all_nodes = get_all_nodes(model)
    assert buff_model.input.metadata.is_tensor
    ref_all_nodes = {buff_model.input.metadata.shape}
    assert all_nodes == ref_all_nodes


def test_node_count_8():
    model = Model()
    add_model1 = Add()
    add_model1.set_types(left=Tensor, right=Tensor)
    add_model2 = Add()
    add_model3 = Add()
    model |= add_model1.connect(left="left", right="right")
    model |= add_model2.connect(left="left", right=add_model1.output)
    model |= add_model3.connect(left="left", right=add_model2.output)
    model.set_shapes(left=[])
    ref_all_nodes = {model.left.metadata.shape, model.right.metadata.shape}  # type: ignore
    all_nodes = get_all_nodes(model)
    assert all_nodes == ref_all_nodes


def test_node_count_9():
    model = Model()
    add_model1 = Add()
    add_model1.set_types(left=Tensor, right=Tensor)
    add_model2 = Add()
    add_model3 = Add()
    model |= add_model1.connect(left="left", right="right")
    model |= add_model2.connect(left="left", right=add_model1.output)
    model |= add_model3.connect(left="left", right=add_model2.output)
    model.set_shapes(left=[])
    ref_all_nodes = {model.left.metadata.shape, model.right.metadata.shape}  # type: ignore
    all_nodes = get_all_nodes(model)
    assert all_nodes == ref_all_nodes


def test_node_count_10():
    submodel1 = Model()

    buff_model1 = Buffer()
    buff_model2 = Buffer()
    buff_model3 = Buffer()

    submodel1 |= buff_model1.connect(
        input=IOKey("input1", type=Tensor),
        output=IOKey(name="output1"),
    )
    submodel1 |= buff_model2.connect(
        input=IOKey("input2", type=Tensor),
        output=IOKey(name="output2"),
    )
    submodel1 |= buff_model3.connect(
        input=IOKey("input3", type=Tensor),
        output=IOKey(name="output3"),
    )

    model = Model()
    submodel2 = deepcopy(submodel1)
    submodel3 = deepcopy(submodel1)

    model |= submodel1.connect(input1="input1", input2="input2", input3="input3")
    model |= submodel2.connect(
        input1=submodel1.output1,  # type: ignore
        input2=submodel1.output2,  # type: ignore
        input3=submodel1.output3,  # type: ignore
    )
    model |= submodel3.connect(
        input1=submodel2.output1,  # type: ignore
        input2=submodel2.output2,  # type: ignore
        input3=submodel2.output3,  # type: ignore
    )

    all_nodes = get_all_nodes(model)
    ref_all_nodes = {
        model.input1.metadata.shape,  # type: ignore
        model.input2.metadata.shape,  # type: ignore
        model.input3.metadata.shape,  # type: ignore
    }
    assert all_nodes == ref_all_nodes


def test_node_count_11():
    composite_3 = Model()
    m1 = Model()
    m1 |= Add().connect(
        left=IOKey("input1", type=Tensor),
        right=IOKey("input2", type=Tensor),
        output=IOKey(name="output"),
    )
    m2 = Model()
    m2 |= m1.connect(input1="input1", input2="input2")
    m2 |= Add().connect(left="input1", right=m1.output, output=IOKey(name="output"))  # type: ignore
    m3 = Model()
    m3 |= m2.connect(input1="input1", input2="input2")
    m3 |= (add3 := Add()).connect(
        left="input1",
        right=m2.output,  # type: ignore
        output=IOKey(name="output"),
    )
    m4 = Model()
    m4 |= m3.connect(input1="input1", input2="input2")
    m4 |= Add().connect(left="input1", right=m3.output, output=IOKey(name="output"))  # type: ignore
    composite_3 |= m4.connect(input1="input1", input2="input2")
    composite_3 |= Add().connect(
        left="input1",
        right=m4.output,  # type: ignore
        output=IOKey(name="output"),
    )

    add3.set_shapes(left=[])

    all_nodes = get_all_nodes(composite_3)
    ref_all_nodes = {
        composite_3.input1.metadata.shape,  # type: ignore
        composite_3.input2.metadata.shape,  # type: ignore
    }
    assert all_nodes == ref_all_nodes


def test_node_count_12():
    model = Model()
    buff_model1 = Buffer()
    buff_model2 = Buffer()
    model |= buff_model1.connect(input="input1", output=IOKey(name="output1"))
    model |= buff_model2.connect(input="input2", output=IOKey(name="output2"))
    shape_1: dict[str, list] = {
        "input1": ["x", "y", 1],
        "input2": ["x", "y", 1],
    }
    model.set_shapes(**shape_1)

    all_nodes = get_all_nodes(model)
    ref_all_nodes = {model.input1.metadata.shape}  # type: ignore
    assert all_nodes == ref_all_nodes


def test_node_count_13():
    model = Model()
    buff_model1 = Buffer()
    buff_model2 = Buffer()
    model |= buff_model1.connect(input="input1", output=IOKey(name="output1"))
    model |= buff_model2.connect(input="input2", output=IOKey(name="output2"))
    shape_1: dict[str, list] = {
        "input1": [("Var1", ...), "a"],
        "input2": [("Var1", ...), "a"],
    }
    model.set_shapes(**shape_1)
    all_nodes = get_all_nodes(model)
    ref_all_nodes = {model.input1.metadata.shape}  # type: ignore
    assert all_nodes == ref_all_nodes


def test_node_count_14():
    model = Model()
    buff_model1 = Buffer()
    buff_model2 = Buffer()
    model |= buff_model1.connect(input="input1", output=IOKey(name="output1"))
    model |= buff_model2.connect(input="input2", output=IOKey(name="output2"))
    model.set_shapes(input1=["x", "y", "z"], input2=["x", "y", "z"])

    all_nodes = get_all_nodes(model)
    ref_all_nodes = {model.input1.metadata.shape}  # type: ignore
    assert all_nodes == ref_all_nodes


def test_node_count_15():
    model = Model()
    buff_model1 = Buffer()
    buff_model2 = Buffer()
    model |= buff_model1.connect(input="input1", output=IOKey(name="output1"))
    model |= buff_model2.connect(input="input2", output=IOKey(name="output2"))
    model.set_shapes(input1=[1, 1], input2=[1, 1])

    all_nodes = get_all_nodes(model)
    ref_all_nodes = {model.input1.metadata.shape}  # type: ignore
    assert all_nodes == ref_all_nodes


def test_node_count_16() -> None:
    class MyModel(PrimitiveModel):
        input: Connection
        output: Connection

        def __init__(self) -> None:
            super().__init__(
                formula_key="buffer",
                input=BaseKey(shape=[5, 5], type=Tensor),
                output=BaseKey(shape=[5, 5], type=Tensor),
            )

        def connect(  # type: ignore[override]
            self, input: ConnectionType = NOT_GIVEN, output: ConnectionType = NOT_GIVEN
        ):
            return ExtendInfo(self, {"input": input, "output": output})

    model = Model()
    model |= (test_model := MyModel())
    model += MyModel()
    model += MyModel()
    model += MyModel()

    all_nodes = get_all_nodes(model)
    data = test_model.input.metadata
    assert data.is_tensor
    ref_all_nodes = {data.shape}
    assert all_nodes == ref_all_nodes


def test_node_count_18() -> None:
    class MyModel(PrimitiveModel):
        input: Connection
        output: Connection

        def __init__(self) -> None:
            super().__init__(
                formula_key="buffer",
                input=BaseKey(shape=["a", "b"], type=Tensor),
                output=BaseKey(shape=["a", "b"], type=Tensor),
            )

        def connect(  # type: ignore[override]
            self, input: ConnectionType = NOT_GIVEN, output: ConnectionType = NOT_GIVEN
        ):
            return ExtendInfo(self, {"input": input, "output": output})

    model = Model()
    model += (test_model := MyModel())
    model += MyModel()
    model += MyModel()
    model += MyModel()

    all_nodes = get_all_nodes(model)
    assert test_model.input.metadata.is_tensor
    ref_all_nodes = {test_model.input.metadata.shape}
    assert all_nodes == ref_all_nodes


def test_node_count_19() -> None:
    class MyModel(PrimitiveModel):
        input: Connection
        output: Connection

        def __init__(self) -> None:
            super().__init__(
                formula_key="buffer",
                input=BaseKey(shape=["a", ("V1", ...), "b", "c"], type=Tensor),
                output=BaseKey(shape=["c", ("V1", ...), "a", "b"], type=Tensor),
            )

        def connect(  # type: ignore[override]
            self, input: ConnectionType = NOT_GIVEN, output: ConnectionType = NOT_GIVEN
        ):
            return ExtendInfo(self, {"input": input, "output": output})

    model = Model()
    model += (test_model1 := MyModel())
    model += (test_model2 := MyModel())
    model += (test_model3 := MyModel())
    for _ in range(400):
        model += MyModel()

    all_nodes = get_all_nodes(model)
    assert test_model1.input.metadata.is_tensor
    assert test_model2.input.metadata.is_tensor
    assert test_model3.input.metadata.is_tensor
    ref_all_nodes = {
        test_model1.input.metadata.shape,
        test_model2.input.metadata.shape,
        test_model3.input.metadata.shape,
    }
    assert all_nodes == ref_all_nodes


def test_node_count_20() -> None:
    class MyModel(PrimitiveModel):
        input: Connection
        output: Connection

        def __init__(self) -> None:
            super().__init__(
                formula_key="buffer",
                input=BaseKey(shape=["a", "b", "c"], type=Tensor),
                output=BaseKey(shape=["b", "c", "a"], type=Tensor),
            )

        def connect(  # type: ignore[override]
            self, input: ConnectionType = NOT_GIVEN, output: ConnectionType = NOT_GIVEN
        ):
            return ExtendInfo(self, {"input": input, "output": output})

    model = Model()
    model += (test_model1 := MyModel())
    model += (test_model2 := MyModel())
    model += (test_model3 := MyModel())
    for _ in range(400):
        model += MyModel()

    all_nodes = get_all_nodes(model)
    assert test_model1.input.metadata.is_tensor
    assert test_model2.input.metadata.is_tensor
    assert test_model3.input.metadata.is_tensor
    ref_all_nodes = {
        test_model1.input.metadata.shape,
        test_model2.input.metadata.shape,
        test_model3.input.metadata.shape,
    }
    assert all_nodes == ref_all_nodes


def test_node_count_21() -> None:
    class MyModel(PrimitiveModel):
        input: Connection
        output: Connection

        def __init__(self) -> None:
            super().__init__(
                formula_key="buffer",
                input=BaseKey(shape=[], type=Tensor),
                output=BaseKey(shape=[], type=Tensor),
            )

        def connect(  # type: ignore[override]
            self, input: ConnectionType = NOT_GIVEN, output: ConnectionType = NOT_GIVEN
        ):
            return ExtendInfo(self, {"input": input, "output": output})

    model = Model()
    model += (test_model1 := MyModel())
    model += (test_model2 := MyModel())
    model += (test_model3 := MyModel())
    for _ in range(20):
        model += MyModel()

    all_nodes = get_all_nodes(model)
    assert test_model1.input.metadata.is_tensor
    assert test_model2.input.metadata.is_tensor
    assert test_model3.input.metadata.is_tensor
    ref_all_nodes = {
        test_model1.input.metadata.shape,
        test_model2.input.metadata.shape,
        test_model3.input.metadata.shape,
    }
    assert all_nodes == ref_all_nodes


def test_uniadic_repr_count_1():
    model = Model()
    buff_model = Buffer()
    buff_model.set_shapes(input=["a", "b", "c"])
    model += Buffer()
    model += Buffer()
    model += Buffer()
    model += Buffer()
    model += Buffer()
    model += Buffer()
    model += Buffer()
    all_symbols = get_all_symbols(buff_model)

    for symbol in all_symbols:
        assert symbol is not None
        assert len(symbol.reprs) == 1


def test_uniadic_repr_count_2():
    model = Model()
    model |= (buff_model1 := Buffer()).connect(
        input="input1", output=IOKey(name="output1")
    )
    model |= Buffer().connect(input="input2", output=IOKey(name="output2"))
    model |= Buffer().connect(input="input3", output=IOKey(name="output3"))
    shape_1: dict[str, list] = {
        "input1": [1, "u1", "u2"],
        "input2": [1, "u1", "u2"],
        "input3": [1, "u1"],
    }
    model.set_shapes(**shape_1)

    assert buff_model1.input.metadata.is_tensor
    data_shape = buff_model1.input.metadata.shape

    assert data_shape is not None
    input_1_prefix = next(iter(data_shape.reprs)).prefix

    uni1 = input_1_prefix[0]
    uni2 = input_1_prefix[1]
    uni3 = input_1_prefix[2]

    assert len(uni1.reprs) == 2
    assert len(uni2.reprs) == 2
    assert len(uni3.reprs) == 1


def test_uniadic_repr_count_3():
    model = Model()
    model |= (buff_model1 := Buffer()).connect(
        input="input1", output=IOKey(name="output1")
    )
    model |= Buffer().connect(input="input2", output=IOKey(name="output2"))
    model |= Buffer().connect(input="input3", output=IOKey(name="output3"))
    model |= Buffer().connect(input="input4", output="output4")
    model |= Buffer().connect(input="input5", output="output5")
    model |= Buffer().connect(input="input6", output="output6")

    model.set_shapes(
        input1=["u1", "u2", "u3", "u4", "u5"],
        input2=["u1", "u2", "u3", "u4"],
        input3=["u1", "u2", "u3"],
        input4=["u1", "u2"],
        input5=["u1"],
        input6=["u1", "u2", "u3"],
    )

    assert buff_model1.input.metadata.is_tensor
    data_shape = buff_model1.input.metadata.shape

    assert data_shape is not None

    input_1_prefix = next(iter(data_shape.reprs)).prefix

    uni1 = input_1_prefix[0]
    uni2 = input_1_prefix[1]
    uni3 = input_1_prefix[2]
    uni4 = input_1_prefix[3]
    uni5 = input_1_prefix[4]

    assert len(uni1.reprs) == 5
    assert len(uni2.reprs) == 4
    assert len(uni3.reprs) == 3
    assert len(uni4.reprs) == 2
    assert len(uni5.reprs) == 1


def test_uniadic_repr_count_4():
    model = Model()
    model |= Buffer().connect(input="input1", output=IOKey(name="output1"))
    model |= Buffer().connect(input="input2", output=IOKey(name="output2"))
    model |= Buffer().connect(input="input3", output=IOKey(name="output3"))
    model |= Buffer().connect(input="input4", output=IOKey(name="output4"))
    model |= Buffer().connect(input="input5", output=IOKey(name="output5"))
    model |= Buffer().connect(input="input6", output=IOKey(name="output6"))

    main_model = Model()
    main_model |= (model1 := deepcopy(model)).connect(
        **{key: key for key in model.input_keys}
    )

    main_model |= (model2 := deepcopy(model)).connect(
        input1=model1.output1,  # type: ignore
        input2=model1.output2,  # type: ignore
        input3=model1.output3,  # type: ignore
        input4=model1.output4,  # type: ignore
        input5=model1.output5,  # type: ignore
        input6=model1.output6,  # type: ignore
    )

    main_model |= (model3 := deepcopy(model)).connect(
        input1=model2.output1,  # type: ignore
        input2=model2.output2,  # type: ignore
        input3=model2.output3,  # type: ignore
        input4=model2.output4,  # type: ignore
        input5=model2.output5,  # type: ignore
        input6=model2.output6,  # type: ignore
    )

    main_model |= deepcopy(model).connect(
        input1=model3.output1,  # type: ignore
        input2=model3.output2,  # type: ignore
        input3=model3.output3,  # type: ignore
        input4=model3.output4,  # type: ignore
        input5=model3.output5,  # type: ignore
        input6=model3.output6,  # type: ignore
    )

    model1.set_shapes(
        input1=["u1", "u2", "u3", "u4", "u5"],
        input2=["u1", "u2", "u3", "u4"],
        input3=["u1", "u2", "u3"],
        input4=["u1", "u2"],
        input5=["u1"],
        input6=["u1", "u2", "u3"],
    )

    data_shape = model1.input1.metadata.shape  # type: ignore

    assert data_shape is not None

    input_1_prefix = next(iter(data_shape.reprs)).prefix

    uni1 = input_1_prefix[0]
    uni2 = input_1_prefix[1]
    uni3 = input_1_prefix[2]
    uni4 = input_1_prefix[3]
    uni5 = input_1_prefix[4]

    assert len(uni1.reprs) == 5
    assert len(uni2.reprs) == 4
    assert len(uni3.reprs) == 3
    assert len(uni4.reprs) == 2
    assert len(uni5.reprs) == 1


def test_uniadic_repr_count_5():
    model = Model()
    model |= (buff_model1 := Buffer()).connect(
        input="input1", output=IOKey(name="output1")
    )
    model |= Buffer().connect(input="input2", output=IOKey(name="output2"))
    model |= Buffer().connect(input="input3", output=IOKey(name="output3"))
    model |= Buffer().connect(input="input4", output="output4")
    model |= Buffer().connect(input="input5", output="output5")
    model |= Buffer().connect(input="input6", output="output6")

    model.set_shapes(
        input1=[1, "u1"],
        input2=[1, "u2"],
        input3=[1, "u3"],
        input4=[1, "u4"],
        input5=[1, "u5"],
        input6=[1, "u6"],
    )

    assert buff_model1.input.metadata.is_tensor
    data_shape = buff_model1.input.metadata.shape

    assert data_shape is not None

    input_1_prefix = next(iter(data_shape.reprs)).prefix

    uni1 = input_1_prefix[0]
    uni2 = input_1_prefix[1]

    assert len(uni1.reprs) == 6
    assert len(uni2.reprs) == 1

    model.set_shapes(
        input1=[1, 2],
        input2=[1, 2],
        input3=[1, 2],
        input4=[1, 2],
        input5=[1, 2],
        input6=[1, 2],
    )

    assert len(uni1.reprs) == 1
    assert len(uni2.reprs) == 1


def test_repr_count_mlp():
    layer_num = 50
    model = MLP(
        activations=[Relu() for _ in range(layer_num)],
        dimensions=[10 for _ in range(layer_num)],
    )
    all_symbols = get_all_symbols(model)

    for symbol in all_symbols:
        assert symbol is not None
        all_reprs = symbol.reprs
        for repr1, repr2 in combinations(all_reprs, 2):
            assert not repr1.is_equal(repr2)


def test_different_constsolver_objects():
    model = Model()
    relu1 = Relu()
    relu1.set_shapes(input=[1, 2])
    relu2 = deepcopy(relu1)

    model |= relu1.connect(input="input1", output=IOKey(name="output1"))
    model |= relu2.connect(input="input2", output=IOKey(name="output2"))
    assert model.input2.metadata.shape == model.input1.metadata.shape  # type: ignore


def test_symbol_store():
    """Testcase of different constraintsolver objects

    Add model, wrapped with two models, model1 and model2

    Firstly, set shapes of output of model2 as [2, 3, 4, 5]
    Secondly, set shapes of left of model1 as [2, 3, 4, 5]
    Thirdly set shapes of right of add_model as [2, 3, 4, 5]

    Note that all of the models share same left, right and output

    (model1.left == model2.left == add_model.left)
    (model1.right == model2.right == add_model.right)
    (model1.output == model2.output == add_model.output)

    Therefore, in the end, all of them should share same shaperepr
    objects.

    However, all of them have different Constraintsolver objects. Therefore,
    Their symbolstore also hold different objects, causing same valued uniadics
    to have different symbols in each level.
    """
    model1 = Model()
    model2 = Model()
    add_model = Add()
    model1 += add_model.connect(left="left", right="right", output=IOKey(name="output"))
    model2 += model1.connect(left="left", right="right", output=IOKey(name="output"))
    model2.set_shapes(output=[2, 3, 4, 5])
    model1.set_shapes(left=[2, 3, 4, 5])
    add_model.set_shapes(right=[2, 3, 4, 5])
    assert_all_nodes_unique(model2)


def test_multi_repr_with_integer_uni():
    """Testscase of [V1, 1]
                 [1, V1]

    1's should be same uniadic objects
    """

    model = Model()
    relu_model = Relu()
    model += relu_model.connect(input="input", output=IOKey(name="output"))
    model.set_shapes(input=[("Var1", ...), 1])
    model.set_shapes(input=[1, ("Var1", ...)])
    input_shape_node = model.input.metadata.shape  # type: ignore
    repr1, repr2 = tuple(input_shape_node.reprs)

    uni1 = (repr1.prefix + repr1.suffix)[0]
    uni2 = (repr2.prefix + repr2.suffix)[0]

    assert uni1.value == 1
    assert uni2.value == 1

    assert uni1 is uni2


def test_scalar_empty_node():
    model = Model()
    reduce_model = Mean(axis=1)
    reduce_model_2 = Mean(axis=1)
    model += reduce_model
    model += reduce_model_2
    assert_all_nodes_unique(model)


def test_add_model_with_scalar_input():
    model = Model()
    add1 = Add()

    left_input: Tensor[float] = Tensor(np.ones((3, 4, 5, 6, 7)).tolist())
    right_input: Tensor[float] = Tensor(np.ones((3, 4, 5, 6, 7)).tolist())
    model += add1.connect(
        left=left_input, right=right_input, output=IOKey(name="output")
    )
    assert_all_nodes_unique(model)


def test_add_model_set_shapes():
    model = Model()
    add1 = Add()

    model += add1.connect(left="left", right="right", output=IOKey(name="output"))
    model.set_shapes(left=[3, 4, 5, 6, 7], right=[3, 4, 5, 6, 7])
    assert_all_nodes_unique(model)


def test_possible_uniadic_values_directed_1():
    uni = Uniadic()
    uni.update_possible_values({1, 2, 3, 4, 5})
    assert uni.metadata.possible_values == {1, 2, 3, 4, 5}


def test_possible_uniadic_values_directed_2():
    uni = Uniadic()
    uni.update_possible_values({1, 2, 3, 4, 5})

    with pytest.raises(ValueError) as err_info:
        uni.update_possible_values({6})
    assert str(err_info.value) == "Possible values mismatch!"


def test_possible_uniadic_values_directed_3():
    uni1 = Uniadic()
    uni2 = Uniadic()
    uni1.update_possible_values({1, 2, 3, 4, 5})
    uni2.update_possible_values({5, 6})
    uni1.match(uni2)

    assert uni1.metadata.possible_values == {5}
    assert uni1.value == 5


def test_possible_uniadic_values_directed_4():
    uni = Uniadic()
    uni.update_possible_values({1, 2, 3, 4, 5})

    with pytest.raises(ValueError) as err_info:
        uni.update_possible_values({6})
    assert str(err_info.value) == "Possible values mismatch!"


def test_possible_uniadic_values_directed_5():
    uni1 = Uniadic()
    uni1.update_possible_values({5})

    assert uni1.value == 5


def test_possible_uniadic_values_directed_7():
    uni = Uniadic()

    with pytest.raises(ValueError) as err_info:
        uni.update_possible_values(set())
    assert str(err_info.value) == "Possible value set could not be empty!"


def test_possible_uniadic_values_directed_8():
    buff_model = Buffer()
    buff_model.set_shapes(input=["a", "b"])

    assert buff_model.input.metadata.is_tensor
    data_shape = buff_model.input.metadata.shape

    assert data_shape is not None

    input_repr = next(iter(data_shape.reprs))

    input_repr[0].update_possible_values({1, 2, 3, 4})
    input_repr[1].update_possible_values({1, 2})

    buff_model.set_shapes(input=["a", "a"])

    assert input_repr[0].possible_values == {1, 2}
    assert input_repr[1].possible_values == {1, 2}


def test_possible_uniadic_values_directed_9():
    buff_model = Buffer()
    buff_model.set_shapes(input=["a", "b", "c", "d"])

    assert buff_model.input.metadata.is_tensor
    data_shape = buff_model.input.metadata.shape

    assert data_shape is not None

    input_repr = next(iter(data_shape.reprs))

    input_repr[0].update_possible_values({1, 2, 3, 4})
    input_repr[1].update_possible_values({1, 2, 3})
    input_repr[2].update_possible_values({1, 2})
    input_repr[3].update_possible_values({2, 3})

    buff_model.set_shapes(input=["a", "a", "a", "a"])

    ref_shapes = {"input": [2, 2, 2, 2], "output": [2, 2, 2, 2]}

    assert_shapes(buff_model, ref_shapes)


def test_possible_uniadic_values_directed_10():
    uni1 = Uniadic({1, 2})
    uni2 = Uniadic({1, 2, 3})
    uni3 = Uniadic({2, 3})

    repr1 = ShapeRepr(prefix=[uni1], root=None, suffix=None)
    repr2 = ShapeRepr(prefix=[uni2], root=None, suffix=None)
    repr3 = ShapeRepr(prefix=[uni3], root=None, suffix=None)

    repr1_node = repr1.node
    repr2_node = repr2.node
    repr3_node = repr3.node

    repr1_node.merge(repr2_node)
    repr1_node.merge(repr3_node)

    assert repr1_node.get_shapes() == [2]


def test_possible_uniadic_values_directed_11():
    uni1 = Uniadic({1, 2})
    uni2 = Uniadic({1, 3})
    uni3 = Uniadic({2, 3})

    repr1 = ShapeRepr(prefix=[uni1], root=None, suffix=None)
    repr2 = ShapeRepr(prefix=[uni2], root=None, suffix=None)
    repr3 = ShapeRepr(prefix=[uni3], root=None, suffix=None)

    repr1_node = repr1.node
    repr2_node = repr2.node
    repr3_node = repr3.node

    repr1_node.merge(repr2_node)
    with pytest.raises(ValueError) as err_info:
        repr1_node.merge(repr3_node)
    assert str(err_info.value) == "Possible values mismatch!"


def test_possible_uniadic_values_directed_12():
    uni1 = Uniadic({1, 2})
    uni2 = Uniadic({1, 3})
    uni3 = Uniadic({2, 3})

    uni4 = Uniadic({1, 2})
    uni5 = Uniadic({1, 3})
    uni6 = Uniadic({2, 3})

    repr1 = ShapeRepr(prefix=[uni1, uni2], root=Variadic(), suffix=[uni3])
    repr2 = ShapeRepr(prefix=[uni6], root=Variadic(), suffix=[uni5, uni4])

    repr1_node = repr1.node
    repr2_node = repr2.node
    repr1_node.merge(repr2_node)

    assert repr1_node.get_shapes() == [2, "u1", "(V1, ...)", 2]
    assert repr2_node.get_shapes() == [2, "(V1, ...)", "u1", 2]


def test_possible_uniadic_values_1():
    model = Model()
    model |= Add().connect(left="l1", right="r1", output="o1")
    model |= Add().connect(left="l1", right="r2", output="o2")
    shapes: dict[str, list] = {
        "o1": [4],
        "l1": [None],
        "r1": [None],
        "o2": [5],
        "r2": [None],
    }
    model.set_shapes(**shapes)
    ref_shapes = {"l1": [1], "r1": [4], "r2": [5], "o1": [4], "o2": [5]}
    assert_shapes(model, ref_shapes)


def assert_possibles(possibles, ref_possibles):
    assert possibles == ref_possibles
    if possibles is not None:
        for _len, pos in possibles.items():
            assert pos._is_applicable == ref_possibles[_len]._is_applicable
            assert pos.dnf_lookup_table == ref_possibles[_len].dnf_lookup_table


def test_update_possible_values_1():
    var = Variadic()
    uni1 = Uniadic({1, 4})
    var.update_possible_values(PossibleValues(), PossibleValues((uni1,)))

    ref_possibles = {0: PossibleValues(), 1: PossibleValues((uni1,))}
    assert_possibles(var.possibles, ref_possibles)


def test_update_possible_values2():
    var = Variadic()

    var.update_possible_values(
        PossibleValues(),
        PossibleValues((uni1_1 := Uniadic({1, 4}), uni1_2 := Uniadic(5))),
    )
    var.update_possible_values(
        PossibleValues(), PossibleValues((uni2_1 := Uniadic(4), uni2_2 := Uniadic()))
    )
    dnf1 = DNF([AND({uni1_1: uni2_1})])
    dnf2 = DNF([AND({uni1_2: uni2_2})])

    ref_possibles = {
        0: PossibleValues(),
        2: PossibleValues((uni1_1, uni1_2), [dnf1, dnf2]),
    }
    eq1 = Equivalences({uni1_1, uni2_1}, {4})
    eq2 = Equivalences({uni1_2, uni2_2}, {5})

    ref_possibles[2].dnf_lookup_table = {
        uni1_1: eq1,
        uni2_1: eq1,
        uni1_2: eq2,
        uni2_2: eq2,
    }

    assert_possibles(var.possibles, ref_possibles)


def test_update_possible_values_2_2():
    var = Variadic()
    uni3 = Uniadic({3, 5})
    dnf1 = DNF([AND({uni3: 5})])
    var.update_possible_values(
        PossibleValues(),
        PossibleValues(
            (uni1_1 := Uniadic({1, 4, 7}), uni1_2 := Uniadic({5, 6})), [dnf1]
        ),
    )

    uniadics = (uni2_1 := Uniadic({1, 4}), uni2_2 := Uniadic({5, 6}))
    dnf2 = DNF([AND({uni3: 6, uni2_1: 4, uni2_2: 6}), AND({uni2_1: 1, uni2_2: 5})])
    var.update_possible_values(PossibleValues(), PossibleValues(uniadics, [dnf2]))
    dnf3 = DNF([AND({uni1_1: uni2_1})])
    dnf4 = DNF([AND({uni1_2: uni2_2})])

    ref_possibles = {
        0: PossibleValues(),
        2: PossibleValues((uni1_1, uni1_2), [dnf1, dnf2, dnf3, dnf4]),
    }
    eq1 = Equivalences({uni1_1, uni2_1}, {1})
    eq2 = Equivalences({uni1_2, uni2_2}, {5})
    eq3 = Equivalences({uni3}, {5})

    ref_possibles[2].dnf_lookup_table = {
        uni1_1: eq1,
        uni2_1: eq1,
        uni1_2: eq2,
        uni2_2: eq2,
        uni3: eq3,
    }
    assert_possibles(var.possibles, ref_possibles)


def test_update_possible_values_3():
    var = Variadic()
    var.update_possible_values(
        PossibleValues(), PossibleValues((Uniadic({1, 5}), Uniadic({4, 5})))
    )

    with pytest.raises(ValueError) as err_info:
        var.update_possible_values(PossibleValues((Uniadic(), Uniadic(), Uniadic())))
    assert str(err_info.value) == "Incompatible possible values for Variadic!"


def test_update_possible_values_4():
    var = Variadic()
    var.update_possible_values(
        PossibleValues(), PossibleValues((Uniadic({1, 5}), Uniadic({4, 5})))
    )

    with pytest.raises(ValueError) as err_info:
        var.update_possible_values(PossibleValues((Uniadic({2, 5}),)))
    assert str(err_info.value) == "Incompatible possible values for Variadic!"


def test_update_possible_values_5():
    var = Variadic()
    var.update_possible_values(
        PossibleValues(), PossibleValues((uni1 := Uniadic({1, 4}), uni2 := Uniadic()))
    )
    var.update_possible_values(
        PossibleValues((uni3 := Uniadic(), uni4 := Uniadic({5})))
    )
    assert_possibles(var.possibles, None)
    assert uni3.possible_values == uni1.possible_values == {1, 4}
    assert uni4.possible_values == uni2.possible_values == {5}


def test_update_possible_values_6():
    var = Variadic()
    var.update_possible_values(
        PossibleValues(),
        PossibleValues(
            (uni1 := Uniadic({1, 4}), uni2 := Uniadic(), uni3 := Uniadic({6}))
        ),
    )
    var.update_possible_values(
        PossibleValues((uni4 := Uniadic(), uni5 := Uniadic(5), uni6 := Uniadic()))
    )

    assert_possibles(var.possibles, None)
    assert uni4.possible_values == uni1.possible_values == {1, 4}
    assert uni5.possible_values == uni2.possible_values == {5}
    assert uni6.possible_values == uni3.possible_values == {6}


def test_update_possible_values_7():
    var = Variadic()

    var.update_possible_values(
        PossibleValues(),
        PossibleValues(
            (uni1 := Uniadic({1, 4}), uni2 := Uniadic({1, 4}), uni3 := Uniadic({1, 4}))
        ),
    )
    var.update_possible_values(
        PossibleValues(),
        PossibleValues(
            (uni4 := Uniadic(), uni5 := Uniadic(4), uni6 := Uniadic({1, 5}))
        ),
    )
    dnf1 = DNF([AND({uni1: uni4})])
    dnf2 = DNF([AND({uni2: uni5})])
    dnf3 = DNF([AND({uni3: uni6})])

    ref_possibles = {
        0: PossibleValues(),
        3: PossibleValues((uni1, uni2, uni3), [dnf1, dnf2, dnf3]),
    }
    eq1 = Equivalences({uni1, uni4}, {1, 4})
    eq2 = Equivalences({uni2, uni5}, {4})
    eq3 = Equivalences({uni3, uni6}, {1})

    ref_possibles[3].dnf_lookup_table = {
        uni1: eq1,
        uni2: eq2,
        uni3: eq3,
        uni4: eq1,
        uni5: eq2,
        uni6: eq3,
    }
    assert_possibles(var.possibles, ref_possibles)


def test_update_possible_values_8():
    var = Variadic()
    var.update_possible_values(
        PossibleValues(),
        PossibleValues((uni1 := Uniadic(1), uni2 := Uniadic(2), uni3 := Uniadic(5))),
    )
    var.update_possible_values(
        PossibleValues(),
        PossibleValues(
            (uni4 := Uniadic(), uni5 := Uniadic(2), uni6 := Uniadic({1, 5}))
        ),
    )

    dnf1 = DNF([AND({uni1: uni4})])
    dnf2 = DNF([AND({uni2: uni5})])
    dnf3 = DNF([AND({uni3: uni6})])

    ref_possibles = {
        0: PossibleValues(),
        3: PossibleValues((uni1, uni2, uni3), [dnf1, dnf2, dnf3]),
    }
    eq1 = Equivalences({uni1, uni4}, {1})
    eq2 = Equivalences({uni2, uni5}, {2})
    eq3 = Equivalences({uni3, uni6}, {5})

    ref_possibles[3].dnf_lookup_table = {
        uni1: eq1,
        uni2: eq2,
        uni3: eq3,
        uni4: eq1,
        uni5: eq2,
        uni6: eq3,
    }
    assert_possibles(var.possibles, ref_possibles)


def test_update_possible_values_9():
    var = Variadic()
    var.update_possible_values(
        PossibleValues(),
        PossibleValues((uni1 := Uniadic(), uni2 := Uniadic(), uni3 := Uniadic())),
    )
    var.update_possible_values(
        PossibleValues(),
        PossibleValues(
            (uni4 := Uniadic(), uni5 := Uniadic({2, 1}), uni6 := Uniadic({2, 5}))
        ),
    )

    dnf1 = DNF([AND({uni1: uni4})])
    dnf2 = DNF([AND({uni2: uni5})])
    dnf3 = DNF([AND({uni3: uni6})])

    ref_possibles = {
        0: PossibleValues(),
        3: PossibleValues((uni1, uni2, uni3), [dnf1, dnf2, dnf3]),
    }
    eq1 = Equivalences({uni1, uni4}, None)
    eq2 = Equivalences({uni2, uni5}, {2, 1})
    eq3 = Equivalences({uni3, uni6}, {2, 5})

    ref_possibles[3].dnf_lookup_table = {
        uni1: eq1,
        uni2: eq2,
        uni3: eq3,
        uni4: eq1,
        uni5: eq2,
        uni6: eq3,
    }
    assert_possibles(var.possibles, ref_possibles)


def test_update_possible_values_10():
    var = Variadic()
    var.update_possible_values(
        PossibleValues(),
        PossibleValues(
            (uni1 := Uniadic(), uni2 := Uniadic(4), uni3 := Uniadic({1, 5}))
        ),
    )
    var.update_possible_values(
        PossibleValues(),
        PossibleValues(
            (uni4 := Uniadic({1, 4}), uni5 := Uniadic({1, 4}), uni6 := Uniadic({1, 4}))
        ),
    )

    dnf1 = DNF([AND({uni1: uni4})])
    dnf2 = DNF([AND({uni2: uni5})])
    dnf3 = DNF([AND({uni3: uni6})])

    ref_possibles = {
        0: PossibleValues(),
        3: PossibleValues((uni1, uni2, uni3), [dnf1, dnf2, dnf3]),
    }
    eq1 = Equivalences({uni1, uni4}, {1, 4})
    eq2 = Equivalences({uni2, uni5}, {4})
    eq3 = Equivalences({uni3, uni6}, {1})

    ref_possibles[3].dnf_lookup_table = {
        uni1: eq1,
        uni2: eq2,
        uni3: eq3,
        uni4: eq1,
        uni5: eq2,
        uni6: eq3,
    }
    assert_possibles(var.possibles, ref_possibles)


def test_update_possible_values_11():
    var = Variadic()
    var.update_possible_values(
        PossibleValues(),
        PossibleValues(
            (uni1 := Uniadic(), uni2 := Uniadic(2), uni3 := Uniadic({1, 5}))
        ),
    )
    var.update_possible_values(
        PossibleValues(),
        PossibleValues((uni4 := Uniadic(1), uni5 := Uniadic(2), uni6 := Uniadic(5))),
    )

    dnf1 = DNF([AND({uni1: uni4})])
    dnf2 = DNF([AND({uni2: uni5})])
    dnf3 = DNF([AND({uni3: uni6})])

    ref_possibles = {
        0: PossibleValues(),
        3: PossibleValues((uni1, uni2, uni3), [dnf1, dnf2, dnf3]),
    }
    eq1 = Equivalences({uni1, uni4}, {1})
    eq2 = Equivalences({uni2, uni5}, {2})
    eq3 = Equivalences({uni3, uni6}, {5})

    ref_possibles[3].dnf_lookup_table = {
        uni1: eq1,
        uni2: eq2,
        uni3: eq3,
        uni4: eq1,
        uni5: eq2,
        uni6: eq3,
    }
    assert_possibles(var.possibles, ref_possibles)


def test_update_possible_values_12():
    var = Variadic()
    var.update_possible_values(
        PossibleValues(),
        PossibleValues(
            (uni1 := Uniadic(), uni2 := Uniadic(2), uni3 := Uniadic({1, 5}))
        ),
    )
    var.update_possible_values(
        PossibleValues(),
        PossibleValues((uni4 := Uniadic(), uni5 := Uniadic(), uni6 := Uniadic())),
    )

    dnf1 = DNF([AND({uni1: uni4})])
    dnf2 = DNF([AND({uni2: uni5})])
    dnf3 = DNF([AND({uni3: uni6})])

    ref_possibles = {
        0: PossibleValues(),
        3: PossibleValues((uni1, uni2, uni3), [dnf1, dnf2, dnf3]),
    }
    eq1 = Equivalences({uni1, uni4}, None)
    eq2 = Equivalences({uni2, uni5}, {2})
    eq3 = Equivalences({uni3, uni6}, {1, 5})

    ref_possibles[3].dnf_lookup_table = {
        uni1: eq1,
        uni2: eq2,
        uni3: eq3,
        uni4: eq1,
        uni5: eq2,
        uni6: eq3,
    }
    assert_possibles(var.possibles, ref_possibles)


# @pytest.mark.skip("Known missing feature")
def test_possible_variadic_values_1():
    var = Variadic()
    repr = ShapeRepr(root=var)
    # var.update_possible_values([[], [None, 2, {1, 5}]])
    var.update_possible_values(
        PossibleValues(),
        PossibleValues(
            (uni1 := Uniadic(), uni2 := Uniadic(2), uni3 := Uniadic({1, 5}))
        ),
    )
    # var.update_possible_values([[None, None, None]])
    var.update_possible_values(
        PossibleValues((uni4 := Uniadic(), uni5 := Uniadic(), uni6 := Uniadic()))
    )

    # assert var.possible_values == [[None, 2, {1, 5}]]
    assert uni4.possible_values == uni1.possible_values is None
    assert uni5.possible_values == uni2.possible_values == {2}
    assert uni6.possible_values == uni3.possible_values == {1, 5}

    assert repr.get_shapes() == ["u1", 2, "u2"]


def test_possible_variadic_values_14():
    # V1 <- a, V2
    # V1 = [[], [4], [1, 5], [2, 6, 4]]
    # V2 = [[3], [6, None], [3, None, None]]

    # aV2 = [[a, 3], [a, 6, None], [a, 3, None, None]]
    # matching possibilities = [[2, 6, 4]]
    # possible values for a = {2}

    var1 = Variadic()
    repr1 = ShapeRepr(root=var1)
    # var1.update_possible_values({(), (4,), (1, 5), (2, 6, 4)})
    var1.update_possible_values(
        PossibleValues(),
        PossibleValues((Uniadic(4),)),
        PossibleValues((Uniadic(1), Uniadic(5))),
        PossibleValues((Uniadic(2), Uniadic(6), Uniadic(4))),
    )

    var2 = Variadic()
    repr2 = ShapeRepr(prefix=[Uniadic()], root=var2)
    var2.update_possible_values(
        PossibleValues((Uniadic(3),)),
        PossibleValues((Uniadic(6), Uniadic())),
        PossibleValues((Uniadic(3), Uniadic(), Uniadic())),
    )
    repr1.match(repr2)
    assert repr1.get_shapes() == repr2.get_shapes() == [2, 6, 4]


def test_possible_variadic_values_14_1():
    # V1 <- a, V2 where a.possible_values = {2}
    # V1 = [[], [4], [1, 5], [2, 6, 4]]
    # V2 = [[3], [6, None], [3, None, None]]

    # aV2 = [[{2}, 3], [{2}, 6, None], [{2}, 3, None, None]]
    # matching possibilities = [[2, 6, 4]]
    # possible values for a = {2}

    var1 = Variadic()
    # var1.update_possible_values({(), (4,), (1, 5), (2, 6, 4)})
    var1.update_possible_values(
        PossibleValues(),
        PossibleValues((Uniadic(4),)),
        PossibleValues((Uniadic(1), Uniadic(5))),
        PossibleValues((Uniadic(2), Uniadic(6), Uniadic(4))),
    )
    repr1 = ShapeRepr(root=var1)

    var2 = Variadic()
    var2.update_possible_values(
        PossibleValues((Uniadic(3),)),
        PossibleValues((Uniadic(6), Uniadic())),
        PossibleValues((Uniadic(3), Uniadic(), Uniadic())),
    )
    uni = Uniadic(2)
    repr2 = ShapeRepr(root=var2, prefix=[uni])
    repr1.match(repr2)

    assert repr1.get_shapes() == repr2.get_shapes() == [2, 6, 4]


def test_possible_variadic_values_14_2():
    # V1 <- a, V2 where a.possible_values = {2, 3}
    # V1 = [[], [4], [1, 5], [2, 6, 4]]
    # V2 = [[3], [6, None], [3, None, None]]

    # aV2 = [[{2, 3}, 3], [{2, 3}, 6, None], [{2, 3}, 3, None, None]]
    # matching possibilities = [[2, 6, 4]]
    # possible values for a = {2}

    var1 = Variadic()
    var1.update_possible_values(
        PossibleValues(),
        PossibleValues((Uniadic(4),)),
        PossibleValues((Uniadic(1), Uniadic(5))),
        PossibleValues((Uniadic(2), Uniadic(6), Uniadic(4))),
    )
    repr1 = ShapeRepr(root=var1)

    var2 = Variadic()
    var2.update_possible_values(
        PossibleValues((Uniadic(3),)),
        PossibleValues((Uniadic(6), Uniadic())),
        PossibleValues((Uniadic(3), Uniadic(), Uniadic())),
    )
    uni = Uniadic({2, 3})
    repr2 = ShapeRepr(root=var2, prefix=[uni])
    repr1.match(repr2)

    assert repr1.get_shapes() == repr2.get_shapes() == [2, 6, 4]


def test_possible_variadic_values_14_2_1():
    # V1 <- a, V2 where a.possible_values = {2, 3}
    # V1 = [[], [4], [1, 5], [{2, 10}, 6, 4]]
    # V2 = [[3], [6, None], [3, None, None]]

    # aV2 = [[{2, 3}, 3], [{2, 3}, 6, None], [{2, 3}, 3, None, None]]
    # matching possibilities = [[2, 6, 4]]
    # possible values for a = {2}

    var1 = Variadic()
    # var1.update_possible_values({(), (4,), (1, 5), (2, 6, 4)})
    var1.update_possible_values(
        PossibleValues(),
        PossibleValues((Uniadic(4),)),
        PossibleValues((Uniadic(1), Uniadic(5))),
        PossibleValues((Uniadic(2), Uniadic(6), Uniadic(4))),
    )
    repr1 = ShapeRepr(root=var1)

    var2 = Variadic()
    var2.update_possible_values(
        PossibleValues((Uniadic(3),)),
        PossibleValues((Uniadic(6), Uniadic())),
        PossibleValues((Uniadic(3), Uniadic(), Uniadic())),
    )
    uni = Uniadic({2, 3})
    repr2 = ShapeRepr(root=var2, prefix=[uni])
    repr1.match(repr2)

    assert repr1.get_shapes() == repr2.get_shapes() == [2, 6, 4]


def test_possible_variadic_values_14_2_2():
    # V1 <- a, V2 where a.possible_values = {2, 3, 10}
    # V1 = [[], [4], [1, 5], [{2, 10}, 6, 4]]
    # V2 = [[3], [6, None], [3, None, None]]

    # aV2 = [[{2, 3, 10}, 3], [{2, 3, 10}, 6, None], [{2, 3, 10}, 3, None, None]]
    # matching possibilities = [[{2, 10}, 6, 4]]
    # possible values for a = {2, 10}

    var1 = Variadic()
    var1.update_possible_values(
        PossibleValues(),
        PossibleValues((Uniadic(4),)),
        PossibleValues((Uniadic(1), Uniadic(5))),
        PossibleValues((Uniadic({2, 10}), Uniadic(6), Uniadic(4))),
    )
    repr1 = ShapeRepr(root=var1)

    var2 = Variadic()
    var2.update_possible_values(
        PossibleValues((Uniadic(3),)),
        PossibleValues((Uniadic(6), Uniadic())),
        PossibleValues((Uniadic(3), Uniadic(), Uniadic())),
    )
    uni = Uniadic({2, 3, 10})
    repr2 = ShapeRepr(root=var2, prefix=[uni])
    repr1.match(repr2)

    assert repr1.get_shapes() == repr2.get_shapes() == ["u1", 6, 4]
    assert uni.possible_values == {2, 10}


def test_possible_variadic_values_14_3():
    # V1 <- a, V2 where a.possible_values = {10}
    # V1 = [[], [4], [1, 5], [2, 6, 4]]
    # V2 = [[3], [6, None], [3, None, None]]

    # aV2 = [[{10}, 3], [{10}, 6, None], [{10}, 3, None, None]]
    # matching possibilities = NOT FOUND

    var1 = Variadic()
    var1.update_possible_values(
        PossibleValues(),
        PossibleValues((Uniadic(4),)),
        PossibleValues((Uniadic(1), Uniadic(5))),
        PossibleValues((Uniadic(2), Uniadic(6), Uniadic(4))),
    )
    repr1 = ShapeRepr(root=var1)

    var2 = Variadic()
    var2.update_possible_values(
        PossibleValues((Uniadic(3),)),
        PossibleValues((Uniadic(6), Uniadic())),
        PossibleValues((Uniadic(3), Uniadic(), Uniadic())),
    )
    uni = Uniadic(10)
    repr2 = ShapeRepr(root=var2, prefix=[uni])

    with pytest.raises(ValueError) as err_info:
        repr1.match(repr2)
    assert str(err_info.value) == "Incompatible possible values for Variadic!"


def test_possible_variadic_values_14_4():
    # V1 <- a, V2 where a.possible_values = {10, 20}
    # V1 = [[], [4], [1, 5], [2, 6, 4]]
    # V2 = [[3], [6, None], [3, None, None]]

    # aV2 = [[{10, 20}, 3], [{10, 20}, 6, None], [{10, 20}, 3, None, None]]
    # matching possibilities = NOT FOUND
    var1 = Variadic()
    var1.update_possible_values(
        PossibleValues(),
        PossibleValues((Uniadic(4),)),
        PossibleValues((Uniadic(1), Uniadic(5))),
        PossibleValues((Uniadic(2), Uniadic(6), Uniadic(4))),
    )
    repr1 = ShapeRepr(root=var1)

    var2 = Variadic()
    var2.update_possible_values(
        PossibleValues((Uniadic(3),)),
        PossibleValues((Uniadic(6), Uniadic())),
        PossibleValues((Uniadic(3), Uniadic(), Uniadic())),
    )
    uni = Uniadic({10, 20})
    repr2 = ShapeRepr(root=var2, prefix=[uni])

    with pytest.raises(ValueError) as err_info:
        repr1.match(repr2)
    assert str(err_info.value) == "Incompatible possible values for Variadic!"


def test_possible_variadic_values_15():
    # V1 <- a, V2, b, c

    # V1 = [[], [4], [1, 5], [2, 6, 4], [10, 11, 12, 13, 14, 15]]
    # V2 = [[3], [3, None], [11, None, None]]

    # aV2bc = [[None, 3, None, None], [None, 3, None, None, None],
    # [None, 11, None, None, None, None]]
    # matching possibilities = [[10, 11, 12, 13, 14, 15]]
    var1 = Variadic()
    var1.update_possible_values(
        PossibleValues(),
        PossibleValues((uni1 := Uniadic(4),)),
        PossibleValues((uni2 := Uniadic(1), uni3 := Uniadic(5))),
        PossibleValues((Uniadic(2), Uniadic(6), Uniadic(4))),
        PossibleValues(
            (
                Uniadic(10),
                Uniadic(11),
                Uniadic(12),
                Uniadic(13),
                Uniadic(14),
                Uniadic(15),
            )
        ),
    )
    repr1 = ShapeRepr(root=var1)

    var2 = Variadic()
    var2.update_possible_values(
        PossibleValues((Uniadic(3),)),
        PossibleValues((Uniadic(3), Uniadic())),
        PossibleValues((Uniadic(11), Uniadic(), Uniadic())),
    )
    uni1 = Uniadic()
    uni2 = Uniadic()
    uni3 = Uniadic()
    repr2 = ShapeRepr(root=var2, prefix=[uni1], suffix=[uni2, uni3])
    repr1.match(repr2)

    assert repr1.get_shapes() == repr2.get_shapes() == [10, 11, 12, 13, 14, 15]


def test_possible_variadic_values_16():
    # V1 <- a, V2

    # a = 2 | 6 | 4
    # V1 = [[3], [2, 1], [1, 2, 3]]

    # In this case, "a" should be 2 as it is the only possibility.
    # V1 should also be determined with value of [2, 1]

    # Final result should be [2, 1]
    v2 = Variadic()
    repr2 = ShapeRepr(prefix=[Uniadic({2, 6, 4})], root=v2, suffix=[])

    var = Variadic()
    var.update_possible_values(
        PossibleValues((Uniadic(3),)),
        PossibleValues((Uniadic(2), Uniadic(1))),
        PossibleValues((Uniadic(1), Uniadic(2), Uniadic(3))),
    )
    repr1 = ShapeRepr(root=var)

    repr1.match(repr2)
    assert repr1[0].value == 2
    assert repr1.get_shapes() == [2, 1]


def test_possible_variadic_values_17():
    # V1 <- V2

    # V2 = [[1, {2, 3}, None]]
    # V1 = [[3], [2, 1], [{1, 2, 3}, 2, 3]]

    # Final Result => [1, 2, 3]

    var1 = Variadic()
    var2 = Variadic()

    var1.update_possible_values(
        PossibleValues(()),
        PossibleValues((Uniadic(1), Uniadic({2, 3}), Uniadic())),
    )
    var2.update_possible_values(
        PossibleValues((Uniadic(3),)),
        PossibleValues((Uniadic(2), Uniadic(1))),
        PossibleValues((Uniadic({1, 2, 3}), Uniadic(2), Uniadic(3))),
    )

    repr1 = ShapeRepr(prefix=[], root=var1, suffix=[])
    repr2 = ShapeRepr(prefix=[], root=var2, suffix=[])

    repr1.match(repr2)
    assert repr2.get_shapes() == repr1.get_shapes() == [1, 2, 3]


def test_possible_variadic_values_18():
    # [V1] <- [V2, a]
    # a = {2}
    # V1 = [[], [1, 4], [4, 2, 3, 2]]

    # Final Result => [4, 2, 3, 2]

    repr1 = ShapeRepr(prefix=[], root=Variadic(), suffix=[Uniadic(2)])

    var = Variadic()
    var.update_possible_values(
        PossibleValues(),
        PossibleValues((Uniadic(1), Uniadic(4))),
        PossibleValues((Uniadic(4), Uniadic(2), Uniadic(3), Uniadic(2))),
    )

    repr2 = ShapeRepr(prefix=[], root=var, suffix=[])

    repr1.match(repr2)
    assert repr1.get_shapes() == [4, 2, 3, 2]
    assert repr2.get_shapes() == [4, 2, 3, 2]


def test_possible_variadic_values_19():
    # [V1] <- [V2, a]
    # a = {2}
    # V1 = [[], [1, 4], [4, 2, 3, {2, 3}]]

    # Final Result => [4, 2, 3, 2]

    repr1 = ShapeRepr(prefix=[], root=Variadic(), suffix=[Uniadic(2)])

    var = Variadic()
    var.update_possible_values(
        PossibleValues(),
        PossibleValues((Uniadic(1), Uniadic(4))),
        PossibleValues((Uniadic(4), Uniadic(2), Uniadic(3), Uniadic({2, 3}))),
    )
    repr2 = ShapeRepr(prefix=[], root=var, suffix=[])

    repr1.match(repr2)
    assert repr1.get_shapes() == [4, 2, 3, 2]
    assert repr2.get_shapes() == [4, 2, 3, 2]


def test_possible_variadic_values_20():
    # [V1] <- [V2, a]
    # a = {2, 5}
    # V1 = [[], [1, 4], [5, 2, 3, {2, 3}]]

    # Final Result => [5, 2, 3, 2]

    repr1 = ShapeRepr(prefix=[], root=Variadic(), suffix=[Uniadic({2, 5})])

    var = Variadic()
    var.update_possible_values(
        PossibleValues(),
        PossibleValues((Uniadic(1), Uniadic(4))),
        PossibleValues((Uniadic(5), Uniadic(2), Uniadic(3), Uniadic({2, 3}))),
    )
    repr2 = ShapeRepr(prefix=[], root=var, suffix=[])

    repr1.match(repr2)
    assert repr1.get_shapes() == [5, 2, 3, 2]
    assert repr2.get_shapes() == [5, 2, 3, 2]


def test_possible_variadic_values_21():
    # [V1] <- [V2]
    # V1 = [[{2, 3}, {2, 3}], [{4,5}, {6, 7}, {8, 9}]]
    # V2 = [[1, {2,3}], [5, 6, 9]]

    var1 = Variadic()
    var1.update_possible_values(
        PossibleValues((Uniadic({2, 3}), Uniadic({2, 3}))),
        PossibleValues((Uniadic({4, 5}), Uniadic({6, 7}), Uniadic({8, 9}))),
    )
    repr1 = ShapeRepr(prefix=[], root=var1, suffix=[])

    var2 = Variadic()
    var2.update_possible_values(
        PossibleValues((Uniadic(1), Uniadic({2, 3}))),
        PossibleValues((Uniadic(5), Uniadic(6), Uniadic(9))),
    )

    repr2 = ShapeRepr(prefix=[], root=var2, suffix=[])

    repr1.match(repr2)
    assert repr1.get_shapes() == [5, 6, 9]
    assert repr2.get_shapes() == [5, 6, 9]


def test_possible_variadic_values_22():
    # [V1] <- [V2]
    # V1 = [[{2, 3}, {2, 3}], [{4,5}, {6, 7}, {8, 9}]]
    # V2 = [[1, {2,3}], [5, 6, 9==uni1]]

    # Final Result

    # V1 = [5, 6, 9]
    var1 = Variadic()
    uniadics = (Uniadic({4, 5}), Uniadic({6, 7}), _uni := Uniadic({8, 9}))
    dnf = DNF([AND({(uni1 := Uniadic()): _uni})])
    var1.update_possible_values(
        PossibleValues((Uniadic({2, 3}), Uniadic({2, 3}))),
        PossibleValues(uniadics, [dnf]),
    )
    repr1 = ShapeRepr(prefix=[], root=var1, suffix=[])

    var2 = Variadic()
    var2.update_possible_values(
        PossibleValues((Uniadic(1), Uniadic({2, 3}))),
        PossibleValues((Uniadic(5), Uniadic(6), Uniadic(9))),
    )

    repr2 = ShapeRepr(prefix=[], root=var2, suffix=[])

    repr1.match(repr2)
    assert repr2.get_shapes() == repr1.get_shapes() == [5, 6, 9]
    assert uni1.possible_values == {9}


# @pytest.mark.skip("Known Bug")
def test_possible_variadic_values_23():
    # [aV1] <- [V2b]

    # a = {2, 3, 4}
    # b = {5, 6}
    # V1 = [[], [5], [{9, 1}, {6, 7}]]
    # V2 = [[], [{7, 8}], [3, {9, 10}]]

    # * V1 and V2 should not be equal to [] as {2, 3, 4} & {5, 6} = None

    # * V1 can be equal to 5 because {5} & {5, 6} is not equal to None,
    # However, if V1 is equal to 5. This would also mean V2 should be
    # equal to {7, 8}, However possibilites of a {2, 3, 4} & {7, 8} is None.
    # Therefore, len(V1) cannot be equal to 1

    # * Only one possibiliy is left. V1 is equal to [{9, 10}, {6, 7}] and
    # V2 is equal to [3, {9, 1}].

    # Since {6, 7} & {5, 6} = {6}, then b == 6 (V1[1] should be equal to b)

    # {2, 3, 4} & {3} = {3}, then a == 3 (V2[0] should be equal to a)

    # In that case, first element of V1 should be equal to second element of
    # V2. Therefore, {9, 10} & {9, 1} = {9}, Then V1[0] == V2[1] == 9

    # Combining all, Final shape should be [3, 9, 6]

    a = Uniadic({2, 3, 4})
    b = Uniadic({5, 6})
    V1 = Variadic()
    V1.update_possible_values(
        PossibleValues(),
        PossibleValues((Uniadic(5),)),
        PossibleValues((Uniadic({9, 6}), Uniadic({6, 7}))),
    )

    V2 = Variadic()
    V2.update_possible_values(
        PossibleValues(),
        PossibleValues((Uniadic({7, 8}),)),
        PossibleValues((Uniadic(3), Uniadic({9, 10}))),
    )

    repr1 = ShapeRepr(prefix=[a], root=V1, suffix=[])
    repr2 = ShapeRepr(prefix=[], root=V2, suffix=[b])

    repr1.match(repr2)
    assert repr1.get_shapes() == [3, 9, 6]
    assert repr2.get_shapes() == [3, 9, 6]


@pytest.mark.skip(
    "Consider to add a more general match operation \
        which handles handle_numerical_incompatibility method's work"
)
def test_possible_variadic_values_23_1():
    # [aV1] <- [V2b]

    # a = {2, 3, 4}
    # b = {5, 6}
    # V1 = [[], [5], [{9, 1}, {6, 7}]]
    # V2 = [[], [{7, 8}], [3, {9, 10}]]

    # * V1 and V2 should not be equal to [] as {2, 3, 4} & {5, 6} = None

    # * V1 can be equal to 5 because {5} & {5, 6} is not equal to None,
    # However, if V1 is equal to 5. This would also mean V2 should be
    # equal to {7, 8}, However possibilites of a {2, 3, 4} & {7, 8} is None.
    # Therefore, len(V1) cannot be equal to 1

    # * Only one possibiliy is left. V1 is equal to [{9, 1}, {6, 7}] and
    # V2 is equal to [3, {9, 10}].

    # Since {6, 7} & {5, 6} = {6}, then b == 6 (V1[1] should be equal to b)

    # {2, 3, 4} & {3} = {3}, then a == 3 (V2[0] should be equal to a)

    # In that case, first element of V1 should be equal to second element of
    # V2. Therefore, {9, 10} & {9, 1} = {9}, Then V1[0] == V2[1] == 9

    # Combining all, Final shape should be [3, 9, 6]

    a = Uniadic({2, 3, 4})
    b = Uniadic()
    V1 = Variadic()
    V1.update_possible_values(
        PossibleValues(),
        PossibleValues((Uniadic(5),)),
        PossibleValues((Uniadic({9, 6}), Uniadic({6, 7}))),
    )

    V2 = Variadic()
    V2.update_possible_values(
        PossibleValues(),
        PossibleValues((Uniadic({7, 8}),)),
        PossibleValues((Uniadic(3), Uniadic({9, 10}))),
    )

    repr1 = ShapeRepr(prefix=[a], root=V1, suffix=[])
    repr2 = ShapeRepr(prefix=[], root=V2, suffix=[b])

    updates = repr1.match(repr2)

    updates |= b.update_possible_values({5, 6})

    ConstraintSolver()(updates)

    assert repr1.get_shapes() == [3, 9, 6]
    assert repr2.get_shapes() == [3, 9, 6]


# @pytest.mark.skip("Known Bug")
def test_possible_variadic_values_24():
    # [aV1] <- [V2b]
    # a = {2, 3, 4}
    # V1 = [[], [{5, 7}]]

    # V2 =  [[], [{2, 7, 8}]]
    # b = {5, 6}

    a = Uniadic({2, 3, 4})
    b = Uniadic({5, 6})
    V1 = Variadic()
    V1.update_possible_values(PossibleValues(), PossibleValues((Uniadic({5, 7}),)))

    V2 = Variadic()
    V2.update_possible_values(PossibleValues(), PossibleValues((Uniadic({2, 7, 8}),)))

    repr1 = ShapeRepr(prefix=[a], root=V1, suffix=[])
    repr2 = ShapeRepr(prefix=[], root=V2, suffix=[b])

    repr1.match(repr2)
    assert repr1.get_shapes() == [2, 5]
    assert repr2.get_shapes() == [2, 5]


def test_possible_variadic_values_26():
    # [aV1] <- [V2b]
    # a = 2 | 3 | 4
    # V1 = None

    # V2 = None
    # b = 5 | 6

    # Result should be [a, V3, b]

    a = Uniadic({2, 3, 4})
    b = Uniadic({5, 6})

    repr1 = ShapeRepr(prefix=[a], root=Variadic(), suffix=[])
    repr2 = ShapeRepr(prefix=[], root=Variadic(), suffix=[b])

    repr1.match(repr2)
    assert repr1.get_shapes() == ["u1", "(V1, ...)", "u2"]
    assert repr2.get_shapes() == ["u1", "(V1, ...)", "u2"]


def test_possible_variadic_values_27():
    # [V1] <- [aa]
    # a = 2 | 3 | 4
    # V1 = [None, [None], [{2, 3}, {2, 4}]]

    # a should be 2 as {2, 3, 4} & {2, 3} & {2, 4} == {2}

    # Result should be [2, 2]

    a = Uniadic({2, 3, 4})
    v1 = Variadic()
    # v1.update_possible_values({(), (None,), (2, 2), (2, 4), (3, 2), (3, 4)})
    v1.update_possible_values(
        PossibleValues(),
        PossibleValues((Uniadic(),)),
        PossibleValues((Uniadic({2, 3}), Uniadic({2, 4}))),
    )

    repr1 = ShapeRepr(prefix=[a, a], root=None, suffix=[])
    repr2 = ShapeRepr(prefix=[], root=v1, suffix=[])

    repr2.match(repr1)
    assert repr1.get_shapes() == [2, 2]
    assert repr2.get_shapes() == [2, 2]


def test_possible_variadic_values_28():
    # [V1] <- [V2]
    # V1 = [[{2, 3, 4}], [{1, 2}, {2, 3}, {3, 4}]]
    # V2 = [[{5, 7}], [{2, 6}, {3, 7}, {1, 4, 8}]]
    # Result should be [2, 3, 4]

    v1 = Variadic()
    v1.update_possible_values(
        PossibleValues((Uniadic({2, 3, 4}),)),
        PossibleValues((Uniadic({1, 2}), Uniadic({2, 3}), Uniadic({3, 4}))),
    )

    v2 = Variadic()
    v2.update_possible_values(
        PossibleValues((Uniadic({5, 7}),)),
        PossibleValues((Uniadic({2, 6}), Uniadic({3, 7}), Uniadic({1, 4, 8}))),
    )

    repr1 = ShapeRepr(prefix=[], root=v2, suffix=[])
    repr2 = ShapeRepr(prefix=[], root=v1, suffix=[])

    repr2.match(repr1)
    assert repr1.get_shapes() == [2, 3, 4]
    assert repr2.get_shapes() == [2, 3, 4]


def test_possible_variadic_values_29():
    # V1 <- V2

    # V1 = [[3], [2, 1], [1, 2, 3]]

    v2 = Variadic()
    v2.update_possible_values(
        PossibleValues((uni1 := Uniadic({3, 4}),)),
        PossibleValues((uni2 := Uniadic(2), uni3 := Uniadic({1, 5}))),
        PossibleValues((uni4 := Uniadic(1), uni5 := Uniadic(2), uni6 := Uniadic())),
    )

    repr1 = ShapeRepr(root=v2)

    var = Variadic()
    var.update_possible_values(
        PossibleValues((uni7 := Uniadic(3),)),
        PossibleValues((uni8 := Uniadic({2, 4}), uni9 := Uniadic(1))),
        PossibleValues((uni10 := Uniadic(), uni11 := Uniadic(2), uni12 := Uniadic(3))),
    )

    repr2 = ShapeRepr(root=var)
    repr1.match(repr2)

    dnf1 = DNF([AND({uni1: uni7})])
    dnf2 = DNF([AND({uni2: uni8})])
    dnf3 = DNF([AND({uni3: uni9})])
    dnf4 = DNF([AND({uni4: uni10})])
    dnf5 = DNF([AND({uni5: uni11})])
    dnf6 = DNF([AND({uni6: uni12})])

    ref_possibles = {
        1: PossibleValues((uni1,), [dnf1]),
        2: PossibleValues((uni2, uni3), [dnf2, dnf3]),
        3: PossibleValues((uni4, uni5, uni6), [dnf4, dnf5, dnf6]),
    }
    eq1 = Equivalences({uni1, uni7}, {3})
    eq2 = Equivalences({uni2, uni8}, {2})
    eq3 = Equivalences({uni3, uni9}, {1})
    eq4 = Equivalences({uni4, uni10}, {1})
    eq5 = Equivalences({uni5, uni11}, {2})
    eq6 = Equivalences({uni6, uni12}, {3})

    ref_possibles[1].dnf_lookup_table = {uni1: eq1, uni7: eq1}
    ref_possibles[2].dnf_lookup_table = {
        uni2: eq2,
        uni3: eq3,
        uni8: eq2,
        uni9: eq3,
    }
    ref_possibles[3].dnf_lookup_table = {
        uni4: eq4,
        uni5: eq5,
        uni6: eq6,
        uni10: eq4,
        uni11: eq5,
        uni12: eq6,
    }
    assert_possibles(v2.possibles, ref_possibles)


# @pytest.mark.skip("Known missing feature")
def test_impossible():
    m1 = Add()
    m1.set_types(left=Tensor, right=Tensor)
    m1.set_shapes(left=[1, 1])
    m2 = Add()
    m2.set_types(left=Tensor, right=Tensor)
    m2.set_shapes(output=["a", "b"])
    model = Model()
    model.extend(m1, left="left", right="right", output="o1")
    model.extend(m2, left="o1", right="right", output="output")

    ref_shapes: dict[str, list] = {
        "left": [1, 1],
        "right": ["(V2, ...)"],
        "o1": ["a", "b"],
        "output": ["a", "b"],
    }

    assert_shapes(model, ref_shapes)


# @pytest.mark.skip("Known missing feature")
def test_less_impossible_yet_not_possible():
    m1 = Add()
    m1.set_types(left=Tensor, right=Tensor)
    m1.set_shapes(left=[1, 1])
    m2 = Add()
    m2.set_types(left=Tensor, right=Tensor)
    m2.set_shapes(output=[2, 3])
    model = Model()
    model.extend(m1, left="left", right="right", output="o1")
    model.extend(m2, left="o1", right="right", output="output")

    ref_shapes = {"left": [1, 1], "right": [2, 3], "o1": [2, 3], "output": [2, 3]}

    assert_shapes(model, ref_shapes)


# @pytest.mark.skip("Known missing feature")
def test_impossible_variadic_values_1():
    # [aV1] <- [V2b]
    V1 = Variadic()
    a = Uniadic({5, 6})
    b = Uniadic({3})
    repr1 = ShapeRepr(root=V1, prefix=[b])

    V1.update_possible_values(
        PossibleValues((Uniadic(5),)), PossibleValues((Uniadic(9), Uniadic({6, 7})))
    )

    V2 = Variadic()
    repr2 = ShapeRepr(root=V2, suffix=[a])

    V2.update_possible_values(PossibleValues((Uniadic(3), Uniadic({9, 10}))))

    repr1.match(repr2)
    assert repr1.get_shapes() == [3, 9, 6]
    assert repr2.get_shapes() == [3, 9, 6]


def test_update_possible_values_8_possibilities_2_1():
    var = Variadic()

    uniadics1 = (uni1_1 := Uniadic({1, 4, 7}), uni1_2 := Uniadic({5, 6}))
    dnf1 = DNF(
        [
            AND({uni1_1: 1, uni1_2: 5}),
            AND({uni1_1: 4, uni1_2: 5}),
            AND({uni1_1: 1, uni1_2: 6}),
            AND({uni1_1: 4, uni1_2: 6}),
        ]
    )
    var.update_possible_values(PossibleValues(), PossibleValues(uniadics1, [dnf1]))

    uniadics2 = (uni2_1 := Uniadic({1, 4}), uni2_2 := Uniadic({5, 6}))
    dnf2 = DNF([AND({uni2_1: 4, uni2_2: 6}), AND({uni2_1: 1, uni2_2: 5})])
    var.update_possible_values(PossibleValues(), PossibleValues(uniadics2, [dnf2]))

    assert var.possibles is not None
    assert (
        var.possibles[2].dnf_lookup_table[uni2_1]
        == var.possibles[2].dnf_lookup_table[uni1_1]
        == Equivalences({uni1_1, uni2_1}, values={1, 4})
    )
    assert (
        var.possibles[2].dnf_lookup_table[uni2_2]
        == var.possibles[2].dnf_lookup_table[uni1_2]
        == Equivalences({uni1_2, uni2_2}, values={5, 6})
    )


def test_update_possible_values_8_possibilities_2_2():
    # resulting_cnf = [
    #     # {(uni2_1, 4), (uni2_2, 6), (uni1_1, 1), (uni1_2, 5)},
    #     # {(uni2_1, 4), (uni2_2, 6), (uni1_1, 4), (uni1_2, 5)},
    #     # {(uni2_1, 4), (uni2_2, 6), (uni1_1, 1), (uni1_2, 6)},
    #     {(uni2_1, 1), (uni2_2, 5), (uni1_1, 1), (uni1_2, 5)},
    #     {(uni2_1, 4), (uni2_2, 6), (uni1_1, 4), (uni1_2, 6)},
    #     # {(uni2_1, 1), (uni2_2, 5), (uni1_1, 4), (uni1_2, 5)},
    #     # {(uni2_1, 1), (uni2_2, 5), (uni1_1, 1), (uni1_2, 6)},
    #     # {(uni2_1, 1), (uni2_2, 5), (uni1_1, 4), (uni1_2, 6)},
    # ]
    var = Variadic()

    uniadics1 = (uni1_1 := Uniadic({1, 4, 7}), uni1_2 := Uniadic({5, 6}))
    dnf1 = DNF(
        [
            AND({uni1_1: 1, uni1_2: 5}),
            AND({uni1_1: 4, uni1_2: 5}),
            AND({uni1_1: 7, uni1_2: 6}),
            AND({uni1_1: 4, uni1_2: 6}),
        ]
    )
    var.update_possible_values(PossibleValues(), PossibleValues(uniadics1, [dnf1]))
    uniadics2 = (uni2_1 := Uniadic({1, 4}), uni2_2 := Uniadic({5, 6}))
    dnf2 = DNF([AND({uni2_1: 1, uni2_2: 6}), AND({uni2_1: 1, uni2_2: 5})])
    # dnf3 = DNF([AND({uni2_1: 1, uni2_2: 6})]) ->

    var.update_possible_values(
        # PossibleValues(),
        PossibleValues(uniadics2, [dnf2])
    )

    assert var.possibles is None
    assert uni1_1.possible_values == uni2_1.possible_values == {1}
    assert uni1_2.possible_values == uni2_2.possible_values == {5}


def test_update_possible_values_2_3():
    var = Variadic()
    repr = ShapeRepr(root=var)
    my_uni = Uniadic()

    uniadics = (
        uni1 := Uniadic({1, 4}),
        uni2 := Uniadic({5, 6}),
        uni3 := Uniadic({7, 8}),
    )
    dnf1 = DNF([AND({uni1: 1, uni2: 5}), AND({uni1: 4, uni3: 8})])
    dnf2 = DNF([AND({uni2: my_uni})])
    pos_val = PossibleValues(uniadics, [dnf1, dnf2])
    var.update_possible_values(pos_val)

    my_uni.set_value(6)
    updates = Updates()
    updates.add(my_uni)
    solver = ConstraintSolver()
    solver(updates)

    assert uni1.possible_values == {4}
    assert uni2.possible_values == {6}
    assert uni3.possible_values == {8}
    # assert var.possibles is None
    assert repr.get_shapes() == [4, 6, 8]


def test_update_possible_values_30():
    var = Variadic()
    repr = ShapeRepr(root=var)

    my_uni = Uniadic()

    uniadics = (
        uni1 := Uniadic({1, 4}),
        uni2 := Uniadic({5, 6}),
        uni3 := Uniadic({7, 8}),
    )
    dnf1 = DNF([AND({uni1: 1, uni2: 5}), AND({uni1: 4, uni3: 8})])
    dnf2 = DNF([AND({uni2: my_uni})])
    pos_val = PossibleValues(uniadics, [dnf1, dnf2])

    var.update_possible_values(
        PossibleValues(), PossibleValues((Uniadic(), Uniadic())), pos_val
    )
    var.match(new_prefix=[Uniadic(4), Uniadic({5, 6}), Uniadic({7, 8})])
    assert var.possibles is None
    assert uni1.possible_values == {4}
    assert my_uni.possible_values == uni2.possible_values == {5, 6}
    assert uni3.possible_values == {8}

    assert var.possibles is None
    assert repr.get_shapes() == [4, "u1", 8]


def test_update_possible_values_31():
    var = Variadic()
    repr = ShapeRepr(root=var)

    my_uni = Uniadic()

    uniadics = (
        uni1 := Uniadic({1, 4}),
        uni2 := Uniadic({5, 6}),
        uni3 := Uniadic({7, 8}),
    )
    dnf1 = DNF([AND({uni1: 1, uni2: 5}), AND({uni1: 4, uni3: 8})])
    dnf2 = DNF([AND({uni2: my_uni})])

    var.update_possible_values(PossibleValues(uniadics, [dnf1, dnf2]))

    # Check vars_dicts are filled with right lengths
    # assert uni1.metadata.vars_dict == {var: {3}}
    # assert uni2.metadata.vars_dict == {var: {3}}
    # assert uni3.metadata.vars_dict == {var: {3}}
    # assert my_uni.metadata.vars_dict == {var: {3}}
    root = repr.root  # var is updated by extract variadic
    assert root is not None
    assert uni1.metadata.vars_dict[root] == {0}
    assert uni2.metadata.vars_dict[root] == {0}
    assert uni3.metadata.vars_dict[root] == {0}
    assert my_uni.metadata.vars_dict[root] == {0}

    var2 = Variadic()
    var2.update_possible_values(PossibleValues((Uniadic(), Uniadic(), Uniadic())))
    var2.match(new_root=root)

    my_uni.set_value(6)
    updates = Updates()
    updates.add(my_uni)
    solver = ConstraintSolver()
    solver(updates)

    assert uni1.possible_values == {4}
    assert uni2.possible_values == {6}
    assert uni3.possible_values == {8}

    # Check vars_dicts are emptied since no possibilities are left
    assert uni1.metadata.vars_dict == {}
    assert uni2.metadata.vars_dict == {}
    assert uni3.metadata.vars_dict == {}
    assert my_uni.metadata.vars_dict == {}

    # assert var.possibles is None
    assert repr.get_shapes() == [4, 6, 8]


def test_update_possible_values2_removed_zero_length():
    var = Variadic()
    repr = ShapeRepr(root=var)

    var.update_possible_values(
        PossibleValues(),
        PossibleValues((uni1_1 := Uniadic({1, 4}), uni1_2 := Uniadic(5))),
    )
    var.update_possible_values(
        PossibleValues(), PossibleValues((uni2_1 := Uniadic(4), uni2_2 := Uniadic()))
    )
    var.update_possible_values(PossibleValues((Uniadic(), Uniadic())))

    assert uni1_1.possible_values == uni2_1.possible_values == {4}
    assert uni1_2.possible_values == uni2_2.possible_values == {5}

    assert var.possibles is None
    assert repr.get_shapes() == [4, 5]


def test_connect_shapes():
    relu1 = Relu()
    relu2 = Relu()
    relu3 = Relu()
    relu1.set_shapes(input=[5, 7, ("Var1", ...)])
    relu2.set_shapes(input=[("Var1", ...), 5, 7])
    relu3.set_shapes(input=[5, 7])

    model = Model()
    model |= relu1
    model |= relu2
    model.merge_connections(relu1.input, relu2.input)
    model |= relu3.connect(input="input", output=relu1.input)

    assert model.shapes["input"] == [5, 7]


def test_remove_variadic():
    model = Model()
    sig2 = Sigmoid()
    model += sig2.connect(input="input", output="output")
    model.set_shapes(input=[7, ("Var1", ...), 5])
    with pytest.raises(Exception) as err_info:
        # model.shape_map["output"].remove_variadic([Uniadic(5)])
        data = model.conns.get_data("output")
        assert data.is_tensor
        data_shape = data.shape
        assert data_shape is not None
        next(iter(data_shape.reprs)).remove_variadic([Uniadic(5)])

    assert str(err_info.value) == "Requires minimum of 2 dimensionality, got 1."


# @pytest.mark.skip(reason= "Known Bugs")
def test_bcast_left():
    model = Add()
    model.set_shapes(
        left=[2, 1, ("V1", ...)],
        right=[2, 1, ("V2", ...)],
    )

    assert model.output.metadata.is_tensor
    data_shape = model.output.metadata.shape
    assert data_shape is not None
    assert data_shape.get_shapes() == [2, "u1", "(V1, ...)"]
    model.set_shapes(left=[2, 1], right=[2, 1, 3])
    assert data_shape.get_shapes() == [2, 2, 3]


def test_bcast_right():
    model = MatrixMultiply()
    model.set_shapes(
        output=[("V1", ...), "x", "k"],
        left=["y", "l"],
        right=[("V2", ...), "z", "m"],
    )


def test_bcast_left_2():
    model = Add()
    model.set_shapes(
        output=["a", ("V1", ...)],
        left=[3, 1, ("V2", ...)],
        right=[2, ("V3", ...)],
    )

    assert model.output.metadata.is_tensor
    data_shape = model.output.metadata.shape

    assert data_shape is not None
    assert data_shape.reprs[0][0].possible_values == {3, 2}


@pytest.mark.skip(
    reason="could be testable after set_shapes will accept possible values for uniadics"
)
def test_bcast_left_3():
    model = Add()
    model.set_shapes(
        {
            model.output: ["a", ("V1", ...)],
            model.left: [{3, 4, 5}, 1, ("V2", ...)],  # type: ignore
            model.right: [{2, 3}, ("V3", ...)],  # type: ignore
        }
    )

    assert model.output.metadata.is_tensor
    data_shape = model.output.metadata.shape

    assert data_shape is not None
    assert data_shape.reprs[0][0].possible_values == {2, 3, 4, 5}


# @pytest.mark.skip(reason="Known bug")
def test_bcast_4():
    model = Model()
    add1 = Add()
    add2 = Add()
    add1.set_types(left=Tensor, right=Tensor)
    add2.set_types(left=Tensor, right=Tensor)
    add1.set_cin("left")
    add2.set_cin("left")
    add1.set_shapes(left=[1, 1])

    add2.set_shapes(output=["a", "b"])

    # add2.set_shapes({
    #     "output": [3, 5]
    # })

    model |= add1.connect()
    model |= add2.connect(left=add1.output, right=add1.right)
    ref_shapes: dict[str, list] = {
        "$_Add_0_output": ["a", "b"],
        "$_Add_1_output": ["a", "b"],
        "$input": [1, 1],
        "$right_1": ["(V1, ...)"],
    }
    # ref_shapes = {
    #     '$_Add_0_output': [3, 5],
    #     '$_Add_1_output': [3, 5],
    #     '$input': [1, 1],
    #     '$right_1': ['(V1, ...)']
    # }
    assert_shapes(model, ref_shapes)


def test_bcast_4_len1():
    model = Model()
    add1 = Add()
    add2 = Add()
    add1.set_types(left=Tensor, right=Tensor)
    add2.set_types(left=Tensor, right=Tensor)
    add1.set_cin("left")
    add2.set_cin("left")

    add1.set_shapes(left=[1])

    add2.set_shapes(output=["a"])

    model |= add1.connect()
    model |= add2.connect(left=add1.output, right=add1.right)
    ref_shapes: dict[str, list] = {
        "$_Add_0_output": ["a"],
        "$_Add_1_output": ["a"],
        "$input": [1],
        "$right_1": ["(V1, ...)"],
    }
    assert_shapes(model, ref_shapes)


def test_bcast_pos_val_1():
    model = Model()
    add1 = Add()
    add2 = Add()
    add1.set_types(left=Tensor, right=Tensor)
    add2.set_types(left=Tensor, right=Tensor)
    add1.set_cin("left")
    add2.set_cin("left")

    add1.set_shapes(left=[1, 1])
    add2.set_shapes(right=[1, 1], output=["a", "b"])

    model |= add1
    model |= add2.connect(left=add1.output)
    ref_shapes: dict[str, list] = {
        "$_Add_0_output": ["u1", "u2"],
        "$_Add_1_output": ["u1", "u2"],
        "$input": [1, 1],
        "$right_0": ["(V1, ...)"],
        "$right_1": [1, 1],
    }
    assert_shapes(model, ref_shapes)


def test_var_empty_pos():
    v1 = Variadic()
    with pytest.raises(ValueError) as err_info:
        v1.update_possible_values()

    assert str(err_info.value) == "Variadic possible values could not be empty!"


def test_bcast_align_match():
    model = Add()
    model.set_shapes(left=[3, 4, 5, 1], right=[1, 7])
    ref_shapes = {
        "left": [3, 4, 5, 1],
        "right": [1, 7],
        "output": [3, 4, 5, 7],
    }
    assert_shapes(model, ref_shapes)


def test_bcast_uniadics():
    from mithril.framework.constraints import bcast_uniadics

    left = ShapeRepr(root=Variadic(), suffix=[Uniadic(), Uniadic()])
    right = ShapeRepr(root=Variadic())
    output = ShapeRepr(root=Variadic(), suffix=[Uniadic(1), Uniadic(2)])

    bcast_uniadics(output, left, right, 0)
    uni_cache: dict[UniadicRecord, str] = {}
    var_cache: dict[Variadic, str] = {}
    assert left.get_shapes(uni_cache, var_cache) == ["(V1, ...)", 1, "u1"]
    assert right.get_shapes(uni_cache, var_cache) == ["(V2, ...)"]
    assert output.get_shapes(uni_cache, var_cache) == ["(V3, ...)", 1, 2]


def test_bcast_align():
    from mithril.framework.constraints import bacast_align_output

    left = ShapeRepr(root=Variadic(), suffix=[Uniadic(), Uniadic()])
    right = ShapeRepr(root=Variadic())
    output = ShapeRepr(root=Variadic(), suffix=[Uniadic(1), Uniadic(2)])

    bacast_align_output(output, left, right, 0)
    uni_cache: dict[UniadicRecord, str] = {}
    var_cache: dict[Variadic, str] = {}
    assert left.get_shapes(uni_cache, var_cache) == ["(V1, ...)", "u1", "u2"]
    assert right.get_shapes(uni_cache, var_cache) == ["(V2, ...)"]
    assert output.get_shapes(uni_cache, var_cache) == ["(V3, ...)", 1, 2]


def test_ands_equality_1():
    a = Uniadic()
    b = Uniadic()
    c = Uniadic()
    d = Uniadic()
    e = Uniadic()
    f = Uniadic()
    g = Uniadic()
    h = Uniadic()

    and1 = AND({a: b, c: d})
    and2 = AND({e: f, g: h})

    a.metadata = e.metadata
    b.metadata = f.metadata

    c.metadata = g.metadata
    d.metadata = h.metadata

    assert and1.is_equal(and2)


def test_ands_equality_2():
    a = Uniadic()
    b = Uniadic()
    c = Uniadic()
    d = Uniadic()
    e = Uniadic()
    f = Uniadic()
    g = Uniadic()
    h = Uniadic()
    i = Uniadic()
    j = Uniadic()

    and1 = AND({a: b, c: d, i: 1})
    and2 = AND({e: f, g: h, j: 1})

    a.metadata = e.metadata
    b.metadata = f.metadata

    c.metadata = g.metadata
    d.metadata = h.metadata

    i.metadata = j.metadata

    assert and1.is_equal(and2)


def test_ands_inequality_1():
    a = Uniadic()
    b = Uniadic()
    c = Uniadic()
    d = Uniadic()
    e = Uniadic()
    f = Uniadic()
    g = Uniadic()
    h = Uniadic()
    i = Uniadic()
    j = Uniadic()

    and1 = AND({a: b, c: d, i: 1})
    and2 = AND({e: f, g: h, j: 2})

    a.metadata = e.metadata
    b.metadata = f.metadata

    c.metadata = g.metadata
    d.metadata = h.metadata

    i.metadata = j.metadata

    assert and1.is_equal(and2)


def test_ands_inequality_2():
    a = Uniadic()
    b = Uniadic()
    c = Uniadic()
    d = Uniadic()
    e = Uniadic()
    f = Uniadic()
    g = Uniadic()
    h = Uniadic()
    i = Uniadic()
    j = Uniadic()

    and1 = AND({a: b, c: d, i: 1})
    and2 = AND({e: f, g: h, j: 1})

    a.metadata = e.metadata
    b.metadata = f.metadata

    c.metadata = g.metadata

    assert and1.is_equal(and2)


def test_var_update_ands_equality_1():
    a = Uniadic()
    b = Uniadic()
    c = Uniadic()
    d = Uniadic()
    e = Uniadic()
    f = Uniadic()
    g = Uniadic()
    h = Uniadic()

    and1 = AND({a: b, c: d})
    and2 = AND({e: f, g: h})

    a.match(e)
    b.match(f)
    c.match(g)
    d.match(h)

    and3 = AND({Uniadic(): 2})
    pos1 = PossibleValues(dnf_list=[DNF([and1]), DNF([and3])])
    pos2 = PossibleValues((Uniadic(),), dnf_list=[DNF([and2])])
    v = Variadic()
    v.update_possible_values(pos1, pos2)

    assert a.metadata == e.metadata == b.metadata == f.metadata
    assert c.metadata == d.metadata == g.metadata == h.metadata


def test_var_update_possibles1():
    a = Uniadic()
    b = Uniadic()
    c = Uniadic()
    d = Uniadic()
    e = Uniadic()

    var = Variadic()
    var.update_possible_values(
        PossibleValues(), PossibleValues((a,)), PossibleValues((b, c))
    )

    var.update_possible_values(
        PossibleValues(), PossibleValues((d,)), PossibleValues((e, d))
    )
    assert var.possibles is not None
    assert (
        var.possibles[1].dnf_lookup_table[a].uniadics
        == var.possibles[1].dnf_lookup_table[d].uniadics
        == {a, d}
    )
    assert (
        var.possibles[2].dnf_lookup_table[b].uniadics
        == var.possibles[2].dnf_lookup_table[e].uniadics
        == {b, e}
    )
    assert (
        var.possibles[2].dnf_lookup_table[d].uniadics
        == var.possibles[2].dnf_lookup_table[c].uniadics
        == {d, c}
    )


def test_extract_uni_from_possibles():
    a = Uniadic()
    b = Uniadic()
    pos2 = PossibleValues((a, b))
    pos1 = PossibleValues((a,))

    v = Variadic()
    repr = ShapeRepr(root=v)
    assert repr.get_shapes() == ["(V1, ...)"]
    v.update_possible_values(pos1, pos2)
    assert repr.get_shapes() == ["u1", "(V1, ...)"]


def test_extract_uni_from_possibles_rev():
    a = Uniadic()
    b = Uniadic()
    pos2 = PossibleValues((a, b))
    pos1 = PossibleValues((b,))

    v = Variadic()
    repr = ShapeRepr(root=v)
    assert repr.get_shapes() == ["(V1, ...)"]
    v.update_possible_values(pos1, pos2)
    assert repr.get_shapes() == ["(V1, ...)", "u1"]


def test_extract_uni_from_possibles_both():
    a = Uniadic()
    b = Uniadic()
    c = Uniadic()
    pos1 = PossibleValues((a, b, c))
    pos2 = PossibleValues((a, c))

    v = Variadic()
    repr = ShapeRepr(root=v)
    assert repr.get_shapes() == ["(V1, ...)"]
    v.update_possible_values(pos1, pos2)
    assert repr.get_shapes() == ["u1", "(V1, ...)", "u2"]

    ref_possibles = {0: PossibleValues(), 1: PossibleValues((b,))}
    assert repr.root is not None
    assert repr.root.possibles is not None
    assert repr.root.possibles.keys() == ref_possibles.keys()
    assert repr.root.possibles[1].uniadics == ref_possibles[1].uniadics


def test_shapes_tensor_item_numeric():
    model = Model()
    relu_model1 = Relu()
    relu_model2 = Relu()
    relu_model1.set_shapes(input=[("V1", ...), "u1", "u2"])
    model |= relu_model1.connect(input="input", output="output")
    model |= relu_model2.connect(
        input=relu_model1.output[:, None, :, 2:4], output="output2"
    )
    model.set_shapes(input=[3, 4, 5])

    ref = {
        "$_Slice_1_output": None,
        "$_Slice_2_output": None,
        "$_Slice_3_output": None,
        "output": [3, 4, 5],
        "$_ToTuple_4_output": None,
        "$_Indexer_5_output": [3, 1, 4, 2],
        "input": [3, 4, 5],
        "$_Slice_1_start": None,
        "$_Slice_1_stop": None,
        "$_Slice_1_step": None,
        "$_Slice_2_start": None,
        "$_Slice_2_stop": None,
        "$_Slice_2_step": None,
        "$_Slice_3_start": None,
        "$_Slice_3_stop": None,
        "$_Slice_3_step": None,
        "$_ToTuple_4_input2": None,
        "output2": [3, 1, 4, 2],
    }
    check_shapes_semantically(model.get_shapes(), ref)


def test_shapes_tensor_item_symbolic():
    model = Model()
    relu_model1 = Relu()
    relu_model2 = Relu()
    relu_model1.set_shapes(input=[("V1", ...), "u1", "u2"])
    model |= relu_model1.connect(input="input", output="output")
    model |= relu_model2.connect(
        input=relu_model1.output[:, None, :, 2:4], output="output2"
    )

    ref: Mapping[str, list | None] = {
        "$_Slice_1_output": None,
        "$_Slice_2_output": None,
        "$_Slice_3_output": None,
        "output": ["u1", "(V1, ...)", "u2", "u3"],
        "$_ToTuple_4_output": None,
        "$_Indexer_5_output": ["u1", 1, "u4", "u5", "(V2, ...)"],
        "input": ["u1", "(V1, ...)", "u2", "u3"],
        "$_Slice_1_start": None,
        "$_Slice_1_stop": None,
        "$_Slice_1_step": None,
        "$_Slice_2_start": None,
        "$_Slice_2_stop": None,
        "$_Slice_2_step": None,
        "$_Slice_3_start": None,
        "$_Slice_3_stop": None,
        "$_Slice_3_step": None,
        "$_ToTuple_4_input2": None,
        "output2": ["u1", 1, "u4", "u5", "(V2, ...)"],
    }
    check_shapes_semantically(model.get_shapes(), ref)


def test_tensor_item_with_single_tensor_index():
    model = Model()
    relu_model = Relu()
    model |= relu_model.connect(input="input", output="output")
    model.set_shapes(input=[7, 4, 5])
    output = model.cout[Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])]
    model |= Buffer().connect(input=output, output="output2")

    ref: Mapping[str, list | None] = {
        "output": [7, 4, 5],
        "$_Indexer_1_output": [3, 3, 4, 5],
        "input": [7, 4, 5],
        "$_Indexer_1_index": [3, 3],
        "output2": [3, 3, 4, 5],
    }
    check_shapes_semantically(model.get_shapes(), ref)


def test_tensor_item_with_multiple_tensor_index():
    model = Model()
    relu_model = Relu()
    model |= relu_model.connect(input="input", output="output")
    model.set_shapes(input=[7, 4, 5])
    output = model.cout[Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), Tensor([[[1]]])]  # type: ignore
    model |= Buffer().connect(input=output, output="output2")

    ref: Mapping[str, list | None] = {
        "output": [7, 4, 5],
        "$_ToTuple_1_output": None,
        "$_Indexer_2_output": [1, 3, 3, 5],
        "input": [7, 4, 5],
        "$_ToTuple_1_input1": [3, 3],
        "$_ToTuple_1_input2": [1, 1, 1],
        "output2": [1, 3, 3, 5],
    }

    check_shapes_semantically(model.get_shapes(), ref)


def test_tensor_item_with_slice_and_multiple_tensor_index():
    model = Model()
    relu_model = Relu()
    model |= relu_model.connect(input="input", output="output")
    model.set_shapes(input=[7, 4, 5])
    output = model.cout[
        1:7,  # type: ignore
        Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
        Tensor([[[1]], [[2]], [[3]], [[1]], [[0]]]),
    ]
    model |= Buffer().connect(input=output, output="output2")

    ref: Mapping[str, list | None] = {
        "$_Slice_1_output": None,
        "output": [7, 4, 5],
        "$_ToTuple_2_output": None,
        "$_Indexer_3_output": [6, 5, 3, 3],
        "input": [7, 4, 5],
        "$_Slice_1_start": None,
        "$_Slice_1_stop": None,
        "$_Slice_1_step": None,
        "$_ToTuple_2_input2": [3, 3],
        "$_ToTuple_2_input3": [5, 1, 1],
        "output2": [6, 5, 3, 3],
    }
    check_shapes_semantically(model.get_shapes(), ref)


def test_tensor_item_with_slice_and_non_consecutive_tensors():
    model = Model()
    relu_model = Relu()
    model |= relu_model.connect(input="input", output="output")
    model.set_shapes(input=[7, 4, 5])
    output = model.cout[
        1:7,  # type: ignore
        Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
        None,
        Tensor([[[1]], [[2]], [[3]], [[1]], [[0]]]),
    ]
    model |= Buffer().connect(input=output, output="output2")

    ref: Mapping[str, list | None] = {
        "$_Slice_1_output": None,
        "output": [7, 4, 5],
        "$_ToTuple_2_output": None,
        "$_Indexer_3_output": [5, 3, 3, 6, 1],
        "input": [7, 4, 5],
        "$_Slice_1_start": None,
        "$_Slice_1_stop": None,
        "$_Slice_1_step": None,
        "$_ToTuple_2_input2": [3, 3],
        "$_ToTuple_2_input3": None,
        "$_ToTuple_2_input4": [5, 1, 1],
        "output2": [5, 3, 3, 6, 1],
    }

    check_shapes_semantically(model.get_shapes(), ref)


def test_tensor_item_with_post_ellipsis_non_consecutive_tensors_and_int():
    model = Model()
    relu_model = Relu()
    model |= relu_model.connect(input="input", output="output")
    model.set_shapes(input=[7, 4, 5])
    output = model.cout[..., Tensor([[1, 1, 0], [1, 0, 2], [2, 2, 2]]), None, 2]  # type: ignore
    model |= Buffer().connect(input=output, output="output2")

    ref: Mapping[str, list | None] = {
        "output": [7, 4, 5],
        "$_ToTuple_1_output": None,
        "$_Indexer_2_output": [3, 3, 7, 1],
        "input": [7, 4, 5],
        "$_ToTuple_1_input1": None,
        "$_ToTuple_1_input2": [3, 3],
        "$_ToTuple_1_input3": None,
        "$_ToTuple_1_input4": None,
        "output2": [3, 3, 7, 1],
    }

    check_shapes_semantically(model.get_shapes(), ref)


def test_index_with_two_consec_arange():
    model = Model()
    model |= (arr_1 := Arange(stop=7))
    model |= (arr_2 := Arange(stop=8))

    model |= (relu := Relu())

    tensor1 = arr_1.output[None, ...]
    tensor2 = arr_2.output[..., None]

    model.set_shapes({relu.input: [2, 3, 4]})

    output = relu.output[None, None, None, tensor1, tensor2]
    model |= Buffer().connect(input=output, output="output")

    ref: Mapping[str, list | None] = {
        "$_Arange_0_output": [7],
        "$_ToTuple_3_output": None,
        "$_Arange_1_output": [8],
        "$_ToTuple_5_output": None,
        "$_Indexer_4_output": [1, 7],
        "$_Indexer_6_output": [8, 1],
        "$_Relu_2_output": [2, 3, 4],
        "$_ToTuple_7_output": None,
        "$_Indexer_8_output": [1, 1, 1, 8, 7, 4],
        "$input": [2, 3, 4],
        "$_Arange_0_start": None,
        "$_Arange_0_stop": None,
        "$_Arange_0_step": None,
        "$_Arange_0_dtype": None,
        "$_Arange_1_start": None,
        "$_Arange_1_stop": None,
        "$_Arange_1_step": None,
        "$_Arange_1_dtype": None,
        "$_ToTuple_3_input1": None,
        "$_ToTuple_3_input2": None,
        "$_ToTuple_5_input1": None,
        "$_ToTuple_5_input2": None,
        "$_ToTuple_7_input1": None,
        "$_ToTuple_7_input2": None,
        "$_ToTuple_7_input3": None,
        "output": [1, 1, 1, 8, 7, 4],
    }

    check_shapes_semantically(model.get_shapes(), ref)


def test_index_with_two_non_consec_arange():
    model = Model()
    model |= (arr_1 := Arange(stop=7))
    model |= (arr_2 := Arange(stop=8))

    model |= (relu := Relu())

    tensor1 = arr_1.output[None, ...]
    tensor2 = arr_2.output[..., None]

    model.set_shapes({relu.input: [2, 3, 4]})

    output = relu.output[None, None, None, tensor1, 1:3, tensor2]
    model |= Buffer().connect(input=output, output="output")

    ref: Mapping[str, list | None] = {
        "$_Arange_0_output": [7],
        "$_ToTuple_3_output": None,
        "$_Arange_1_output": [8],
        "$_ToTuple_6_output": None,
        "$_Indexer_4_output": [1, 7],
        "$_Slice_5_output": None,
        "$_Indexer_7_output": [8, 1],
        "$_Relu_2_output": [2, 3, 4],
        "$_ToTuple_8_output": None,
        "$_Indexer_9_output": [8, 7, 1, 1, 1, 2],
        "$input": [2, 3, 4],
        "$_Arange_0_start": None,
        "$_Arange_0_stop": None,
        "$_Arange_0_step": None,
        "$_Arange_0_dtype": None,
        "$_Arange_1_start": None,
        "$_Arange_1_stop": None,
        "$_Arange_1_step": None,
        "$_Arange_1_dtype": None,
        "$_ToTuple_3_input1": None,
        "$_ToTuple_3_input2": None,
        "$_Slice_5_start": None,
        "$_Slice_5_stop": None,
        "$_Slice_5_step": None,
        "$_ToTuple_6_input1": None,
        "$_ToTuple_6_input2": None,
        "$_ToTuple_8_input1": None,
        "$_ToTuple_8_input2": None,
        "$_ToTuple_8_input3": None,
        "output": [8, 7, 1, 1, 1, 2],
    }

    check_shapes_semantically(model.get_shapes(), ref)


def test_index_with_list_int():
    model = Model()
    relu_model = Relu()
    model |= relu_model.connect(input="input", output="output")
    model.set_shapes(input=[7, 4, 5])
    output = model.cout[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]
    model |= Buffer().connect(input=output, output="output2")

    ref: Mapping[str, list | None] = {
        "output": [7, 4, 5],
        "$_Indexer_1_output": [3, 3, 4, 5],
        "input": [7, 4, 5],
        "$_Indexer_1_index": None,
        "output2": [3, 3, 4, 5],
    }
    check_shapes_semantically(model.get_shapes(), ref)


def test_index_with_list_of_tuple_ints():
    model = Model()
    relu_model = Relu()
    model |= relu_model.connect(input="input", output="output")
    model.set_shapes(input=[7, 4, 5])
    output = model.cout[None, None, [[1, 2, 3, 4, 5, 6]], None, [[1], [2], [3]]]  # type: ignore
    model |= Buffer().connect(input=output, output="output2")

    ref: Mapping[str, list | None] = {
        "output": [7, 4, 5],
        "$_ToTuple_1_output": None,
        "$_Indexer_2_output": [3, 6, 1, 1, 1, 5],
        "input": [7, 4, 5],
        "$_ToTuple_1_input1": None,
        "$_ToTuple_1_input2": None,
        "$_ToTuple_1_input3": None,
        "$_ToTuple_1_input4": None,
        "$_ToTuple_1_input5": None,
        "output2": [3, 6, 1, 1, 1, 5],
    }
    check_shapes_semantically(model.get_shapes(), ref)


def test_partial_shape_propagation():
    model = Model()
    relu_model = Relu()
    model |= relu_model.connect(input="input", output="output1")
    model.set_shapes(input=["u1", "u2", 3, 4])
    shp_1 = model.cout.shape[-1]
    shp_2 = model.cout.shape[-2]
    out = shp_1 * shp_2
    model |= Buffer().connect(input=out, output="output2")
    assert model.cout.metadata.value == 12
