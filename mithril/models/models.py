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

from __future__ import annotations

import math
from abc import abstractmethod
from collections.abc import Sequence
from copy import deepcopy
from types import EllipsisType

from .. import types
from ..common import PaddingType
from ..framework.common import (
    NOT_GIVEN,
    TBD,
    MainValueType,
    ShapeTemplateType,
    Tensor,
    ToBeDetermined,
)
from ..framework.constraints import polynomial_kernel_constraint
from ..framework.logical.model import (
    Connection,
    ConnectionType,
    ExtendInfo,
    IOKey,
    Model,
)
from ..framework.logical.operator import Operator
from ..types import Constant
from ..utils.utils import convert_to_list, convert_to_tuple
from .primitives import (
    Absolute,
    Add,
    ArgMax,
    AUCCore,
    Buffer,
    CartesianDifference,
    Cast,
    Cholesky,
    Clamp,
    Concat,
    DistanceMatrix,
    Divide,
    Dtype,
    Eigvalsh,
    Exponential,
    Eye,
    EyeComplement,
    Floor,
    GPRAlpha,
    GPRVOuter,
    Greater,
    Indexer,
    KLDivergence,
    Length,
    Log,
    MatrixMultiply,
    Mean,
    Multiply,
    Negate,
    NormModifier,
    PaddingConverter1D,
    PaddingConverter2D,
    PermuteTensor,
    PolynomialFeatures,
    Power,
    PrimitiveAvgPool2D,
    PrimitiveConvolution1D,
    PrimitiveConvolution2D,
    PrimitiveMaxPool1D,
    PrimitiveMaxPool2D,
    PrimitiveRandInt,
    PrimitiveRandn,
    Reshape,
    Shape,
    Sigmoid,
    Sign,
    Size,
    Slice,
    Sqrt,
    Square,
    Squeeze,
    StableReciprocal,
    StrideConverter,
    Subtract,
    Sum,
    Tanh,
    ToList,
    Transpose,
    TransposedDiagonal,
    Trapezoid,
    TsnePJoint,
    TupleConverter,
    Unique,
    Variance,
    Where,
)

__all__ = [
    "Linear",
    "ElementWiseAffine",
    "Layer",
    "LayerNorm",
    "GroupNorm",
    "BatchNorm2D",
    "L1",
    "L2",
    "QuadraticFormRegularizer",
    "RBFKernel",
    "PolynomialKernel",
    "KernelizedSVM",
    "LinearSVM",
    "LogisticRegression",
    "MLP",
    "Cell",
    "RNNCell",
    "LSTMCell",
    "RNN",
    "OneToMany",
    "ManyToOne",
    "EncoderDecoder",
    "EncoderDecoderInference",
    "EncoderDistanceMatrix",
    "PolynomialRegression",
    "MDSCore",
    "MDS",
    "TSNE",
    "GaussProcessRegressionCore",
    "GPRLoss",
    "F1",
    "Precision",
    "Recall",
    "MaxPool1D",
    "MaxPool2D",
    "Convolution1D",
    "Convolution2D",
    "LSTMCellBody",
    "Cast",
    "Dtype",
    "Metric",
    "Accuracy",
    "AUC",
    "SiLU",
    "AvgPool2D",
    "Split",
    "Randn",
    "RandInt",
    "Clamp",
    "Floor",
]


class Pool1D(Model):
    input: Connection
    kernel_size: Connection
    stride: Connection
    padding: Connection
    dilation: Connection
    output: Connection

    @property
    def pool_model(self) -> type[Model]:
        raise NotImplementedError("Pool Model should be indicated!")

    def __init__(
        self,
        kernel_size: int | ToBeDetermined,
        stride: int | None | ToBeDetermined = None,
        padding: int | PaddingType | tuple[int, int] | ToBeDetermined = (0, 0),
        dilation: int | ToBeDetermined = 1,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.factory_args = {
            "kernel_size": convert_to_list(kernel_size),
            "stride": convert_to_list(stride),
            "padding": convert_to_list(padding),
            "dilation": convert_to_list(dilation),
        }
        pad = convert_to_tuple(padding) if isinstance(padding, list) else padding

        stride_conv = StrideConverter()
        pad_conv = PaddingConverter1D()

        self |= stride_conv.connect(
            input=IOKey(name="stride", value=stride),
            kernel_size=IOKey(name="kernel_size", value=kernel_size),
        )

        self |= pad_conv.connect(
            input=IOKey(name="padding", value=pad), kernel_size="kernel_size"
        )

        self |= self.pool_model().connect(
            input=IOKey("input", value=input),
            kernel_size="kernel_size",
            stride=stride_conv.output,
            padding=pad_conv.output,
            dilation=IOKey(name="dilation", value=dilation),
            output="output",
        )

        self.expose_keys("stride", "padding", "dilation", "kernel_size", "output")
        self._set_cin("input", safe=False)
        self._freeze()

    def connect(  # type: ignore[override]
        self,
        input: ConnectionType | Tensor[int | float | bool] = NOT_GIVEN,
        kernel_size: ConnectionType | int = NOT_GIVEN,
        stride: ConnectionType | int | None = NOT_GIVEN,
        padding: ConnectionType | int | PaddingType | tuple[int, int] = NOT_GIVEN,
        dilation: ConnectionType | int = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().connect(
            input=input,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            output=output,
        )


class MaxPool1D(Pool1D):
    @property
    def pool_model(self) -> type[Model]:
        return PrimitiveMaxPool1D


# TODO: Implement MinPool1D and AvgPool1D
# class MinPool1D(Pool1D):
#     @property
#     def pool_model(self):
#         return PrimitiveMinPool1D

# class AvgPool1D(Pool1D):
#     @property
#     def pool_model(self):
#         return PrimitiveAvgPool1D


class Pool2D(Model):
    input: Connection
    kernel_size: Connection
    stride: Connection
    padding: Connection
    dilation: Connection
    output: Connection

    @property
    def pool_model(self) -> type[Model]:
        raise NotImplementedError("Pool Model should be indicated!")

    def __init__(
        self,
        kernel_size: int | tuple[int, int] | ToBeDetermined,
        stride: int | None | tuple[int, int] | ToBeDetermined = None,
        padding: int | PaddingType | tuple[int, int] | ToBeDetermined = (0, 0),
        dilation: int | ToBeDetermined = 1,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.factory_args = {
            "kernel_size": convert_to_list(kernel_size),
            "stride": convert_to_list(stride),
            "padding": convert_to_list(padding),
            "dilation": convert_to_list(dilation),
        }

        pad = convert_to_tuple(padding) if isinstance(padding, list) else padding
        if isinstance(stride, list):
            stride = convert_to_tuple(stride)
        if isinstance(kernel_size, list):
            kernel_size = convert_to_tuple(kernel_size)

        kt_converter = TupleConverter()
        s_converter = StrideConverter()
        st_converter = TupleConverter()
        p_converter = PaddingConverter2D()
        pt_converter = TupleConverter()
        dt_converter = TupleConverter()

        self |= kt_converter.connect(input=IOKey(name="kernel_size", value=kernel_size))
        self |= s_converter.connect(
            input=IOKey(name="stride", value=stride), kernel_size=kt_converter.output
        )
        self |= st_converter.connect(input=s_converter.output)
        self |= p_converter.connect(
            input=IOKey(name="padding", value=pad), kernel_size=kt_converter.output
        )
        self |= pt_converter.connect(input=p_converter.output)
        self |= dt_converter.connect(input=IOKey(name="dilation", value=dilation))
        self |= self.pool_model().connect(
            input=IOKey("input", value=input),
            kernel_size=kt_converter.output,
            stride=st_converter.output,
            padding=pt_converter.output,
            dilation=dt_converter.output,
            output="output",
        )

        self.expose_keys("stride", "padding", "dilation", "kernel_size", "output")
        self._set_cin("input", safe=False)
        self._freeze()

    def connect(  # type: ignore[override]
        self,
        input: ConnectionType | Tensor[int | float | bool] = NOT_GIVEN,
        kernel_size: ConnectionType | int | tuple[int, int] = NOT_GIVEN,
        stride: ConnectionType | int | None | tuple[int, int] = NOT_GIVEN,
        padding: ConnectionType | int | PaddingType | tuple[int, int] = NOT_GIVEN,
        dilation: ConnectionType | int = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().connect(
            input=input,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            output=output,
        )


class MaxPool2D(Pool2D):
    @property
    def pool_model(self) -> type[Model]:
        return PrimitiveMaxPool2D


# TODO: Implement MinPool2D
# class MinPool2D(Pool2D):
#     @property
#     def pool_model(self):
#         return PrimitiveMinPool2D


class AvgPool2D(Pool2D):
    @property
    def pool_model(self) -> type[Model]:
        return PrimitiveAvgPool2D


class Convolution1D(Model):
    input: Connection
    weight: Connection
    stride: Connection
    padding: Connection
    dilation: Connection
    output: Connection

    def __init__(
        self,
        kernel_size: int | None = None,
        out_channels: int | None = None,
        stride: int | ToBeDetermined = 1,
        padding: int | PaddingType | tuple[int, int] | ToBeDetermined = 0,
        dilation: int | ToBeDetermined = 1,
        use_bias: bool = True,
        input: Tensor[int | float] | ToBeDetermined = TBD,
        weight: Tensor[int | float] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.factory_args = {
            "kernel_size": convert_to_list(kernel_size),
            "out_channels": out_channels,
            "stride": convert_to_list(stride),
            "padding": convert_to_list(padding),
            "dilation": convert_to_list(dilation),
            "use_bias": use_bias,
        }

        pad = convert_to_tuple(padding) if isinstance(padding, list) else padding

        k_shp = Shape()
        p_converter = PaddingConverter1D()

        self |= k_shp.connect(
            input=IOKey(name="weight", shape=[out_channels, "C_in", kernel_size])
        )
        self |= p_converter.connect(
            input=IOKey(name="padding", value=pad), kernel_size=k_shp.output[-1]
        )

        conv_connections: dict[str, ConnectionType] = {
            "output": "output",
            "input": IOKey("input", value=input),
            "weight": IOKey("weight", value=weight, differentiable=True),
            "stride": IOKey(name="stride", value=stride),
            "padding": p_converter.output,
            "dilation": IOKey(name="dilation", value=dilation),
        }
        if use_bias:
            conv_connections["bias"] = IOKey("bias", differentiable=True)

        self |= PrimitiveConvolution1D(use_bias=use_bias).connect(**conv_connections)

        self.expose_keys("input", "weight", "padding", "stride", "dilation", "output")
        self._set_cin("input", safe=False)
        self._freeze()

    def connect(  # type: ignore[override]
        self,
        input: ConnectionType | Tensor[int | float] = NOT_GIVEN,
        weight: ConnectionType | Tensor[float] = NOT_GIVEN,
        stride: ConnectionType | int = NOT_GIVEN,
        padding: ConnectionType | int | tuple[int, int] = NOT_GIVEN,
        dilation: ConnectionType | int = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().connect(
            input=input,
            weight=weight,
            stride=stride,
            padding=padding,
            dilation=dilation,
            output=output,
        )


class Convolution2D(Model):
    input: Connection
    weight: Connection
    stride: Connection
    padding: Connection
    dilation: Connection
    groups: Connection
    output: Connection

    def __init__(
        self,
        kernel_size: int | tuple[int, int] | None = None,
        out_channels: int | None = None,
        stride: int | tuple[int, int] | ToBeDetermined = (1, 1),
        padding: int
        | PaddingType
        | tuple[int, int]
        | tuple[tuple[int, int], tuple[int, int]]
        | ToBeDetermined = (0, 0),
        dilation: int | tuple[int, int] | ToBeDetermined = (1, 1),
        groups: int | ToBeDetermined = 1,
        use_bias: bool = True,
        input: Tensor[int | float] | ToBeDetermined = TBD,
        weight: Tensor[int | float] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.factory_args = {
            "kernel_size": convert_to_list(kernel_size),
            "out_channels": out_channels,
            "stride": convert_to_list(stride),
            "padding": convert_to_list(padding),
            "dilation": convert_to_list(dilation),
            "groups": groups,
            "use_bias": use_bias,
        }

        if isinstance(kernel_size, int | None):
            k_size = (kernel_size, kernel_size)
        else:
            k_size = kernel_size

        if isinstance(stride, list):
            stride = convert_to_tuple(stride)
        pad = convert_to_tuple(padding) if isinstance(padding, list) else padding
        if isinstance(dilation, list):
            dilation = convert_to_tuple(dilation)

        k_shp = Shape()
        p_converter = PaddingConverter2D()
        st_converter = TupleConverter()
        pt_converter = TupleConverter()
        dt_converter = TupleConverter()

        self |= k_shp.connect(
            input=IOKey(name="weight", shape=[out_channels, "C_in", *k_size])
        )
        self |= p_converter.connect(
            input=IOKey(name="padding", value=pad), kernel_size=k_shp.output[-2:]
        )
        self |= st_converter.connect(input=IOKey(name="stride", value=stride))
        self |= pt_converter.connect(input=p_converter.output)
        self |= dt_converter.connect(input=IOKey(name="dilation", value=dilation))

        conv_connections: dict[str, ConnectionType] = {
            "output": "output",
            "input": IOKey("input", value=input),
            "weight": IOKey("weight", value=weight, differentiable=True),
            "stride": st_converter.output,
            "padding": pt_converter.output,
            "dilation": dt_converter.output,
            "groups": IOKey(name="groups", value=groups),
        }
        if use_bias:
            conv_connections["bias"] = IOKey("bias", differentiable=True)

        self |= PrimitiveConvolution2D(use_bias=use_bias).connect(**conv_connections)

        self.expose_keys("output")
        self._set_cin("input", safe=False)
        self._freeze()

    def connect(  # type: ignore[override]
        self,
        input: ConnectionType | Tensor[int | float] = NOT_GIVEN,
        weight: ConnectionType | Tensor[float] = NOT_GIVEN,
        stride: ConnectionType | int | tuple[int, int] = NOT_GIVEN,
        padding: ConnectionType
        | int
        | PaddingType
        | tuple[int, int]
        | tuple[tuple[int, int], tuple[int, int]] = NOT_GIVEN,
        dilation: ConnectionType | int | tuple[int, int] = NOT_GIVEN,
        groups: ConnectionType | int = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().connect(
            input=input,
            weight=weight,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            output=output,
        )


class Linear(Model):
    output: Connection
    input: Connection
    weight: Connection
    bias: Connection

    def __init__(
        self,
        dimension: int | None = None,
        use_bias: bool = True,
        input: Tensor[int | float] | ToBeDetermined = TBD,
        weight: Tensor[float] | ToBeDetermined = TBD,
        bias: Tensor[float] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.factory_args = {"dimension": dimension, "use_bias": use_bias}
        dim: int | str = "d_out" if dimension is None else dimension
        shapes: dict[str, ShapeTemplateType] = {
            "input": ["N", ("Var_inter", ...), "d_in"],
            "weight": [dim, "d_in"],
            "output": ["N", ("Var_inter", ...), dim],
        }

        mult = MatrixMultiply()

        output = "output"
        input_key = IOKey(name="input", value=input)
        weight_key = IOKey(name="weight", value=weight, differentiable=True).transpose()

        if use_bias:
            bias_key = IOKey(
                name="bias",
                value=bias,
                type=Tensor[float],
                differentiable=True,
            )
            self |= mult.connect(left=input_key, right=weight_key)
            self |= Add().connect(left=mult.output, right=bias_key, output=output)
            shapes["bias"] = [dim]
            self._formula_key = "linear_bias"
        else:
            self |= mult.connect(left=input_key, right=weight_key, output=output)
            self._formula_key = "linear"

        self.expose_keys("output")
        self._set_shapes(**shapes)
        self._set_cin("input", safe=False)
        self._freeze()

    def connect(  # type: ignore[override]
        self,
        input: ConnectionType | Tensor[int | float] = NOT_GIVEN,
        weight: ConnectionType | Tensor[float] = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
        *,
        bias: ConnectionType | Tensor[float] = NOT_GIVEN,
    ) -> ExtendInfo:
        kwargs = {"input": input, "weight": weight, "output": output}

        if "bias" not in self.input_keys and bias != NOT_GIVEN:
            raise KeyError("bias is not a valid input when 'use_bias' is False!")
        elif "bias" in self.input_keys:
            kwargs["bias"] = bias

        return super().connect(**kwargs)


class ElementWiseAffine(Model):
    input: Connection
    weight: Connection
    bias: Connection
    output: Connection

    def __init__(
        self,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        weight: Tensor[int | float | bool] | ToBeDetermined = TBD,
        bias: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)

        mult_model = Multiply()
        sum_model = Add()

        self |= mult_model.connect(
            left=IOKey("input", value=input), right=IOKey("weight", value=weight)
        )
        self += sum_model.connect(
            right=IOKey(name="bias", value=bias),
            output=IOKey(name="output"),
        )
        self._set_cin("input", safe=False)
        self._freeze()

    def connect(  # type: ignore[override]
        self,
        input: ConnectionType | Tensor[int | float | bool] = NOT_GIVEN,
        weight: ConnectionType | Tensor[int | float | bool] = NOT_GIVEN,
        bias: ConnectionType | Tensor[int | float | bool] = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().connect(
            input=input,
            weight=weight,
            bias=bias,
            output=output,
        )


class Layer(Model):
    input: Connection
    weight: Connection
    bias: Connection
    output: Connection

    def __init__(
        self,
        activation: Model,
        dimension: int | None = None,
        input: Tensor[int | float] | ToBeDetermined = TBD,
        weight: Tensor[float] | ToBeDetermined = TBD,
        bias: Tensor[float] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.factory_args = {"activation": activation, "dimension": dimension}
        linear_model = Linear(dimension=dimension)
        self |= linear_model.connect(
            input=IOKey("input", value=input),
            weight=IOKey("weight", value=weight),
            bias=IOKey("bias", value=bias),
        )
        self += activation.connect(output="output")

        self.expose_keys("output")
        self._set_cin("input", safe=False)
        self._freeze()

    def connect(  # type: ignore[override]
        self,
        input: ConnectionType | Tensor[int | float] = NOT_GIVEN,
        weight: ConnectionType | Tensor[float] = NOT_GIVEN,
        bias: ConnectionType | Tensor[float] = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().connect(
            input=input,
            weight=weight,
            bias=bias,
            output=output,
        )


class LayerNorm(Model):
    input: Connection
    output: Connection
    weight: Connection
    b: Connection

    def __init__(
        self,
        use_scale: bool = True,
        use_bias: bool = True,
        eps: float | Tensor[float] | ToBeDetermined = 1e-5,
        input: Tensor[float] | ToBeDetermined = TBD,
        weight: Tensor[float] | ToBeDetermined = TBD,
        bias: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.factory_args = {"use_scale": use_scale, "use_bias": use_bias, "eps": eps}

        # Expects its input shape as [B, ..., d] d refers to normalized dimension
        mean = Mean(axis=-1, keepdim=True)
        numerator = Subtract()
        var = Variance(axis=-1, correction=0, keepdim=True)
        add = Add()
        denominator = Sqrt()
        in_key = IOKey("input", value=input, type=Tensor[float])
        self |= mean.connect(input=in_key)
        self |= numerator.connect(left=in_key, right=mean.output)
        self |= var.connect(input=in_key)
        self |= add.connect(
            left=var.output, right=IOKey("eps", value=eps, type=Tensor[float] | float)
        )
        self |= denominator.connect(input=add.output)
        self |= Divide().connect(
            numerator=numerator.output, denominator=denominator.output
        )

        self._set_shapes(input=["B", "C", "d"])

        shapes: dict[str, ShapeTemplateType] = {
            "left": ["B", "C", "d"],
            "right": ["d"],
        }

        if use_scale:
            mult = Multiply()
            self += mult.connect(
                right=IOKey("weight", value=weight, differentiable=True)
            )
            mult._set_shapes(**shapes)

        if use_bias:
            add = Add()
            self += add.connect(right=IOKey("bias", value=bias, differentiable=True))
            add._set_shapes(**shapes)
        # TODO: Remove below Buffer after required naming-related changes are done.
        self |= Buffer().connect(input=self.cout, output=IOKey(name="output"))
        self._set_cin("input", safe=False)
        self._freeze()

    def connect(  # type: ignore[override]
        self,
        input: ConnectionType | Tensor[float] = NOT_GIVEN,
        output: ConnectionType | Tensor[float] = NOT_GIVEN,
        eps: ConnectionType | float | Tensor[float] = NOT_GIVEN,
        *,
        weight: ConnectionType | Tensor[float] = NOT_GIVEN,
        bias: ConnectionType | Tensor[float] = NOT_GIVEN,
    ) -> ExtendInfo:
        kwargs = {"input": input, "output": output, "eps": eps}

        if "weight" not in self.input_keys and weight != NOT_GIVEN:
            raise KeyError("weight is not a valid input when 'use_scale' is False!")
        elif "weight" in self.input_keys:
            kwargs["weight"] = weight

        if "bias" not in self.input_keys and bias != NOT_GIVEN:
            raise KeyError("bias is not a valid input when 'use_bias' is False!")
        elif "bias" in self.input_keys:
            kwargs["bias"] = bias

        return super().connect(**kwargs)


class GroupNorm(Model):
    input: Connection
    output: Connection

    def __init__(
        self,
        num_groups: int = 32,
        use_scale: bool = True,
        use_bias: bool = True,
        eps: float = 1e-5,
        input: Tensor[float] | ToBeDetermined = TBD,
        *,
        weight: Tensor[float] | ToBeDetermined = TBD,
        bias: Tensor[float] | ToBeDetermined = TBD,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)

        # Assumed input shape is [N, C, H, W]
        input_key = IOKey(name="input", value=input, type=Tensor[float])
        input_shape = input_key.shape
        B = input_shape[0]

        _input_key = input_key.reshape((B, num_groups, -1))

        mean = _input_key.mean(axis=-1, keepdim=True)
        var = _input_key.var(axis=-1, keepdim=True)

        _input_key = (_input_key - mean) / (
            var + IOKey("eps", value=eps, type=Tensor[float] | float)
        ).sqrt()
        self |= Reshape().connect(input=_input_key, shape=input_shape)

        self._set_shapes(input=["B", "C", "H", "W"])

        shapes: dict[str, ShapeTemplateType] = {
            "left": ["B", "C", "H", "W"],
            "right": [1, "C", 1, 1],
        }

        if use_scale:
            weight_key = IOKey(
                name="weight", type=Tensor[float], value=weight, differentiable=True
            )
            mult = Multiply()
            self |= mult.connect(left=self.cout, right=weight_key)
            mult._set_shapes(**shapes)

        if use_bias:
            bias_key = IOKey(
                name="bias", type=Tensor[float], value=bias, differentiable=True
            )
            add = Add()
            self |= add.connect(left=self.cout, right=bias_key)
            add._set_shapes(**shapes)

        self |= Buffer().connect(input=self.cout, output=IOKey(name="output"))

        self.expose_keys("output", "eps")
        self._set_cin("input", safe=False)
        self._freeze()

    def connect(  # type: ignore[override]
        self,
        input: ConnectionType | Tensor[float] = NOT_GIVEN,
        output: ConnectionType | Tensor[float] = NOT_GIVEN,
        eps: ConnectionType | Tensor[float] | float = NOT_GIVEN,
        *,
        weight: ConnectionType | Tensor[float] = NOT_GIVEN,
        bias: ConnectionType | Tensor[float] = NOT_GIVEN,
    ) -> ExtendInfo:
        kwargs = {"input": input, "output": output, "eps": eps}

        if "weight" not in self.input_keys and weight != NOT_GIVEN:
            raise KeyError("weight is not a valid input when 'use_scale' is False!")
        elif "weight" in self.input_keys:
            kwargs["weight"] = weight

        if "bias" not in self.input_keys and bias != NOT_GIVEN:
            raise KeyError("bias is not a valid input when 'use_bias' is False!")
        elif "bias" in self.input_keys:
            kwargs["bias"] = bias

        return super().connect(**kwargs)


class BatchNorm2D(Model):
    input: Connection
    output: Connection
    running_mean: Connection
    running_var: Connection

    def __init__(
        self,
        num_features: int | None = None,
        use_scale: bool = True,
        use_bias: bool = True,
        eps: float = 1e-5,
        momentum: float | ToBeDetermined = 0.1,
        inference: bool = True,
        *,
        input: Tensor[float] | ToBeDetermined = TBD,
        weight: Tensor[float] | ToBeDetermined = TBD,
        bias: Tensor[float] | ToBeDetermined = TBD,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)

        # Assumed input shape is [N, C, H, W]
        input_key = IOKey(
            name="input", value=input, shape=["N", num_features, "H", "W"]
        )
        shp = input_key.shape

        running_mean = IOKey(name="running_mean")
        running_var = IOKey(name="running_var")

        if inference:
            norm = (input_key - running_mean.reshape((1, shp[1], 1, 1))) / (
                running_var.reshape((1, shp[1], 1, 1)) + eps
            ).sqrt()
            # Compute mean and variance over the spatial dimensions
        else:
            size = shp[0] * shp[2] * shp[3]
            mean = input_key.mean(axis=(0, 2, 3), keepdim=True)  # Shape: [1, C, 1, 1]
            var = input_key.var(axis=(0, 2, 3), keepdim=True)  # Shape: [1, C, 1, 1]

            m_key = IOKey(name="momentum", value=momentum)
            running_mean_out = (1 - m_key) * running_mean + m_key * mean.reshape(
                (mean.shape[1],)
            )
            running_var_out = (1 - m_key) * running_var + m_key * (
                var * (size / (size - 1))
            ).reshape((mean.shape[1],))
            # NOTE: multiplication (size / (size - 1)) is added to make the
            # running_variance similar to the BatchNorm2d module in PyTorch

            self |= Buffer().connect(running_mean_out, "running_mean_out")
            self |= Buffer().connect(running_var_out, "running_var_out")
            self.bind_state_keys(running_mean, "running_mean_out", Constant.ZEROS)
            self.bind_state_keys(running_var, "running_var_out", Constant.ONES)

            # Normalize the input
            norm = (input_key - mean) / (var + eps).sqrt()

        self |= Buffer().connect(input=norm)
        shapes: dict[str, ShapeTemplateType] = {
            "left": ["B", "C", "H", "W"],
            "right": [1, "C", 1, 1],
        }

        if use_scale:
            weight_key = IOKey(
                name="weight", type=Tensor[float], value=weight, differentiable=True
            )
            mult = Multiply()
            mult._set_shapes(**shapes)
            self |= mult.connect(left=self.cout, right=weight_key)

        if use_bias:
            bias_key = IOKey(
                name="bias", type=Tensor[float], value=bias, differentiable=True
            )
            add = Add()
            add._set_shapes(**shapes)
            self |= add.connect(left=self.cout, right=bias_key)

        self |= Buffer().connect(input=self.cout, output=IOKey(name="output"))

        _num_features: str | int | None = num_features
        if _num_features is None:
            _num_features = "num_features"
        self.set_shapes(
            input=["N", _num_features, "H", "W"],
            running_mean=[_num_features],
            running_var=[_num_features],
        )
        self.set_cin("input", safe=False)
        self._freeze()


class L1(Model):
    input: Connection
    output: Connection

    def __init__(
        self,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)

        abs_model = Absolute()

        self |= abs_model.connect(input=IOKey("input", value=input))
        self += Sum().connect(output=IOKey(name="output"))

        self._set_cin("input", safe=False)
        self._set_cout("output", safe=False)
        self._freeze()

    def connect(  # type: ignore[override]
        self,
        input: ConnectionType | Tensor[int | float | bool] = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().connect(
            input=input,
            output=output,
        )


class L2(Model):
    input: Connection
    output: Connection

    def __init__(
        self,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)

        self += Square().connect(input=IOKey("input", value=input))
        self += Sum().connect()
        self += Multiply().connect(right=0.5, output=IOKey(name="output"))

        self.expose_keys("output")
        self._set_cin("input", safe=False)
        self._set_cout("output", safe=False)
        self._freeze()

    def connect(  # type: ignore[override]
        self,
        input: ConnectionType | Tensor[int | float | bool] = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().connect(
            input=input,
            output=output,
        )


class QuadraticFormRegularizer(Model):
    input: Connection
    kernel: Connection
    output: Connection

    def __init__(
        self,
        input: Tensor[int | float] | ToBeDetermined = TBD,
        kernel: Tensor[int | float] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        transpose_model = Transpose()
        dot_model1 = MatrixMultiply()
        dot_model2 = MatrixMultiply()

        self |= transpose_model.connect(input=IOKey("input", value=input))
        self |= dot_model1.connect(
            left=transpose_model.input, right=IOKey("kernel", value=kernel)
        )
        self |= dot_model2.connect(left=dot_model1.output, right=transpose_model.output)
        self |= Multiply().connect(
            left=dot_model2.output, right=0.5, output=IOKey(name="output")
        )
        shapes: dict[str, ShapeTemplateType] = {"input": [1, "N"], "kernel": ["N", "N"]}
        self._set_shapes(**shapes)
        self._set_cin("input", safe=False)
        self._freeze()

    def connect(  # type: ignore[override]
        self,
        input: ConnectionType | Tensor[int | float] = NOT_GIVEN,
        kernel: ConnectionType | Tensor[int | float] = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().connect(
            input=input,
            kernel=kernel,
            output=output,
        )


class RBFKernel(Model):
    input1: Connection
    input2: Connection
    l_scale: Connection
    sigma: Connection
    output: Connection

    def __init__(
        self,
        input1: Tensor[int | float] | ToBeDetermined = TBD,
        input2: Tensor[int | float] | ToBeDetermined = TBD,
        l_scale: Tensor[int | float | bool] | ToBeDetermined = TBD,
        sigma: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)

        euclidean_model = CartesianDifference()
        square_model1 = Square()
        square_model2 = Square()
        sum_model = Sum(axis=2)
        mult_model1 = Multiply()
        div_model = Divide()
        exp_model = Exponential()
        mult_model2 = Multiply()
        l_square = Multiply()
        l_key = IOKey("l_scale", value=l_scale, type=Tensor[int | float | bool])

        self |= euclidean_model.connect(
            left=IOKey("input1", value=input1, type=Tensor[int | float]),
            right=IOKey("input2", value=input2, type=Tensor[int | float]),
        )
        self |= square_model1.connect(input=euclidean_model.output)
        self |= sum_model.connect(input=square_model1.output)
        self |= mult_model1.connect(left=sum_model.output, right=-0.5)
        self |= square_model2.connect(
            input=IOKey("sigma", value=sigma, type=Tensor[int | float | bool])
        )
        self |= div_model.connect(
            numerator=mult_model1.output, denominator=square_model2.output
        )
        self |= exp_model.connect(input=div_model.output)
        self |= l_square.connect(left=l_key, right=l_key)
        self |= mult_model2.connect(
            left=l_square.output,
            right=exp_model.output,
            output=IOKey(name="output"),
        )

        # self.set_canonical_input("input1")
        shapes: dict[str, ShapeTemplateType] = {
            "input1": ["N", "dim"],
            "input2": ["M", "dim"],
            "l_scale": [1],
            "sigma": [1],
            "output": ["N", "M"],
        }

        self._set_shapes(**shapes)
        self._set_cin("input1", "input2", safe=False)
        self._freeze()

    def connect(  # type: ignore[override]
        self,
        input1: ConnectionType | Tensor[int | float] = NOT_GIVEN,
        input2: ConnectionType | Tensor[int | float] = NOT_GIVEN,
        l_scale: ConnectionType | Tensor[int | float | bool] = NOT_GIVEN,
        sigma: ConnectionType | Tensor[int | float | bool] = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().connect(
            input1=input1,
            input2=input2,
            l_scale=l_scale,
            sigma=sigma,
            output=output,
        )


class PolynomialKernel(Model):
    input1: Connection
    input2: Connection
    poly_coef: Connection
    degree: Connection
    output: Connection

    def __init__(
        self,
        robust: bool = True,
        input1: Tensor[int | float | bool] | ToBeDetermined = TBD,
        input2: Tensor[int | float | bool] | ToBeDetermined = TBD,
        poly_coef: Tensor[int | float | bool] | ToBeDetermined = TBD,
        degree: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)

        transpose_model = Transpose()
        mult_model = MatrixMultiply()
        sum_model = Add()
        power_model = Power(robust=robust)  # TODO: Should it be usual Power or not???

        self |= transpose_model.connect(input=IOKey("input2", value=input2))
        self |= mult_model.connect(
            left=IOKey("input1", value=input1), right=transpose_model.output
        )
        self |= sum_model.connect(
            left=mult_model.output, right=IOKey("poly_coef", value=poly_coef)
        )
        self |= power_model.connect(
            base=sum_model.output,
            exponent=IOKey("degree", value=degree),
            output=IOKey(name="output"),
        )
        self._set_shapes(input1=["N", "d"], input2=["M", "d"], output=["N", "M"])
        self._add_constraint(
            fn=polynomial_kernel_constraint,
            keys=["poly_coef", "degree"],
        )
        self._set_cin("input1", "input2", safe=False)
        self._freeze()

    def connect(  # type: ignore[override]
        self,
        input1: ConnectionType | Tensor[int | float | bool] = NOT_GIVEN,
        input2: ConnectionType | Tensor[int | float | bool] = NOT_GIVEN,
        poly_coef: ConnectionType | Tensor[int | float | bool] = NOT_GIVEN,
        degree: ConnectionType | Tensor[int | float | bool] = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().connect(
            input1=input1,
            input2=input2,
            poly_coef=poly_coef,
            degree=degree,
            output=output,
        )


class KernelizedSVM(Model):
    input1: Connection
    input2: Connection
    weight: Connection
    bias: Connection
    output: Connection

    def __init__(
        self,
        kernel: Model,
        weight: Tensor[float] | ToBeDetermined = TBD,
        bias: Tensor[float] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
        **kwargs: MainValueType | Tensor[int | float | bool] | ToBeDetermined,
    ) -> None:
        if len(kernel.input_keys) < 2:
            raise KeyError("Kernel requires at least two inputs!")
        if len(kernel.conns.output_keys) != 1:
            raise KeyError("Kernel requires single output!")
        super().__init__(name=name)

        self.factory_args = {"kernel": kernel}

        linear_model = Linear()
        # Get kernel inputs from given model.
        kernel_input_args = {}
        for key in kernel.input_keys:
            conn = kernel.conns.get_connection(key)
            if conn and conn.metadata.is_tensor and not key.startswith("$"):
                kernel_input_args[key] = IOKey(key, value=kwargs.get(key, TBD))

        (kernel_output_name,) = kernel.conns.output_keys  # NOTE: Assumes single output!
        kernel_output_args = {kernel_output_name: "kernel"}

        self |= kernel.connect(**kernel_input_args, **kernel_output_args)
        self |= linear_model.connect(
            input=kernel.cout,
            weight=IOKey("weight", value=weight),
            bias=IOKey("bias", value=bias),
            output="output",
        )

        # TODO: It is not clear where these "input1" and "input2" names come from.
        # It assumes kernel model has two inputs named "input1" and "input2".
        shapes: dict[str, ShapeTemplateType] = {
            "input1": ["N", "d_in"],
            "input2": ["M", "d_in"],
            "weight": [1, "M"],
            "bias": [1],
            "output": ["N", 1],
            "kernel": ["N", "M"],
        }

        self.expose_keys("kernel", "output")
        self._set_shapes(**shapes)
        self._set_cin("input1", "input2", safe=False)
        self._freeze()

    def connect(  # type: ignore[override]
        self,
        input1: ConnectionType | Tensor[float] = NOT_GIVEN,
        input2: ConnectionType | Tensor[float] = NOT_GIVEN,
        weight: ConnectionType | Tensor[float] = NOT_GIVEN,
        bias: ConnectionType | Tensor[float] = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().connect(
            input1=input1,
            input2=input2,
            weight=weight,
            bias=bias,
            output=output,
        )


class LinearSVM(Model):
    input: Connection
    weight: Connection
    bias: Connection
    output: Connection
    decision_output: Connection

    def __init__(
        self,
        input: Tensor[int | float] | ToBeDetermined = TBD,
        weight: Tensor[float] | ToBeDetermined = TBD,
        bias: Tensor[float] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)

        linear_model = Linear(dimension=1)
        decision_model = Sign()

        self |= linear_model.connect(
            input=IOKey("input", value=input),
            weight=IOKey("weight", value=weight),
            bias=IOKey("bias", value=bias),
            output="output",
        )
        self += decision_model.connect(output="decision_output")

        self.expose_keys("output", "decision_output")
        self._set_cout(linear_model.output)
        self._freeze()

    def connect(  # type: ignore[override]
        self,
        input: ConnectionType | Tensor[int | float] = NOT_GIVEN,
        weight: ConnectionType | Tensor[float] = NOT_GIVEN,
        bias: ConnectionType | Tensor[float] = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
        decision_output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().connect(
            input=input,
            weight=weight,
            bias=bias,
            output=output,
            decision_output=decision_output,
        )


class LogisticRegression(Model):
    input: Connection
    weight: Connection
    bias: Connection
    output: Connection
    probs_output: Connection

    def __init__(
        self,
        input: Tensor[int | float] | ToBeDetermined = TBD,
        weight: Tensor[float] | ToBeDetermined = TBD,
        bias: Tensor[float] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)

        linear_model = Linear(dimension=1)
        sigmoid_model = Sigmoid()

        self |= linear_model.connect(
            input=IOKey("input", value=input),
            weight=IOKey("weight", value=weight),
            bias=IOKey("bias", value=bias),
            output="output",
        )
        self |= sigmoid_model.connect(input=linear_model.output, output="probs_output")

        self.expose_keys("output", "probs_output")
        self._set_cout(linear_model.output)
        self._freeze()

    def connect(  # type: ignore[override]
        self,
        input: ConnectionType | Tensor[int | float] = NOT_GIVEN,
        weight: ConnectionType | Tensor[float] = NOT_GIVEN,
        bias: ConnectionType | Tensor[float] = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
        probs_output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().connect(
            input=input,
            weight=weight,
            bias=bias,
            output=output,
            probs_output=probs_output,
        )


class MLP(Model):
    input: Connection
    output: Connection

    def __init__(
        self,
        activations: list[Model],
        dimensions: Sequence[int | None],
        input_name_templates: dict[str, str] | None = None,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
        **weights_biases: Tensor[int | float | bool] | ToBeDetermined,
    ) -> None:
        super().__init__(name=name)
        self.factory_args = {"activations": activations, "dimensions": dimensions}
        if len(activations) != len(dimensions):
            raise ValueError("Lengths of activations and dimensions must be equal!")
        assert len(activations) > 0, "At least one layer must be defined!"

        # Extract the keys to be used. Use "w" and "b" for default case.
        input_name_templates = input_name_templates or {}
        weight = input_name_templates.get("weight", "weight")
        bias = input_name_templates.get("bias", "bias")

        # Create first layer.
        prev_layer = Layer(activation=activations[0], dimension=dimensions[0])

        # Add first layer to the model in order to use as base for the
        # second model in the model extention loop.
        weight_key = weight + "0"
        bias_key = bias + "0"
        extend_kwargs: dict[str, ConnectionType] = {
            "input": IOKey("input", value=input),
            "weight": IOKey(weight_key, weights_biases.get(weight_key, TBD)),
            "bias": IOKey(bias_key, weights_biases.get(bias_key, TBD)),
        }
        if len(activations) == 1:
            extend_kwargs["output"] = IOKey(name="output")
        self |= prev_layer.connect(**extend_kwargs)

        # Add layers sequentially starting from second elements.
        for idx, (activation, dim) in enumerate(
            zip(activations[1:], dimensions[1:], strict=False)
        ):
            current_layer = Layer(activation=activation, dimension=dim)

            # Prepare the kwargs for the current layer.
            kwargs: dict[str, ConnectionType] = {
                "weight": f"{weight}{idx + 1}",
                "bias": f"{bias}{idx + 1}",
            }

            # In order to make last layer output as model output we must name it.
            if idx == (
                len(activations) - 2
            ):  # Loop starts to iterate from second elemets, so it is -2.
                kwargs |= {"output": "output"}

            # Add current layer to the model.
            self += current_layer.connect(**kwargs)
            prev_layer = current_layer

        self.expose_keys("output")
        self._set_cin("input", safe=False)
        self._freeze()

    def connect(  # type: ignore[override]
        self,
        input: ConnectionType | Tensor[int | float] = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
        **weights_biases: ConnectionType | Tensor[float],
    ) -> ExtendInfo:
        return super().connect(
            input=input,
            output=output,
            **weights_biases,
        )


class Cell(Model):
    shared_keys: set[str]
    private_keys: set[str]
    state_keys: set[str]
    hidden_key: str

    input: Connection
    prev_hidden: Connection
    hidden: Connection
    hidden_compl: Connection
    output: Connection

    out_key = "output"

    @abstractmethod
    def connect(
        self, **kwargs: ConnectionType | Tensor[int | float | bool] | MainValueType
    ) -> ExtendInfo:
        raise NotImplementedError("connect method not implemented!")


class RNNCell(Cell):
    input: Connection
    prev_hidden: Connection
    w_ih: Connection
    w_hh: Connection
    w_ho: Connection
    bias_h: Connection
    bias_o: Connection
    hidden: Connection
    hidden_compl: Connection
    output: Connection

    shared_keys = {"w_ih", "w_hh", "w_ho", "bias_h", "bias_o"}
    state_keys = {"hidden"}
    out_key = "output"
    # output_keys = {out, hidden_compl}

    def __init__(
        self,
        input: Tensor[int | float] | ToBeDetermined = TBD,
        w_ih: Tensor[float] | ToBeDetermined = TBD,
        w_hh: Tensor[float] | ToBeDetermined = TBD,
        w_ho: Tensor[float] | ToBeDetermined = TBD,
        bias_h: Tensor[float] | ToBeDetermined = TBD,
        bias_o: Tensor[float] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)

        shape = Shape()
        scalar_item = Indexer()
        slice_1 = Slice(stop=None, step=None)
        slice_2 = Slice(start=None, step=None)
        tensor_item_1 = Indexer()
        tensor_item_2 = Indexer()
        mult_model_1 = Linear(use_bias=False)
        mult_model_2 = Linear(use_bias=False)
        mult_model_3 = Linear(use_bias=False)
        sum_model_1 = Add()
        sum_model_2 = Add()

        self |= shape.connect(input=IOKey("input", value=input))
        self |= scalar_item.connect(input=shape.output, index=0)
        self |= slice_1.connect(start=scalar_item.output)
        self |= tensor_item_1.connect(
            input="prev_hidden",
            index=slice_1.output,
            output="hidden_compl",
        )
        self |= slice_2.connect(stop=scalar_item.output)
        self |= tensor_item_2.connect(input="prev_hidden", index=slice_2.output)
        self |= mult_model_1.connect(
            input=tensor_item_2.output,
            weight=IOKey("w_hh", value=w_hh, differentiable=True),
        )
        self |= mult_model_2.connect(
            input="input", weight=IOKey("w_ih", value=w_ih, differentiable=True)
        )
        self |= sum_model_1.connect(left=mult_model_1.output, right=mult_model_2.output)
        self |= sum_model_2.connect(
            left=sum_model_1.output,
            right=IOKey("bias_h", value=bias_h, differentiable=True),
        )
        self |= Tanh().connect(input=sum_model_2.output, output="hidden")
        self |= mult_model_3.connect(
            input="hidden", weight=IOKey("w_ho", value=w_ho, differentiable=True)
        )
        self |= Add().connect(
            left=mult_model_3.output,
            right=IOKey("bias_o", value=bias_o, differentiable=True),
            output="output",
        )
        shapes: dict[str, ShapeTemplateType] = {
            "input": ["N", 1, "d_in"],
            "prev_hidden": ["M", 1, "d_hid"],
            "w_ih": ["d_hid", "d_in"],
            "w_hh": ["d_hid", "d_hid"],
            "w_ho": ["d_out", "d_hid"],
            "bias_h": ["d_hid"],
            "bias_o": ["d_out"],
        }

        self.expose_keys("output", "hidden", "hidden_compl")
        self._set_shapes(**shapes)
        self._set_cin("input", safe=False)
        self._set_cout("output")
        self._freeze()

    def connect(  # type: ignore[override]
        self,
        input: ConnectionType | Tensor[int | float] = NOT_GIVEN,
        prev_hidden: ConnectionType | Tensor[float] = NOT_GIVEN,
        w_ih: ConnectionType | Tensor[float] = NOT_GIVEN,
        w_hh: ConnectionType | Tensor[float] = NOT_GIVEN,
        w_ho: ConnectionType | Tensor[float] = NOT_GIVEN,
        bias_h: ConnectionType | Tensor[float] = NOT_GIVEN,
        bias_o: ConnectionType | Tensor[float] = NOT_GIVEN,
        hidden: ConnectionType | Tensor[float] = NOT_GIVEN,
        hidden_compl: ConnectionType | Tensor[float] = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super(Cell, self).connect(
            input=input,
            prev_hidden=prev_hidden,
            w_ih=w_ih,
            w_hh=w_hh,
            w_ho=w_ho,
            bias_h=bias_h,
            bias_o=bias_o,
            hidden=hidden,
            hidden_compl=hidden_compl,
            output=output,
        )


class LSTMCell(Cell):
    input: Connection
    prev_hidden: Connection
    prev_cell: Connection
    w_i: Connection
    w_f: Connection
    w_c: Connection
    w_o: Connection
    w_out: Connection
    bias_f: Connection
    bias_i: Connection
    bias_c: Connection
    bias_o: Connection
    bias_out: Connection
    hidden: Connection
    cell: Connection
    hidden_compl: Connection
    output: Connection

    shared_keys = {
        "w_f",
        "w_i",
        "w_o",
        "w_c",
        "w_out",
        "bias_f",
        "bias_i",
        "bias_o",
        "bias_c",
        "bias_out",
    }
    state_keys = {"hidden", "cell"}
    out_key = "output"

    def __init__(
        self,
        input: Tensor[int | float] | ToBeDetermined = TBD,
        w_i: Tensor[float] | ToBeDetermined = TBD,
        w_f: Tensor[float] | ToBeDetermined = TBD,
        w_c: Tensor[float] | ToBeDetermined = TBD,
        w_o: Tensor[float] | ToBeDetermined = TBD,
        w_out: Tensor[float] | ToBeDetermined = TBD,
        bias_f: Tensor[float] | ToBeDetermined = TBD,
        bias_i: Tensor[float] | ToBeDetermined = TBD,
        bias_c: Tensor[float] | ToBeDetermined = TBD,
        bias_o: Tensor[float] | ToBeDetermined = TBD,
        bias_out: Tensor[float] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        factory_inputs = {
            "input": input,
            "w_i": w_i,
            "w_f": w_f,
            "w_c": w_c,
            "w_o": w_o,
            "w_out": w_out,
            "bias_f": bias_f,
            "bias_i": bias_i,
            "bias_c": bias_c,
            "bias_o": bias_o,
            "bias_out": bias_out,
        }

        cell_body = LSTMCellBody()
        shape_model = Shape()
        scalar_item = Indexer()

        slice_1 = Slice(start=None, step=None)
        slice_2 = Slice(start=None, step=None)
        slice_3 = Slice(stop=None, step=None)
        slice_4 = Slice(start=None, step=None)
        slice_5 = Slice(stop=None, step=None)

        tensor_item_1 = Indexer()
        tensor_item_2 = Indexer()
        tensor_item_3 = Indexer()
        tensor_item_4 = Indexer()
        tensor_item_5 = Indexer()

        self |= shape_model.connect(input=IOKey("input", value=input))
        self |= scalar_item.connect(input=shape_model.output, index=0)

        # Forget gate processes.
        self |= slice_1.connect(stop=scalar_item.output)
        self |= tensor_item_1.connect(input="prev_cell", index=slice_1.output)

        self |= slice_2.connect(stop=scalar_item.output)
        self |= tensor_item_2.connect(input="prev_hidden", index=slice_2.output)

        body_kwargs: dict[str, ConnectionType] = {
            key: IOKey(key, value=factory_inputs.get(key, TBD))
            for key in cell_body.input_keys
            if key[0] != "$"
        }
        body_kwargs["prev_cell"] = tensor_item_1.output
        body_kwargs["prev_hidden"] = tensor_item_2.output

        self |= cell_body.connect(**body_kwargs)

        self |= slice_3.connect(start=scalar_item.output)
        self |= tensor_item_3.connect(
            input=cell_body.output, index=slice_3.output, output="hidden"
        )

        self |= slice_4.connect(stop=scalar_item.output)
        self |= tensor_item_4.connect(
            input=cell_body.output, index=slice_4.output, output="cell"
        )

        # Slice complement process.
        self |= slice_5.connect(start=scalar_item.output)
        self |= tensor_item_5.connect(
            input="prev_hidden",
            index=slice_5.output,
            output="hidden_compl",
        )
        # Final output.
        self |= Linear().connect(
            input="hidden",
            weight=IOKey("w_out", value=w_out),
            bias=IOKey("bias_out", value=bias_out),
            output="output",
        )
        shapes: dict[str, ShapeTemplateType] = {
            "input": ["N", 1, "d_in"],
            "prev_hidden": ["M", 1, "d_hid"],
            "prev_cell": ["M", 1, "d_hid"],
            "w_i": ["d_hid", "d_sum"],
            "w_f": ["d_hid", "d_sum"],
            "w_c": ["d_hid", "d_sum"],
            "w_o": ["d_hid", "d_sum"],
            "w_out": ["d_out", "d_hid"],
            "bias_f": ["d_hid"],
            "bias_i": ["d_hid"],
            "bias_c": ["d_hid"],
            "bias_o": ["d_hid"],
            "bias_out": ["d_out"],
            "hidden": ["N", 1, "d_hid"],
            "cell": ["N", 1, "d_hid"],
        }

        self.expose_keys("hidden", "cell", "hidden_compl", "output")
        self._set_shapes(**shapes)
        self._set_cin("input", safe=False)
        self._set_cout("output")
        self._freeze()

    def connect(  # type: ignore[override]
        self,
        input: ConnectionType | Tensor[int | float] = NOT_GIVEN,
        prev_hidden: ConnectionType | Tensor[float] = NOT_GIVEN,
        prev_cell: ConnectionType | Tensor[float] = NOT_GIVEN,
        w_i: ConnectionType | Tensor[float] = NOT_GIVEN,
        w_f: ConnectionType | Tensor[float] = NOT_GIVEN,
        w_c: ConnectionType | Tensor[float] = NOT_GIVEN,
        w_o: ConnectionType | Tensor[float] = NOT_GIVEN,
        w_out: ConnectionType | Tensor[float] = NOT_GIVEN,
        bias_f: ConnectionType | Tensor[float] = NOT_GIVEN,
        bias_i: ConnectionType | Tensor[float] = NOT_GIVEN,
        bias_c: ConnectionType | Tensor[float] = NOT_GIVEN,
        bias_o: ConnectionType | Tensor[float] = NOT_GIVEN,
        bias_out: ConnectionType | Tensor[float] = NOT_GIVEN,
        hidden: ConnectionType | Tensor[float] = NOT_GIVEN,
        cell: ConnectionType | Tensor[float] = NOT_GIVEN,
        hidden_compl: ConnectionType | Tensor[float] = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super(Cell, self).connect(
            input=input,
            prev_hidden=prev_hidden,
            prev_cell=prev_cell,
            w_i=w_i,
            w_f=w_f,
            w_c=w_c,
            w_o=w_o,
            w_out=w_out,
            bias_f=bias_f,
            bias_i=bias_i,
            bias_c=bias_c,
            bias_o=bias_o,
            bias_out=bias_out,
            hidden=hidden,
            cell=cell,
            hidden_compl=hidden_compl,
            output=output,
        )


class LSTMCellBody(Model):
    input: Connection
    prev_hidden: Connection
    prev_cell: Connection
    w_i: Connection
    w_f: Connection
    w_c: Connection
    w_o: Connection
    bias_f: Connection
    bias_i: Connection
    bias_c: Connection
    bias_o: Connection
    output: Connection

    def __init__(
        self,
        input: Tensor[int | float] | ToBeDetermined = TBD,
        prev_hidden: Tensor[float] | ToBeDetermined = TBD,
        prev_cell: Tensor[float] | ToBeDetermined = TBD,
        w_i: Tensor[float] | ToBeDetermined = TBD,
        w_f: Tensor[float] | ToBeDetermined = TBD,
        w_c: Tensor[float] | ToBeDetermined = TBD,
        w_o: Tensor[float] | ToBeDetermined = TBD,
        bias_f: Tensor[float] | ToBeDetermined = TBD,
        bias_i: Tensor[float] | ToBeDetermined = TBD,
        bias_c: Tensor[float] | ToBeDetermined = TBD,
        bias_o: Tensor[float] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)

        matrix_concat_model = Concat(axis=-1)
        forward_lin = Linear()
        sigmoid_model_1 = Sigmoid()
        mult_model_1 = Multiply()
        input_lin = Linear()
        sigmoid_model_2 = Sigmoid()
        cell_lin = Linear()
        tanh_model_1 = Tanh()
        mult_model_2 = Multiply()
        sum_model_4 = Add()
        tanh_model_2 = Tanh()
        out_gate_lin = Linear()
        sigmoid_model_3 = Sigmoid()
        mult_model_3 = Multiply()

        self += matrix_concat_model.connect(
            input=[
                IOKey("input", value=input),
                IOKey("prev_hidden", value=prev_hidden),
            ],
        )
        self |= forward_lin.connect(
            input=matrix_concat_model.output,
            weight=IOKey("w_f", value=w_f),
            bias=IOKey("bias_f", value=bias_f),
        )
        self |= sigmoid_model_1.connect(input=forward_lin.output)
        self |= mult_model_1.connect(
            left=IOKey("prev_cell", value=prev_cell), right=sigmoid_model_1.output
        )
        # Input gate processes.
        self |= input_lin.connect(
            input=matrix_concat_model.output,
            weight=IOKey("w_i", value=w_i),
            bias=IOKey("bias_i", value=bias_i),
        )
        self |= sigmoid_model_2.connect(input=input_lin.output)
        # Cell state gate processes.
        self |= cell_lin.connect(
            input=matrix_concat_model.output,
            weight=IOKey("w_c", value=w_c),
            bias=IOKey("bias_c", value=bias_c),
        )
        self |= tanh_model_1.connect(input=cell_lin.output)
        # Input-cell gate multiplication.
        self |= mult_model_2.connect(
            left=sigmoid_model_2.output, right=tanh_model_1.output
        )
        # Addition to cell state.
        self |= sum_model_4.connect(left=mult_model_1.output, right=mult_model_2.output)
        # Cell state to hidden state info.
        self |= tanh_model_2.connect(input=sum_model_4.output)
        # Output gate process.
        self |= out_gate_lin.connect(
            input=matrix_concat_model.output,
            weight=IOKey("w_o", value=w_o),
            bias=IOKey("bias_o", value=bias_o),
        )
        self |= sigmoid_model_3.connect(input=out_gate_lin.output)
        # Final hidden state.
        self |= mult_model_3.connect(
            left=tanh_model_2.output, right=sigmoid_model_3.output
        )
        self |= Concat(axis=0).connect(
            input=[sum_model_4.output, mult_model_3.output],
            output=IOKey(name="output"),
        )
        shapes: dict[str, ShapeTemplateType] = {
            "input": ["N", 1, "d_in"],
            "prev_hidden": ["N", 1, "d_hid"],
            "prev_cell": ["N", 1, "d_hid"],
            "w_i": ["d_hid", "d_sum"],
            "w_f": ["d_hid", "d_sum"],
            "w_c": ["d_hid", "d_sum"],
            "w_o": ["d_hid", "d_sum"],
            "bias_f": ["d_hid"],
            "bias_i": ["d_hid"],
            "bias_c": ["d_hid"],
            "bias_o": ["d_hid"],
        }

        self._set_shapes(**shapes)
        self._freeze()

    def connect(  # type: ignore[override]
        self,
        input: ConnectionType | Tensor[int | float] = NOT_GIVEN,
        prev_hidden: ConnectionType | Tensor[float] = NOT_GIVEN,
        prev_cell: ConnectionType | Tensor[float] = NOT_GIVEN,
        w_i: ConnectionType | Tensor[float] = NOT_GIVEN,
        w_f: ConnectionType | Tensor[float] = NOT_GIVEN,
        w_c: ConnectionType | Tensor[float] = NOT_GIVEN,
        w_o: ConnectionType | Tensor[float] = NOT_GIVEN,
        bias_f: ConnectionType | Tensor[float] = NOT_GIVEN,
        bias_i: ConnectionType | Tensor[float] = NOT_GIVEN,
        bias_c: ConnectionType | Tensor[float] = NOT_GIVEN,
        bias_o: ConnectionType | Tensor[float] = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().connect(
            input=input,
            prev_hidden=prev_hidden,
            prev_cell=prev_cell,
            w_i=w_i,
            w_f=w_f,
            w_c=w_c,
            w_o=w_o,
            bias_f=bias_f,
            bias_i=bias_i,
            bias_c=bias_c,
            bias_o=bias_o,
            output=output,
        )


class RNN(Model):
    def __init__(
        self,
        cell_type: Cell,
        *,
        name: str | None = None,
        # **kwargs: Tensor[int | float | bool] | MainValueType,
    ) -> None:
        self.cell_type = cell_type
        super().__init__(name=name)
        # self.set_values(**kwargs)

    def connect(self, **kwargs: ConnectionType) -> ExtendInfo:  # type: ignore[override]
        raise NotImplementedError("connect method not implemented!")


class OneToMany(RNN):
    input: Connection

    def __init__(
        self,
        cell_type: Cell,
        max_sequence_length: int,
        teacher_forcing: bool = False,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
        **kwargs: Tensor[int | float | bool] | MainValueType,
    ) -> None:
        super().__init__(cell_type=cell_type, name=name)

        cell = deepcopy(cell_type)
        prev_cell = cell

        shared_keys_kwargs = {
            key: IOKey(key, value=kwargs.get(key, TBD)) for key in cell_type.shared_keys
        }
        output_kwargs = {cell_type.out_key: "output0"}
        input_kwargs: dict[str, ConnectionType] = {"input": IOKey("input", value=input)}
        initial_state_kwargs = {
            f"prev_{key}": IOKey(
                f"initial_{key}", value=kwargs.get(f"initial_{key}", TBD)
            )
            for key in cell_type.state_keys
        }
        exposed_keys = ["output0"]

        self += prev_cell.connect(
            **(input_kwargs | shared_keys_kwargs | output_kwargs | initial_state_kwargs)
        )

        for idx in range(1, max_sequence_length):
            current_cell = deepcopy(cell_type)
            state_keys_kwargs = {
                f"prev_{key}": getattr(prev_cell, key) for key in cell_type.state_keys
            }
            # Create slicing model which filters unnecessary data for
            # current time step.
            shape_model = Shape()
            item_model = Indexer()
            slice_model = Slice(start=None, step=None)
            tensor_item = Indexer()

            self |= shape_model.connect(input=f"target{idx}")
            self |= item_model.connect(input=shape_model.output, index=0)

            # Create slicing model which filters unnecessary data for
            # current time step.
            if teacher_forcing:
                # Teacher forcing approach requires targets of previous
                # time step as inputs to the current time step.
                slice_input_1 = f"target{idx - 1}"
            else:
                # When not using teacher forcing, simply take outputs
                # of previous time step as inputs to the current time step.
                slice_input_1 = getattr(prev_cell, prev_cell.out_key)

            self |= slice_model.connect(stop=item_model.output)
            self |= tensor_item.connect(input=slice_input_1, index=slice_model.output)

            input_kwargs = {"input": tensor_item.output}
            out_name = f"output{idx}"
            output_kwargs = {cell_type.out_key: out_name}
            exposed_keys.append(out_name)

            self |= current_cell.connect(
                **(
                    input_kwargs
                    | shared_keys_kwargs
                    | state_keys_kwargs
                    | output_kwargs
                )
            )

            prev_cell = current_cell

        self.expose_keys(*exposed_keys)
        self._set_cin("input")
        self._set_cout(current_cell.output)
        self._freeze()

    def connect(  # type: ignore[override]
        self,
        input: ConnectionType | Tensor[int | float] = NOT_GIVEN,
        **model_keys: ConnectionType | MainValueType | Tensor[int | float | bool],
    ) -> ExtendInfo:
        return super(RNN, self).connect(input=input, **model_keys)


class OneToManyInference(RNN):
    input: Connection

    def __init__(
        self,
        cell_type: Cell,
        max_sequence_length: int,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
        **kwargs: Tensor[int | float | bool] | ToBeDetermined,
    ) -> None:
        super().__init__(cell_type=cell_type, name=name)

        cell = deepcopy(cell_type)
        prev_cell = cell

        shared_keys_kwargs = {key: key for key in cell_type.shared_keys}
        output_kwargs = {cell_type.out_key: "output0"}
        input_kwargs: dict[str, ConnectionType] = {"input": IOKey("input", value=input)}
        initial_state_kwargs = {
            f"prev_{key}": IOKey(
                f"initial_{key}", value=kwargs.get(f"initial_{key}", TBD)
            )
            for key in cell_type.state_keys
        }
        self += prev_cell.connect(
            **(input_kwargs | shared_keys_kwargs | output_kwargs | initial_state_kwargs)
        )
        exposed_keys = ["output0"]
        for idx in range(1, max_sequence_length):
            current_cell = deepcopy(cell_type)

            state_keys_kwargs = {
                f"prev_{key}": getattr(prev_cell, key) for key in cell_type.state_keys
            }
            output_kwargs = {cell_type.out_key: f"output{idx}"}

            self += current_cell.connect(
                **(shared_keys_kwargs | state_keys_kwargs | output_kwargs)
            )

            prev_cell = current_cell
            exposed_keys.append(f"output{idx}")

        self.expose_keys(*exposed_keys)
        self._freeze()

    def connect(  # type: ignore[override]
        self,
        input: ConnectionType | Tensor[int | float] = NOT_GIVEN,
        **model_keys: ConnectionType | Tensor[int | float | bool] | MainValueType,
    ) -> ExtendInfo:
        return super(RNN, self).connect(input=input, **model_keys)


class ManyToOne(RNN):
    hidden_concat: Connection

    def __init__(
        self,
        cell_type: Cell,
        max_sequence_length: int,
        hidden_concat: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
        **kwargs: Tensor[int | float | bool] | ToBeDetermined,
    ) -> None:
        super().__init__(cell_type, name=name)

        prev_cell = deepcopy(cell_type)

        concat_model = Concat()
        concat_input_args: list[ConnectionType] = []
        shared_keys_kwargs = {key: key for key in cell_type.shared_keys}
        output_kwargs = {cell_type.out_key: "output0"}
        input_kwargs = {"input": IOKey("input0", value=kwargs.get("input0", TBD))}
        initial_state_kwargs = {
            f"prev_{key}": IOKey(
                f"initial_{key}", value=kwargs.get(f"initial_{key}", TBD)
            )
            for key in cell_type.state_keys
        }
        exposed_keys = ["output0"]

        self |= prev_cell.connect(
            **(input_kwargs | shared_keys_kwargs | output_kwargs | initial_state_kwargs)
        )

        for idx in range(1, max_sequence_length):
            cur_cell = deepcopy(cell_type)
            state_keys_kwargs = {
                f"prev_{key}": getattr(prev_cell, key) for key in cell_type.state_keys
            }
            input_kwargs = {
                "input": IOKey(f"input{idx}", value=kwargs.get(f"input{idx}", TBD))
            }
            output_kwargs = {cell_type.out_key: f"output{idx}"}

            # For the last cell, include hidden
            self |= cur_cell.connect(
                **(
                    input_kwargs
                    | shared_keys_kwargs
                    | state_keys_kwargs
                    | output_kwargs
                )
            )

            self.expose_keys(f"output{idx}")
            # For the last cell, include hidden
            if idx < max_sequence_length - 1:
                concat_input_args.append(cur_cell.hidden_compl)
            else:
                concat_input_args.extend([cur_cell.hidden, cur_cell.hidden_compl])

            prev_cell = cur_cell
            exposed_keys.append(f"output{idx}")

        # Add concat model with accumulated hidden states.
        self |= concat_model.connect(
            input=concat_input_args,
            output=IOKey(name="hidden_concat", value=hidden_concat),
        )
        exposed_keys.append("hidden_concat")

        self.expose_keys(*exposed_keys)
        self._set_cin("input0")
        self._set_cout("hidden_concat")
        self._freeze()

    def connect(  # type: ignore[override]
        self,
        hidden_concat: ConnectionType | Tensor[float] = NOT_GIVEN,
        **model_keys: ConnectionType,
    ) -> ExtendInfo:
        return super(RNN, self).connect(hidden_concat=hidden_concat, **model_keys)


class EncoderDecoder(Model):
    indices: Connection

    def __init__(
        self,
        cell_type: Cell,
        max_input_sequence_length: int,
        max_target_sequence_length: int,
        teacher_forcing: bool = False,
        indices: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)

        # Encoder Model
        encoder = ManyToOne(
            cell_type=cell_type, max_sequence_length=max_input_sequence_length
        )

        # Decoder Model
        decoder = OneToMany(
            cell_type=cell_type,
            max_sequence_length=max_target_sequence_length,
            teacher_forcing=teacher_forcing,
        )

        permutation_model = PermuteTensor()

        enc_input_mapping = {key: key for key in encoder.input_keys if "$" not in key}

        dec_input_mapping = {
            key: "decoder_" + key if "target" not in key else key
            for key in decoder.input_keys
            if "$" not in key and key != "initial_hidden"
        }

        dec_output_mapping = {key: key for key in decoder.conns.output_keys}

        self |= encoder.connect(**enc_input_mapping)
        self |= permutation_model.connect(
            input=encoder.hidden_concat, indices=IOKey("indices", value=indices)
        )
        self |= decoder.connect(
            initial_hidden=permutation_model.output,
            **(dec_input_mapping | dec_output_mapping),
        )

        self.expose_keys(*dec_output_mapping.values())
        self._set_cout(decoder.cout)
        self._freeze()

    def connect(  # type: ignore[override]
        self,
        indices: ConnectionType | Tensor[int] = NOT_GIVEN,
        **model_keys: ConnectionType,
    ) -> ExtendInfo:
        return super().connect(indices=indices, **model_keys)


class EncoderDecoderInference(Model):
    indices: Connection

    def __init__(
        self,
        cell_type: Cell,
        max_input_sequence_length: int,
        max_target_sequence_length: int,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)

        # Encoder Model
        encoder = ManyToOne(
            cell_type=cell_type, max_sequence_length=max_input_sequence_length
        )

        # Decoder Model
        decoder = OneToManyInference(
            cell_type=cell_type, max_sequence_length=max_target_sequence_length
        )

        enc_input_mapping = {key: key for key in encoder.input_keys if "$" not in key}

        dec_input_mapping = {
            key: "decoder_" + key if "target" not in key else key
            for key in decoder.input_keys
            if "$" not in key and key != "initial_hidden"
        }

        dec_output_mapping = {key: key for key in decoder.conns.output_keys}

        self |= encoder.connect(**enc_input_mapping)
        self |= decoder.connect(
            initial_hidden=encoder.hidden_concat,
            **(dec_input_mapping | dec_output_mapping),
        )
        self.expose_keys(*dec_output_mapping.values())
        self._set_cout(decoder.cout)
        self._freeze()

    def connect(self, **model_keys: ConnectionType) -> ExtendInfo:  # type: ignore[override]
        return super().connect(**model_keys)


class EncoderDistanceMatrix(Model):
    input1: Connection
    input2: Connection
    norm: Connection
    output: Connection

    def __init__(
        self,
        get_final_distance: bool = True,
        robust: bool = True,
        input1: Tensor[int | float | bool] | ToBeDetermined = TBD,
        input2: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.factory_args = {"get_final_distance": get_final_distance, "robust": robust}

        dist_model = DistanceMatrix()
        modifier_model = NormModifier()
        input1_key = IOKey("input1", value=input1)
        input2_key = IOKey("input2", value=input2)
        if get_final_distance:
            reciprocal_model = Divide()
            power_model = Power(robust=robust)

            self |= modifier_model.connect(input="norm")
            self |= dist_model.connect(
                left=input1_key, right=input2_key, norm=modifier_model.output
            )
            self |= reciprocal_model.connect(
                numerator=1.0, denominator=modifier_model.output
            )
            self |= power_model.connect(
                base=dist_model.output,
                exponent=reciprocal_model.output,
                output=IOKey(name="output"),
            )

        else:
            self |= modifier_model.connect(input="norm")
            self |= dist_model.connect(
                left="input1",
                right="input2",
                norm=modifier_model.output,
                output=IOKey(name="output"),
            )
        self._set_cin("input1", "input2", safe=False)
        self._freeze()

    def connect(  # type: ignore[override]
        self,
        input1: ConnectionType | Tensor[int | float | bool] = NOT_GIVEN,
        input2: ConnectionType | Tensor[int | float | bool] = NOT_GIVEN,
        norm: ConnectionType | Tensor[int | float | bool] = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().connect(input1=input1, input2=input2, norm=norm, output=output)


class PolynomialRegression(Model):
    input: Connection
    weight: Connection
    bias: Connection
    output: Connection

    def __init__(
        self,
        degree: int,
        dimension: int | None = None,
        input: Tensor[int | float] | ToBeDetermined = TBD,
        weight: Tensor[float] | ToBeDetermined = TBD,
        bias: Tensor[float] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.factory_args = {"degree": degree, "dimension": dimension}

        linear_model = Linear(dimension=dimension)
        feature_model = PolynomialFeatures(degree=degree)

        self |= feature_model.connect(input=IOKey("input", value=input))
        self |= linear_model.connect(
            input=feature_model.output,
            weight=IOKey("weight", value=weight),
            bias=IOKey("bias", value=bias),
            output=IOKey(name="output"),
        )

        self._freeze()

    def connect(  # type: ignore[override]
        self,
        input: ConnectionType | Tensor[int | float] = NOT_GIVEN,
        weight: ConnectionType | Tensor[float] = NOT_GIVEN,
        bias: ConnectionType | Tensor[float] = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().connect(input=input, weight=weight, bias=bias, output=output)


class MDSCore(Model):
    distances: Connection
    pred_distances: Connection
    norm: Connection
    output: Connection

    requires_norm: bool = True

    def __init__(
        self,
        exact_distances: bool = True,
        robust: bool = True,
        distances: Tensor[float] | ToBeDetermined = TBD,
        pred_distances: Tensor[float] | ToBeDetermined = TBD,
        norm: Tensor[float] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        self.factory_args = {"exact_distances": exact_distances, "robust": robust}
        super().__init__(name=name)

        # Prepare models used in MDS.
        subtract_model = Subtract()
        abs_model = Absolute()
        norm_model = NormModifier()
        power_model_1 = Power(robust=robust)
        power_model_2 = Power(robust=robust)
        power_model_3 = Power(robust=robust)
        power_model_4 = Power(robust=robust)
        sum_model_1 = Sum()
        sum_model_2 = Sum()
        reciprocal_model_1 = StableReciprocal()
        reciprocal_model_2 = StableReciprocal()
        mult_model = Multiply()

        if exact_distances:
            self |= norm_model.connect(input=IOKey("norm", value=norm))
            self |= reciprocal_model_1.connect(input=norm_model.output)
            self |= power_model_4.connect(
                base=IOKey("pred_distances", value=pred_distances),
                exponent=reciprocal_model_1.output,
            )
            self |= subtract_model.connect(
                left=IOKey("distances", value=distances), right=power_model_4.output
            )
            self |= abs_model.connect(input=subtract_model.output)
            self |= power_model_1.connect(
                base=abs_model.output, exponent=norm_model.output
            )
            self |= sum_model_1.connect(input=power_model_1.output)
            self |= power_model_2.connect(
                base=self.distances, exponent=norm_model.output
            )
            self |= sum_model_2.connect(input=power_model_2.output)
            self |= reciprocal_model_2.connect(input=sum_model_2.output)
            self |= mult_model.connect(
                left=sum_model_1.output, right=reciprocal_model_2.output
            )
            self |= power_model_3.connect(
                base=mult_model.output,
                exponent=reciprocal_model_1.output,
                output=IOKey(name="output"),
            )

        else:
            self |= norm_model.connect(input="norm")
            self |= reciprocal_model_1.connect(input=norm_model.output)
            self |= power_model_1.connect(
                base="distances", exponent=reciprocal_model_1.output
            )
            self |= power_model_4.connect(
                base="pred_distances", exponent=reciprocal_model_1.output
            )
            self |= subtract_model.connect(
                left=power_model_1.output, right=power_model_4.output
            )
            self |= abs_model.connect(input=subtract_model.output)
            self |= power_model_2.connect(
                base=abs_model.output, exponent=norm_model.output
            )
            self |= sum_model_1.connect(input=power_model_2.output)
            self |= sum_model_2.connect(input=self.distances)
            self |= reciprocal_model_2.connect(input=sum_model_2.output)
            self |= mult_model.connect(
                left=sum_model_1.output, right=reciprocal_model_2.output
            )
            self |= power_model_3.connect(
                base=mult_model.output,
                exponent=reciprocal_model_1.output,
                output=IOKey(name="output"),
            )

        self._set_shapes(distances=["N", "N"], pred_distances=["N", "N"])
        self._freeze()

    def connect(  # type: ignore[override]
        self,
        distances: ConnectionType | Tensor[float] = NOT_GIVEN,
        pred_distances: ConnectionType | Tensor[float] = NOT_GIVEN,
        norm: ConnectionType | Tensor[float] = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().connect(
            distances=distances,
            pred_distances=pred_distances,
            norm=norm,
            output=output,
        )


class TSNECore(Model):
    distances: Connection
    pred_distances: Connection
    p_joint: Connection
    output: Connection

    requires_norm: bool = False

    def __init__(
        self,
        exact_distances: bool = True,
        calculate_p_joint: bool = False,
        perplexity: float = 20.0,
        distances: Tensor[float] | ToBeDetermined = TBD,
        pred_distances: Tensor[float] | ToBeDetermined = TBD,
        p_joint: Tensor[float] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.factory_args = {
            "exact_distances": exact_distances,
            "perplexity": perplexity,
        }

        p_joint_model = TsnePJoint()
        divide_model_1 = Divide()
        divide_model_2 = Divide()
        sum_model_1 = Add()
        sum_model_2 = Sum()
        sum_model_3 = Sum()
        size_model = Size(dim=0)
        zero_diagonal_model = EyeComplement()
        mult_model = Multiply()
        kl_divergence_model = KLDivergence()

        dist_key = IOKey("distances", value=distances)
        pred_dist_key = IOKey("pred_distances", value=pred_distances)
        # Always process with squared distances in TSNE calculations.
        if exact_distances:
            square_model = Square()
            self |= square_model.connect(input=dist_key)
            if calculate_p_joint:
                self |= p_joint_model.connect(
                    squared_distances=square_model.output, target_perplexity=perplexity
                )
        else:
            if calculate_p_joint:
                self |= p_joint_model.connect(
                    squared_distances=dist_key, target_perplexity=perplexity
                )
        self |= sum_model_1.connect(left=1.0, right=pred_dist_key)
        self |= divide_model_1.connect(numerator=1.0, denominator=sum_model_1.output)
        self |= size_model.connect(input=dist_key)
        self |= zero_diagonal_model.connect(N=size_model.output)
        self |= mult_model.connect(
            left=divide_model_1.output, right=zero_diagonal_model.output
        )
        self |= sum_model_2.connect(input=mult_model.output)
        self |= divide_model_2.connect(
            numerator=mult_model.output, denominator=sum_model_2.output
        )
        self |= kl_divergence_model.connect(
            input=divide_model_2.output,
            target=p_joint_model.output
            if calculate_p_joint
            else IOKey("p_joint", value=p_joint),
        )
        self |= sum_model_3.connect(
            input=kl_divergence_model.output, output=IOKey(name="output")
        )

        self._set_shapes(distances=["N", "N"], pred_distances=["N", "N"])
        self._set_cin("distances", safe=False)
        self._set_cout("output")
        self._freeze()

    def connect(  # type: ignore[override]
        self,
        distances: ConnectionType | Tensor[float] = NOT_GIVEN,
        pred_distances: ConnectionType | Tensor[float] = NOT_GIVEN,
        p_joint: ConnectionType | Tensor[float] = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        kwargs = {
            "distances": distances,
            "pred_distances": pred_distances,
            "output": output,
        }

        if "p_joint" in self.input_keys:
            kwargs["p_joint"] = p_joint
        elif p_joint != NOT_GIVEN:
            raise ValueError("p_joint is only required when calculate_p_joint is True!")

        return super().connect(**kwargs)


class DistanceEncoder(Model):
    input: Connection
    coords: Connection
    norm: Connection
    predicted_coords: Connection
    output: Connection

    ephemeral: bool = True

    def __init__(
        self,
        base_model: MDSCore | TSNECore,
        input_type: str = "distances",
        input: Tensor[float] | ToBeDetermined = TBD,
        coords: Tensor[float] | ToBeDetermined = TBD,
        norm: Tensor[float] | ToBeDetermined = TBD,
        predicted_coords: Tensor[float] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.factory_args = {"base_model": base_model, "input_type": input_type}

        assert input_type in ["distances", "powered_distances", "points"]

        coords_distance_matrix = EncoderDistanceMatrix(get_final_distance=False)
        buffer_model = Buffer()

        # NOTE: We should assert a standard naming for inputs to
        #  the base model (i.e. "distances", "pred_distances")
        if input_type == "points":
            input_distance_matrix = EncoderDistanceMatrix(get_final_distance=False)
            self |= input_distance_matrix.connect(
                input1=IOKey("input", value=input),
                input2="input",
                norm=IOKey("norm", value=norm),
            )
            self |= coords_distance_matrix.connect(
                input1=IOKey("coords", value=coords), input2="coords", norm="norm"
            )

            base_kwargs: dict[str, ConnectionType] = {
                "distances": input_distance_matrix.output,
                "pred_distances": coords_distance_matrix.output,
                "output": IOKey(name="output"),
            }
            # Create inputs taking "requires_norm" attribute of base model class.
            if base_model.requires_norm:
                base_kwargs["norm"] = "norm"

            for key in base_model.input_keys:
                con = base_model.conns.get_connection(key)
                assert con is not None
                if key not in base_kwargs and not con.is_autogenerated:
                    base_kwargs[key] = key

            self |= base_model.connect(**base_kwargs)
            self |= buffer_model.connect(
                input=self.coords,
                output=IOKey(name="predicted_coords", value=predicted_coords),
            )

        else:
            self |= coords_distance_matrix.connect(
                input1="coords", input2="coords", norm="norm"
            )

            # Create inputs taking "requires_norm" attribute of base model class.
            base_kwargs = {
                "distances": "input",
                "pred_distances": coords_distance_matrix.output,
                "output": IOKey(name="output"),
            }
            if base_model.requires_norm:
                base_kwargs["norm"] = "norm"

            self |= base_model.connect(**base_kwargs)
            self |= buffer_model.connect(
                input=self.coords, output=IOKey(name="predicted_coords")
            )

        self._freeze()
        # self._set_shapes(trace=False,
        #     input = ["N", "M"], # NOTE: Here "M" denotes input dim or
        #     sample size ("N") depending on input_type.
        #     coords = ["N", "d"]
        # )

    def connect(  # type: ignore[override]
        self,
        input: ConnectionType | Tensor[float] = NOT_GIVEN,
        coords: ConnectionType | Tensor[float] = NOT_GIVEN,
        norm: ConnectionType | Tensor[float] = NOT_GIVEN,
        predicted_coords: ConnectionType | Tensor[float] = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        kwargs = {
            "coords": coords,
            "norm": norm,
            "predicted_coords": predicted_coords,
            "output": output,
        }
        if "input" in self.input_keys:
            kwargs["input"] = input
        elif coords != NOT_GIVEN:
            raise ValueError("coords is only required when input_type is 'points'!")

        return super().connect(**kwargs)


class MDS(DistanceEncoder):
    input: Connection
    coords: Connection
    norm: Connection
    predicted_coords: Connection
    output: Connection

    def __init__(
        self,
        prediction_dim: int,
        input_type: str = "distances",
        input: Tensor[float] | ToBeDetermined = TBD,
        coords: Tensor[float] | ToBeDetermined = TBD,
        norm: Tensor[float] | ToBeDetermined = TBD,
        predicted_coords: Tensor[float] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        assert input_type in ["distances", "powered_distances", "points"]
        base_model = MDSCore(exact_distances=(input_type == "distances"))
        super().__init__(
            base_model=base_model,
            input_type=input_type,
            name=name,
            input=input,
            coords=coords,
            norm=norm,
            predicted_coords=predicted_coords,
        )
        self.factory_args = {"prediction_dim": prediction_dim, "input_type": input_type}
        self._set_shapes(coords=[None, prediction_dim])
        self._freeze()

    def connect(  # type: ignore[override]
        self,
        input: ConnectionType | Tensor[float] = NOT_GIVEN,
        coords: ConnectionType | Tensor[float] = NOT_GIVEN,
        norm: ConnectionType | Tensor[float] = NOT_GIVEN,
        predicted_coords: ConnectionType | Tensor[float] = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        kwargs: dict[str, ConnectionType | Tensor[float]] = {
            "input": input,
            "norm": norm,
            "predicted_coords": predicted_coords,
            "output": output,
        }

        if "coords" in self.input_keys:
            kwargs["coords"] = coords
        elif coords != NOT_GIVEN:
            raise ValueError("coords is only required when input_type is 'points'!")

        return super().connect(**kwargs)  # type: ignore


class TSNE(DistanceEncoder):
    input: Connection
    norm: Connection
    predicted_coords: Connection
    output: Connection

    # TODO: TSNE norm is always 2. Should we handle this automatically?
    def __init__(
        self,
        prediction_dim: int,
        input_type: str = "distances",
        preplexity: float = 20.0,
        calculate_p_joint: bool = False,
        input: Tensor[float] | ToBeDetermined = TBD,
        norm: Tensor[float] | ToBeDetermined = TBD,
        predicted_coords: Tensor[float] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        assert input_type in ["distances", "powered_distances", "points"]
        base_model = TSNECore(
            calculate_p_joint=calculate_p_joint,
            perplexity=preplexity,
            exact_distances=(input_type == "distances"),
        )
        super().__init__(
            base_model=base_model,
            input_type=input_type,
            name=name,
            input=input,
            norm=norm,
            predicted_coords=predicted_coords,
        )
        self.factory_args = {
            "prediction_dim": prediction_dim,
            "input_type": input_type,
            "preplexity": preplexity,
        }

        self._set_shapes(coords=[None, prediction_dim])
        self._freeze()

    def connect(  # type: ignore[override]
        self,
        input: ConnectionType | Tensor[float] = NOT_GIVEN,
        norm: ConnectionType | Tensor[float] = NOT_GIVEN,
        predicted_coords: ConnectionType | Tensor[float] = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().connect(
            input=input,
            norm=norm,
            predicted_coords=predicted_coords,
            output=output,
        )


class GaussProcessRegressionCore(Model):
    label: Connection
    s: Connection
    k: Connection
    k_star: Connection
    mu: Connection
    loss: Connection
    prediction: Connection
    confidence: Connection

    def __init__(
        self,
        s: Tensor[int | float | bool] | ToBeDetermined = TBD,
        k: Tensor[int | float | bool] | ToBeDetermined = TBD,
        k_star: Tensor[int | float | bool] | ToBeDetermined = TBD,
        mu: Tensor[int | float | bool] | ToBeDetermined = TBD,
        label: Tensor[int | float | bool] | ToBeDetermined = TBD,
        loss: Tensor[int | float | bool] | ToBeDetermined = TBD,
        prediction: Tensor[int | float | bool] | ToBeDetermined = TBD,
        confidence: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)

        # Prepare models used in GPR.
        size_model = Size(dim=0)
        K_term_eye_model = Eye()
        K_term_mult_model = Multiply()
        K_term_model = Add()
        L_term_model = Cholesky()
        label_mu_diff_model = Subtract()
        alpha_model = GPRAlpha()
        gprloss_model = GPRLoss()
        pred_t_model = Transpose()
        pred_dot_model = MatrixMultiply()
        pred_model = Add()
        conf_v_outer_model = GPRVOuter()
        conf_sub_model = Subtract()
        conf_abs_model = Absolute()
        conf_diag_model = TransposedDiagonal()
        conf_model = Add()

        self |= size_model.connect(input=IOKey("k", value=k))
        self += K_term_eye_model.connect(N=size_model.output)
        self += K_term_mult_model.connect(
            left=IOKey("s", value=s), right=K_term_eye_model.output
        )
        self += K_term_model.connect(left=self.k, right=K_term_mult_model.output)
        self += L_term_model.connect(input=K_term_model.output)
        self += label_mu_diff_model.connect(
            left=IOKey("label", value=label), right=IOKey("mu", value=mu)
        )
        self += alpha_model.connect(
            label_mu_diff=label_mu_diff_model.output,
            L=L_term_model.output,
            K_term=K_term_model.output,
        )
        # Loss Model.
        self += gprloss_model.connect(
            labels=self.label,
            mu=self.mu,
            L=L_term_model.output,
            K_term=K_term_model.output,
            alpha=alpha_model.output,
            output=IOKey(name="loss", value=loss),
        )
        # Prediction Pipeline.
        self += pred_t_model.connect(input=self.k)
        self += pred_dot_model.connect(
            left=pred_t_model.output, right=alpha_model.output
        )
        self += pred_model.connect(
            left=self.mu,
            right=pred_dot_model.output,
            output=IOKey(name="prediction", value=prediction),
        )
        # Confidence Pipeline.
        self += conf_v_outer_model.connect(
            K=self.k, L=L_term_model.output, K_term=K_term_model.output
        )
        self += conf_sub_model.connect(
            left=IOKey("k_star", value=k_star), right=conf_v_outer_model.output
        )
        self += conf_diag_model.connect(input=conf_sub_model.output)
        self += conf_abs_model.connect(input=conf_diag_model.output)
        self += conf_model.connect(
            left=self.s,
            right=conf_abs_model.output,
            output=IOKey(name="confidence", value=confidence),
        )

        self._set_cout(pred_model.output)
        shapes: dict[str, ShapeTemplateType] = {
            "label": ["N", 1],
            "s": [1],
            "k": ["N", "M"],
            "k_star": ["N", "M_test"],
            "mu": ["N", 1],
            "loss": [1],
            "prediction": ["N", 1],
            "confidence": ["N", 1],
        }

        self._set_shapes(**shapes)
        self._freeze()

    def connect(  # type: ignore[override]
        self,
        label: ConnectionType | Tensor[int | float | bool] = NOT_GIVEN,
        s: ConnectionType | Tensor[int | float | bool] = NOT_GIVEN,
        k: ConnectionType | Tensor[int | float | bool] = NOT_GIVEN,
        k_star: ConnectionType | Tensor[int | float | bool] = NOT_GIVEN,
        mu: ConnectionType | Tensor[int | float | bool] = NOT_GIVEN,
        loss: ConnectionType | Tensor[int | float | bool] = NOT_GIVEN,
        prediction: ConnectionType | Tensor[int | float | bool] = NOT_GIVEN,
        confidence: ConnectionType | Tensor[int | float | bool] = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().connect(
            label=label,
            s=s,
            k=k,
            k_star=k_star,
            mu=mu,
            loss=loss,
            prediction=prediction,
            confidence=confidence,
        )


class GPRLoss(Model):
    labels: Connection
    mu: Connection
    L: Connection
    K_term: Connection
    alpha: Connection
    output: Connection

    def __init__(
        self,
        robust: bool = False,
        labels: Tensor[float] | ToBeDetermined = TBD,
        mu: Tensor[float] | ToBeDetermined = TBD,
        L: Tensor[float] | ToBeDetermined = TBD,
        K_term: Tensor[float] | ToBeDetermined = TBD,
        alpha: Tensor[float] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.factory_args = {"robust": robust}

        diff_model = Subtract()
        transpose_model = Transpose()
        dot_model = MatrixMultiply()
        squeeze_model = Squeeze()
        mult_model = Multiply()
        mult_model_2 = Multiply()
        eig_model = Eigvalsh()
        log_model = Log(robust=robust)
        sum_reduce_model = Sum()
        length_model = Length()
        sum_model_1 = Add()

        self += diff_model.connect(
            left=IOKey("labels", value=labels), right=IOKey("mu", value=mu)
        )
        self += transpose_model.connect(input=diff_model.output)
        self += dot_model.connect(
            left=transpose_model.output, right=IOKey("alpha", value=alpha)
        )
        self += squeeze_model.connect(input=dot_model.output)
        self += mult_model.connect(left=squeeze_model.output, right=0.5)
        self += eig_model.connect(
            K_term=IOKey("K_term", value=K_term), L=IOKey("L", value=L)
        )
        self += log_model.connect(input=eig_model.output)
        self += sum_reduce_model.connect(input=log_model.output)
        self += length_model.connect(input=self.labels)
        self += mult_model_2.connect(
            left=length_model.output, right=math.log(2 * math.pi) / 2
        )
        self += sum_model_1.connect(
            left=mult_model.output, right=sum_reduce_model.output
        )
        self += Add().connect(
            left=sum_model_1.output,
            right=mult_model_2.output,
            output=IOKey(name="output"),
        )

        shapes: dict[str, ShapeTemplateType] = {
            "labels": ["N", 1],
            "mu": ["N", 1],
            "L": ["N", "N"],
            "K_term": ["N", "N"],
            "alpha": ["N", 1],
            "output": [1],
        }

        self._set_shapes(**shapes)
        self._freeze()

    def connect(  # type: ignore[override]
        self,
        labels: ConnectionType | Tensor[float] = NOT_GIVEN,
        mu: ConnectionType | Tensor[float] = NOT_GIVEN,
        L: ConnectionType | Tensor[float] = NOT_GIVEN,
        K_term: ConnectionType | Tensor[float] = NOT_GIVEN,
        alpha: ConnectionType | Tensor[float] = NOT_GIVEN,
        output: ConnectionType | Tensor[float] = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().connect(
            labels=labels,
            mu=mu,
            L=L,
            K_term=K_term,
            alpha=alpha,
            output=output,
        )


class Metric(Model):
    pred: Connection
    label: Connection
    output: Connection
    pred_formatted: Connection
    label_formatted: Connection
    label_argmax: Connection
    pred_argmax: Connection
    greater_out: Connection
    pred_comp: Connection

    def __init__(
        self,
        threshold: float | None = None,
        is_binary: bool = False,
        is_pred_one_hot: bool = True,
        is_label_one_hot: bool = True,
        pred: Tensor[int | float | bool] | ToBeDetermined = TBD,
        label: Tensor[bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)

        assert (
            not is_binary or threshold is not None
        ), "Probs must be False if threshold is not None"

        pred_key: IOKey | Connection = IOKey(name="pred", type=Tensor, value=pred)
        label_key: IOKey | Connection = IOKey(name="label", type=Tensor, value=label)

        if is_label_one_hot:
            self += ArgMax(axis=-1).connect(label_key, output="label_argmax")
            label_key = self.label_argmax

        if is_binary and is_pred_one_hot:
            self |= ArgMax(axis=-1).connect(pred_key, output="pred_argmax")
            pred_key = self.pred_argmax
        elif is_binary and not is_pred_one_hot:
            assert threshold is not None
            self |= Greater().connect(
                left=pred_key, right=threshold, output="greater_out"
            )
            self |= Where().connect(
                cond="greater_out",
                input1=Tensor(1),
                input2=Tensor(0),
                output="pred_comp",
            )
            pred_key = self.pred_comp
        elif is_pred_one_hot:
            self |= ArgMax(axis=-1).connect(pred_key, output="pred_argmax")
            pred_key = self.pred_argmax

        result = pred_key - label_key
        self |= Buffer().connect(input=pred_key, output="pred_formatted")
        self |= Buffer().connect(input=label_key, output="label_formatted")
        self |= Buffer().connect(input=result, output="output")

        self.expose_keys("pred_formatted", "label_formatted", "output")
        self._set_cin(self.pred)
        self._freeze()

    def connect(  # type: ignore[override]
        self,
        pred: ConnectionType | Tensor[int | float | bool] = NOT_GIVEN,
        label: ConnectionType | Tensor[int | float | bool] = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
        pred_formatted: ConnectionType | Tensor[int | float | bool] = NOT_GIVEN,
        label_formatted: ConnectionType | Tensor[int | float | bool] = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().connect(
            pred=pred,
            label=label,
            output=output,
            pred_formatted=pred_formatted,
            label_formatted=label_formatted,
        )


class Accuracy(Model):
    pred: Connection
    label: Connection
    output: Connection
    metric_out: Connection
    pred_formatted: Connection
    label_formatted: Connection

    def __init__(
        self,
        threshold: float | None = None,
        is_binary: bool = False,
        is_pred_one_hot: bool = True,
        is_label_one_hot: bool = True,
        pred: Tensor[int | float | bool] | ToBeDetermined = TBD,
        label: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self |= Metric(
            threshold=threshold,
            is_binary=is_binary,
            is_pred_one_hot=is_pred_one_hot,
            is_label_one_hot=is_label_one_hot,
        ).connect(
            IOKey("pred", value=pred),
            IOKey("label", value=label),
            "metric_out",
            "pred_formatted",
            "label_formatted",
        )

        true_predictions = self.metric_out.eq(0)
        n_prediction = self.label_formatted.shape[0]

        self |= Sum().connect(input=true_predictions, output="n_true_predictions")
        self |= Divide().connect(
            numerator="n_true_predictions",
            denominator=n_prediction.tensor(),
            output="output",
        )

        self.expose_keys("output")
        self._set_cin(self.pred)
        self._set_cout(self.output)
        self._freeze()

    def connect(  # type: ignore[override]
        self,
        pred: ConnectionType | Tensor[int | float | bool] = NOT_GIVEN,
        label: ConnectionType | Tensor[int | float | bool] = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().connect(
            pred=pred,
            label=label,
            output=output,
        )


class Precision(Model):
    pred: Connection
    label: Connection
    output: Connection
    metric_out: Connection
    pred_formatted: Connection
    label_formatted: Connection
    n_true_positive: Connection
    n_false_positive: Connection
    n_classes: Connection

    def __init__(
        self,
        average: str = "micro",
        n_classes: int | None = None,
        threshold: float | None = None,
        is_binary: bool = False,
        is_pred_one_hot: bool = True,
        is_label_one_hot: bool = True,
        pred: Tensor[int | float | bool] | ToBeDetermined = TBD,
        label: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.factory_args = {"threshold": threshold}

        assert average in [
            "micro",
            "macro",
            "weighted",
        ], "average must be one of ['micro', 'macro', 'weighted']"
        # assert (
        #     average not in ["weighted", "macro"] or n_classes is not None
        # ), "n_classes must be provided if average is 'weighted' or 'macro'"

        self |= Metric(
            threshold=threshold,
            is_binary=is_binary,
            is_pred_one_hot=is_pred_one_hot,
            is_label_one_hot=is_label_one_hot,
        ).connect(
            IOKey("pred", value=pred),
            IOKey("label", value=label),
            "metric_out",
            "pred_formatted",
            "label_formatted",
        )

        if average == "micro":
            true_positive = self.metric_out.eq(Tensor(0))
            false_positive = self.metric_out.ne(Tensor(0))
            self |= Sum().connect(input=true_positive, output="n_true_positive")
            self |= Sum().connect(input=false_positive, output="n_false_positive")

            self |= Buffer().connect(
                input=self.n_true_positive
                / (self.n_true_positive + self.n_false_positive),
                output=IOKey(name="output"),
            )

        if average == "macro":
            sum_precision = None
            assert (
                n_classes is not None
            ), "n_classes must be provided if average is or 'macro'"
            for idx in range(n_classes):
                class_idxs = self.label_formatted.eq(Tensor(idx))
                true_positive = (self.metric_out.eq(Tensor(0))) & class_idxs
                false_positive = (self.pred_formatted.eq(Tensor(idx))) & ~class_idxs

                self |= Sum().connect(
                    input=true_positive, output=f"true_positive_{idx}"
                )
                self |= Sum().connect(
                    input=false_positive, output=f"false_positive_{idx}"
                )
                denominator = getattr(self, f"true_positive_{idx}") + getattr(
                    self, f"false_positive_{idx}"
                )
                self |= Where().connect(
                    denominator.eq(Tensor(0)),
                    Tensor(1),
                    denominator,
                    f"denominator_{idx}",
                )
                self |= Divide().connect(
                    numerator=f"true_positive_{idx}",
                    denominator=getattr(self, f"denominator_{idx}"),
                    output=f"precision_{idx}",
                )

                if sum_precision is None:
                    sum_precision = getattr(self, f"precision_{idx}")
                else:
                    sum_precision += getattr(self, f"precision_{idx}")

            self |= Unique().connect(input=self.label_formatted, output="n_classes")

            self |= Divide().connect(
                numerator=sum_precision,  # type: ignore
                denominator=self.n_classes.shape[0].tensor(),
                output=IOKey(name="output"),
            )

        elif average == "weighted":
            precision = None
            n_element = self.label_formatted.shape[0]
            assert (
                n_classes is not None
            ), "n_classes must be provided if average is or 'weighted'"
            for idx in range(n_classes):
                class_idxs = self.label_formatted.eq(Tensor(idx))
                true_positive = (self.metric_out.eq(Tensor(0))) & class_idxs
                false_positive = (self.pred_formatted.eq(Tensor(idx))) & ~class_idxs
                self |= Sum().connect(input=class_idxs, output=f"n_class_{idx}")

                self |= Sum().connect(
                    input=true_positive, output=f"true_positive_{idx}"
                )
                self |= Sum().connect(
                    input=false_positive, output=f"false_positive_{idx}"
                )
                denominator = getattr(self, f"true_positive_{idx}") + getattr(
                    self, f"false_positive_{idx}"
                )
                self |= Where().connect(
                    denominator.eq(Tensor(0)),
                    Tensor(1),
                    denominator,
                    f"denominator_{idx}",
                )
                self |= Divide().connect(
                    numerator=f"true_positive_{idx}",
                    denominator=(getattr(self, f"denominator_{idx}")),
                    output=f"precision_{idx}",
                )
                self |= Divide().connect(
                    numerator=getattr(self, f"precision_{idx}")
                    * getattr(self, f"n_class_{idx}"),
                    denominator=n_element.tensor(),
                    output=f"weighted_precision_{idx}",
                )

                if precision is None:
                    precision = getattr(self, f"weighted_precision_{idx}")
                else:
                    precision += getattr(self, f"weighted_precision_{idx}")

            self |= Buffer().connect(input=precision, output=IOKey(name="output"))

        self.expose_keys("output")
        self._set_cin(self.pred)
        self._set_cout(self.output)
        self._freeze()

    def connect(  # type: ignore[override]
        self,
        pred: ConnectionType | Tensor[int | float | bool] = NOT_GIVEN,
        label: ConnectionType | Tensor[int | float | bool] = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().connect(
            pred=pred,
            label=label,
            output=output,
        )


class Recall(Model):
    pred: Connection
    label: Connection
    output: Connection
    metric_out: Connection
    pred_formatted: Connection
    label_formatted: Connection
    n_true_positive: Connection
    n_false_negative: Connection
    n_classes: Connection

    def __init__(
        self,
        average: str = "micro",
        n_classes: int | None = None,
        threshold: float | None = None,
        is_binary: bool = False,
        is_pred_one_hot: bool = True,
        is_label_one_hot: bool = True,
        pred: Tensor[int | float | bool] | ToBeDetermined = TBD,
        label: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.factory_args = {"threshold": threshold}

        assert average in [
            "micro",
            "macro",
            "weighted",
        ], "average must be one of ['micro', 'macro', 'weighted']"
        # assert (
        #     average not in ["weighted", "macro"] or n_classes is not None
        # ), "n_classes must be provided if average is 'weighted' or 'macro'"

        self |= Metric(
            threshold=threshold,
            is_binary=is_binary,
            is_pred_one_hot=is_pred_one_hot,
            is_label_one_hot=is_label_one_hot,
        ).connect(
            IOKey("pred", value=pred),
            IOKey("label", value=label),
            "metric_out",
            "pred_formatted",
            "label_formatted",
        )

        if average == "micro":
            true_positive = self.metric_out.eq(Tensor(0))
            false_negative = self.metric_out.ne(Tensor(0))
            self |= Sum().connect(input=true_positive, output="n_true_positive")
            self |= Sum().connect(input=false_negative, output="n_false_negative")

            self |= Buffer().connect(
                input=self.n_true_positive
                / (self.n_true_positive + self.n_false_negative),
                output=IOKey(name="output"),
            )

        if average == "macro":
            sum_recall = None
            assert (
                n_classes is not None
            ), "n_classes must be provided if average is or 'macro'"
            for idx in range(n_classes):
                class_idxs = self.label_formatted.eq(Tensor(idx))
                true_positive = (self.metric_out.eq(Tensor(0))) & class_idxs
                false_negative = (self.pred_formatted.ne(Tensor(idx))) & class_idxs

                self |= Sum().connect(
                    input=true_positive, output=f"true_positive_{idx}"
                )
                self |= Sum().connect(
                    input=false_negative, output=f"false_negative_{idx}"
                )
                denominator = getattr(self, f"true_positive_{idx}") + getattr(
                    self, f"false_negative_{idx}"
                )
                self |= Where().connect(
                    denominator.eq(Tensor(0)),
                    Tensor(1),
                    denominator,
                    f"denominator_{idx}",
                )
                self |= Divide().connect(
                    numerator=f"true_positive_{idx}",
                    denominator=getattr(self, f"denominator_{idx}"),
                    output=f"recall_{idx}",
                )

                if sum_recall is None:
                    sum_recall = getattr(self, f"recall_{idx}")
                else:
                    sum_recall += getattr(self, f"recall_{idx}")

            self |= Unique().connect(input=self.label_formatted, output="n_classes")

            self |= Divide().connect(
                numerator=sum_recall,  # type: ignore
                denominator=self.n_classes.shape[0].tensor(),
                output=IOKey(name="output"),
            )

        elif average == "weighted":
            recall = None
            assert (
                n_classes is not None
            ), "n_classes must be provided if average is or 'weighted'"
            n_element = self.label_formatted.shape[0]
            for idx in range(n_classes):
                class_idxs = self.label_formatted.eq(Tensor(idx))
                true_positive = (self.metric_out.eq(Tensor(0))) & class_idxs
                false_negative = (self.pred_formatted.ne(Tensor(idx))) & class_idxs
                self |= Sum().connect(input=class_idxs, output=f"n_class_{idx}")

                self |= Sum().connect(
                    input=true_positive, output=f"true_positive_{idx}"
                )
                self |= Sum().connect(
                    input=false_negative, output=f"false_negative_{idx}"
                )
                denominator = getattr(self, f"true_positive_{idx}") + getattr(
                    self, f"false_negative_{idx}"
                )
                self |= Where().connect(
                    denominator.eq(Tensor(0)),
                    Tensor(1),
                    denominator,
                    f"denominator_{idx}",
                )
                self |= Divide().connect(
                    numerator=f"true_positive_{idx}",
                    denominator=getattr(self, f"denominator_{idx}"),
                    output=f"recall_{idx}",
                )
                self |= Divide().connect(
                    numerator=getattr(self, f"recall_{idx}")
                    * getattr(self, f"n_class_{idx}"),
                    denominator=n_element.tensor(),
                    output=f"weighted_recall_{idx}",
                )

                if recall is None:
                    recall = getattr(self, f"weighted_recall_{idx}")
                else:
                    recall += getattr(self, f"weighted_recall_{idx}")

            self |= Buffer().connect(input=recall, output=IOKey(name="output"))

        self.expose_keys("output")
        self._set_cin(self.pred)
        self._set_cout(self.output)
        self._freeze()

    def connect(  # type: ignore[override]
        self,
        pred: ConnectionType | Tensor[int | float | bool] = NOT_GIVEN,
        label: ConnectionType | Tensor[int | float | bool] = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().connect(
            pred=pred,
            label=label,
            output=output,
        )


class F1(Model):
    pred: Connection
    label: Connection
    output: Connection
    metric_out: Connection
    pred_formatted: Connection
    label_formatted: Connection
    n_true_positive: Connection
    n_false_positive: Connection
    n_classes: Connection

    def __init__(
        self,
        average: str = "micro",
        n_classes: int | None = None,
        threshold: float | None = None,
        is_binary: bool = False,
        is_pred_one_hot: bool = True,
        is_label_one_hot: bool = True,
        pred: Tensor[int | float | bool] | ToBeDetermined = TBD,
        label: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.factory_args = {"threshold": threshold}

        assert average in [
            "micro",
            "macro",
            "weighted",
        ], "average must be one of ['micro', 'macro', 'weighted']"
        assert (
            average not in ["weighted", "macro"] or n_classes is not None
        ), "n_classes must be provided if average is 'weighted' or 'macro'"

        self |= Metric(
            threshold=threshold,
            is_binary=is_binary,
            is_pred_one_hot=is_pred_one_hot,
            is_label_one_hot=is_label_one_hot,
        ).connect(
            IOKey("pred", value=pred),
            IOKey("label", value=label),
            "metric_out",
            "pred_formatted",
            "label_formatted",
        )

        if average == "micro":
            true_positive = self.metric_out.eq(Tensor(0))
            false_positive = self.metric_out.ne(Tensor(0))
            self |= Sum().connect(input=true_positive, output="n_true_positive")
            self |= Sum().connect(input=false_positive, output="n_false_positive")

            self |= Buffer().connect(
                input=self.n_true_positive
                / (self.n_true_positive + self.n_false_positive),
                output="output",
            )

        if average == "macro":
            sum_precision = None
            assert (
                n_classes is not None
            ), "n_classes must be provided if average is or 'macro'"
            for idx in range(n_classes):
                class_idxs = self.label_formatted.eq(Tensor(idx))
                true_positive = (self.metric_out.eq(Tensor(0))) & class_idxs
                false_negative = (self.pred_formatted.ne(Tensor(idx))) & class_idxs
                false_positive = (self.pred_formatted.eq(Tensor(idx))) & ~class_idxs

                self |= Sum().connect(
                    input=true_positive, output=f"true_positive_{idx}"
                )
                self |= Sum().connect(
                    input=false_positive, output=f"false_positive_{idx}"
                )
                self |= Sum().connect(
                    input=false_negative, output=f"false_negative_{idx}"
                )
                denominator = getattr(self, f"true_positive_{idx}") + Tensor(0.5) * (
                    getattr(self, f"false_positive_{idx}")
                    + getattr(self, f"false_negative_{idx}")
                )
                self |= Where().connect(
                    denominator.eq(Tensor(0)),
                    Tensor(1),
                    denominator,
                    f"denominator_{idx}",
                )
                self |= Divide().connect(
                    numerator=f"true_positive_{idx}",
                    denominator=getattr(self, f"denominator_{idx}"),
                    output=f"precision_{idx}",
                )

                if sum_precision is None:
                    sum_precision = getattr(self, f"precision_{idx}")
                else:
                    sum_precision += getattr(self, f"precision_{idx}")

            self |= Unique().connect(input=self.label_formatted, output="n_classes")
            self |= Divide().connect(
                numerator=sum_precision,  # type: ignore
                denominator=self.n_classes.shape[0].tensor(),
                output=IOKey(name="output"),
            )

        elif average == "weighted":
            precision = None
            assert (
                n_classes is not None
            ), "n_classes must be provided if average is or 'weighted'"
            n_element = self.label_formatted.shape[0].tensor()
            for idx in range(n_classes):
                class_idxs = self.label_formatted.eq(Tensor(idx))
                true_positive = (self.metric_out.eq(Tensor(0))) & class_idxs
                false_negative = (self.pred_formatted.ne(Tensor(idx))) & class_idxs
                false_positive = (self.pred_formatted.eq(Tensor(idx))) & ~class_idxs
                self |= Sum().connect(input=class_idxs, output=f"n_class_{idx}")

                self |= Sum().connect(
                    input=true_positive, output=f"true_positive_{idx}"
                )
                self |= Sum().connect(
                    input=false_positive, output=f"false_positive_{idx}"
                )
                self |= Sum().connect(
                    input=false_negative, output=f"false_negative_{idx}"
                )
                denominator = getattr(self, f"true_positive_{idx}") + Tensor(0.5) * (
                    getattr(self, f"false_positive_{idx}")
                    + getattr(self, f"false_negative_{idx}")
                )
                self |= Where().connect(
                    denominator.eq(Tensor(0)),
                    Tensor(1),
                    denominator,
                    f"denominator_{idx}",
                )
                self |= Divide().connect(
                    numerator=f"true_positive_{idx}",
                    denominator=getattr(self, f"denominator_{idx}"),
                    output=f"precision_{idx}",
                )
                self |= Divide().connect(
                    numerator=getattr(self, f"precision_{idx}")
                    * getattr(self, f"n_class_{idx}"),
                    denominator=n_element,
                    output=f"weighted_precision_{idx}",
                )

                if precision is None:
                    precision = getattr(self, f"weighted_precision_{idx}")
                else:
                    precision += getattr(self, f"weighted_precision_{idx}")

            self |= Buffer().connect(input=precision, output=IOKey(name="output"))

        self.expose_keys("output")
        self._set_cin(self.pred)
        self._set_cout(self.output)
        self._freeze()

    def connect(  # type: ignore[override]
        self,
        pred: ConnectionType | Tensor[int | float | bool] = NOT_GIVEN,
        label: ConnectionType | Tensor[int | float | bool] = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().connect(
            pred=pred,
            label=label,
            output=output,
        )


class AUC(Model):
    pred: Connection
    label: Connection
    output: Connection
    label_argmax: Connection

    def __init__(
        self,
        n_classes: int,
        is_label_one_hot: bool = True,
        pred: Tensor[float] | ToBeDetermined = TBD,
        label: Tensor[bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)

        assert n_classes > 0, ""
        assert isinstance(n_classes, int)

        label_key: IOKey | Connection = IOKey(name="label", type=Tensor, value=label)
        pred_key: IOKey | Connection = IOKey(name="pred", type=Tensor, value=pred)

        if is_label_one_hot:
            self |= ArgMax(axis=-1).connect(label_key, output="label_argmax")
            label_key = self.label_argmax

        auc_score = None
        for class_idx in range(n_classes):
            class_label = label_key.eq(Tensor(class_idx))
            pred_class = pred_key[:, class_idx] if n_classes != 1 else pred_key

            self |= AUCCore().connect(pred_class, class_label, f"auc_core_{class_idx}")
            self |= Trapezoid().connect(
                y=getattr(self, f"auc_core_{class_idx}")[0],
                x=getattr(self, f"auc_core_{class_idx}")[1],
                output=IOKey(f"auc_class_{class_idx}"),
            )
            if auc_score is None:
                auc_score = getattr(self, f"auc_class_{class_idx}") / Tensor(n_classes)
            else:
                auc_score += getattr(self, f"auc_class_{class_idx}") / Tensor(n_classes)

        self |= Buffer().connect(auc_score, IOKey("output"))

        self._set_cin(self.pred)
        self._freeze()

    def connect(  # type: ignore[override]
        self,
        pred: ConnectionType | Tensor[float] = NOT_GIVEN,
        label: ConnectionType | Tensor[bool] = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().connect(
            pred=pred,
            label=label,
            output=output,
        )


class SiLU(Model):
    input: Connection
    output: Connection

    def __init__(
        self,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)

        self |= Negate().connect(input=IOKey("input", value=input), output="negate")
        self |= Exponential().connect(input="negate", output="exp")
        self |= Add().connect(left=Tensor(1), right="exp", output="add")
        self |= Divide().connect(
            numerator="input", denominator="add", output=IOKey(name="output")
        )
        self._set_shapes(input=[("Var", ...)], output=[("Var", ...)])

        self._freeze()

    def connect(  # type: ignore[override]
        self,
        input: ConnectionType | Tensor[int | float | bool] = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().connect(input=input, output=output)


class Randn(Model):
    shape: Connection
    key: Connection
    dtype: Connection
    output: Connection

    def __init__(
        self,
        shape: tuple[int, ...] | ToBeDetermined = TBD,
        key: int | ToBeDetermined = TBD,
        dtype: types.Dtype | None = None,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        _shape = IOKey("shape", value=shape)
        _key = IOKey("key", key)
        _dtype = IOKey("dtype", value=dtype)

        if key is TBD:
            _key = IOKey("key")
            self |= PrimitiveRandInt(shape=tuple(), low=0, high=2**14).connect(
                key=_key, output="new_key"
            )
            self.bind_state_keys(_key, "new_key", Tensor(42))
            self.expose_keys("new_key")
        self |= PrimitiveRandn().connect(_shape, _key, _dtype, output="output")

        self.expose_keys("output", "shape", "dtype")
        self._freeze()

    def connect(  # type: ignore[override]
        self,
        shape: ConnectionType | tuple[int | ConnectionType, ...] = NOT_GIVEN,
        dtype: ConnectionType | types.Dtype | None = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().connect(shape=shape, dtype=dtype, output=output)


class RandInt(Model):
    shape: Connection
    key: Connection
    dtype: Connection
    output: Connection

    def __init__(
        self,
        shape: tuple[int, ...] | ToBeDetermined = TBD,
        key: int | Tensor[int] | ToBeDetermined = TBD,
        low: int | ToBeDetermined = TBD,
        high: int | ToBeDetermined = TBD,
        dtype: types.Dtype | None = None,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        _shape = IOKey("shape", value=shape)
        _low = IOKey("low", value=low)
        _high = IOKey("high", value=high)
        _key = IOKey("key", key)
        _dtype = IOKey("dtype", value=dtype)
        output = IOKey("output")

        if key is TBD:
            _key = IOKey("key")
            self |= PrimitiveRandInt(shape=tuple(), low=0, high=2**14).connect(
                key=_key, output="new_key"
            )
            self.bind_state_keys(_key, "new_key", Tensor(42))
        self |= PrimitiveRandInt().connect(
            _shape, _key, _low, _high, _dtype, output=output
        )

        self._freeze()

    def connect(  # type: ignore[override]
        self,
        shape: ConnectionType = NOT_GIVEN,
        low: ConnectionType = NOT_GIVEN,
        high: ConnectionType = NOT_GIVEN,
        dtype: ConnectionType = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().connect(
            shape=shape, low=low, high=high, dtype=dtype, output=output
        )


class Split(Model):
    split_size: Connection
    axis: Connection
    input: Connection
    output: Connection

    def __init__(
        self,
        split_size: int,  # TODO: should we add default for split_size?
        axis: int,
        input: Tensor[int | float | bool] | ToBeDetermined = TBD,
        *,
        name: str | None = None,
    ):
        # TODO: Raise error if input is not exactly divisible by split_size.
        # Think about if this will be a logical error block or a runtime error.

        super().__init__(name=name, formula_key="split")

        input_key = IOKey("input", value=input, type=Tensor[int | float | bool])
        split_size_key = IOKey("split_size", value=split_size)
        axis_key = IOKey("axis", value=axis)
        # Find the length of the each tensor along the specified axis
        # after splitting.
        len_per_split = input_key.shape[axis_key] // split_size_key
        to_list_kwargs = {}
        for idx in range(split_size):
            slc_model = Slice()
            indexer = Indexer()
            # Calculate start and stop indices for each split.
            start_idx = len_per_split * idx
            end_idx = start_idx + len_per_split
            if axis < 0:
                index_list: list[Connection | slice | EllipsisType] = [
                    slc_model.output if i == -(axis + 1) else slice(None)
                    for i in range(-axis)
                ]
                index_list.append(...)
                index = tuple(index_list[::-1])
            else:
                index = tuple(
                    slc_model.output if i == axis else slice(None)
                    for i in range(axis + 1)
                )
            # Add corresponding Indexer model for each split using
            # corresponding Slice model output.
            self |= slc_model.connect(start_idx, end_idx, None)
            self |= indexer.connect(
                input=input_key,
                index=index,
            )
            to_list_kwargs[f"input{idx+1}"] = indexer.output
        # Finally collect all the split tensors into a list.
        self |= ToList(n=split_size).connect(**to_list_kwargs, output="output")

        self.expose_keys("split_size", "axis", "output")
        self._freeze()

    def connect(  # type: ignore[override]
        self,
        input: ConnectionType | Tensor[int | float | bool] = NOT_GIVEN,
        output: ConnectionType = NOT_GIVEN,
    ) -> ExtendInfo:
        return super().connect(input=input, output=output)

    def infer_differentiability(
        self, values: dict[str, Tensor[int | float | bool]]
    ) -> list[bool | None]:
        val = values[Operator.output_key]
        assert isinstance(val, list)
        return [values["input"].differentiable] * len(val)
