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

import json

import numpy as np
import pytest

import mithril
from mithril import JaxBackend, TorchBackend
from mithril.models import PhysicalModel, TrainModel
from tests.scripts.helper import evaluate_case
from tests.scripts.test_utils import finalize_model

# Read Data
# shape_inference_tests_path = "tests/json_files/shape_directed_test.json"
# with open(shape_inference_tests_path) as f:
#     shape_inference_tests = json.load(f)

discard_keys_inference_tests = "tests/json_files/discard_keys_directed_test.json"
with open(discard_keys_inference_tests) as f:
    discard_keys_inference_tests_dict: dict = json.load(f)

static_keys_inference_tests = "tests/json_files/static_keys_directed_test.json"
with open(static_keys_inference_tests) as f:
    static_keys_inference_tests_dict: dict = json.load(f)

no_grad_inference_tests = "tests/json_files/no_grad_inference_test.json"
with open(no_grad_inference_tests) as f:
    no_grad_inference_tests_dict: dict = json.load(f)


@pytest.mark.parametrize("case", discard_keys_inference_tests_dict)
def test_discard_keys_inference(case: str) -> None:
    backend = TorchBackend(dtype=mithril.float64)
    current_case = discard_keys_inference_tests_dict[case]

    results = current_case["results"]
    discard_keys = set(current_case.get("discard_keys", []))

    model = finalize_model(current_case)
    if isinstance(model, TrainModel):
        model.finalize()

    reference_output_keys = sorted(results.get("output_keys", {}))
    reference_discard_keys = sorted(results.get("discard_keys", {}))

    pm = PhysicalModel(
        model,
        backend,
        discard_keys=discard_keys,
        data_keys=set(),
        constant_keys=dict(),
        trainable_keys=set(),
        shapes=dict(),
        inference=True,
        safe_shapes=True,
        safe_names=True,
        use_short_namings=True,
        jit=True,
    )

    discarded_keys = pm.discarded_keys
    output_keys = pm.output_keys
    hanging_keys = pm.flat_graph.hanging_keys
    discard_keys |= {key for key in hanging_keys if key not in pm.output_keys}

    assert sorted(discarded_keys) == reference_discard_keys
    assert sorted(output_keys) == reference_output_keys


@pytest.mark.parametrize("case", static_keys_inference_tests_dict)
def test_static_keys_inference(case: str) -> None:
    backend = JaxBackend(dtype=mithril.float64)
    current_case = static_keys_inference_tests_dict[case]

    base_static_inputs = {
        key: np.random.randn(*value)
        for key, value in current_case.get("static_input_shapes", {}).items()
    }
    static_inputs = {
        key: backend.array(value) for key, value in base_static_inputs.items()
    }

    results = current_case["results"]
    results = results.get("static_keys", {})
    discard_keys = set(current_case.get("discard_keys", []))

    model = finalize_model(current_case)

    compiled_model = mithril.compile(
        model,
        backend=backend,
        discard_keys=discard_keys,
        constant_keys=static_inputs,
        inference=True,
        jit=True,
    )
    model_static_keys = sorted(compiled_model.flat_graph.all_static_keys)
    assert model_static_keys == sorted(results)


@pytest.mark.parametrize("case", no_grad_inference_tests_dict)
def test_no_grad_inference(
    case: str, tolerance: float = 1e-14, relative_tolerance: float = 1e-14
) -> None:
    current_case = no_grad_inference_tests_dict[case]
    evaluate_case(
        JaxBackend(dtype=mithril.float64),
        current_case,
        tolerance=tolerance,
        relative_tolerance=relative_tolerance,
    )
    evaluate_case(
        TorchBackend(dtype=mithril.float64),
        current_case,
        tolerance=tolerance,
        relative_tolerance=relative_tolerance,
    )
