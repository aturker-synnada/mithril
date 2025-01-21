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

import dataclasses
from typing import Any

from ..backends.backend import Backend
from ..framework.logical.model import Model


@dataclasses.dataclass
class EvaluationContext:
    backend: Backend[Any]
    model: Model
    inputs: dict[str, Any]
    data: dict[str, Any]
    test_model: Model | None = None
    test_data: dict[str, Any] | None = None
