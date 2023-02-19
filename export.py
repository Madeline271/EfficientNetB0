# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
efficientnet export.
"""
import os
import argparse
import numpy as np
import mindspore as ms
from mindspore import context, Tensor, load_checkpoint, load_param_into_net, export
from src.models.effnet import EfficientNet

import moxing as mox


parser = argparse.ArgumentParser(description='Image classification')
parser.add_argument('--checkpoint_path', type=str, required=True, default=" ",help='Checkpoint file path')
parser.add_argument("--output_path", type=str, default="efficientnet", help="output file name.")
parser.add_argument('--width', type=int, default=224, help='input width')
parser.add_argument('--height', type=int, default=224, help='input height')
parser.add_argument("--file_format", type=str, choices=["AIR", "ONNX", "MINDIR"], default="AIR", help="file format")
args_opt = parser.parse_args()

if __name__ == '__main__':
    local_ckpt_url = "/cache/ckpt"
    local_output_url = "/cache/output"
    mox.file.copy_parallel(args_opt.checkpoint_path, local_ckpt_url)
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")

    net = EfficientNet(1, 1)

    param_dict = load_checkpoint(os.path.join(local_ckpt_url, "efficientnet-b0.ckpt"))
    load_param_into_net(net, param_dict)
    input_data = Tensor(np.ones([1, 3, 224, 224]), ms.float32)
    export(net, input_data, file_name=os.path.join(local_output_url, "efficientnet-b0"), file_format=args_opt.file_format)
    mox.file.copy_parallel(local_output_url, args_opt.output_path)
