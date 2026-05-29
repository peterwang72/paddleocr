# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from paddle import nn
from ppocr.modeling.transforms import build_transform
from ppocr.modeling.backbones import build_backbone
from ppocr.modeling.necks import build_neck
from ppocr.modeling.heads import build_head

__all__ = ["BaseModel"]


class BaseModel(nn.Layer):
    def __init__(self, config):
        """
        the module for OCR.
        args:
            config (dict): the super parameters for module.
        """
        super(BaseModel, self).__init__()
        in_channels = config.get("in_channels", 3)
        model_type = config["model_type"]
        # build transform,
        # for rec, transform can be TPS,None
        # for det and cls, transform should to be None,
        # if you make model differently, you can use transform in det and cls
        if "Transform" not in config or config["Transform"] is None:
            self.use_transform = False
        else:
            self.use_transform = True
            config["Transform"]["in_channels"] = in_channels
            self.transform = build_transform(config["Transform"])
            in_channels = self.transform.out_channels

        # build backbone, backbone is need for del, rec and cls
        if "Backbone" not in config or config["Backbone"] is None:
            self.use_backbone = False
        else:
            self.use_backbone = True
            config["Backbone"]["in_channels"] = in_channels
            self.backbone = build_backbone(config["Backbone"], model_type)
            in_channels = self.backbone.out_channels

        # build neck
        # for rec, neck can be cnn,rnn or reshape(None)
        # for det, neck can be FPN, BIFPN and so on.
        # for cls, neck should be none
        if "Neck" not in config or config["Neck"] is None:
            self.use_neck = False
        else:
            self.use_neck = True
            config["Neck"]["in_channels"] = in_channels
            self.neck = build_neck(config["Neck"])
            in_channels = self.neck.out_channels

        # # build head, head is need for det, rec and cls
        if "Head" not in config or config["Head"] is None:
            self.use_head = False
        else:
            self.use_head = True
            config["Head"]["in_channels"] = in_channels
            self.head = build_head(config["Head"])

        if "annot_head" not in config or config["annot_head"] is None:
            self.use_annot_head = False
        else:
            self.use_annot_head = True
            config["annot_head"]["in_channels"] = in_channels
            self.annot_head = build_head(config["annot_head"])

        self.return_all_feats = config.get("return_all_feats", False)

    def forward(self, x, data=None):
        y = dict()
        if self.use_transform:
            x = self.transform(x)
        if self.use_backbone:
            x = self.backbone(x)
        if isinstance(x, dict):
            y.update(x)
        else:
            y["backbone_out"] = x
        final_name = "backbone_out"
        
        if self.use_neck:
            x = self.neck(x)
            if isinstance(x, dict):
                y.update(x)
            else:
                y["neck_out"] = x
            final_name = "neck_out"

        neck_feat = x

        # 1. 运行主检测头 (文字检测)
        if self.use_head:
            out = self.head(neck_feat, targets=data)
            if isinstance(out, dict):
                if "ctc_neck" in out.keys():
                    y["neck_out"] = out["ctc_neck"]
                    y["head_out"] = out
                y.update(out)
            else:
                y["head_out"] = out
            x = out
            final_name = "head_out"

        # 2. 运行新增符号检测头 (符号检测)
        if self.use_annot_head:
            annot_out = self.annot_head(neck_feat, targets=data)
            if isinstance(annot_out, dict):
                # 训练兼容性逻辑：
                # 为符号检测头的输出增加前缀（例如 "annot_maps"）
                # 这样 y 中将同时拥有 "maps" 和 "annot_maps" 供不同 Loss 监督
                for k, v in annot_out.items():
                    y["annot_" + k] = v
            else:
                y["annot_maps"] = annot_out
            x = annot_out
            final_name = "annot_head_out"
        
        res = {}
        for k, v in out.items():
            res[k] = v
        for k, v in annot_out.items():
            res["annot_" + k] = v

        # 3. 结果返回逻辑 (针对训练与并行推理进行增强)
        if self.return_all_feats:
            if self.training:
                return y
            
            # 推理阶段：如果是并行模式，返回 y；否则返回当前 x
            if self.use_head and self.use_annot_head:
                return y
            elif isinstance(x, dict):
                return x
            else:
                return {final_name: x}
        else:
            # 如果配置了两头，为了让训练器能同时获取两个 maps 进行梯度回传，必须返回字典 y
            if self.use_head and self.use_annot_head:
                return res
            return x