# copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
import paddle.nn.functional as F
from paddle import nn


def _as_logits(predicts):
    if isinstance(predicts, (list, tuple)):
        predicts = predicts[-1]
    if isinstance(predicts, dict):
        predicts = predicts.get("head_out", next(iter(predicts.values())))
    return predicts


class ClsLoss(nn.Layer):
    def __init__(self, **kwargs):
        super(ClsLoss, self).__init__()
        self.loss_func = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, predicts, batch):
        predicts = _as_logits(predicts)
        label = batch[1].astype("int64")
        loss = self.loss_func(input=predicts, label=label)
        return {"loss": loss}


class ClsFocalLoss(nn.Layer):
    """Multi-class focal loss with optional per-class alpha (class-balanced)."""

    def __init__(
        self,
        class_dim=3,
        gamma=2.0,
        class_weights=None,
        normalize_class_weights=True,
        eps=1e-6,
        **kwargs,
    ):
        super(ClsFocalLoss, self).__init__()
        self.class_dim = int(class_dim)
        self.gamma = float(gamma)
        self.eps = float(eps)
        if class_weights is not None:
            alpha = paddle.to_tensor(class_weights, dtype="float32")
            if normalize_class_weights:
                alpha = alpha * len(alpha) / paddle.sum(alpha)
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(self, predicts, batch):
        predicts = _as_logits(predicts)
        label = batch[1].astype("int64")
        log_probs = F.log_softmax(predicts, axis=-1)
        probs = F.softmax(predicts, axis=-1)
        one_hot = F.one_hot(label, num_classes=self.class_dim).astype(probs.dtype)
        pt = paddle.sum(probs * one_hot, axis=1)
        log_pt = paddle.sum(log_probs * one_hot, axis=1)
        focal = paddle.pow(1.0 - pt + self.eps, self.gamma)
        if self.alpha is not None:
            alpha_t = paddle.index_select(self.alpha, label, axis=0)
            focal = alpha_t * focal
        loss = -focal * log_pt
        return {"loss": loss.mean()}
