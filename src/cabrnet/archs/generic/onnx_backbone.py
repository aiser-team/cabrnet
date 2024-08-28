from __future__ import annotations
from os import PathLike
from typing import Optional, Tuple

import numpy as np
import onnx
import onnx_graphsurgeon as gs
import onnxruntime as rt
import torch
import torch.nn as nn


# TODO:
# create a collection of ORT running sessions with provided layer keys

class GenericONNXModel(nn.Module):
    r"""A class describing generic ONNX models to be used as backbone.

    Can optionally be trimed upto a specific given layer in accordance with the
    ConvExtractor.
    Provide some metadata about the ONNX graph for bookeeping.

    Provides a forward function relying on the onnxruntime.

    Attributes:
        backbone: a ONNX file
    """

    def __init__(self, onnx_path: PathLike, layer_cut: Optional[str] = None):
        super(GenericONNXModel, self).__init__()
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)
        self.backbone = model
        self.backbone_path = onnx_path
        self.feat_shape = (-1, 1, 1, 1)  # TODO: compute that from layer_cut
        self.out_shape = (-1, 1, 1, 1)
        if len(gs.import_onnx(self.backbone).inputs[0].shape) != 4:
            (
                self.backbone_batched,
                self.backbone_batched_path,
            ) = self.allow_batched_inputs()
        if layer_cut is not None:
            print("toto")
            self.backbone_trimed, self.backbone_trimed_path = self.trim_backbone(
                self.backbone_batched, layer_cut
            )

    def allow_batched_inputs(self) -> Tuple[onnx.ModelProto, str]:
        graph = gs.import_onnx(self.backbone)
        (h, w, d) = graph.inputs[0].shape
        (y) = graph.outputs[0].shape
        graph.inputs[0].shape = (-1, h, w, d)
        graph.outputs[0].shape = (-1, y)
        graph.cleanup()
        batched_name = "onnx_batched.onnx"
        model_proto = gs.export_onnx(graph)
        onnx.save_model(model_proto, batched_name)
        return (model_proto, batched_name)

    # TODO: probably more the role of the ConvExtractor, that may need to be
    # adapted for this purpose
    def trim_backbone(
        self, model: onnx.ModelProto, layer_cut: str
    ) -> Tuple[onnx.ModelProto, str]:
        r"""Trim the ONNX model upto the provided layer name, and returns the
        modified ONNX model.

        """
        to_trim = False
        graph = gs.import_onnx(model)
        n_to_trim = []
        for node in graph.nodes:
            if node.name == layer_cut:
                to_trim = True
                last_node = node
            if to_trim:
                n_to_trim.append(node)
        for node in n_to_trim:
            last_node.outputs = node.outputs
            node.outputs.clear()
        # create a new output for the graph
        graph.outputs.append(
            gs.Variable("features", dtype=np.float32, shape=self.feat_shape)
        )
        last_node.outputs = graph.outputs
        graph.cleanup()
        trimed_name = "onnx_trimed.onnx"
        model_proto = gs.export_onnx(graph)
        onnx.save_model(model_proto, trimed_name)
        return model_proto, trimed_name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        providers = ["CPUExecutionProvider", "CUDAExecutionProvider"]
        if device == torch.device("cpu"):
            device_type = "cpu"
        else:
            device_type = "cuda"
        sess = rt.InferenceSession(self.backbone_path, providers=providers)
        sess_f = rt.InferenceSession(self.backbone_trimed_path, providers=providers)
        input_name = sess.get_inputs()[0].name
        bind = sess.io_binding()
        bind_f = sess_f.io_binding()
        bind.bind_input(
            name=input_name,
            device_type=device_type,
            device_id=0,
            element_type=x.dtype,
            shape=tuple(x.shape),
            buffer_ptr=x.data_ptr(),
        )
        bind_f.bind_input(
            name=input_name,
            device_type=device_type,
            device_id=0,
            element_type=x.dtype,
            shape=tuple(x.shape),
            buffer_ptr=x.data_ptr(),
        )
        score_shape = torch.Size((x.size()[0], *self.out_shape))
        feat_shape = torch.Size((x.size()[0], *self.feat_shape))
        score_tensor = torch.empty(
            score_shape, dtype=torch.uint8, device=device
        ).contiguous()
        feat_tensor = torch.empty(
            feat_shape, dtype=torch.float32, device=device
        ).contiguous()
        bind.bind_output(
            name="scores",
            device_type=device_type,
            device_id=0,
            element_type=np.uint8,
            shape=tuple(score_tensor.shape),
            buffer_ptr=score_tensor.data_ptr(),
        )
        bind_f.bind_output(
            name="features",
            device_type=device_type,
            device_id=0,
            element_type=np.float32,
            shape=tuple(feat_tensor.shape),
            buffer_ptr=feat_tensor.data_ptr(),
        )
        sess.run_with_iobinding(bind)
        sess_f.run_with_iobinding(bind_f)
        return score_tensor, feat_tensor
