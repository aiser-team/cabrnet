from __future__ import annotations
from os import PathLike
from typing import Optional, Tuple
from loguru import logger

import numpy as np
import onnx
import onnx_graphsurgeon as gs
import onnxruntime as rt
import torch
import torch.nn as nn

"""
TODO:
* [ ] path sanitization and file permissions for ONNX models generation
* [ ] Saving generated ONNX omdels
      (they must be saved with the rest of the configuration once generated
      to enable reproduceability)
* [ ] create a collection of ORT running sessions with provided layer keys,
      mirroring the ConvExtractor class capability to provide multi-output
      models
* [ ] type sanitization conversions for the ONNX sessions bindings
"""


class GenericONNXModel(nn.Module):
    r"""A class describing generic ONNX models to be used as backbone.

    Can optionally be trimed upto a specific given layer in accordance with the
    ConvExtractor.
    Provide some metadata about the ONNX graph for bookeeping.

    Provides a forward function relying on the onnxruntime.

    Attributes:
        backbone: a ONNX file
    """

    def __init__(self, onnx_path: PathLike[str], layer_cut: Optional[str] = None):
        super(GenericONNXModel, self).__init__()
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)
        model = onnx.shape_inference.infer_shapes(model)
        self.backbone: onnx.ModelProto = model
        self.backbone_path: PathLike[str] = onnx_path
        # To be inferred with shape inference
        self.out_shape: Optional[Tuple[int, int, int, int]] = None
        if len(gs.import_onnx(self.backbone).inputs[0].shape) != 4:
            (
                self.backbone_batched,
                self.backbone_batched_path,
            ) = self.allow_batched_inputs()
        if layer_cut is not None:
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
        r"""
        Trim the ONNX graph upto the provided layer name (included), and saves
        the corresponding ONNX graph on disk.
        Returns both the modified model and the path of the saved ONNX file.

        Args:
            model (onnx.ModelProto): ONNX model to trim.
            layer_cut (str): Layer to trim the model to.


        """
        # shape inference must have been run beforehand
        assert len(model.graph.value_info) > 0
        to_trim = False
        node_candidates = [
            x for x in model.graph.value_info if x.name.__contains__(layer_cut)
        ]
        # otherwise, layer_cut is underspecified
        assert len(node_candidates) == 1
        assert len(node_candidates[0].type.tensor_type.shape.dim) == 4
        [b, x, y, z] = node_candidates[0].type.tensor_type.shape.dim
        self.out_shape = (b.dim_value, x.dim_value, y.dim_value, z.dim_value)
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
        # and remove the previous one
        graph.outputs = [
            gs.Variable("features", dtype=np.float32, shape=self.out_shape)
        ]
        print(graph.outputs)
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
        sess = rt.InferenceSession(self.backbone_trimed_path, providers=providers)
        input_name = sess.get_inputs()[0].name
        output_name = sess.get_outputs()[0].name
        bind = sess.io_binding()
        bind.bind_input(
            name=input_name,
            device_type=device_type,
            device_id=0,
            element_type=np.float32,
            shape=tuple(x.shape),
            buffer_ptr=x.data_ptr(),
        )
        out_shape = torch.Size(self.out_shape)
        out_tensor = torch.empty(
            out_shape, dtype=torch.float32, device=device
        ).contiguous()
        bind.bind_output(
            name=output_name,
            device_type=device_type,
            device_id=0,
            element_type=np.float32,
            shape=tuple(out_tensor.shape),
            buffer_ptr=out_tensor.data_ptr(),
        )
        sess.run_with_iobinding(bind)
        return out_tensor
