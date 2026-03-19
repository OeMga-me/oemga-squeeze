import json
import os
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple, Set
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.fx as fx
import subprocess
import tempfile

@dataclass
class CodegenArtifacts:
    output_root: str
    include_dir: str
    src_dir: str
    generated_files: List[str]

@dataclass
class MemoryBuffer:
    name: str
    kind: str              # "activation" | "scratch"
    size_bytes: int
    notes: Dict[str, Any]


@dataclass
class LoweredOp:
    name: str
    source_ir_node: str
    backend_op: str
    inputs: List[str]
    outputs: List[str]
    input_buffer: Optional[str]
    output_buffer: Optional[str]
    scratch_buffer: Optional[str]
    attrs: Dict[str, Any]
    notes: Dict[str, Any]


@dataclass
class LoweredPlan:
    model_name: str
    backend: str
    buffers: List[MemoryBuffer]
    tensor_locations: Dict[str, Dict[str, Any]]
    ops: List[LoweredOp]
    inputs: List[str]
    outputs: List[str]
    summary: Dict[str, Any]

@dataclass
class EmbeddedEstimates:
    weight_storage_int8_bytes: int
    bias_storage_int32_bytes: int
    requant_param_bytes: int
    estimated_model_data_flash_bytes: int
    peak_activation_int8_bytes: int
    double_buffer_int8_bytes: int
    max_conv_scratch_int8_bytes: int
    estimated_runtime_ram_bytes: int
    notes: List[str]

@dataclass
class TensorCost:
    name: str
    kind: str
    shape: List[int]
    dtype: str
    num_elements: int
    bytes: int

@dataclass
class NodeCost:
    name: str
    op: str
    input_tensors: List[str]
    output_tensors: List[str]
    macs: int
    param_bytes: int
    output_bytes: int
    notes: Dict[str, Any]


@dataclass
class CostReport:
    model_name: str
    total_param_bytes: int
    total_activation_bytes: int
    peak_activation_bytes: int
    total_macs: int
    tensor_costs: List[TensorCost]
    node_costs: List[NodeCost]
    summary: Dict[str, Any]
    embedded_estimates: Optional[EmbeddedEstimates] = None

@dataclass
class IRTensor:
    name: str
    shape: List[int]
    dtype: str
    kind: str              # "input" | "activation" | "param" | "output"
    source: Optional[str] = None


@dataclass
class IRNode:
    name: str
    op: str
    inputs: List[str]
    outputs: List[str]
    attrs: Dict[str, Any]
    debug: Dict[str, Any]


@dataclass
class IRGraph:
    model_name: str
    tensors: List[IRTensor]
    nodes: List[IRNode]
    inputs: List[str]
    outputs: List[str]

@dataclass
class TensorSpec:
    name: str
    shape: Optional[List[int]]
    dtype: Optional[str]


@dataclass
class ConstantSpec:
    name: str
    kind: str
    dtype: Optional[str]
    shape: Optional[List[int]]
    values: Any


@dataclass
class NodeSpec:
    name: str
    op: str
    inputs: List[str]
    outputs: List[str]
    attrs: Dict[str, Any]
    output_shape: Optional[List[int]]
    output_dtype: Optional[str]
    debug: Dict[str, Any]


@dataclass
class GraphSpec:
    model_name: str
    inputs: List[TensorSpec]
    outputs: List[TensorSpec]
    nodes: List[NodeSpec]
    constants: List[ConstantSpec]


class OemgaSqueezeError(Exception):
    pass


class UnsupportedGraphError(OemgaSqueezeError):
    pass


class GraphValidationError(OemgaSqueezeError):
    pass


class OemgaSqueeze:
    SUPPORTED_OPS = {
        "conv1d",
        "linear",
        "maxpool1d",
        "relu",
        "reshape",
    }

    def __init__(
        self,
        model: nn.Module,
        example_input: torch.Tensor,
        calibration_data: Optional[torch.Tensor] = None,
        output_dir: str = "oemga_out",
        nn_layers_path: str = "nn_layers.h",
        optimizations: Optional[List[str]] = None,
        prefer_int8_inference: bool = True,
        strict: bool = True,
    ):
        self.model = model.eval()
        self.example_input = example_input
        self.calibration_data = calibration_data if calibration_data is not None else example_input

        self.output_dir = output_dir
        # If the user didn't provide a custom path, and it's not in the current directory,
        # look for it inside the package folder (next to core.py)
        if nn_layers_path == "nn_layers.h" and not os.path.exists(nn_layers_path):
            pkg_dir = os.path.dirname(os.path.abspath(__file__))
            self.nn_layers_path = os.path.join(pkg_dir, "nn_layers.h")
        else:
            self.nn_layers_path = nn_layers_path
        self.optimizations = optimizations if optimizations else []
        self.prefer_int8_inference = prefer_int8_inference
        self.strict = strict

        os.makedirs(self.output_dir, exist_ok=True)

        self.modules = dict(self.model.named_modules())
        self.graph_module: Optional[fx.GraphModule] = None
        self.graph_spec: Optional[GraphSpec] = None
        self.ir_graph: Optional[IRGraph] = None
        self.cost_report: Optional[CostReport] = None
        self.lowered_plan: Optional[LoweredPlan] = None
        self.codegen_artifacts: Optional[CodegenArtifacts] = None

    @staticmethod
    def _dtype_to_str(dtype: Any) -> Optional[str]:
        if dtype is None:
            return None
        return str(dtype).replace("torch.", "")

    @staticmethod
    def _jsonable(x: Any) -> Any:
        if isinstance(x, (str, int, float, bool)) or x is None:
            return x
        if isinstance(x, (list, tuple)):
            return [OemgaSqueeze._jsonable(v) for v in x]
        if isinstance(x, dict):
            return {str(k): OemgaSqueeze._jsonable(v) for k, v in x.items()}
        if isinstance(x, torch.dtype):
            return OemgaSqueeze._dtype_to_str(x)
        if isinstance(x, torch.device):
            return str(x)
        return str(x)

    def _node_meta_val(self, node: fx.Node) -> Any:
        return getattr(node, "meta", {}).get("val", None)

    def _shape_from_node(self, node: fx.Node) -> Optional[List[int]]:
        meta = getattr(node, "meta", {})
        tm = meta.get("tensor_meta", None)
        if tm is not None and hasattr(tm, "shape"):
            return list(tm.shape)

        val = meta.get("val", None)
        if isinstance(val, torch.Tensor):
            return list(val.shape)

        return None

    def _dtype_from_node(self, node: fx.Node) -> Optional[str]:
        meta = getattr(node, "meta", {})
        tm = meta.get("tensor_meta", None)
        if tm is not None and hasattr(tm, "dtype"):
            return self._dtype_to_str(tm.dtype)

        val = meta.get("val", None)
        if isinstance(val, torch.Tensor):
            return self._dtype_to_str(val.dtype)

        return None

    def _flatten_input_names(self, args: Any) -> List[str]:
        names: List[str] = []

        def visit(x: Any) -> None:
            if isinstance(x, fx.Node):
                names.append(x.name)
            elif isinstance(x, (list, tuple)):
                for y in x:
                    visit(y)

        visit(args)
        return names

    def _tensor_to_constant_spec(self, name: str, t: torch.Tensor) -> ConstantSpec:
        return ConstantSpec(
            name=name,
            kind="tensor",
            dtype=self._dtype_to_str(t.dtype),
            shape=list(t.shape),
            values=t.detach().cpu().tolist(),
        )

    def _scalar_to_constant_spec(self, name: str, x: Any) -> ConstantSpec:
        return ConstantSpec(
            name=name,
            kind="scalar",
            dtype=type(x).__name__ if x is not None else None,
            shape=None,
            values=x,
        )

    def export_graph(self) -> fx.GraphModule:
        self.model.eval()

        try:
            gm = fx.symbolic_trace(self.model)

            from torch.fx.passes.shape_prop import ShapeProp
            ShapeProp(gm).propagate(self.example_input)

            self.graph_module = gm
            return gm

        except Exception as e:
            raise OemgaSqueezeError(f"Failed to trace/propagate graph: {e}")

    def _normalize_call_module(self, node: fx.Node) -> Tuple[str, Dict[str, Any]]:
        layer = self.graph_module.get_submodule(str(node.target))

        if isinstance(layer, nn.Conv1d):
            return "conv1d", {
                "in_channels": int(layer.in_channels),
                "out_channels": int(layer.out_channels),
                "kernel_size": [int(layer.kernel_size[0])] if isinstance(layer.kernel_size, tuple) else [int(layer.kernel_size)],
                "stride": [int(layer.stride[0])] if isinstance(layer.stride, tuple) else [int(layer.stride)],
                "padding": [int(layer.padding[0])] if isinstance(layer.padding, tuple) else [int(layer.padding)],
                "dilation": [int(layer.dilation[0])] if isinstance(layer.dilation, tuple) else [int(layer.dilation)],
                "groups": int(layer.groups),
                "bias": layer.bias is not None,
                "weight_name": f"{node.target}.weight",
                "bias_name": f"{node.target}.bias" if layer.bias is not None else None,
            }

        if isinstance(layer, nn.Linear):
            return "linear", {
                "in_features": int(layer.in_features),
                "out_features": int(layer.out_features),
                "bias": layer.bias is not None,
                "weight_name": f"{node.target}.weight",
                "bias_name": f"{node.target}.bias" if layer.bias is not None else None,
            }

        if isinstance(layer, nn.MaxPool1d):
            def to_1list(v):
                return [int(v)] if isinstance(v, int) else [int(v[0])]

            stride = layer.stride if layer.stride is not None else layer.kernel_size

            return "maxpool1d", {
                "kernel_size": to_1list(layer.kernel_size),
                "stride": to_1list(stride),
                "padding": to_1list(layer.padding),
                "dilation": to_1list(layer.dilation),
                "ceil_mode": bool(layer.ceil_mode),
                "return_indices": bool(layer.return_indices),
            }

        if isinstance(layer, nn.ReLU):
            return "relu", {}

        if isinstance(layer, nn.Flatten):
            return "reshape", {
                "kind": "flatten",
                "start_dim": int(layer.start_dim),
                "end_dim": int(layer.end_dim),
            }

        raise UnsupportedGraphError(
            f"Unsupported module op: {type(layer).__name__} at node '{node.name}'"
        )

    def _normalize_call_function(self, node: fx.Node) -> Tuple[str, Dict[str, Any]]:
        target_str = str(node.target)

        if "aten.conv1d.default" in target_str:
            args = node.args
            return "conv1d", {
                "stride": self._jsonable(args[3]) if len(args) > 3 else None,
                "padding": self._jsonable(args[4]) if len(args) > 4 else None,
                "dilation": self._jsonable(args[5]) if len(args) > 5 else None,
                "groups": self._jsonable(args[6]) if len(args) > 6 else 1,
            }

        if "aten.linear.default" in target_str:
            return "linear", {}

        if "aten.relu.default" in target_str or "aten.relu_.default" in target_str:
            return "relu", {}

        if "aten.max_pool1d.default" in target_str:
            args = node.args
            return "maxpool1d", {
                "kernel_size": self._jsonable(args[1]) if len(args) > 1 else None,
                "stride": self._jsonable(args[2]) if len(args) > 2 else None,
                "padding": self._jsonable(args[3]) if len(args) > 3 else None,
                "dilation": self._jsonable(args[4]) if len(args) > 4 else None,
                "ceil_mode": self._jsonable(args[5]) if len(args) > 5 else False,
            }

        if "aten.view.default" in target_str:
            shape = node.args[1] if len(node.args) > 1 else None
            return "reshape", {"shape": self._jsonable(shape)}

        if "aten.reshape.default" in target_str:
            shape = node.args[1] if len(node.args) > 1 else None
            return "reshape", {"shape": self._jsonable(shape)}

        if "aten.flatten.using_ints" in target_str:
            start_dim = node.args[1] if len(node.args) > 1 else 0
            end_dim = node.args[2] if len(node.args) > 2 else -1
            return "reshape", {
                "kind": "flatten",
                "start_dim": self._jsonable(start_dim),
                "end_dim": self._jsonable(end_dim),
            }

        raise UnsupportedGraphError(
            f"Unsupported function op: target='{target_str}' node='{node.name}'"
        )

    def _normalize_call_method(self, node: fx.Node) -> Tuple[str, Dict[str, Any]]:
        target = str(node.target)

        if target in {"view", "reshape"}:
            shape = list(node.args[1:]) if len(node.args) > 1 else None
            return "reshape", {"shape": self._jsonable(shape)}

        if target == "flatten":
            start_dim = node.args[1] if len(node.args) > 1 else 0
            end_dim = node.args[2] if len(node.args) > 2 else -1
            return "reshape", {
                "kind": "flatten",
                "start_dim": self._jsonable(start_dim),
                "end_dim": self._jsonable(end_dim),
            }

        if target == "relu":
            return "relu", {}

        raise UnsupportedGraphError(
            f"Unsupported method op: target='{target}' node='{node.name}'"
        )

    def normalize_graph(self) -> GraphSpec:
        if self.graph_module is None:
            self.export_graph()

        gm = self.graph_module

        inputs: List[TensorSpec] = []
        outputs: List[TensorSpec] = []
        nodes: List[NodeSpec] = []
        constants: List[ConstantSpec] = []

        for node in gm.graph.nodes:
            if node.op == "placeholder":
                inputs.append(
                    TensorSpec(
                        name=node.name,
                        shape=self._shape_from_node(node),
                        dtype=self._dtype_from_node(node),
                    )
                )

            elif node.op == "get_attr":
                obj = getattr(gm, str(node.target))
                if isinstance(obj, torch.Tensor):
                    constants.append(self._tensor_to_constant_spec(node.name, obj))
                else:
                    constants.append(self._scalar_to_constant_spec(node.name, obj))

            elif node.op == "call_module":
                op, attrs = self._normalize_call_module(node)
                nodes.append(
                        NodeSpec(
                            name=node.name,
                            op=op,
                            inputs=self._flatten_input_names(node.args),
                            outputs=[node.name],
                            attrs=attrs,
                            output_shape=self._shape_from_node(node),
                            output_dtype=self._dtype_from_node(node),
                            debug={"node_op": node.op, "target": str(node.target)},
                        )
                )

            elif node.op == "call_function":
                op, attrs = self._normalize_call_function(node)
                nodes.append(
                    NodeSpec(
                            name=node.name,
                            op=op,
                            inputs=self._flatten_input_names(node.args),
                            outputs=[node.name],
                            attrs=attrs,
                            output_shape=self._shape_from_node(node),
                            output_dtype=self._dtype_from_node(node),
                            debug={"node_op": node.op, "target": str(node.target)},
                    )
                )

            elif node.op == "call_method":
                op, attrs = self._normalize_call_method(node)
                nodes.append(
                    NodeSpec(
                            name=node.name,
                            op=op,
                            inputs=self._flatten_input_names(node.args),
                            outputs=[node.name],
                            attrs=attrs,
                            output_shape=self._shape_from_node(node),
                            output_dtype=self._dtype_from_node(node),
                            debug={"node_op": node.op, "target": str(node.target)},
                    )
                )

            elif node.op == "output":
                out = node.args[0]

                def collect(v: Any) -> None:
                    if isinstance(v, fx.Node):
                        outputs.append(
                            TensorSpec(
                                name=v.name,
                                shape=self._shape_from_node(v),
                                dtype=self._dtype_from_node(v),
                            )
                        )
                    elif isinstance(v, (list, tuple)):
                        for y in v:
                            collect(y)

                collect(out)

            else:
                raise UnsupportedGraphError(
                    f"Unsupported FX node type '{node.op}' at node '{node.name}'"
                )

        self.graph_spec = GraphSpec(
            model_name=self.model.__class__.__name__,
            inputs=inputs,
            outputs=outputs,
            nodes=nodes,
            constants=constants,
        )
        return self.graph_spec

    def validate_graph(self) -> None:
        if self.graph_spec is None:
            self.normalize_graph()

        graph = self.graph_spec

        if not graph.inputs:
            raise GraphValidationError("Graph has no inputs")
        if not graph.outputs:
            raise GraphValidationError("Graph has no outputs")

        known_values = set()

        for t in graph.inputs:
            known_values.add(t.name)

        for c in graph.constants:
            known_values.add(c.name)

        for node in graph.nodes:
            if node.op not in self.SUPPORTED_OPS:
                raise GraphValidationError(f"Unsupported normalized op '{node.op}'")

            for inp in node.inputs:
                if inp not in known_values:
                    raise GraphValidationError(
                        f"Node '{node.name}' refers to unknown input '{inp}'"
                    )

            for out in node.outputs:
                if out in known_values:
                    raise GraphValidationError(
                        f"Value '{out}' already exists before node '{node.name}'"
                    )
                known_values.add(out)

            if node.op == "conv1d":
                groups = node.attrs.get("groups", 1)
                if groups not in (1, [1]):
                    raise GraphValidationError(
                        f"Only groups=1 is supported in v1, got {groups} at '{node.name}'"
                    )

            if node.op == "maxpool1d":
                if node.attrs.get("return_indices", False):
                    raise GraphValidationError(
                        f"return_indices=True is not supported at '{node.name}'"
                    )
                if node.attrs.get("ceil_mode", False):
                    raise GraphValidationError(
                        f"ceil_mode=True is not supported at '{node.name}'"
                    )

        for out in graph.outputs:
            if out.name not in known_values:
                raise GraphValidationError(f"Output '{out.name}' is invalid")

    def save_graph_json(self, filename: str = "normalized_graph.json") -> str:
        if self.graph_spec is None:
            self.normalize_graph()
            self.validate_graph()

        path = os.path.join(self.output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self.graph_spec), f, indent=2)
        return path

    def print_graph_summary(self) -> None:
        if self.graph_spec is None:
            self.normalize_graph()
            self.validate_graph()

        graph = self.graph_spec

        print("\n========== OEMGA-SQUEEZE STEP 1 ==========")
        print(f"Model: {graph.model_name}")

        print("\nInputs:")
        for x in graph.inputs:
            print(f"  - {x.name}: shape={x.shape}, dtype={x.dtype}")

        print("\nNodes:")
        for i, n in enumerate(graph.nodes):
            print(
                f"  [{i:02d}] {n.name:<20} op={n.op:<10} "
                f"in={n.inputs} out={n.outputs} attrs={n.attrs}"
            )

        print("\nOutputs:")
        for x in graph.outputs:
            print(f"  - {x.name}: shape={x.shape}, dtype={x.dtype}")

        print(f"\nConstants captured: {len(graph.constants)}")

    def run_step1(self) -> GraphSpec:
        print("Starting OEMGA-SQUEEZE step 1...")
        self.export_graph()
        self.normalize_graph()
        self.validate_graph()
        path = self.save_graph_json()
        self.print_graph_summary()
        print(f"\n✅ Step 1 complete. Graph saved to: {path}")
        return self.graph_spec

    def _normalize_dtype_str(self, dtype: Optional[str]) -> str:
        return dtype if dtype is not None else "unknown"

    def _normalize_shape_list(self, shape: Optional[List[int]]) -> List[int]:
        if shape is None:
            raise GraphValidationError("IR lowering requires known tensor shapes")
        return [int(x) for x in shape]


    def _find_node_output_spec(self, value_name: str) -> TensorSpec:
        """
        Look for a value either in graph inputs or as node outputs.
        """
        if self.graph_spec is None:
            raise OemgaSqueezeError("graph_spec is not ready")

        for inp in self.graph_spec.inputs:
            if inp.name == value_name:
                return inp

        for node in self.graph_spec.nodes:
            if value_name in node.outputs:
                return TensorSpec(
                    name=value_name,
                    shape=node.output_shape,
                    dtype=node.output_dtype,
                )

        for out in self.graph_spec.outputs:
            if out.name == value_name:
                return out

        raise GraphValidationError(f"Could not find TensorSpec for value '{value_name}'")


    def _get_module_tensor(self, dotted_name: str) -> torch.Tensor:
        obj = self.model
        for part in dotted_name.split("."):
            if part.isdigit():
                obj = obj[int(part)]
            else:
                obj = getattr(obj, part)
        if not isinstance(obj, torch.Tensor) and not isinstance(obj, nn.Parameter):
            raise GraphValidationError(f"Expected tensor/parameter at '{dotted_name}', got {type(obj)}")
        return obj.detach().cpu()


    def _add_ir_tensor(
        self,
        tensors: List[IRTensor],
        seen: Set[str],
        name: str,
        shape: List[int],
        dtype: str,
        kind: str,
        source: Optional[str] = None,
    ) -> None:
        if name in seen:
            return
        tensors.append(
            IRTensor(
                name=name,
                shape=shape,
                dtype=dtype,
                kind=kind,
                source=source,
            )
        )
        seen.add(name)
    
    def lower_to_ir(self) -> IRGraph:
        """
        Step 2:
        Convert the step-1 normalized frontend graph into a backend-independent IR.

        Goals:
        - explicit tensors
        - explicit parameters
        - canonical node names
        - canonical tensor names
        - no PyTorch module naming in core compiler structure
        """
        if self.graph_spec is None:
            self.run_step1()

        graph = self.graph_spec

        tensors: List[IRTensor] = []
        ir_nodes: List[IRNode] = []
        seen_tensors: Set[str] = set()

        # Canonical counters
        op_counters = {
            "conv1d": 0,
            "relu": 0,
            "maxpool1d": 0,
            "reshape": 0,
            "linear": 0,
        }

        # Map frontend value names -> IR tensor names
        value_map: Dict[str, str] = {}

        # --------------------------------------------------------
        # Inputs
        # --------------------------------------------------------
        if len(graph.inputs) != 1:
            raise GraphValidationError("v1 currently expects exactly one model input")

        for i, inp in enumerate(graph.inputs):
            ir_name = f"input{i}"
            shape = self._normalize_shape_list(inp.shape)
            dtype = self._normalize_dtype_str(inp.dtype)

            self._add_ir_tensor(
                tensors=tensors,
                seen=seen_tensors,
                name=ir_name,
                shape=shape,
                dtype=dtype,
                kind="input",
                source=inp.name,
            )
            value_map[inp.name] = ir_name

        # --------------------------------------------------------
        # Nodes
        # --------------------------------------------------------
        for node in graph.nodes:
            op = node.op
            if op not in self.SUPPORTED_OPS:
                raise GraphValidationError(f"Unsupported op during IR lowering: {op}")

            idx = op_counters[op]
            op_counters[op] += 1

            # Canonical node name
            if op == "conv1d":
                ir_node_name = f"conv{idx}"
            elif op == "relu":
                ir_node_name = f"relu{idx}"
            elif op == "maxpool1d":
                ir_node_name = f"pool{idx}"
            elif op == "reshape":
                ir_node_name = f"reshape{idx}"
            elif op == "linear":
                ir_node_name = f"fc{idx}"
            else:
                ir_node_name = f"{op}{idx}"

            # Activation output tensor name
            out_tensor_name = f"t{len(ir_nodes)}"

            # Resolve activation inputs
            ir_inputs: List[str] = []
            for inp_name in node.inputs:
                if inp_name not in value_map:
                    raise GraphValidationError(
                        f"Frontend value '{inp_name}' has no IR tensor mapping"
                    )
                ir_inputs.append(value_map[inp_name])

            # Add parameter tensors explicitly for conv/linear
            attrs = dict(node.attrs)

            if op in {"conv1d", "linear"}:
                weight_name = attrs.get("weight_name", None)
                bias_name = attrs.get("bias_name", None)

                if weight_name is None:
                    raise GraphValidationError(f"{node.name} missing weight_name in attrs")

                w_tensor = self._get_module_tensor(weight_name)
                w_ir_name = f"{ir_node_name}_w"

                self._add_ir_tensor(
                    tensors=tensors,
                    seen=seen_tensors,
                    name=w_ir_name,
                    shape=list(w_tensor.shape),
                    dtype=self._dtype_to_str(w_tensor.dtype) or "float32",
                    kind="param",
                    source=weight_name,
                )
                ir_inputs.append(w_ir_name)

                if bias_name is not None:
                    b_tensor = self._get_module_tensor(bias_name)
                    b_ir_name = f"{ir_node_name}_b"

                    self._add_ir_tensor(
                        tensors=tensors,
                        seen=seen_tensors,
                        name=b_ir_name,
                        shape=list(b_tensor.shape),
                        dtype=self._dtype_to_str(b_tensor.dtype) or "float32",
                        kind="param",
                        source=bias_name,
                    )
                    ir_inputs.append(b_ir_name)

                # Remove frontend-only param references from attrs
                attrs.pop("weight_name", None)
                attrs.pop("bias_name", None)

            # Output tensor
            out_shape = self._normalize_shape_list(node.output_shape)
            out_dtype = self._normalize_dtype_str(node.output_dtype)

            self._add_ir_tensor(
                tensors=tensors,
                seen=seen_tensors,
                name=out_tensor_name,
                shape=out_shape,
                dtype=out_dtype,
                kind="activation",
                source=node.outputs[0],
            )

            ir_nodes.append(
                IRNode(
                    name=ir_node_name,
                    op=op,
                    inputs=ir_inputs,
                    outputs=[out_tensor_name],
                    attrs=attrs,
                    debug={
                        "frontend_node": node.name,
                        "frontend_target": node.debug.get("target"),
                    },
                )
            )

            # Map frontend value -> IR tensor
            value_map[node.outputs[0]] = out_tensor_name

        # --------------------------------------------------------
        # Outputs
        # --------------------------------------------------------
        ir_outputs: List[str] = []
        for i, out in enumerate(graph.outputs):
            if out.name not in value_map:
                raise GraphValidationError(f"Frontend output '{out.name}' missing from IR map")

            src_tensor_name = value_map[out.name]
            final_name = f"output{i}"

            # Find the existing tensor entry and clone it as output alias
            out_shape = self._normalize_shape_list(out.shape)
            out_dtype = self._normalize_dtype_str(out.dtype)

            self._add_ir_tensor(
                tensors=tensors,
                seen=seen_tensors,
                name=final_name,
                shape=out_shape,
                dtype=out_dtype,
                kind="output",
                source=out.name,
            )

            # Add a terminal identity-like mapping by renaming through value map.
            # To keep IR simple, we just treat output0 as the formal graph output name.
            # Later lowering/codegen can resolve output0 <- src_tensor_name if needed.
            ir_outputs.append(final_name)

            # Add a tiny passthrough node only if names differ
            if src_tensor_name != final_name:
                ir_nodes.append(
                    IRNode(
                        name=f"output_alias{i}",
                        op="reshape",
                        inputs=[src_tensor_name],
                        outputs=[final_name],
                        attrs={"kind": "identity"},
                        debug={"note": "formal graph output alias"},
                    )
                )

        ir_inputs = [f"input{i}" for i in range(len(graph.inputs))]

        self.ir_graph = IRGraph(
            model_name=graph.model_name,
            tensors=tensors,
            nodes=ir_nodes,
            inputs=ir_inputs,
            outputs=ir_outputs,
        )
        return self.ir_graph

    def validate_ir(self) -> None:
        if self.ir_graph is None:
            self.lower_to_ir()

        ir = self.ir_graph

        if not ir.inputs:
            raise GraphValidationError("IR has no inputs")
        if not ir.outputs:
            raise GraphValidationError("IR has no outputs")

        tensor_map = {t.name: t for t in ir.tensors}
        if len(tensor_map) != len(ir.tensors):
            raise GraphValidationError("Duplicate tensor names found in IR")

        for name in ir.inputs:
            if name not in tensor_map:
                raise GraphValidationError(f"IR input '{name}' missing in tensor table")
            if tensor_map[name].kind != "input":
                raise GraphValidationError(f"IR input '{name}' is not marked as input")

        for name in ir.outputs:
            if name not in tensor_map:
                raise GraphValidationError(f"IR output '{name}' missing in tensor table")
            if tensor_map[name].kind != "output":
                raise GraphValidationError(f"IR output '{name}' is not marked as output")

        produced = set(ir.inputs)
        produced.update([t.name for t in ir.tensors if t.kind == "param"])

        for node in ir.nodes:
            for inp in node.inputs:
                if inp not in tensor_map:
                    raise GraphValidationError(f"Node '{node.name}' uses unknown tensor '{inp}'")

            for out in node.outputs:
                if out not in tensor_map:
                    raise GraphValidationError(f"Node '{node.name}' produces unknown tensor '{out}'")

            if node.op in {"conv1d", "linear"}:
                if len(node.inputs) < 2:
                    raise GraphValidationError(f"{node.op} node '{node.name}' must have activation + weight")
            elif node.op in {"relu", "maxpool1d", "reshape"}:
                if len(node.inputs) != 1:
                    raise GraphValidationError(f"{node.op} node '{node.name}' must have exactly one input")
    
    def save_ir_json(self, filename: str = "ir_graph.json") -> str:
        if self.ir_graph is None:
            self.lower_to_ir()
            self.validate_ir()

        path = os.path.join(self.output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self.ir_graph), f, indent=2)
        return path


    def print_ir_summary(self) -> None:
        if self.ir_graph is None:
            self.lower_to_ir()
            self.validate_ir()

        ir = self.ir_graph

        print("\n========== OEMGA-SQUEEZE STEP 2 IR ==========")
        print(f"Model: {ir.model_name}")

        print("\nIR Inputs:")
        for name in ir.inputs:
            print(f"  - {name}")

        print("\nIR Tensors:")
        for t in ir.tensors:
            print(f"  - {t.name:<12} kind={t.kind:<10} shape={t.shape} dtype={t.dtype}")

        print("\nIR Nodes:")
        for i, n in enumerate(ir.nodes):
            print(
                f"  [{i:02d}] {n.name:<12} op={n.op:<10} "
                f"in={n.inputs} out={n.outputs} attrs={n.attrs}"
            )

        print("\nIR Outputs:")
        for name in ir.outputs:
            print(f"  - {name}")

    def run_step2(self) -> IRGraph:
        print("Starting OEMGA-SQUEEZE step 2: lowering to backend-independent IR...")
        if self.graph_spec is None:
            self.run_step1()
        self.lower_to_ir()
        self.validate_ir()
        path = self.save_ir_json()
        self.print_ir_summary()
        print(f"\n✅ Step 2 complete. IR saved to: {path}")
        return self.ir_graph


    def _dtype_size_bytes(self, dtype: str) -> int:
        table = {
            "float32": 4,
            "float": 4,
            "int32": 4,
            "uint32": 4,
            "int16": 2,
            "uint16": 2,
            "int8": 1,
            "uint8": 1,
            "bool": 1,
        }
        if dtype not in table:
            raise GraphValidationError(f"Unknown dtype size for '{dtype}'")
        return table[dtype]


    def _num_elements(self, shape: List[int]) -> int:
        n = 1
        for x in shape:
            n *= int(x)
        return n


    def _tensor_map(self) -> Dict[str, IRTensor]:
        if self.ir_graph is None:
            raise OemgaSqueezeError("IR graph not ready")
        return {t.name: t for t in self.ir_graph.tensors}


    def _tensor_bytes(self, tensor: IRTensor) -> int:
        return self._num_elements(tensor.shape) * self._dtype_size_bytes(tensor.dtype)
    
    def analyze_ir(self) -> CostReport:
        """
        Step 3:
        Analyze backend-independent IR and produce a detailed cost report.

        Current report includes:
        - tensor sizes
        - parameter bytes
        - activation bytes
        - peak activation bytes
        - per-node MACs
        - total MACs
        """
        if self.ir_graph is None:
            self.run_step2()

        ir = self.ir_graph
        tensor_map = self._tensor_map()

        tensor_costs: List[TensorCost] = []
        node_costs: List[NodeCost] = []

        total_param_bytes = 0
        total_activation_bytes = 0
        peak_activation_bytes = 0
        total_macs = 0

        # --------------------------------------------------------
        # Tensor costs
        # --------------------------------------------------------
        for t in ir.tensors:
            n_elem = self._num_elements(t.shape)
            n_bytes = n_elem * self._dtype_size_bytes(t.dtype)

            tensor_costs.append(
                TensorCost(
                    name=t.name,
                    kind=t.kind,
                    shape=t.shape,
                    dtype=t.dtype,
                    num_elements=n_elem,
                    bytes=n_bytes,
                )
            )

            if t.kind == "param":
                total_param_bytes += n_bytes
            elif t.kind in {"activation", "output"}:
                total_activation_bytes += n_bytes
                peak_activation_bytes = max(peak_activation_bytes, n_bytes)

        # --------------------------------------------------------
        # Node costs
        # --------------------------------------------------------
        for node in ir.nodes:
            input_tensors = [tensor_map[name] for name in node.inputs]
            output_tensors = [tensor_map[name] for name in node.outputs]

            macs = 0
            param_bytes = 0
            output_bytes = sum(self._tensor_bytes(t) for t in output_tensors)
            notes: Dict[str, Any] = {}

            if node.op == "conv1d":
                # inputs = [activation, weight, bias?]
                x = input_tensors[0]
                w = input_tensors[1]
                y = output_tensors[0]

                # For Conv1d:
                # input  [N, Cin, Lin]
                # weight [Cout, Cin/groups, K]
                # output [N, Cout, Lout]
                N, Cin, Lin = x.shape
                Cout, Cin_per_group, K = w.shape
                Ny, Cout_y, Lout = y.shape

                groups = int(node.attrs.get("groups", 1))
                if Cout != Cout_y or N != Ny:
                    raise GraphValidationError(
                        f"Conv node '{node.name}' has inconsistent tensor shapes"
                    )

                macs = int(N * Cout * Lout * Cin_per_group * K * groups / groups)
                # Equivalent to N * Cout * Lout * Cin_per_group * K

                param_bytes = sum(
                    self._tensor_bytes(t)
                    for t in input_tensors[1:]
                    if t.kind == "param"
                )

                notes = {
                    "input_shape": x.shape,
                    "weight_shape": w.shape,
                    "output_shape": y.shape,
                    "kernel_size": K,
                    "groups": groups,
                }

            elif node.op == "linear":
                # inputs = [activation, weight, bias?]
                x = input_tensors[0]
                w = input_tensors[1]
                y = output_tensors[0]

                # Typical shapes:
                # x [N, InF]
                # w [OutF, InF]
                # y [N, OutF]
                if len(x.shape) != 2 or len(w.shape) != 2 or len(y.shape) != 2:
                    raise GraphValidationError(
                        f"Linear node '{node.name}' expects rank-2 tensors"
                    )

                N, InF = x.shape
                OutF, WInF = w.shape
                Ny, YOutF = y.shape

                if N != Ny or InF != WInF or OutF != YOutF:
                    raise GraphValidationError(
                        f"Linear node '{node.name}' has inconsistent tensor shapes"
                    )

                macs = int(N * InF * OutF)

                param_bytes = sum(
                    self._tensor_bytes(t)
                    for t in input_tensors[1:]
                    if t.kind == "param"
                )

                notes = {
                    "input_shape": x.shape,
                    "weight_shape": w.shape,
                    "output_shape": y.shape,
                }

            elif node.op in {"relu", "maxpool1d", "reshape"}:
                macs = 0
                param_bytes = 0
                notes = {
                    "input_shape": input_tensors[0].shape if input_tensors else None,
                    "output_shape": output_tensors[0].shape if output_tensors else None,
                }

            else:
                raise GraphValidationError(f"Unsupported op in analyzer: {node.op}")

            total_macs += macs

            node_costs.append(
                NodeCost(
                    name=node.name,
                    op=node.op,
                    input_tensors=node.inputs,
                    output_tensors=node.outputs,
                    macs=macs,
                    param_bytes=param_bytes,
                    output_bytes=output_bytes,
                    notes=notes,
                )
            )

        largest_param = None
        largest_activation = None

        param_tensors = [tc for tc in tensor_costs if tc.kind == "param"]
        act_tensors = [tc for tc in tensor_costs if tc.kind in {"activation", "output"}]

        if param_tensors:
            largest_param = max(param_tensors, key=lambda x: x.bytes).name
        if act_tensors:
            largest_activation = max(act_tensors, key=lambda x: x.bytes).name

        embedded_estimates = self.estimate_embedded_deployment()

        self.cost_report = CostReport(
            model_name=ir.model_name,
            total_param_bytes=total_param_bytes,
            total_activation_bytes=total_activation_bytes,
            peak_activation_bytes=peak_activation_bytes,
            total_macs=total_macs,
            tensor_costs=tensor_costs,
            node_costs=node_costs,
            summary={
                "num_tensors": len(ir.tensors),
                "num_nodes": len(ir.nodes),
                "largest_param_tensor": largest_param,
                "largest_activation_tensor": largest_activation,
            },
            embedded_estimates=embedded_estimates,
        )
        return self.cost_report

    def save_cost_report_json(self, filename: str = "cost_report.json") -> str:
        if self.cost_report is None:
            self.analyze_ir()

        path = os.path.join(self.output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self.cost_report), f, indent=2)
        return path

    def print_cost_report(self) -> None:
        if self.cost_report is None:
            self.analyze_ir()

        report = self.cost_report
        emb = report.embedded_estimates

        print("\n========== OEMGA-SQUEEZE STEP 3 ANALYZER ==========")
        print(f"Model: {report.model_name}")
        print(f"Total parameter bytes   : {report.total_param_bytes} ({report.total_param_bytes / 1024.0:.2f} KB)")
        print(f"Total activation bytes  : {report.total_activation_bytes} ({report.total_activation_bytes / 1024.0:.2f} KB)")
        print(f"Peak activation bytes   : {report.peak_activation_bytes} ({report.peak_activation_bytes / 1024.0:.2f} KB)")
        print(f"Total MACs              : {report.total_macs}")
        print(f"Largest param tensor    : {report.summary.get('largest_param_tensor')}")
        print(f"Largest activation tens.: {report.summary.get('largest_activation_tensor')}")

        if emb is not None:
            print("\n--- Embedded deployment estimates ---")
            print(
                f"Int8 weight storage     : {emb.weight_storage_int8_bytes} "
                f"({emb.weight_storage_int8_bytes / 1024.0:.2f} KB)"
            )
            print(
                f"Int32 bias storage      : {emb.bias_storage_int32_bytes} "
                f"({emb.bias_storage_int32_bytes / 1024.0:.2f} KB)"
            )
            print(
                f"Requant param storage   : {emb.requant_param_bytes} "
                f"({emb.requant_param_bytes / 1024.0:.2f} KB)"
            )
            print(
                f"Est. model-data flash   : {emb.estimated_model_data_flash_bytes} "
                f"({emb.estimated_model_data_flash_bytes / 1024.0:.2f} KB)"
            )
            print(
                f"Peak int8 activation    : {emb.peak_activation_int8_bytes} "
                f"({emb.peak_activation_int8_bytes / 1024.0:.2f} KB)"
            )
            print(
                f"Double-buffer int8 RAM  : {emb.double_buffer_int8_bytes} "
                f"({emb.double_buffer_int8_bytes / 1024.0:.2f} KB)"
            )
            print(
                f"Max conv scratch RAM    : {emb.max_conv_scratch_int8_bytes} "
                f"({emb.max_conv_scratch_int8_bytes / 1024.0:.2f} KB)"
            )
            print(
                f"Est. runtime RAM        : {emb.estimated_runtime_ram_bytes} "
                f"({emb.estimated_runtime_ram_bytes / 1024.0:.2f} KB)"
            )

            print("\nDeployment notes:")
            for note in emb.notes:
                print(f"  - {note}")

        print("\nPer-node costs:")
        for n in report.node_costs:
            print(
                f"  - {n.name:<12} op={n.op:<10} "
                f"macs={n.macs:<10} "
                f"param_bytes={n.param_bytes:<8} "
                f"output_bytes={n.output_bytes}"
            )

    def run_step3(self) -> CostReport:
        print("Starting OEMGA-SQUEEZE step 3: IR analyzer...")
        if self.ir_graph is None:
            self.run_step2()
        self.analyze_ir()
        path = self.save_cost_report_json()
        self.print_cost_report()
        print(f"\n✅ Step 3 complete. Cost report saved to: {path}")
        return self.cost_report

    def _tensor_shape_rank(self, tensor: IRTensor) -> int:
        return len(tensor.shape)

    def _conv_output_channels_from_weight(self, weight_tensor: IRTensor) -> int:
        if len(weight_tensor.shape) != 3:
            raise GraphValidationError(
                f"Conv weight tensor '{weight_tensor.name}' must be rank-3, got {weight_tensor.shape}"
            )
        return int(weight_tensor.shape[0])

    def _linear_output_features_from_weight(self, weight_tensor: IRTensor) -> int:
        if len(weight_tensor.shape) != 2:
            raise GraphValidationError(
                f"Linear weight tensor '{weight_tensor.name}' must be rank-2, got {weight_tensor.shape}"
            )
        return int(weight_tensor.shape[0])

    def estimate_embedded_deployment(self) -> EmbeddedEstimates:
        """
        Estimate embedded deployment cost assuming a typical int8 inference pipeline:
        - weights stored as int8
        - bias stored as int32
        - per-output-channel requant params:
            * out_mult_q31 : int32
            * out_shift    : int8
        - activations stored as int8
        - simple double-buffer working model
        - conv scratch for padded input in int8 domain

        This is an estimate, not final lowering/runtime memory planning.
        """
        if self.ir_graph is None:
            self.run_step2()

        ir = self.ir_graph
        tensor_map = self._tensor_map()

        weight_storage_int8_bytes = 0
        bias_storage_int32_bytes = 0
        requant_param_bytes = 0

        peak_activation_int8_bytes = 0
        max_conv_scratch_int8_bytes = 0

        notes: List[str] = []

        # --------------------------------------------------------
        # Estimate activation bytes in int8 domain
        # --------------------------------------------------------
        for t in ir.tensors:
            if t.kind in {"activation", "output"}:
                n = self._num_elements(t.shape)
                peak_activation_int8_bytes = max(peak_activation_int8_bytes, n)

        double_buffer_int8_bytes = 2 * peak_activation_int8_bytes

        # --------------------------------------------------------
        # Estimate parameter storage and conv scratch
        # --------------------------------------------------------
        for node in ir.nodes:
            if node.op not in {"conv1d", "linear"}:
                continue

            if len(node.inputs) < 2:
                raise GraphValidationError(
                    f"Node '{node.name}' missing weight tensor input for deployment estimate"
                )

            act_tensor = tensor_map[node.inputs[0]]
            weight_tensor = tensor_map[node.inputs[1]]
            bias_tensor = tensor_map[node.inputs[2]] if len(node.inputs) >= 3 else None

            # Weight storage as int8
            weight_storage_int8_bytes += self._num_elements(weight_tensor.shape)

            # Bias storage as int32
            out_channels_or_features = 0

            if node.op == "conv1d":
                out_channels_or_features = self._conv_output_channels_from_weight(weight_tensor)

                if len(act_tensor.shape) != 3:
                    raise GraphValidationError(
                        f"Conv activation tensor '{act_tensor.name}' must be rank-3, got {act_tensor.shape}"
                    )

                _, in_ch, in_len = act_tensor.shape
                padding = node.attrs.get("padding", [0])
                if isinstance(padding, list):
                    pad = int(padding[0])
                else:
                    pad = int(padding)

                # Your planned backend style uses padded input scratch:
                # scratch = in_channels * (length + 2*padding) * 1 byte
                scratch_bytes = int(in_ch) * int(in_len + 2 * pad)
                max_conv_scratch_int8_bytes = max(max_conv_scratch_int8_bytes, scratch_bytes)

            elif node.op == "linear":
                out_channels_or_features = self._linear_output_features_from_weight(weight_tensor)

            if bias_tensor is not None:
                bias_storage_int32_bytes += self._num_elements(bias_tensor.shape) * 4
            else:
                # If no bias tensor, still keep zero
                bias_storage_int32_bytes += 0

            # Requant metadata per output channel/feature:
            # out_mult_q31 -> int32
            # out_shift    -> int8
            requant_param_bytes += out_channels_or_features * (4 + 1)

        estimated_model_data_flash_bytes = (
            weight_storage_int8_bytes
            + bias_storage_int32_bytes
            + requant_param_bytes
        )

        estimated_runtime_ram_bytes = (
            double_buffer_int8_bytes
            + max_conv_scratch_int8_bytes
        )

        # --------------------------------------------------------
        # Helpful notes
        # --------------------------------------------------------
        notes.append(
            "Assumes int8 weights, int8 activations, int32 bias, and per-output-channel requant params."
        )
        notes.append(
            "Runtime RAM estimate assumes simple double buffering plus max Conv1d padded-input scratch."
        )
        notes.append(
            "This does not yet include stack, logging, allocator overhead, or Zephyr/system buffers."
        )

        return EmbeddedEstimates(
            weight_storage_int8_bytes=weight_storage_int8_bytes,
            bias_storage_int32_bytes=bias_storage_int32_bytes,
            requant_param_bytes=requant_param_bytes,
            estimated_model_data_flash_bytes=estimated_model_data_flash_bytes,
            peak_activation_int8_bytes=peak_activation_int8_bytes,
            double_buffer_int8_bytes=double_buffer_int8_bytes,
            max_conv_scratch_int8_bytes=max_conv_scratch_int8_bytes,
            estimated_runtime_ram_bytes=estimated_runtime_ram_bytes,
            notes=notes,
        )

    def _get_ir_tensor(self, name: str) -> IRTensor:
        if self.ir_graph is None:
            raise OemgaSqueezeError("IR graph not ready")
        for t in self.ir_graph.tensors:
            if t.name == name:
                return t
        raise GraphValidationError(f"IR tensor '{name}' not found")


    def _estimate_tensor_int8_bytes(self, tensor: IRTensor) -> int:
        return self._num_elements(tensor.shape)


    def _max_activation_int8_bytes(self) -> int:
        if self.ir_graph is None:
            raise OemgaSqueezeError("IR graph not ready")
        peak = 0
        for t in self.ir_graph.tensors:
            if t.kind in {"activation", "output"}:
                peak = max(peak, self._estimate_tensor_int8_bytes(t))
        return peak


    def _max_conv_scratch_int8_bytes(self) -> int:
        if self.ir_graph is None:
            raise OemgaSqueezeError("IR graph not ready")

        max_scratch = 0
        tensor_map = self._tensor_map()

        for node in self.ir_graph.nodes:
            if node.op != "conv1d":
                continue

            x = tensor_map[node.inputs[0]]
            if len(x.shape) != 3:
                raise GraphValidationError(f"Conv input tensor '{x.name}' must be rank-3")

            _, in_ch, in_len = x.shape
            padding = node.attrs.get("padding", [0])
            if isinstance(padding, list):
                pad = int(padding[0])
            else:
                pad = int(padding)

            scratch = int(in_ch) * int(in_len + 2 * pad)
            max_scratch = max(max_scratch, scratch)

        return max_scratch

    def lower_to_backend(self, backend: str = "oemga_native_int8") -> LoweredPlan:
        """
        Step 4:
        Lower backend-independent IR into a backend-specific execution plan.

        Current backend assumptions:
        - int8 activations
        - int8 weights
        - int32 bias
        - ping-pong activation buffers
        - one shared scratch buffer
        """
        if self.ir_graph is None:
            self.run_step2()

        if backend != "oemga_native_int8":
            raise OemgaSqueezeError(f"Unsupported backend '{backend}' in v1")

        ir = self.ir_graph
        tensor_map = self._tensor_map()

        peak_act_bytes = self._max_activation_int8_bytes()
        max_scratch_bytes = self._max_conv_scratch_int8_bytes()

        buffers = [
            MemoryBuffer(
                name="buf0",
                kind="activation",
                size_bytes=peak_act_bytes,
                notes={"role": "ping"},
            ),
            MemoryBuffer(
                name="buf1",
                kind="activation",
                size_bytes=peak_act_bytes,
                notes={"role": "pong"},
            ),
        ]

        if max_scratch_bytes > 0:
            buffers.append(
                MemoryBuffer(
                    name="scratch0",
                    kind="scratch",
                    size_bytes=max_scratch_bytes,
                    notes={"role": "shared_conv_scratch"},
                )
            )

        tensor_locations: Dict[str, Dict[str, Any]] = {}

        # Params are not in ping-pong buffers
        for t in ir.tensors:
            if t.kind == "param":
                tensor_locations[t.name] = {
                    "storage": "param",
                    "buffer": None,
                    "offset": None,
                    "notes": {"source": t.source},
                }

        # Inputs start in buf0
        for inp in ir.inputs:
            tensor_locations[inp] = {
                "storage": "activation",
                "buffer": "buf0",
                "offset": 0,
                "notes": {"role": "graph_input"},
            }

        lowered_ops: List[LoweredOp] = []

        cur_buf = "buf0"
        nxt_buf = "buf1"

        for node in ir.nodes:
            input_buffer = None
            output_buffer = None
            scratch_buffer = None
            backend_op = ""
            notes: Dict[str, Any] = {}

            # Primary activation input buffer
            if node.inputs:
                first_in = node.inputs[0]
                if first_in in tensor_locations:
                    input_buffer = tensor_locations[first_in]["buffer"]

            if node.op == "conv1d":
                backend_op = "conv1d_s8"
                output_buffer = nxt_buf
                scratch_buffer = "scratch0" if max_scratch_bytes > 0 else None

                notes = {
                    "weight_tensor": node.inputs[1],
                    "bias_tensor": node.inputs[2] if len(node.inputs) > 2 else None,
                    "layout": "NCL",
                }

                out_name = node.outputs[0]
                tensor_locations[out_name] = {
                    "storage": "activation",
                    "buffer": output_buffer,
                    "offset": 0,
                    "notes": {"producer": node.name},
                }

                lowered_ops.append(
                    LoweredOp(
                        name=node.name,
                        source_ir_node=node.name,
                        backend_op=backend_op,
                        inputs=node.inputs,
                        outputs=node.outputs,
                        input_buffer=input_buffer,
                        output_buffer=output_buffer,
                        scratch_buffer=scratch_buffer,
                        attrs=dict(node.attrs),
                        notes=notes,
                    )
                )

                cur_buf, nxt_buf = nxt_buf, cur_buf

            elif node.op == "relu":
                backend_op = "relu_s8"
                output_buffer = nxt_buf

                out_name = node.outputs[0]
                tensor_locations[out_name] = {
                    "storage": "activation",
                    "buffer": output_buffer,
                    "offset": 0,
                    "notes": {"producer": node.name},
                }

                lowered_ops.append(
                    LoweredOp(
                        name=node.name,
                        source_ir_node=node.name,
                        backend_op=backend_op,
                        inputs=node.inputs,
                        outputs=node.outputs,
                        input_buffer=input_buffer,
                        output_buffer=output_buffer,
                        scratch_buffer=None,
                        attrs=dict(node.attrs),
                        notes={"mode": "out_of_place"},
                    )
                )

                cur_buf, nxt_buf = nxt_buf, cur_buf

            elif node.op == "maxpool1d":
                backend_op = "maxpool1d_s8"
                output_buffer = nxt_buf

                out_name = node.outputs[0]
                tensor_locations[out_name] = {
                    "storage": "activation",
                    "buffer": output_buffer,
                    "offset": 0,
                    "notes": {"producer": node.name},
                }

                lowered_ops.append(
                    LoweredOp(
                        name=node.name,
                        source_ir_node=node.name,
                        backend_op=backend_op,
                        inputs=node.inputs,
                        outputs=node.outputs,
                        input_buffer=input_buffer,
                        output_buffer=output_buffer,
                        scratch_buffer=None,
                        attrs=dict(node.attrs),
                        notes={"mode": "out_of_place"},
                    )
                )

                cur_buf, nxt_buf = nxt_buf, cur_buf

            elif node.op == "reshape":
                backend_op = "reshape_view"

                # No new buffer if this is just flatten/identity
                output_buffer = input_buffer

                out_name = node.outputs[0]
                tensor_locations[out_name] = {
                    "storage": "activation_view",
                    "buffer": output_buffer,
                    "offset": 0,
                    "notes": {"producer": node.name, "view_of": node.inputs[0]},
                }

                lowered_ops.append(
                    LoweredOp(
                        name=node.name,
                        source_ir_node=node.name,
                        backend_op=backend_op,
                        inputs=node.inputs,
                        outputs=node.outputs,
                        input_buffer=input_buffer,
                        output_buffer=output_buffer,
                        scratch_buffer=None,
                        attrs=dict(node.attrs),
                        notes={"mode": "view_only"},
                    )
                )

            elif node.op == "linear":
                backend_op = "linear_s8"
                output_buffer = nxt_buf

                out_name = node.outputs[0]
                tensor_locations[out_name] = {
                    "storage": "activation",
                    "buffer": output_buffer,
                    "offset": 0,
                    "notes": {"producer": node.name},
                }

                lowered_ops.append(
                    LoweredOp(
                        name=node.name,
                        source_ir_node=node.name,
                        backend_op=backend_op,
                        inputs=node.inputs,
                        outputs=node.outputs,
                        input_buffer=input_buffer,
                        output_buffer=output_buffer,
                        scratch_buffer=None,
                        attrs=dict(node.attrs),
                        notes={
                            "weight_tensor": node.inputs[1],
                            "bias_tensor": node.inputs[2] if len(node.inputs) > 2 else None,
                            "layout": "flat",
                        },
                    )
                )

                cur_buf, nxt_buf = nxt_buf, cur_buf

            else:
                raise GraphValidationError(f"Unsupported IR op for lowering: {node.op}")

        # Mark formal graph outputs
        for out in ir.outputs:
            if out not in tensor_locations:
                # output alias case already handled by reshape_view node
                tensor_locations[out] = {
                    "storage": "activation",
                    "buffer": cur_buf,
                    "offset": 0,
                    "notes": {"role": "graph_output"},
                }

        self.lowered_plan = LoweredPlan(
            model_name=ir.model_name,
            backend=backend,
            buffers=buffers,
            tensor_locations=tensor_locations,
            ops=lowered_ops,
            inputs=list(ir.inputs),
            outputs=list(ir.outputs),
            summary={
                "num_lowered_ops": len(lowered_ops),
                "num_buffers": len(buffers),
                "activation_buffer_bytes_each": peak_act_bytes,
                "scratch_buffer_bytes": max_scratch_bytes,
            },
        )
        return self.lowered_plan
    
    def validate_lowered_plan(self) -> None:
        if self.lowered_plan is None:
            self.lower_to_backend()

        plan = self.lowered_plan

        buffer_names = {b.name for b in plan.buffers}

        for inp in plan.inputs:
            if inp not in plan.tensor_locations:
                raise GraphValidationError(f"Lowered input '{inp}' missing tensor location")

        for out in plan.outputs:
            if out not in plan.tensor_locations:
                raise GraphValidationError(f"Lowered output '{out}' missing tensor location")

        for op in plan.ops:
            if op.input_buffer is not None and op.input_buffer not in buffer_names:
                raise GraphValidationError(
                    f"Lowered op '{op.name}' uses unknown input buffer '{op.input_buffer}'"
                )
            if op.output_buffer is not None and op.output_buffer not in buffer_names:
                raise GraphValidationError(
                    f"Lowered op '{op.name}' uses unknown output buffer '{op.output_buffer}'"
                )
            if op.scratch_buffer is not None and op.scratch_buffer not in buffer_names:
                raise GraphValidationError(
                    f"Lowered op '{op.name}' uses unknown scratch buffer '{op.scratch_buffer}'"
                )

    def save_lowered_plan_json(self, filename: str = "lowered_plan.json") -> str:
        if self.lowered_plan is None:
            self.lower_to_backend()
            self.validate_lowered_plan()

        path = os.path.join(self.output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self.lowered_plan), f, indent=2)
        return path


    def print_lowered_plan(self) -> None:
        if self.lowered_plan is None:
            self.lower_to_backend()
            self.validate_lowered_plan()

        plan = self.lowered_plan

        print("\n========== OEMGA-SQUEEZE STEP 4 LOWERING ==========")
        print(f"Model: {plan.model_name}")
        print(f"Backend: {plan.backend}")

        print("\nBuffers:")
        for b in plan.buffers:
            print(f"  - {b.name:<10} kind={b.kind:<10} size_bytes={b.size_bytes} notes={b.notes}")

        print("\nTensor locations:")
        for name, loc in plan.tensor_locations.items():
            print(f"  - {name:<12} storage={loc['storage']:<16} buffer={loc['buffer']}")

        print("\nLowered ops:")
        for i, op in enumerate(plan.ops):
            print(
                f"  [{i:02d}] {op.name:<14} backend_op={op.backend_op:<14} "
                f"in_buf={op.input_buffer} out_buf={op.output_buffer} scratch={op.scratch_buffer}"
            )

        print("\nOutputs:")
        for out in plan.outputs:
            print(f"  - {out}")

    def run_step4(self, backend: str = "oemga_native_int8") -> LoweredPlan:
        print("Starting OEMGA-SQUEEZE step 4: lowering to backend + memory planning...")
        if self.ir_graph is None:
            self.run_step2()
        self.lower_to_backend(backend=backend)
        self.validate_lowered_plan()
        path = self.save_lowered_plan_json()
        self.print_lowered_plan()
        print(f"\n✅ Step 4 complete. Lowered plan saved to: {path}")
        return self.lowered_plan

    @staticmethod
    def _safe_scale_from_maxabs(maxabs: float) -> float:
        if maxabs <= 0 or not np.isfinite(maxabs):
            return 1.0
        return float(maxabs / 127.0)


    @staticmethod
    def _quantize_symmetric_fp_to_int8(x: np.ndarray, scale: float) -> np.ndarray:
        if scale <= 0 or not np.isfinite(scale):
            scale = 1.0
        q = np.round(x / scale)
        q = np.clip(q, -128, 127).astype(np.int8)
        return q


    @staticmethod
    def _per_channel_weight_quant(w: torch.Tensor, out_channels: int):
        w_np = w.detach().cpu().numpy().astype(np.float32)
        w_scales = np.zeros((out_channels,), dtype=np.float32)
        w_q = np.zeros_like(w_np, dtype=np.int8)

        for oc in range(out_channels):
            w_oc = w_np[oc]
            maxabs = float(np.max(np.abs(w_oc))) if w_oc.size else 0.0
            scale = (maxabs / 127.0) if maxabs != 0 else 1.0
            w_scales[oc] = scale
            w_q[oc] = np.clip(np.round(w_oc / scale), -128, 127).astype(np.int8)

        return w_q, w_scales


    @staticmethod
    def _real_multiplier_to_q31_shift(real_multiplier: float):
        if real_multiplier <= 0 or not np.isfinite(real_multiplier):
            return 0, 0
        if real_multiplier > 1.0:
            real_multiplier = 1.0

        mantissa, exp = np.frexp(real_multiplier)
        mult_q31 = int(np.round(mantissa * (1 << 31)))
        if mult_q31 == (1 << 31):
            mult_q31 -= 1

        shift = -int(exp)
        if shift < 0:
            shift = 0
        return mult_q31, shift

    def _collect_layer_outputs_for_codegen(self):
        wanted = (nn.Conv1d, nn.Linear)
        outputs = {}
        hooks = []

        def make_hook(name):
            def hook(_m, _inp, out):
                outputs[name] = out.detach().cpu()
            return hook

        for name, m in self.model.named_modules():
            if isinstance(m, wanted):
                hooks.append(m.register_forward_hook(make_hook(name)))

        with torch.no_grad():
            _ = self.model(self.calibration_data)

        for h in hooks:
            h.remove()

        return outputs

    def build_codegen_qparams(self) -> None:
        """
        Build int8/int32 parameters for lowered Conv1d/Linear ops.
        Stores results in self.qparams keyed by lowered op name (conv0, conv1, fc0, fc1 ...).
        """
        if self.lowered_plan is None:
            self.run_step4()

        layer_outputs = self._collect_layer_outputs_for_codegen()

        calib_np = np.abs(self.calibration_data.detach().cpu().numpy().astype(np.float32))
        x_maxabs = float(np.percentile(calib_np, 99.9)) if calib_np.size else 0.0
        current_x_scale = self._safe_scale_from_maxabs(x_maxabs)

        self.qparams = {}
        self.input_scale = float(current_x_scale)

        for op in self.lowered_plan.ops:
            if op.backend_op not in {"conv1d_s8", "linear_s8"}:
                continue

            frontend_target = op.notes.get("weight_tensor")
            if frontend_target is None:
                raise OemgaSqueezeError(f"Lowered op '{op.name}' missing weight_tensor note")

            weight_ir_name = op.inputs[1]
            bias_ir_name = op.inputs[2] if len(op.inputs) > 2 else None

            weight_source = None
            bias_source = None

            for t in self.ir_graph.tensors:
                if t.name == weight_ir_name:
                    weight_source = t.source
                if bias_ir_name is not None and t.name == bias_ir_name:
                    bias_source = t.source

            if weight_source is None:
                raise OemgaSqueezeError(f"Could not find source weight for '{op.name}'")

            weight_t = self._get_module_tensor(weight_source).float()
            bias_t = self._get_module_tensor(bias_source).float() if bias_source is not None else None

            if op.backend_op == "conv1d_s8":
                out_ch = int(weight_t.shape[0])
                w_q, w_scale = self._per_channel_weight_quant(weight_t, out_ch)

                b = (
                    bias_t.numpy().astype(np.float32)
                    if bias_t is not None
                    else np.zeros((out_ch,), dtype=np.float32)
                )

                frontend_target_name = op.debug.get("frontend_target") if hasattr(op, "debug") else None
                module_name = op.notes.get("frontend_target", None)
                layer_name = op.notes.get("layer_name", None)

                # use original frontend target from lowered op source via IR/debug mapping
                # easiest stable lookup: source_ir_node -> step2 debug target
                ir_node = next(n for n in self.ir_graph.nodes if n.name == op.source_ir_node)
                layer_key = ir_node.debug.get("frontend_target")

                y_float = layer_outputs[layer_key].numpy()
                y_maxabs = float(np.percentile(np.abs(y_float), 99.9)) if y_float.size else 0.0
                y_scale_obs = self._safe_scale_from_maxabs(y_maxabs)
                y_scale = max(y_scale_obs, float(current_x_scale * float(np.max(w_scale))))

                bias_q = np.zeros((out_ch,), dtype=np.int32)
                out_mult_q31 = np.zeros((out_ch,), dtype=np.int32)
                out_shift = np.zeros((out_ch,), dtype=np.int8)

                for oc in range(out_ch):
                    denom = float(current_x_scale) * float(w_scale[oc])
                    denom = denom if denom != 0 else 1.0
                    bias_q[oc] = int(np.round(float(b[oc]) / denom))

                    real_mult = (float(current_x_scale) * float(w_scale[oc])) / float(y_scale)
                    mult_q31, sh = self._real_multiplier_to_q31_shift(real_mult)
                    out_mult_q31[oc] = np.int32(mult_q31)
                    out_shift[oc] = np.int8(sh)

                pad = op.attrs.get("padding", [0])
                pad = int(pad[0]) if isinstance(pad, list) else int(pad)
                k = op.attrs.get("kernel_size", [1])
                k = int(k[0]) if isinstance(k, list) else int(k)

                self.qparams[op.name] = dict(
                    kind="conv1d",
                    w_q=w_q,
                    bias_q=bias_q,
                    y_scale=float(y_scale),
                    out_mult_q31=out_mult_q31,
                    out_shift=out_shift,
                    padding=pad,
                    kernel_size=k,
                    in_ch=int(weight_t.shape[1]),
                    out_ch=int(weight_t.shape[0]),
                )
                current_x_scale = float(y_scale)

            elif op.backend_op == "linear_s8":
                out_f = int(weight_t.shape[0])
                w_q, w_scale = self._per_channel_weight_quant(weight_t, out_f)

                b = (
                    bias_t.numpy().astype(np.float32)
                    if bias_t is not None
                    else np.zeros((out_f,), dtype=np.float32)
                )

                ir_node = next(n for n in self.ir_graph.nodes if n.name == op.source_ir_node)
                layer_key = ir_node.debug.get("frontend_target")

                y_float = layer_outputs[layer_key].numpy()
                y_maxabs = float(np.percentile(np.abs(y_float), 99.9)) if y_float.size else 0.0
                y_scale_obs = self._safe_scale_from_maxabs(y_maxabs)
                y_scale = max(y_scale_obs, float(current_x_scale * float(np.max(w_scale))))

                bias_q = np.zeros((out_f,), dtype=np.int32)
                out_mult_q31 = np.zeros((out_f,), dtype=np.int32)
                out_shift = np.zeros((out_f,), dtype=np.int8)

                for oc in range(out_f):
                    denom = float(current_x_scale) * float(w_scale[oc])
                    denom = denom if denom != 0 else 1.0
                    bias_q[oc] = int(np.round(float(b[oc]) / denom))

                    real_mult = (float(current_x_scale) * float(w_scale[oc])) / float(y_scale)
                    mult_q31, sh = self._real_multiplier_to_q31_shift(real_mult)
                    out_mult_q31[oc] = np.int32(mult_q31)
                    out_shift[oc] = np.int8(sh)

                self.qparams[op.name] = dict(
                    kind="linear",
                    w_q=w_q,
                    bias_q=bias_q,
                    y_scale=float(y_scale),
                    out_mult_q31=out_mult_q31,
                    out_shift=out_shift,
                    in_f=int(weight_t.shape[1]),
                    out_f=int(weight_t.shape[0]),
                )
                current_x_scale = float(y_scale)

        self.final_out_scale = float(current_x_scale)

    def _np_to_c_array(self, arr: np.ndarray, c_type: str, name: str) -> str:
        flat = arr.flatten()
        items = ", ".join(str(int(x)) for x in flat)
        return f"static const {c_type} {name}[{flat.size}] = {{{items}}};\n"

    def _prepare_codegen_dirs(self):
        root = self.output_dir
        include_dir = os.path.join(root, "include")
        src_dir = os.path.join(root, "src")

        os.makedirs(include_dir, exist_ok=True)
        os.makedirs(src_dir, exist_ok=True)
        return root, include_dir, src_dir
    
    def generate_model_api_header(self, include_dir: str) -> str:
        # Safely get the input length from the tensor shape
        input_len = self.example_input.shape[-1] 
        output_classes = self.model(self.example_input).shape[-1]
        
        path = os.path.join(include_dir, "oemga_model.h")
        with open(path, "w", encoding="utf-8") as f:
            # Note the f""" and the double {{ }} for C syntax
            f.write(f"""#pragma once
    #include <stdint.h>

    #ifdef __cplusplus
    extern "C" {{
    #endif

    #define OEMGA_INPUT_LENGTH {input_len}
    #define OEMGA_INPUT_CHANNELS 1
    #define OEMGA_OUTPUT_CLASSES {output_classes}

    void oemga_forward_int8(const int8_t* input_q, int8_t* output_q);
    void oemga_forward_f32(const float* input_f, float* output_f, int input_len);

    #ifdef __cplusplus
    }}
    #endif
    """)
        return path

    def generate_weights_header(self, include_dir: str) -> str:
        if not self.qparams:
            self.build_codegen_qparams()

        path = os.path.join(include_dir, "oemga_weights.h")
        with open(path, "w", encoding="utf-8") as f:
            f.write("#pragma once\n#include <stdint.h>\n\n")
            f.write(f"static const float OEMGA_INPUT_SCALE = {self.input_scale:.10e}f;\n")
            f.write(f"static const float OEMGA_FINAL_OUT_SCALE = {self.final_out_scale:.10e}f;\n\n")

            for op_name, qp in self.qparams.items():
                f.write(self._np_to_c_array(qp["w_q"], "int8_t", f"{op_name}_weight_q"))
                f.write(self._np_to_c_array(qp["bias_q"], "int32_t", f"{op_name}_bias_q"))
                f.write(self._np_to_c_array(qp["out_mult_q31"], "int32_t", f"{op_name}_out_mult_q31"))
                f.write(self._np_to_c_array(qp["out_shift"], "int8_t", f"{op_name}_out_shift"))
                f.write(f"static const float {op_name}_y_scale = {qp['y_scale']:.10e}f;\n\n")
        return path

    def generate_backend_header(self, include_dir: str) -> str:
        dst = os.path.join(include_dir, "nn_layers.h")
        if not os.path.exists(self.nn_layers_path):
            raise FileNotFoundError(f"nn_layers.h not found at: {self.nn_layers_path}")
        shutil.copy(self.nn_layers_path, dst)
        return dst

    def generate_model_c(self, src_dir: str) -> str:
        if self.lowered_plan is None:
            self.run_step4()
        if not self.qparams:
            self.build_codegen_qparams()

        act_buf_bytes = self.lowered_plan.summary["activation_buffer_bytes_each"]
        scratch_bytes = self.lowered_plan.summary["scratch_buffer_bytes"]

        lines = []
        lines.append('#include <stdint.h>')
        lines.append('#include "oemga_model.h"')
        lines.append('#include "oemga_weights.h"')
        lines.append('#include "nn_layers.h"')
        lines.append("")
        lines.append(f"static int8_t buf0[{act_buf_bytes}] OEMGA_ALIGN16;")
        lines.append(f"static int8_t buf1[{act_buf_bytes}] OEMGA_ALIGN16;")
        if scratch_bytes > 0:
            lines.append(f"static int8_t scratch0[{scratch_bytes}] OEMGA_ALIGN16;")
        lines.append("")

        lines.append("void oemga_forward_int8(const int8_t* input_q, int8_t* output_q) {")
        lines.append("    for (int i = 0; i < OEMGA_INPUT_LENGTH * OEMGA_INPUT_CHANNELS; i++) buf0[i] = input_q[i];")
        lines.append("")

        tensor_shapes = {t.name: t.shape for t in self.ir_graph.tensors}

        for op in self.lowered_plan.ops:
            if op.backend_op == "conv1d_s8":
                qp = self.qparams[op.name]
                x_shape = tensor_shapes[op.inputs[0]]
                _, _, x_len = x_shape
                lines.append(f"    // {op.name}")
                lines.append(
                    f"    conv1d_s8("
                    f"{op.input_buffer}, "
                    f"{op.name}_weight_q, "
                    f"{op.name}_bias_q, "
                    f"{op.output_buffer}, "
                    f"{qp['in_ch']}, {qp['out_ch']}, {x_len}, {qp['kernel_size']}, {qp['padding']}, "
                    f"{op.name}_out_mult_q31, {op.name}_out_shift, "
                    f"scratch0);"
                )
            elif op.backend_op == "relu_s8":
                x_shape = tensor_shapes[op.inputs[0]]
                count = self._num_elements(x_shape)
                lines.append(f"    relu_s8({op.input_buffer}, {op.output_buffer}, {count});")
            elif op.backend_op == "maxpool1d_s8":
                x_shape = tensor_shapes[op.inputs[0]]
                _, ch, length = x_shape
                k = op.attrs["kernel_size"][0]
                s = op.attrs["stride"][0]
                lines.append(f"    maxpool1d_s8({op.input_buffer}, {op.output_buffer}, {ch}, {length}, {k}, {s});")
            elif op.backend_op == "reshape_view":
                lines.append(f"    // {op.name}: view only")
            elif op.backend_op == "linear_s8":
                qp = self.qparams[op.name]
                lines.append(
                    f"    linear_s8("
                    f"{op.input_buffer}, "
                    f"{op.name}_weight_q, "
                    f"{op.name}_bias_q, "
                    f"{op.output_buffer}, "
                    f"{qp['in_f']}, {qp['out_f']}, "
                    f"{op.name}_out_mult_q31, {op.name}_out_shift);"
                )
            else:
                raise OemgaSqueezeError(f"Unsupported backend op during codegen: {op.backend_op}")

        final_output_buf = self.lowered_plan.tensor_locations[self.lowered_plan.outputs[0]]["buffer"]
        lines.append("")
        lines.append("    for (int i = 0; i < OEMGA_OUTPUT_CLASSES; i++) output_q[i] = " + final_output_buf + "[i];")
        lines.append("}")
        lines.append("")

        lines.append("void oemga_forward_f32(const float* input_f, float* output_f, int input_len) {")
        lines.append("    for (int i = 0; i < input_len; i++) {")
        lines.append("        float scaled = input_f[i] / OEMGA_INPUT_SCALE;")
        lines.append("        int32_t q = (int32_t)(scaled >= 0 ? (scaled + 0.5f) : (scaled - 0.5f));")
        lines.append("        if (q > 127) q = 127;")
        lines.append("        if (q < -128) q = -128;")
        lines.append("        buf0[i] = (int8_t)q;")
        lines.append("    }")
        lines.append("    int8_t out_q[OEMGA_OUTPUT_CLASSES] = {0};")
        lines.append("    oemga_forward_int8(buf0, out_q);")
        lines.append("    dequant_s8_to_f32(out_q, output_f, OEMGA_OUTPUT_CLASSES, OEMGA_FINAL_OUT_SCALE);")
        lines.append("}")

        path = os.path.join(src_dir, "oemga_model.c")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        return path

    def generate_zephyr_main_c(self, src_dir: str) -> str:
        path = os.path.join(src_dir, "main.c")
        with open(path, "w", encoding="utf-8") as f:
            f.write(
                """#include <zephyr/kernel.h>
    #include <zephyr/sys/printk.h>
    #include "oemga_model.h"

    int main(void) {
        static float input_f[OEMGA_INPUT_LENGTH] = {0};
        static float output_f[OEMGA_OUTPUT_CLASSES] = {0};

        oemga_forward_f32(input_f, output_f, OEMGA_INPUT_LENGTH);

        printk("OEMGA output: ");
        for (int i = 0; i < OEMGA_OUTPUT_CLASSES; i++) {
            printk("%d:%f ", i, (double)output_f[i]);
        }
        printk("\\n");
        return 0;
    }
    """
            )
        return path
    
    def generate_zephyr_cmakelists(self, root: str) -> str:
        path = os.path.join(root, "CMakeLists.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(
                """cmake_minimum_required(VERSION 3.20.0)
    find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
    project(oemga_generated_model)

    target_include_directories(app PRIVATE include)

    target_sources(app PRIVATE
        src/main.c
        src/oemga_model.c
    )
    """
            )
        return path

    def generate_zephyr_prj_conf(self, root: str) -> str:
        path = os.path.join(root, "prj.conf")
        with open(path, "w", encoding="utf-8") as f:
            f.write(
                """CONFIG_MAIN_STACK_SIZE=4096
    CONFIG_PRINTK=y
    CONFIG_SERIAL=y
    CONFIG_CONSOLE=y
    """
            )
        return path

    def generate_manifest(self, root: str) -> str:
        emb = self.cost_report.embedded_estimates if self.cost_report else None
        data = {
            "model_name": self.model.__class__.__name__,
            "backend": self.lowered_plan.backend if self.lowered_plan else None,
            "input_scale": getattr(self, "input_scale", None),
            "final_out_scale": getattr(self, "final_out_scale", None),
            "estimated_flash_bytes": emb.estimated_model_data_flash_bytes if emb else None,
            "estimated_runtime_ram_bytes": emb.estimated_runtime_ram_bytes if emb else None,
        }
        path = os.path.join(root, "oemga_manifest.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return path
    
    def run_step5(self) -> CodegenArtifacts:
        print("Starting OEMGA-SQUEEZE step 5: codegen + Zephyr package...")

        if self.cost_report is None:
            self.run_step3()
        if self.lowered_plan is None:
            self.run_step4()

        root, include_dir, src_dir = self._prepare_codegen_dirs()
        self.build_codegen_qparams()

        generated = []
        generated.append(self.generate_model_api_header(include_dir))
        generated.append(self.generate_weights_header(include_dir))
        generated.append(self.generate_backend_header(include_dir))
        generated.append(self.generate_model_c(src_dir))
        generated.append(self.generate_zephyr_main_c(src_dir))
        generated.append(self.generate_zephyr_cmakelists(root))
        generated.append(self.generate_zephyr_prj_conf(root))
        generated.append(self.generate_manifest(root))

        self.codegen_artifacts = CodegenArtifacts(
            output_root=root,
            include_dir=include_dir,
            src_dir=src_dir,
            generated_files=generated,
        )

        print("\nGenerated files:")
        for p in generated:
            print(f"  - {p}")

        print(f"\n✅ Step 5 complete. Zephyr package generated in: {root}")
        return self.codegen_artifacts

    def generate_host_verify_runner(self, src_dir: str, include_dir: str) -> str:
        """
        Generates a tiny host C program that:
        - reads one sample from stdin as float32 text
        - runs oemga_forward_f32
        - prints output logits, one line
        """
        path = os.path.join(src_dir, "verify_runner.c")
        with open(path, "w", encoding="utf-8") as f:
            f.write(
                r'''#include <stdio.h>
    #include "oemga_model.h"

    int main(void) {
        float input_f[OEMGA_INPUT_LENGTH];
        float output_f[OEMGA_OUTPUT_CLASSES];

        for (int i = 0; i < OEMGA_INPUT_LENGTH; i++) {
            if (scanf("%f", &input_f[i]) != 1) {
                return 1;
            }
        }

        oemga_forward_f32(input_f, output_f, OEMGA_INPUT_LENGTH);

        for (int i = 0; i < OEMGA_OUTPUT_CLASSES; i++) {
            printf("%.9f", output_f[i]);
            if (i + 1 < OEMGA_OUTPUT_CLASSES) printf(" ");
        }
        printf("\n");
        return 0;
    }
    '''
            )
        return path

    def build_host_verifier(self) -> str:
        if self.codegen_artifacts is None:
            self.run_step5()

        root = self.codegen_artifacts.output_root
        include_dir = self.codegen_artifacts.include_dir
        src_dir = self.codegen_artifacts.src_dir

        runner_c = self.generate_host_verify_runner(src_dir, include_dir)
        exe_path = os.path.join(root, "verify_model_host")

        cmd = [
            "gcc",
            "-O3",
            "-std=c99",
            "-I", include_dir,
            os.path.join(src_dir, "oemga_model.c"),
            runner_c,
            "-o", exe_path,
        ]

        subprocess.run(cmd, check=True)
        return exe_path

    def run_c_model_on_sample(self, exe_path: str, x_sample: np.ndarray) -> np.ndarray:
        """
        x_sample shape expected:
        [1, 64] or [64]
        """
        x = np.asarray(x_sample, dtype=np.float32).reshape(-1)
        # Get expected length from the graph spec or example input
        expected_len = self.example_input.shape[-1]
        if x.size != expected_len:
            raise ValueError(f"Expected {expected_len} values, got {x.size}")

        stdin_text = " ".join(f"{float(v):.9f}" for v in x) + "\n"

        result = subprocess.run(
            [exe_path],
            input=stdin_text,
            text=True,
            capture_output=True,
            check=True,
        )

        vals = [float(v) for v in result.stdout.strip().split()]
        return np.asarray(vals, dtype=np.float32)

    def run_pytorch_on_dataset(self, X_test_tensor: torch.Tensor) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_test_tensor).detach().cpu().numpy().astype(np.float32)
        return logits

    def run_c_model_on_dataset(self, X_test_tensor: torch.Tensor, exe_path: str) -> np.ndarray:
        X_np = X_test_tensor.detach().cpu().numpy().astype(np.float32)

        outputs = []
        for i in range(X_np.shape[0]):
            x = X_np[i]
            # expected [1, 64] from [1,1,64]
            if x.ndim == 2:
                x = x.reshape(-1)
            elif x.ndim == 1:
                pass
            else:
                raise ValueError(f"Unexpected sample shape: {x.shape}")

            y = self.run_c_model_on_sample(exe_path, x)
            outputs.append(y)

        return np.stack(outputs, axis=0)
    
    def _argmax_preds(self, logits: np.ndarray) -> np.ndarray:
        return np.argmax(logits, axis=1).astype(np.int64)

    def _accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean(y_true == y_pred))


    def _confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
        cm = np.zeros((num_classes, num_classes), dtype=np.int64)
        for t, p in zip(y_true, y_pred):
            cm[int(t), int(p)] += 1
        return cm

    def _macro_f1(self, y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
        cm = self._confusion_matrix(y_true, y_pred, num_classes)
        f1s = []

        for c in range(num_classes):
            tp = cm[c, c]
            fp = cm[:, c].sum() - tp
            fn = cm[c, :].sum() - tp

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            if precision + recall == 0:
                f1 = 0.0
            else:
                f1 = 2.0 * precision * recall / (precision + recall)

            f1s.append(f1)

        return float(np.mean(f1s))

    def evaluate_c_vs_pytorch_f1(self, X_test_tensor: torch.Tensor, y_test) -> Dict[str, Any]:
        """
        Evaluates:
        - PyTorch model on dataset
        - Generated C model on dataset
        - Agreement and classification metrics

        Returns a report dict.
        """
        if self.codegen_artifacts is None:
            self.run_step5()

        y_true = np.asarray(y_test, dtype=np.int64).reshape(-1)
        num_classes = int(self.model(self.example_input).shape[-1])

        exe_path = self.build_host_verifier()

        pt_logits = self.run_pytorch_on_dataset(X_test_tensor)
        c_logits = self.run_c_model_on_dataset(X_test_tensor, exe_path)

        pt_pred = self._argmax_preds(pt_logits)
        c_pred = self._argmax_preds(c_logits)

        pt_acc = self._accuracy(y_true, pt_pred)
        c_acc = self._accuracy(y_true, c_pred)

        pt_f1 = self._macro_f1(y_true, pt_pred, num_classes)
        c_f1 = self._macro_f1(y_true, c_pred, num_classes)

        agreement = self._accuracy(pt_pred, c_pred)

        logit_abs_diff = np.abs(pt_logits - c_logits)
        max_logit_diff = float(np.max(logit_abs_diff)) if logit_abs_diff.size else 0.0
        mean_logit_diff = float(np.mean(logit_abs_diff)) if logit_abs_diff.size else 0.0

        report = {
            "num_samples": int(len(y_true)),
            "num_classes": num_classes,
            "pytorch": {
                "accuracy": pt_acc,
                "macro_f1": pt_f1,
                "confusion_matrix": self._confusion_matrix(y_true, pt_pred, num_classes).tolist(),
            },
            "c_model": {
                "accuracy": c_acc,
                "macro_f1": c_f1,
                "confusion_matrix": self._confusion_matrix(y_true, c_pred, num_classes).tolist(),
            },
            "comparison": {
                "prediction_agreement": agreement,
                "max_logit_abs_diff": max_logit_diff,
                "mean_logit_abs_diff": mean_logit_diff,
                "num_prediction_mismatches": int(np.sum(pt_pred != c_pred)),
            },
        }

        print("\n========== OEMGA VERIFY: PYTORCH VS C ==========")
        print(f"Samples                  : {report['num_samples']}")
        print(f"PyTorch accuracy         : {pt_acc:.4f}")
        print(f"PyTorch macro F1         : {pt_f1:.4f}")
        print(f"C model accuracy         : {c_acc:.4f}")
        print(f"C model macro F1         : {c_f1:.4f}")
        print(f"Prediction agreement     : {agreement:.4f}")
        print(f"Max logit abs diff       : {max_logit_diff:.6f}")
        print(f"Mean logit abs diff      : {mean_logit_diff:.6f}")
        print(f"Prediction mismatches    : {report['comparison']['num_prediction_mismatches']}")

        report_path = os.path.join(self.output_dir, "verify_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        print(f"Verification report saved to: {report_path}")
        return report

    def compile(self, backend: str = "oemga_native_int8") -> CodegenArtifacts:
        """
        Compiles the loaded PyTorch model into static Zephyr C code.
        This is the primary user-facing method, automatically handling the
        entire pipeline from tracing to codegen.
        """
        print(f"\n[OemgaSqueeze] Initiating compilation for backend: {backend}")
        
        # If the user requested a specific backend, ensure Step 4 uses it
        if self.lowered_plan is None or self.lowered_plan.backend != backend:
            self.run_step4(backend=backend)
            
        # run_step5 automatically triggers any missing prerequisites (Steps 1-3)
        return self.run_step5()

    def verify(self, X_test: torch.Tensor, y_test: Any) -> Dict[str, Any]:
        """
        Verifies the generated C model against the original PyTorch model
        using the provided test dataset. Returns a comprehensive metrics report.
        """
        print("\n[OemgaSqueeze] Initiating host-to-target C vs PyTorch verification...")
        
        # Ensure the model is actually compiled before trying to verify it
        if self.codegen_artifacts is None:
            print("Model not compiled yet. Auto-triggering compilation...")
            self.compile()
            
        return self.evaluate_c_vs_pytorch_f1(X_test, y_test)