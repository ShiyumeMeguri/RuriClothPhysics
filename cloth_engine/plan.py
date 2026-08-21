import contextlib

import warp as wp

from . import state as _state


LAUNCH_BLOCK_DIMENSION = 256


def launch_dimension(element_count):
    return (int(element_count),)


def _normalize_dimension(dimension):
    if isinstance(dimension, int):
        return (int(dimension),)
    return tuple(int(extent) for extent in dimension)


class LaunchEntry:
    __slots__ = ("kernel", "dimension", "inputs", "constants")

    def __init__(self, kernel, dimension, inputs):
        self.kernel = kernel
        self.dimension = dimension
        self.inputs = inputs
        self.constants = tuple(value for value in inputs if not isinstance(value, wp.array))

    def identity(self):
        return (self.kernel.module.name, self.kernel.key, self.dimension, self.constants)


class Plan:
    def __init__(self):
        self.entries = []
        self.stages = []
        self.bound_state = None
        self.captured_state = None
        self.graph = None
        self.captured_structure_key = None
        self.graphs = {}
        self.captures = 0

    def reset(self):
        self.entries = []
        self.stages = []
        self.bound_state = None

    def forget_graphs(self):
        self.graphs = {}
        self.graph = None
        self.captured_state = None
        self.captured_structure_key = None
        self.bound_state = None

    def graph_count(self):
        return len(self.graphs)

    def capture_count(self):
        return self.captures

    @contextlib.contextmanager
    def stage(self, stage_name):
        first_entry = len(self.entries)
        yield self
        self.stages.append((stage_name, len(self.entries) - first_entry))

    def record(self, kernel, dimension, inputs):
        if self.bound_state is not None:
            raise RuntimeError("plan is bound to a captured graph, call reset before recording")
        self.entries.append(LaunchEntry(kernel, _normalize_dimension(dimension), list(inputs)))

    def node_count(self):
        return len(self.entries)

    def stage_layout(self):
        return tuple(self.stages)

    def structure_key(self, state):
        return (state.structure_key(), tuple(entry.identity() for entry in self.entries))

    def capture(self, state):
        requested_structure_key = self.structure_key(state)
        if state is not self.captured_state:
            self.forget_graphs()
        held = self.graphs.get(requested_structure_key)
        if held is not None:
            self.graph = held
            self.captured_structure_key = requested_structure_key
            self.captured_state = state
            self.bound_state = state
            return False
        with wp.ScopedCapture(device=_state.DEVICE) as capture:
            for entry in self.entries:
                wp.launch(entry.kernel, dim=entry.dimension, inputs=entry.inputs,
                          device=_state.DEVICE, block_dim=LAUNCH_BLOCK_DIMENSION)
        self.graph = capture.graph
        self.graphs[requested_structure_key] = capture.graph
        self.captures += 1
        self.captured_structure_key = requested_structure_key
        self.captured_state = state
        self.bound_state = state
        return True

    def launch(self):
        if self.bound_state is None:
            raise RuntimeError("plan is not bound to a captured graph, call capture first")
        wp.capture_launch(self.graph)
