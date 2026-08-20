import contextlib

import warp as wp

from . import state as _state


def _normalize_dimension(dimension):
    if isinstance(dimension, int):
        return (int(dimension),)
    return tuple(int(extent) for extent in dimension)


class LaunchEntry:
    __slots__ = ("kernel", "dimension", "inputs")

    def __init__(self, kernel, dimension, inputs):
        self.kernel = kernel
        self.dimension = dimension
        self.inputs = inputs

    def identity(self):
        return (self.kernel.module.name, self.kernel.key, self.dimension)


class Plan:
    def __init__(self):
        self.entries = []
        self.stages = []
        self.bound_state = None
        self.captured_state = None
        self.graph = None
        self.captured_structure_key = None

    def reset(self):
        self.entries = []
        self.stages = []
        self.bound_state = None

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
        if (self.graph is not None
                and state is self.captured_state
                and requested_structure_key == self.captured_structure_key):
            self.bound_state = state
            return False
        with wp.ScopedCapture(device=_state.DEVICE) as capture:
            for entry in self.entries:
                wp.launch(entry.kernel, dim=entry.dimension, inputs=entry.inputs,
                          device=_state.DEVICE)
        self.graph = capture.graph
        self.captured_structure_key = requested_structure_key
        self.captured_state = state
        self.bound_state = state
        return True

    def launch(self):
        if self.bound_state is None:
            raise RuntimeError("plan is not bound to a captured graph, call capture first")
        wp.capture_launch(self.graph)
