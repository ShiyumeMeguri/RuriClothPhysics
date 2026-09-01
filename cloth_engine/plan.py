import contextlib

import warp as wp


LAUNCH_BLOCK_DIMENSION = 64

LAUNCH_BLOCK_DIMENSION_REASON = (
    "most launches in a frame are one element per particle or one per team, so the block "
    "dimension decides how many multiprocessors a launch reaches rather than how many "
    "warps hide latency inside one of them, and a launch narrower than one block runs on a "
    "single multiprocessor whatever the card has; the frozen asset was measured with "
    "stageprofile.py at a fixed shader clock, three runs each, and the frame spends "
    "2.7040 ms at 64, 2.7252 ms at 128 and 2.7956 ms at 256 with the device half at "
    "1.9400 / 1.9908 / 2.0200 ms, while 32 reads the same as 64 inside the run to run "
    "spread and 512 is refused by the module loader on this card; the curve is therefore "
    "flat below 64 and rising above it, and 64 is the wide end of that plateau; warp emits "
    "four bytes of shared memory per thread of the block dimension, so this also drops "
    "every kernel from 1024 to 256 bytes of shared memory, and it changes no register "
    "count, measured entry by entry with kernelregs.py")

GRAPH_CAPTURE_REASON = (
    "warp 1.16 captures on both compile targets and replays with the same call, on CUDA "
    "into a native graph and on the host into an API capture byte stream, and a captured "
    "spatial index refit replays on both, so the frame is recorded once and launched once "
    "and neither target carries a second execution path")

DESCRIPTOR_LAUNCH = "launch"

DESCRIPTOR_REFIT = "refit"

STREAM_COUNT = 4

STREAM_COUNT_REASON = (
    "the CUDA frame records the hazard DAG level by level and fans each level's independent "
    "entries across this many streams so the driver may overlap them, and the width is a "
    "measurement rather than a guess; on the frozen JsspSi asset the frame is 286 entries "
    "with a longest hazard chain of 235 and 16874 edges over three substeps and no self "
    "contacts, so it is nearly a straight line and its level widths fall to one after the "
    "first eleven levels, and the launch segment of perf_baseline.py with --sync-after-launch "
    "(real device execution, 48 frames, three discarded, p50, two alternating rounds) reads "
    "3.254 ms at one stream, 3.242 ms at two, 3.167 ms at four and 3.172 ms at eight; the "
    "fan out therefore only shaves about three percent because the critical path, not the "
    "stream count, sets the floor, and each wide level's fork and join event nodes cost about "
    "what its thin overlap of launch-latency-bound kernels returns; four is the stable bottom "
    "and does not lengthen the total frame, so it is the value carried forward, but the win "
    "this asset is waiting for is fusing the serial tail of tiny launches rather than widening "
    "the streams, which is what the entry level report is measured for; the host target "
    "ignores this width because it records one linear stream")

CAPTURE_LEVELS_REASON = (
    "a level is an entry's depth in the frame's hazard DAG, so entries that share a level "
    "have no hazard edge between them and are safe to fan across streams inside one fork and "
    "join; the levels are computed and proven pairwise conflict free by the dataflow gate "
    "before capture, and this refuses to record a device frame whose entries were never "
    "levelled rather than collapse the fan out silently onto one stream")


def launch_dimension(element_count):
    return (int(element_count),)


def _normalize_dimension(dimension):
    if isinstance(dimension, int):
        return (int(dimension),)
    return tuple(int(extent) for extent in dimension)


def value_identity(value):
    if isinstance(value, wp.array):
        return ("array", value.ptr, tuple(value.shape))
    structure = getattr(value, "_cls", None)
    if structure is not None:
        return ("structure", structure.key,
                tuple((member_name, value_identity(getattr(value, member_name)))
                      for member_name in structure.vars))
    return ("value", value)


class LaunchEntry:
    __slots__ = ("kernel", "dimension", "inputs", "arguments", "family_name")

    def __init__(self, kernel, dimension, inputs, family_name):
        self.kernel = kernel
        self.dimension = dimension
        self.inputs = inputs
        self.family_name = family_name
        self.arguments = tuple(value_identity(value) for value in inputs)

    def identity(self):
        return (self.kernel.module.name, self.kernel.key, self.dimension, self.arguments)

    def descriptor(self):
        return (DESCRIPTOR_LAUNCH, self.family_name)

    def execute(self, device, stream=None):
        wp.launch(self.kernel, dim=self.dimension, inputs=self.inputs, device=device,
                  stream=stream, block_dim=LAUNCH_BLOCK_DIMENSION)


class RefitEntry:
    __slots__ = ("state", "index_name", "identifier")

    def __init__(self, state, index_name):
        self.state = state
        self.index_name = index_name
        self.identifier = state.spatial_index_identifier(index_name)

    def identity(self):
        return ("refit_spatial_index", self.index_name, self.identifier)

    def descriptor(self):
        return (DESCRIPTOR_REFIT, self.index_name)

    def execute(self, device, stream=None):
        if stream is None:
            self.state.refit_spatial_index(self.index_name)
            return
        with wp.ScopedStream(stream):
            self.state.refit_spatial_index(self.index_name)


class Plan:
    def __init__(self):
        self.entries = []
        self.stages = []
        self.levels = None
        self.bound_state = None
        self.captured_state = None
        self.graph = None
        self.captured_structure_key = None
        self.graphs = {}
        self.graph_events = {}
        self.streams = []
        self.stream_count = STREAM_COUNT
        self.captures = 0

    def reset(self):
        self.entries = []
        self.stages = []
        self.levels = None
        self.bound_state = None

    def forget_graphs(self):
        self.graphs = {}
        self.graph_events = {}
        self.streams = []
        self.levels = None
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

    def record(self, kernel, dimension, inputs, family_name):
        if self.bound_state is not None:
            raise RuntimeError("plan is bound to a captured graph, call reset before recording")
        self.entries.append(LaunchEntry(kernel, _normalize_dimension(dimension), list(inputs),
                                        family_name))

    def record_spatial_index_refit(self, state, index_name):
        if self.bound_state is not None:
            raise RuntimeError("plan is bound to a captured graph, call reset before recording")
        self.entries.append(RefitEntry(state, index_name))

    def node_count(self):
        return len(self.entries)

    def stage_layout(self):
        return tuple(self.stages)

    def descriptors(self):
        return tuple(entry.descriptor() for entry in self.entries)

    def assign_levels(self, levels):
        materialized = tuple(int(level) for level in levels)
        assert len(materialized) == len(self.entries), \
            "a level is assigned for every recorded entry, got %d levels for %d entries" \
            % (len(materialized), len(self.entries))
        self.levels = materialized

    def structure_key(self, state):
        return (state.structure_key(), tuple(entry.identity() for entry in self.entries),
                (state.device.is_cuda, self.stream_count))

    def capture(self, state):
        requested_structure_key = self.structure_key(state)
        if self.captured_state is not None and state is not self.captured_state:
            self.forget_graphs()
        held = self.graphs.get(requested_structure_key)
        if held is not None:
            self.graph = held
            self.captured_structure_key = requested_structure_key
            self.captured_state = state
            self.bound_state = state
            return False
        if state.device.is_cuda:
            graph, events = self._capture_streamed(state)
        else:
            graph, events = self._capture_linear(state)
        self.graph = graph
        self.graphs[requested_structure_key] = graph
        self.graph_events[requested_structure_key] = events
        self.captures += 1
        self.captured_structure_key = requested_structure_key
        self.captured_state = state
        self.bound_state = state
        return True

    def _capture_linear(self, state):
        with wp.ScopedCapture(device=state.device) as capture:
            for entry in self.entries:
                entry.execute(state.device)
        return capture.graph, ()

    def _capture_streamed(self, state):
        assert self.levels is not None and len(self.levels) == len(self.entries), \
            "%s\n%d entries carry %r levels" \
            % (CAPTURE_LEVELS_REASON, len(self.entries),
               None if self.levels is None else len(self.levels))
        self._ensure_streams(state.device)
        main = wp.get_stream(state.device)
        grouped = {}
        for position, entry in enumerate(self.entries):
            grouped.setdefault(self.levels[position], []).append(entry)
        events = []
        with wp.ScopedCapture(stream=main) as capture:
            for level in sorted(grouped):
                self._record_level(state, main, grouped[level], events)
        return capture.graph, tuple(events)

    def _ensure_streams(self, device):
        needed = self.stream_count - 1
        if len(self.streams) != needed:
            self.streams = [wp.Stream(device) for _ in range(needed)]

    def _record_level(self, state, main, level_entries, events):
        assignment = tuple((position % self.stream_count, entry)
                           for position, entry in enumerate(level_entries))
        used = sorted({slot for slot, _entry in assignment if slot != 0})
        if used:
            fork = main.record_event()
            events.append(fork)
            for slot in used:
                self.streams[slot - 1].wait_event(fork)
        for slot, entry in assignment:
            if slot == 0:
                entry.execute(state.device)
            else:
                entry.execute(state.device, self.streams[slot - 1])
        for slot in used:
            joined = self.streams[slot - 1].record_event()
            events.append(joined)
            main.wait_event(joined)

    def launch(self):
        if self.bound_state is None:
            raise RuntimeError("plan is not bound to a captured graph, call capture first; %s"
                               % GRAPH_CAPTURE_REASON)
        wp.capture_launch(self.graph)
