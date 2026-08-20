import numpy as np

INDEX_TYPE = np.int32
SCALAR_TYPE = np.float32

MAX_LEVELS = 32
PRIMITIVE_NONE = -1


class BoundingVolumeHierarchy:

    __slots__ = ("aabb_min", "aabb_max", "escape", "primitive", "leaf_nodes",
                 "leaf_primitive", "refit_order", "refit_level_offsets", "level_count")

    def __init__(self, aabb_min, aabb_max, escape, primitive, leaf_nodes, leaf_primitive,
                 refit_order, refit_level_offsets, level_count):
        self.aabb_min = aabb_min
        self.aabb_max = aabb_max
        self.escape = escape
        self.primitive = primitive
        self.leaf_nodes = leaf_nodes
        self.leaf_primitive = leaf_primitive
        self.refit_order = refit_order
        self.refit_level_offsets = refit_level_offsets
        self.level_count = level_count

    @property
    def node_count(self):
        return int(self.escape.shape[0])

    @property
    def primitive_count(self):
        return int(self.leaf_nodes.shape[0])


def _longest_axis(centroid, order, start, stop):
    block = centroid[order[start:stop]]
    extent = block.max(axis=0) - block.min(axis=0)
    return int(np.argmax(extent))


def _emit(state, order, start, stop, depth, centroid):
    index = len(state["primitive"])
    state["primitive"].append(PRIMITIVE_NONE)
    state["escape"].append(PRIMITIVE_NONE)
    state["depth"].append(depth)
    if stop - start == 1:
        state["primitive"][index] = int(order[start])
        state["escape"][index] = index + 1
        return
    axis = _longest_axis(centroid, order, start, stop)
    segment = order[start:stop]
    order[start:stop] = segment[np.argsort(centroid[segment, axis], kind="stable")]
    middle = start + (stop - start) // 2
    _emit(state, order, start, middle, depth + 1, centroid)
    _emit(state, order, middle, stop, depth + 1, centroid)
    state["escape"][index] = len(state["primitive"])


def build(aabb_min, aabb_max):
    aabb_min = np.ascontiguousarray(aabb_min, dtype=SCALAR_TYPE)
    aabb_max = np.ascontiguousarray(aabb_max, dtype=SCALAR_TYPE)
    assert aabb_min.shape == aabb_max.shape and aabb_min.ndim == 2 and aabb_min.shape[1] == 3
    count = int(aabb_min.shape[0])
    assert count > 0, "bounding volume hierarchy needs at least one primitive"

    centroid = (aabb_min.astype(np.float64) + aabb_max.astype(np.float64)) * 0.5
    order = np.arange(count, dtype=np.int64)
    state = {"primitive": [], "escape": [], "depth": []}
    _emit(state, order, 0, count, 0, centroid)

    primitive = np.array(state["primitive"], dtype=INDEX_TYPE)
    escape = np.array(state["escape"], dtype=INDEX_TYPE)
    depth = np.array(state["depth"], dtype=np.int64)
    level_count = int(depth.max()) + 1
    assert level_count <= MAX_LEVELS, \
        "bounding volume hierarchy needs %d levels, limit is %d" % (level_count, MAX_LEVELS)

    node_count = primitive.shape[0]
    assert node_count == 2 * count - 1, "unexpected node count %d for %d primitives" \
        % (node_count, count)

    leaf_nodes = np.flatnonzero(primitive >= 0).astype(INDEX_TYPE)
    leaf_primitive = primitive[leaf_nodes]
    internal = np.flatnonzero(primitive < 0)
    refit_order = internal[np.argsort(-depth[internal], kind="stable")].astype(INDEX_TYPE)
    internal_depth = depth[refit_order]
    boundary = np.flatnonzero(np.diff(internal_depth) != 0) + 1
    refit_level_offsets = np.concatenate(
        [[0], boundary, [refit_order.shape[0]]]).astype(INDEX_TYPE)

    tree = BoundingVolumeHierarchy(
        np.zeros((node_count, 3), dtype=SCALAR_TYPE),
        np.zeros((node_count, 3), dtype=SCALAR_TYPE),
        escape, primitive, leaf_nodes, leaf_primitive,
        refit_order, refit_level_offsets, level_count)
    refit(tree, aabb_min, aabb_max)
    return tree


def refit(tree, primitive_min, primitive_max):
    tree.aabb_min[tree.leaf_nodes] = primitive_min[tree.leaf_primitive]
    tree.aabb_max[tree.leaf_nodes] = primitive_max[tree.leaf_primitive]
    offsets = tree.refit_level_offsets
    for level in range(offsets.shape[0] - 1):
        nodes = tree.refit_order[offsets[level]:offsets[level + 1]]
        if nodes.shape[0] == 0:
            continue
        left = nodes + 1
        right = tree.escape[left]
        tree.aabb_min[nodes] = np.minimum(tree.aabb_min[left], tree.aabb_min[right])
        tree.aabb_max[nodes] = np.maximum(tree.aabb_max[left], tree.aabb_max[right])


def query(tree, query_min, query_max):
    query_min = np.ascontiguousarray(query_min, dtype=SCALAR_TYPE)
    query_max = np.ascontiguousarray(query_max, dtype=SCALAR_TYPE)
    count = int(query_min.shape[0])
    if count == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    node_count = tree.node_count
    cursor = np.zeros(count, dtype=np.int64)
    active = np.arange(count, dtype=np.int64)
    query_parts = []
    primitive_parts = []
    while active.shape[0]:
        node = cursor[active]
        hit = ((query_min[active] <= tree.aabb_max[node])
               & (query_max[active] >= tree.aabb_min[node])).all(axis=1)
        primitive = tree.primitive[node]
        emitted = hit & (primitive >= 0)
        if emitted.any():
            query_parts.append(active[emitted])
            primitive_parts.append(primitive[emitted].astype(np.int64))
        advance = np.where(hit, node + 1, tree.escape[node].astype(np.int64))
        cursor[active] = advance
        active = active[advance < node_count]
    if not query_parts:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    query_index = np.concatenate(query_parts)
    primitive_index = np.concatenate(primitive_parts)
    ordering = np.lexsort((primitive_index, query_index))
    return query_index[ordering], primitive_index[ordering]


def subtree_primitives(tree, node):
    stop = int(tree.escape[node])
    span = tree.primitive[node:stop]
    return np.sort(span[span >= 0].astype(np.int64))
