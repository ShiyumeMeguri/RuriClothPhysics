"""Struct <-> SoA staging on the device.

The per-frame team round-trip (upload the whole host struct, read the whole struct back)
must be a *byte copy of the whole table* -- one H2D / one D2H -- not 167 separate field
transfers (measured 20 ms vs 0.18 ms; the cost is per-transfer launch overhead, not
bandwidth). But the frame megakernel consumes the team as struct-of-arrays (one device
array per field). So we keep a single resident structured staging array, transfer it as
one blob, and bridge to/from the SoA arrays with two data-movement kernels:

* ``explode`` : structured stage -> SoA arrays (after an H2D of the host struct)
* ``implode`` : SoA arrays -> structured stage (before a D2H back to the host)

Both kernels are generated once from the numpy struct dtype (single source of truth --
no hand-listed field table to drift), so a dtype change regenerates them automatically.
They carry no simulation math; they are pure element copies (bool<->uint8 for the device
form). The SoA argument order is exactly ``struct_dtype.names`` (the FieldSet key order).
"""

import numpy as np
from numba import cuda


def _field_copy_lines(struct_dtype, indent, stage_expr, soa_prefix, to_soa):
    """Emit per-field copy statements between the structured ``stage`` row and the SoA
    arrays ``soa_prefix<k>``. ``to_soa`` True = stage->SoA (explode), False = SoA->stage."""
    lines = []
    pad = " " * indent
    for index, name in enumerate(struct_dtype.names):
        sub = struct_dtype[name]
        shape = sub.shape
        is_bool = sub.base == np.bool_
        soa = "%s%d" % (soa_prefix, index)
        field = "%s['%s']" % (stage_expr, name)
        if not shape:
            if to_soa:
                if is_bool:
                    lines.append("%s%s[i] = np.uint8(1) if %s else np.uint8(0)" % (pad, soa, field))
                else:
                    lines.append("%s%s[i] = %s" % (pad, soa, field))
            else:
                if is_bool:
                    lines.append("%s%s = %s[i] != 0" % (pad, field, soa))
                else:
                    lines.append("%s%s = %s[i]" % (pad, field, soa))
        elif len(shape) == 1:
            lines.append("%sfor j0 in range(%d):" % (pad, shape[0]))
            if to_soa:
                lines.append("%s    %s[i, j0] = %s[j0]" % (pad, soa, field))
            else:
                lines.append("%s    %s[j0] = %s[i, j0]" % (pad, field, soa))
        elif len(shape) == 2:
            lines.append("%sfor j0 in range(%d):" % (pad, shape[0]))
            lines.append("%s    for j1 in range(%d):" % (pad, shape[1]))
            if to_soa:
                lines.append("%s        %s[i, j0, j1] = %s[j0][j1]" % (pad, soa, field))
            else:
                lines.append("%s        %s[j0][j1] = %s[i, j0, j1]" % (pad, field, soa))
        else:
            raise ValueError("unsupported field rank for %s: %r" % (name, shape))
    return lines


def _generate(struct_dtype, name, to_soa):
    args = ", ".join("a%d" % k for k in range(len(struct_dtype.names)))
    header = ["def %s(stage, %s):" % (name, args),
              "    i = cuda.grid(1)",
              "    stride = cuda.gridsize(1)",
              "    while i < stage.shape[0]:"]
    body = _field_copy_lines(struct_dtype, 8, "stage[i]", "a", to_soa)
    footer = ["        i += stride"]
    source = "\n".join(header + body + footer)
    namespace = {"cuda": cuda, "np": np}
    exec(compile(source, "<staging.%s>" % name, "exec"), namespace)
    # cache=False: numba has no disk-cache locator for an exec'd function. These are tiny
    # copy-only kernels (no math), so the one-off per-process compile is a few seconds --
    # negligible next to the megakernel, and first-run-slow is acceptable.
    return cuda.jit(cache=False)(namespace[name])


def _aligned_like(struct_dtype):
    """Naturally-aligned twin of a (possibly packed) struct dtype. numpy's default team
    dtype is packed -- int32 fields land at offsets like 14/18 -- and numba refuses
    misaligned record-field access on the device. The aligned twin pads each field to its
    natural boundary; ``astype`` converts host packed<->aligned by field name (bit-exact,
    same values) so the single blob transfer moves a device-legal layout."""
    return np.dtype([(name, struct_dtype[name]) for name in struct_dtype.names], align=True)


# Bridge kernels are compiled once per aligned dtype and shared across every engine /
# staging instance (a fresh compile per GpuEngine would be ruinous -- the dev-harness
# builds hundreds of engines per run). Keyed by the aligned dtype.
_KERNEL_CACHE = {}


def _bridge_kernels(aligned_dtype):
    cached = _KERNEL_CACHE.get(aligned_dtype)
    if cached is None:
        cached = (_generate(aligned_dtype, "explode", to_soa=True),
                  _generate(aligned_dtype, "implode", to_soa=False))
        _KERNEL_CACHE[aligned_dtype] = cached
    return cached


class StructStaging:
    """Resident aligned structured staging array for one struct arena + its generated
    bridge kernels. ``field_order`` is the SoA argument order.

    ``fields=None`` stages the whole struct dtype (the full-table round-trip). A field
    subset stages only those columns -- the narrow per-frame-input / consumable-flag /
    config / output bridges that carry far fewer marshalled array arguments (the measured
    cost of a bridge launch is per-argument, not per-byte: a 145-arg bridge costs the same
    for 2 teams as for 24). A subset repacks source->stage per field, so the source may be a
    structured ndarray (``world.team``) OR a ChunkArena's ``arrays`` dict (colliders /
    particles) -- both support ``source[name]`` -> array."""

    def __init__(self, struct_dtype, count, fields=None):
        self.subset = fields is not None
        self.field_order = list(fields) if self.subset else list(struct_dtype.names)
        sub_dtype = np.dtype([(name, struct_dtype[name]) for name in self.field_order])
        self.dtype = sub_dtype
        self.aligned_dtype = _aligned_like(sub_dtype)
        self.count = max(int(count), 1)
        self.stage = cuda.device_array(self.count, self.aligned_dtype)
        # Page-locked (pinned) host mirror: makes copy_to_device / copy_to_host asynchronous on a
        # stream (pageable memory forces a blocking sync per transfer), so the slim per-frame IO can
        # queue all H2D/kernel/D2H ops on one stream and pay a single synchronize instead of ~9.
        self._host = cuda.pinned_array(self.count, dtype=self.aligned_dtype)
        self._host[:] = 0
        self._explode, self._implode = _bridge_kernels(self.aligned_dtype)

    def _repack_in(self, source):
        if self.subset:
            for name in self.field_order:
                self._host[name] = source[name][:self.count]
        else:
            self._host[:] = source[:self.count]

    def _repack_out(self, target):
        if self.subset:
            for name in self.field_order:
                target[name][:self.count] = self._host[name]
        else:
            target[:self.count] = self._host

    def upload(self, source, fieldset, blocks, threads):
        """Repack the (subset of) columns to the aligned layout, one H2D, then explode into
        the resident SoA arrays. ``source`` is a struct ndarray or an arena arrays dict."""
        self._repack_in(source)
        self.stage.copy_to_device(self._host)
        soa = [fieldset.device[name] for name in self.field_order]
        self._explode[blocks, threads](self.stage, *soa)

    def download(self, target, fieldset, blocks, threads):
        """Implode the resident SoA arrays into the stage, one D2H, then repack the aligned
        layout back into ``target`` (struct ndarray or arena arrays dict)."""
        soa = [fieldset.device[name] for name in self.field_order]
        self._implode[blocks, threads](self.stage, *soa)
        self.stage.copy_to_host(self._host)
        self._repack_out(target)

    # ---- async single-stream variants (slim per-frame IO) -------------------
    def upload_async(self, source, fieldset, blocks, threads, stream):
        """Same as ``upload`` but queued on ``stream`` (async H2D from the pinned host + explode);
        no synchronize -- the caller fences the whole frame once."""
        self._repack_in(source)
        self.stage.copy_to_device(self._host, stream=stream)
        soa = [fieldset.device[name] for name in self.field_order]
        self._explode[blocks, threads, stream](self.stage, *soa)

    def download_issue(self, fieldset, blocks, threads, stream):
        """Queue implode + async D2H on ``stream`` (no synchronize). Pair with ``download_finish``
        after the caller's single stream fence to repack the pinned host into ``target``."""
        soa = [fieldset.device[name] for name in self.field_order]
        self._implode[blocks, threads, stream](self.stage, *soa)
        self.stage.copy_to_host(self._host, stream=stream)

    def download_finish(self, target):
        """CPU repack of the pinned host mirror into ``target`` (call only after the stream fence)."""
        self._repack_out(target)
