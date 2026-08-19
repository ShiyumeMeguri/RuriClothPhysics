import numpy as np
from numba import cuda


def _field_copy_lines(struct_dtype, indent, stage_expr, soa_prefix, to_soa):
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
    return cuda.jit(cache=False)(namespace[name])


def _aligned_like(struct_dtype):
    return np.dtype([(name, struct_dtype[name]) for name in struct_dtype.names], align=True)


_KERNEL_CACHE = {}


def _bridge_kernels(aligned_dtype):
    cached = _KERNEL_CACHE.get(aligned_dtype)
    if cached is None:
        cached = (_generate(aligned_dtype, "explode", to_soa=True),
                  _generate(aligned_dtype, "implode", to_soa=False))
        _KERNEL_CACHE[aligned_dtype] = cached
    return cached


class StructStaging:

    def __init__(self, struct_dtype, count, fields=None):
        self.subset = fields is not None
        self.field_order = list(fields) if self.subset else list(struct_dtype.names)
        sub_dtype = np.dtype([(name, struct_dtype[name]) for name in self.field_order])
        self.dtype = sub_dtype
        self.aligned_dtype = _aligned_like(sub_dtype)
        self.count = max(int(count), 1)
        self.stage = cuda.device_array(self.count, self.aligned_dtype)
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
        self._repack_in(source)
        self.stage.copy_to_device(self._host)
        soa = [fieldset.device[name] for name in self.field_order]
        self._explode[blocks, threads](self.stage, *soa)

    def download(self, target, fieldset, blocks, threads):
        soa = [fieldset.device[name] for name in self.field_order]
        self._implode[blocks, threads](self.stage, *soa)
        self.stage.copy_to_host(self._host)
        self._repack_out(target)

    def upload_async(self, source, fieldset, blocks, threads, stream):
        self._repack_in(source)
        self.stage.copy_to_device(self._host, stream=stream)
        soa = [fieldset.device[name] for name in self.field_order]
        self._explode[blocks, threads, stream](self.stage, *soa)

    def download_issue(self, fieldset, blocks, threads, stream):
        soa = [fieldset.device[name] for name in self.field_order]
        self._implode[blocks, threads, stream](self.stage, *soa)
        self.stage.copy_to_host(self._host, stream=stream)

    def download_finish(self, target):
        self._repack_out(target)
