COMPILE_TARGET_SPECIFICATION = (
    ("CUDA", "cuda:0", "CUDA (kernel 源编译到显卡)",
     "生产路径: 一份 Warp kernel 源编译成 PTX, 整帧一张图一次重放"),
    ("CPU", "cpu", "CPU (同一份 kernel 源编译到主机)",
     "同一份 Warp kernel 源编译成主机代码, 没有显卡时跑, 慢一到两个数量级"),
)

TARGET_NAMES = tuple(row[0] for row in COMPILE_TARGET_SPECIFICATION)

DEFAULT_TARGET = TARGET_NAMES[0]

TARGET_DEVICES = {row[0]: row[1] for row in COMPILE_TARGET_SPECIFICATION}

TARGET_ITEMS = tuple((row[0], row[2], row[3]) for row in COMPILE_TARGET_SPECIFICATION)


def _validate_compile_target_specification():
    seen_targets = set()
    seen_devices = set()
    for row in COMPILE_TARGET_SPECIFICATION:
        assert len(row) == 4, \
            "a compile target row declares the name, the warp device, the label and the " \
            "description, got %r" % (row,)
        target_name, device_name, label, description = row
        assert target_name not in seen_targets, \
            "the compile target %s is declared twice" % target_name
        assert device_name not in seen_devices, \
            "the warp device %r carries two compile target names, so the name would no " \
            "longer identify what is being compiled" % device_name
        seen_targets.add(target_name)
        seen_devices.add(device_name)
        assert label and description, \
            "the compile target %s carries no label or no description and the host shows " \
            "both to the user" % target_name


_validate_compile_target_specification()


def device_of(target_name):
    assert target_name in TARGET_DEVICES, \
        "there is no compile target named %r, the declared targets are %r" \
        % (target_name, list(TARGET_NAMES))
    return TARGET_DEVICES[target_name]
