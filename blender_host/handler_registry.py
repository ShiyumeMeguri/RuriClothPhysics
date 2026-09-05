"""处理器的装卸只此一处。

按对象身份匹配挡不住模块重载: 重载之后函数对象换了新的一份, 旧那份还留在
bpy.app.handlers 里 —— register 的去重看不见它, 于是每重载一次就多叠一份;
unregister 也删不掉它, 于是停用插件之后模拟照样每帧在跑。
只按 __name__ 匹配又会误伤: runtime 和 selection_sync 里都有 _on_depsgraph_update_post,
卸载其中一个会把另一个一起删掉。
所以判据是 (模块, 限定名), 跨重载稳定又不会撞名。

绘制处理器的句柄同理: 模块级全局撑不过重载, 旧句柄就再也拆不掉。句柄存在一个不受
插件重载影响的 sys.modules 条目里。
"""

import sys
import types

_STATE = "_ruri_cloth_physics_registry"


def _identity(function):
    return "%s.%s" % (getattr(function, "__module__", ""),
                      getattr(function, "__qualname__",
                              getattr(function, "__name__", "")))


def _draw_handles():
    holder = sys.modules.get(_STATE)
    if holder is None:
        holder = types.ModuleType(_STATE)
        holder.draw_handles = {}
        sys.modules[_STATE] = holder
    return holder.draw_handles


def install(handlers, function):
    uninstall(handlers, function)
    handlers.append(function)


def uninstall(handlers, function):
    identity = _identity(function)
    for existing in list(handlers):
        if _identity(existing) == identity:
            handlers.remove(existing)


def install_draw(name, add, remove):
    uninstall_draw(name)
    _draw_handles()[name] = (add(), remove)


def uninstall_draw(name):
    entry = _draw_handles().pop(name, None)
    if entry is None:
        return
    handle, remove = entry
    remove(handle)
