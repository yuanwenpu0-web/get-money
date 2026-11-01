"""静态守卫测试：禁止对布尔标量使用按位取反 ``~``。

背景
----
Pine → Python 移植里有一类极难发现的缺陷。开关式门常被写成

    flags((~use_gate_a | liquidity_sweep_long) & ...)

其中 ``use_gate_a`` 是 Python 的 ``bool``。``~True`` 不是 ``False`` 而是 **``-2``**
（按位取反），再与布尔 Series 做 ``|`` 会得到**全 True** —— 于是「开启门」这个
语义被静默吃掉，**门永远不生效**。

实测：:

    >>> s = pd.Series([True, False, True])
    >>> (~True) | s
    0    True
    1    True
    2    True    # 期望 [True, False, True]，实际全 True

这类 bug 的三个特征让它特别危险：

1. **输出看起来完全正常** —— 只是信号多一点，没人会怀疑；
2. **既有测试抓不住** —— 测试往往把门关掉跑，两条路径输出一样也能通过；
3. **人眼 review 抓不住** —— ``~x`` 在布尔上下文里读起来毫无违和感。

本项目在批量移植期间一共踩了 5 次（klinger / market_state_engine /
reversal_engine_score_v1 / market_pressure_scale 等），因此把静态扫描固化成测试。

判定规则
--------
一个名字被认为是「布尔标量」当且仅当它满足以下任一条件：

* 函数签名里标注为 ``bool`` 或 ``bool | None`` / ``Optional[bool]`` 的参数；
* 由 ``boolean(...)`` 或 ``bool(...)`` 调用赋值；
* 由 ``True`` / ``False`` 字面量赋值。

对这类名字施加 ``~`` 即判违规。``~some_series``（操作数是调用或属性）不受影响，
``~np.isnan(...)`` 之类的用法完全合法。
"""
from __future__ import annotations

import ast
import pathlib

import pytest

# 被扫描的包目录：整个 app 包（Pine 移植模块、策略、回测、行情管线一视同仁）。
# 这个陷阱不是移植代码独有的 —— 任何「开关式门 + 布尔标量」的写法都会中招。
APP_DIR = pathlib.Path(__file__).resolve().parents[1] / "app"
SCAN_DIRS = (APP_DIR,)


def _is_bool_annotation(node: ast.AST) -> bool:
    """判断类型注解是否是 ``bool`` / ``bool | None`` / ``Optional[bool]``。"""
    if isinstance(node, ast.Name):
        return node.id == "bool"
    # ``bool | None``（PEP 604）—— BinOp(BitOr)，只要有一侧是 bool 就按布尔标量处理
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return _is_bool_annotation(node.left) or _is_bool_annotation(node.right)
    # Optional[bool] / Union[bool, None]
    if isinstance(node, ast.Subscript):
        base = node.value
        base_name = base.id if isinstance(base, ast.Name) else getattr(base, "attr", None)
        if base_name in {"Optional", "Union"}:
            return _is_bool_annotation(node.slice)
    return False


def _bool_scalar_names(tree: ast.AST) -> set[str]:
    """收集模块里所有「布尔标量」的名字。"""
    names: set[str] = set()

    # 1) 函数签名中标注为布尔的参数（含 positional-only / 普通 / keyword-only）
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = node.args
            candidates = [*args.posonlyargs, *args.args, *args.kwonlyargs]
            for arg in candidates:
                if arg.annotation is not None and _is_bool_annotation(arg.annotation):
                    names.add(arg.arg)

    # 2) 由 boolean(...) / bool(...) / True|False 赋值得到的局部名
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        value = node.value
        is_bool_call = isinstance(value, ast.Call) and _callee_name(value) in {"boolean", "bool"}
        is_bool_literal = isinstance(value, ast.Constant) and isinstance(value.value, bool)
        if is_bool_call or is_bool_literal:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
    return names


def _callee_name(call: ast.Call) -> str | None:
    """取被调用函数的名字（支持 ``boolean(...)`` 与 ``_pine.boolean(...)``）。"""
    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _find_violations(tree: ast.AST, bool_names: set[str]) -> list[tuple[int, str]]:
    """找出所有对布尔标量施加 ``~`` 的位置，返回 ``(行号, 变量名)``。"""
    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Invert):
            operand = node.operand
            if isinstance(operand, ast.Name) and operand.id in bool_names:
                hits.append((node.lineno, operand.id))
    return hits


def _scan() -> list[str]:
    violations: list[str] = []
    for directory in SCAN_DIRS:
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*.py")):
            try:
                source = path.read_text(encoding="utf-8")
                tree = ast.parse(source, filename=str(path))
            except (SyntaxError, UnicodeDecodeError):  # 非 Python / 损坏文件跳过
                continue
            names = _bool_scalar_names(tree)
            for lineno, name in _find_violations(tree, names):
                # 报告相对 app 目录的路径，便于定位
                rel = path.relative_to(APP_DIR).as_posix()
                violations.append(f"app/{rel}:{lineno}  ~{name}")
    return violations


def test_no_bitwise_negation_of_boolean_scalars():
    """扫描全部移植模块：布尔标量一律不得使用 ``~``。"""
    violations = _scan()
    assert not violations, (
        "检测到对布尔标量使用按位取反 ~。~True == -2（不是 False），"
        "与布尔 Series 取 | 会得到全 True，导致门/开关被静默吃掉。\n"
        "请改用 `not`、显式 if 分支，或把标量广播成 Series 后再取反。\n"
        + "\n".join(violations)
    )


# --- 检测器自检：确保守卫本身没有失灵 ------------------------------------------

_BAD_SNIPPET = '''
def f(frame, *, use_gate_a: bool = True, use_gate_b: bool = False):
    use_gate_a = boolean("use_gate_a", use_gate_a)
    constant = True
    return flags((~use_gate_a | cond) & (~use_gate_b | other) & ~constant)
'''

_GOOD_SNIPPET = '''
def g(frame, *, use_gate: bool = True):
    use_gate = boolean("use_gate", use_gate)
    return flags(cond) & ~np.isnan(arr) & ~series.shift(1)
'''


def test_detector_flags_the_known_bad_pattern():
    """自检：三种布尔标量来源（注解参数 / boolean() / 字面量）都要被抓到。"""
    tree = ast.parse(_BAD_SNIPPET)
    names = _bool_scalar_names(tree)
    assert names == {"use_gate_a", "use_gate_b", "constant"}
    caught = {name for _, name in _find_violations(tree, names)}
    assert caught == {"use_gate_a", "use_gate_b", "constant"}


def test_detector_leaves_series_negation_alone():
    """自检：``~np.isnan(...)``、``~series.shift(...)`` 这类合法用法不得误报。"""
    tree = ast.parse(_GOOD_SNIPPET)
    names = _bool_scalar_names(tree)
    assert names == {"use_gate"}
    assert _find_violations(tree, names) == []


@pytest.mark.parametrize(
    "annotation",
    ["bool", "bool | None", "Optional[bool]"],
)
def test_bool_annotation_forms_are_recognised(annotation: str):
    """自检：``bool`` 的等价注解写法都要被识别。"""
    tree = ast.parse(f"def f(*, flag: {annotation} = None): pass")
    assert "flag" in _bool_scalar_names(tree)


@pytest.mark.parametrize(
    "annotation",
    ["int", "float", "str", "int | None"],
)
def test_non_bool_annotations_are_ignored(annotation: str):
    """自检：非布尔注解不得误判。"""
    tree = ast.parse(f"def f(*, flag: {annotation} = None): pass")
    assert _bool_scalar_names(tree) == set()
