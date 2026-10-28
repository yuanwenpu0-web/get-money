from __future__ import annotations

import ast
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd


class PineUnsupportedError(ValueError):
    """The source is outside the supported subset; do not execute it."""


_COLUMNS = {"open", "high", "low", "close", "volume"}
_FUNCTIONS = {"ta.ema", "ta.sma", "ta.rsi", "ta.atr", "ta.crossover", "ta.crossunder", "input.int", "input.float"}
_ALLOWED_NODES = (ast.Expression, ast.Constant, ast.Name, ast.Load, ast.BinOp, ast.UnaryOp, ast.BoolOp,
                  ast.Compare, ast.Call, ast.Attribute, ast.Subscript, ast.Add, ast.Sub, ast.Mult, ast.Div,
                  ast.Mod, ast.USub, ast.UAdd, ast.Not, ast.And, ast.Or, ast.Gt, ast.GtE, ast.Lt,
                  ast.LtE, ast.Eq, ast.NotEq)


def _function(node):
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        return f"{node.value.id}.{node.attr}"
    return ""


def _parse_expr(text, names, line):
    text = re.sub(r"\btrue\b", "True", text)
    text = re.sub(r"\bfalse\b", "False", text)
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise PineUnsupportedError(f"第 {line} 行：不支持的表达式") from exc
    attribute_owners = {id(node.value) for node in ast.walk(tree)
                        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)}
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise PineUnsupportedError(f"第 {line} 行：不支持 {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id not in names | {"ta", "input"}:
            raise PineUnsupportedError(f"第 {line} 行：未知变量 {node.id}")
        if isinstance(node, ast.Name) and node.id in {"ta", "input"} and id(node) not in attribute_owners:
            raise PineUnsupportedError(f"第 {line} 行：不支持单独引用 {node.id}")
        if isinstance(node, ast.Attribute) and _function(node) not in _FUNCTIONS:
            raise PineUnsupportedError(f"第 {line} 行：不支持 {_function(node)}")
        if isinstance(node, ast.Call) and (_function(node.func) not in _FUNCTIONS or node.keywords):
            raise PineUnsupportedError(f"第 {line} 行：不支持的函数或参数")
        if isinstance(node, ast.Call):
            fn = _function(node.func)
            counts = (1, 2) if fn.startswith("input.") else (1,) if fn == "ta.atr" else (2,)
            if len(node.args) not in counts:
                raise PineUnsupportedError(f"第 {line} 行：{fn} 参数数量错误")
        if isinstance(node, ast.Subscript) and (not isinstance(node.slice, ast.Constant) or type(node.slice.value) is not int or node.slice.value < 0):
            raise PineUnsupportedError(f"第 {line} 行：历史引用必须是非负整数")
        if isinstance(node, ast.Compare) and len(node.ops) != 1:
            raise PineUnsupportedError(f"第 {line} 行：不支持链式比较")
    return tree.body


def _rma(values, length):
    """Wilder's moving average, seeded by the first complete SMA."""
    values = pd.Series(values, dtype=float)
    result = pd.Series(np.nan, index=values.index, dtype=float)
    pending = []
    state = None
    for index, value in values.items():
        if pd.isna(value):
            continue
        if state is None:
            pending.append(value)
            if len(pending) == length:
                state = sum(pending) / length
        else:
            state = (state * (length - 1) + value) / length
        if state is not None:
            result.loc[index] = state
    return result


def _series(value, index):
    return value if isinstance(value, pd.Series) else pd.Series(value, index=index)


def _eval(node, env, index):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return env[node.id]
    if isinstance(node, ast.Subscript):
        return _series(_eval(node.value, env, index), index).shift(node.slice.value)
    if isinstance(node, ast.UnaryOp):
        value = _eval(node.operand, env, index)
        if isinstance(node.op, ast.Not):
            if isinstance(value, pd.Series):
                if not pd.api.types.is_bool_dtype(value):
                    raise PineUnsupportedError("not 的操作数必须是布尔表达式")
                return ~value.fillna(False).astype(bool)
            if not isinstance(value, bool):
                raise PineUnsupportedError("not 的操作数必须是布尔表达式")
            return not value
        return -value if isinstance(node.op, ast.USub) else +value
    if isinstance(node, ast.BinOp):
        a, b = _eval(node.left, env, index), _eval(node.right, env, index)
        result = {ast.Add: lambda: a+b, ast.Sub: lambda: a-b, ast.Mult: lambda: a*b,
                  ast.Div: lambda: a/b, ast.Mod: lambda: a%b}[type(node.op)]()
        if np.isinf(result).any() if isinstance(result, pd.Series) else np.isinf(result):
            raise PineUnsupportedError("运算产生无穷值，暂不支持该策略")
        return result
    if isinstance(node, ast.BoolOp):
        values = [_series(_eval(n, env, index), index) for n in node.values]
        if any(not pd.api.types.is_bool_dtype(value) for value in values):
            raise PineUnsupportedError("and/or 的操作数必须是布尔表达式")
        values = [value.fillna(False).astype(bool) for value in values]
        result = values[0]
        for value in values[1:]:
            result = result & value if isinstance(node.op, ast.And) else result | value
        return result
    if isinstance(node, ast.Compare):
        a, b = _eval(node.left, env, index), _eval(node.comparators[0], env, index)
        result = {ast.Gt: lambda: a>b, ast.GtE: lambda: a>=b, ast.Lt: lambda: a<b,
                  ast.LtE: lambda: a<=b, ast.Eq: lambda: a==b, ast.NotEq: lambda: a!=b}[type(node.ops[0])]()
        # Pine comparisons with na do not produce an actionable true signal.
        return result & _series(a, index).notna() & _series(b, index).notna()
    if isinstance(node, ast.Call):
        name = _function(node.func)
        args = [_eval(arg, env, index) for arg in node.args]
        if name in {"input.int", "input.float"}:
            if len(args) not in (1, 2) or not isinstance(args[0], (int, float)) or isinstance(args[0], bool) or (len(args) == 2 and not isinstance(args[1], str)):
                raise PineUnsupportedError(f"{name} 只支持常数默认值和可选标题")
            if name == "input.int" and type(args[0]) is not int:
                raise PineUnsupportedError("input.int 默认值必须为整数")
            return args[0]
        if name in {"ta.crossover", "ta.crossunder"}:
            if len(args) != 2:
                raise PineUnsupportedError(f"{name} 需要两个参数")
            a, b = (_series(arg, index) for arg in args)
            return ((a > b) & (a.shift(1) <= b.shift(1))) if name == "ta.crossover" else ((a < b) & (a.shift(1) >= b.shift(1)))
        expected = 1 if name == "ta.atr" else 2
        if len(args) != expected or type(args[-1]) is not int or args[-1] < 1:
            raise PineUnsupportedError(f"{name} 需要正整数周期")
        length = args[-1]
        if name == "ta.atr":
            high, low, close = env["high"], env["low"], env["close"]
            tr = pd.concat([high-low, (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
            return _rma(tr, length)
        source = _series(args[0], index).astype(float)
        valid = source.notna()
        if not np.isfinite(source[valid]).all() or (valid.any() and source.iloc[np.flatnonzero(valid)[0]:].isna().any()):
            raise PineUnsupportedError(f"{name} 的输入包含中途缺失值或无穷值，暂不支持")
        if name == "ta.ema":
            return source.ewm(span=length, adjust=False, ignore_na=True).mean()
        if name == "ta.sma":
            return source.rolling(length, min_periods=length).mean()
        delta = source.diff()
        gain, loss = _rma(delta.clip(lower=0), length), _rma(-delta.clip(upper=0), length)
        return 100 - 100 / (1 + gain / loss)
    raise PineUnsupportedError("不支持的表达式")


@dataclass
class PineProgram:
    source: str
    statements: list

    def to_python(self):
        """Generate a local Python module that uses this project's restricted runtime."""
        return "from app.pine import compile_pine\n\nSOURCE = " + repr(self.source) + "\n\ndef evaluate(candles):\n    return compile_pine(SOURCE).evaluate(candles)\n"

    def evaluate(self, candles):
        if not _COLUMNS.issubset(candles.columns) or "open_time" not in candles:
            raise ValueError("K线必须包含 open_time、open、high、low、close、volume")
        if candles["open_time"].isna().any() or not candles["open_time"].is_monotonic_increasing or candles["open_time"].duplicated().any():
            raise ValueError("K线时间必须唯一且升序")
        if not candles.index.is_unique:
            raise ValueError("K线索引必须唯一")
        result = candles.copy()
        env = {name: pd.to_numeric(result[name], errors="raise") for name in _COLUMNS}
        if any(not np.isfinite(value).all() for value in env.values()):
            raise ValueError("K线数值必须为有限值")
        for col in ("long_entry", "short_entry", "long_close", "short_close"):
            result[col] = False
        for kind, name, expr in self.statements:
            try:
                value = _eval(expr, env, result.index)
            except PineUnsupportedError:
                raise
            except (TypeError, ValueError, ZeroDivisionError, OverflowError) as exc:
                raise PineUnsupportedError(f"不支持的表达式或数据：{exc}") from exc
            if kind == "assign":
                env[name] = value
                result[name] = value
            else:
                value = _series(value, result.index)
                if not pd.api.types.is_bool_dtype(value):
                    raise PineUnsupportedError("if 条件必须是布尔表达式")
                result[name] |= value.fillna(False).astype(bool)
        if (result[["long_entry", "short_entry", "long_close", "short_close"]].sum(axis=1) > 1).any():
            raise PineUnsupportedError("同一根 K 线出现多条订单指令，其先后顺序尚不支持")
        return result


def compile_pine(source):
    """Compile a strict closed-bar subset; unsupported syntax is always rejected."""
    if not isinstance(source, str) or not source.strip():
        raise PineUnsupportedError("需要完整 Pine 源码")
    lines = source.splitlines()
    if not lines or lines[0].strip() != "//@version=6":
        raise PineUnsupportedError("仅支持显式声明 //@version=6")
    statements, names, entries, side_ids, actions = [], set(_COLUMNS), {}, {}, set()
    header, condition, action_count, block_actions = False, None, 0, 0
    for number, raw in enumerate(lines[1:], 2):
        stripped = raw.strip()
        if not stripped or stripped.startswith("//"):
            continue
        if "//" in stripped:
            raise PineUnsupportedError(f"第 {number} 行：不支持行尾注释")
        indent = len(raw) - len(raw.lstrip())
        if not header:
            if indent or not re.fullmatch(r'strategy\(\s*"[^"\n]+"\s*\)', stripped):
                raise PineUnsupportedError(f"第 {number} 行：仅支持 strategy(\"名称\")，策略选项需人工处理")
            header = True
            continue
        if indent:
            if condition is None or not raw.startswith("    ") or raw.startswith("     "):
                raise PineUnsupportedError(f"第 {number} 行：不支持嵌套或缩进")
            entry = re.fullmatch(r'strategy\.entry\(\s*"([^"\n]+)"\s*,\s*strategy\.(long|short)\s*\)', stripped)
            close = re.fullmatch(r'strategy\.close\(\s*"([^"\n]+)"\s*\)', stripped)
            if entry:
                identifier, side = entry.groups()
                if identifier in entries and entries[identifier] != side:
                    raise PineUnsupportedError(f"第 {number} 行：同一订单 ID 不能对应不同方向")
                if side in side_ids and side_ids[side] != identifier:
                    raise PineUnsupportedError(f"第 {number} 行：同方向多个订单 ID 尚不支持")
                entries[identifier] = side
                side_ids[side] = identifier
                statements.append(("signal", f"{side}_entry", condition))
            elif close and close.group(1) in entries:
                statements.append(("signal", f"{entries[close.group(1)]}_close", condition))
            else:
                raise PineUnsupportedError(f"第 {number} 行：不支持的订单指令或未知订单 ID")
            action = statements[-1][1]
            if action in actions:
                raise PineUnsupportedError(f"第 {number} 行：重复的 {action} 指令尚不支持")
            actions.add(action)
            action_count += 1
            block_actions += 1
            continue
        if condition is not None and not block_actions:
            raise PineUnsupportedError(f"第 {number} 行：if 语句缺少订单指令")
        condition = None
        block_actions = 0
        if stripped.startswith("if "):
            condition = _parse_expr(stripped[3:], names, number)
            continue
        assignment = re.fullmatch(r"([A-Za-z_]\w*)\s*=\s*(.+)", stripped)
        if not assignment:
            raise PineUnsupportedError(f"第 {number} 行：不支持的语句")
        name, expr = assignment.groups()
        if name in names or name in {"ta", "input", "open_time", "long_entry", "short_entry", "long_close", "short_close"}:
            raise PineUnsupportedError(f"第 {number} 行：变量重复或保留名称 {name}")
        statements.append(("assign", name, _parse_expr(expr, names, number)))
        names.add(name)
    if condition is not None and not block_actions:
        raise PineUnsupportedError("if 语句缺少订单指令")
    if not header or not action_count:
        raise PineUnsupportedError("策略必须包含至少一条支持的订单指令")
    return PineProgram(source, statements)
