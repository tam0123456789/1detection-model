import re
import inspect
import ultralytics.nn.tasks as tasks
from src.models.uib import UIB, UIBDown


def register_uib_modules():
    tasks.UIB = UIB
    tasks.UIBDown = UIBDown

    src = inspect.getsource(tasks.parse_model)

    if "base_modules = set(base_modules) | {UIB, UIBDown}" in src:
        tasks.parse_model.__globals__["UIB"] = UIB
        tasks.parse_model.__globals__["UIBDown"] = UIBDown
        tasks.__dict__["UIB"] = UIB
        tasks.__dict__["UIBDown"] = UIBDown
        print("parse_model đã được patch trước đó.")
        return

    lines = src.splitlines()
    out = []
    inserted = False
    in_base = False
    balance = 0
    indent_after = ""

    def delta_balance(line: str):
        return (line.count("{") - line.count("}")) + (line.count("(") - line.count(")"))

    for line in lines:
        out.append(line)

        if (not inserted) and (not in_base) and re.match(r"^\s*base_modules\s*=", line):
            in_base = True
            balance = delta_balance(line)
            indent_after = re.match(r"^(\s*)", line).group(1)

            if balance == 0:
                out.append(indent_after + "base_modules = set(base_modules) | {UIB, UIBDown}")
                out.append(indent_after + "base_modules = frozenset(base_modules)")
                inserted = True
                in_base = False

        elif in_base:
            balance += delta_balance(line)
            if balance == 0:
                out.append(indent_after + "base_modules = set(base_modules) | {UIB, UIBDown}")
                out.append(indent_after + "base_modules = frozenset(base_modules)")
                inserted = True
                in_base = False

    if not inserted:
        raise RuntimeError("Không patch được parse_model(): không tìm thấy base_modules")

    patched = "\n".join(out)

    g = dict(tasks.__dict__)
    g["UIB"] = UIB
    g["UIBDown"] = UIBDown

    exec(patched, g)
    tasks.parse_model = g["parse_model"]

    tasks.parse_model.__globals__["UIB"] = UIB
    tasks.parse_model.__globals__["UIBDown"] = UIBDown
    tasks.__dict__["UIB"] = UIB
    tasks.__dict__["UIBDown"] = UIBDown

    print("Đã patch parse_model() thành công.")