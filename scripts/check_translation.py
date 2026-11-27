"""Check the Kinyarwanda file against the English one."""
import json

try:
    en = json.load(open("app/i18n_en.json", encoding="utf-8"))
    rw = json.load(open("app/i18n_rw.json", encoding="utf-8"))
except json.JSONDecodeError as error:
    raise SystemExit(f"The Kinyarwanda file has a typing problem: {error}")


def pairs(a, b, path=""):
    """Yield (where, english, kinyarwanda) for every text in the English file."""
    if isinstance(a, dict):
        for key, value in a.items():
            if not isinstance(b, dict) or key not in b:
                yield path + "/" + key, value, None
            else:
                yield from pairs(value, b[key], path + "/" + key)
    elif isinstance(a, list):
        if not isinstance(b, list) or len(b) != len(a):
            yield path, a, None
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                yield from pairs(x, y, f"{path}[{i}]")
    else:
        yield path, a, b


total = done = 0
problems = []
for where, english, kinyarwanda in pairs(en, rw):
    total += 1
    if kinyarwanda is None:
        problems.append(f"MISSING     {where}")
    elif kinyarwanda == english:
        problems.append(f"NOT CHANGED {where}")
    else:
        done += 1
        if "{label}" in english and "{label}" not in kinyarwanda:
            problems.append(f"PLACEHOLDER {where}  (keep {{label}} in the text)")

print(f"Translated: {done} of {total}")
for p in problems:
    print(p)
if not problems:
    print("All texts are translated and the structure is correct.")