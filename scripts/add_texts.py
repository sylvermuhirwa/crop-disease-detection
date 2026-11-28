"""Add new screen texts to both language files without touching existing translations."""
import json

NEW = {
    "cassava_only": "This app only checks cassava leaves. If your plant is not cassava, ignore the result.",
}

for path in ["app/i18n_en.json", "app/i18n_rw.json"]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    for key, text in NEW.items():
        data["ui"].setdefault(key, text)        # does nothing if the key already exists
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print("Updated", path)