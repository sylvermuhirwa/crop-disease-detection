"""Create the files used to translate the app into Kinyarwanda.

Writes:
  app/i18n_en.json  - English reference (do not edit)
  app/i18n_rw.json  - a copy of the English text, to be translated
"""
import json

# Texts shown on the app screens
UI_EN = {
    "app_title": "Cassava Leaf Check",
    "app_subtitle": "Take or upload a photo of one cassava leaf to get a first check.",
    "language": "Language",
    "upload": "Upload a photo",
    "camera": "Or take a photo",
    "no_photo": "Please add a photo to start.",
    "checking": "Checking the photo...",
    "your_photo": "Your photo",
    "result": "Result",
    "confidence": "Confidence",
    "what_to_do": "What to do now",
    "get_help": "When to get help",
    "where_looked": "Where the app looked",
    "heatmap_note": "The red and yellow areas are where the app looked most. This is only a guide.",
    "photo_problem": "Problem with the photo",
    "tips_title": "Tips for a good photo",
    "footer": "This is a research prototype. It has not been tested with farmers.",
}

advice = json.load(open("app/advice.json", encoding="utf-8"))["en"]
english = {"ui": UI_EN, "advice": advice}


def count_texts(item):
    """Count the pieces of text inside a nested structure."""
    if isinstance(item, dict):
        return sum(count_texts(v) for v in item.values())
    if isinstance(item, list):
        return sum(count_texts(v) for v in item)
    return 1


for path in ["app/i18n_en.json", "app/i18n_rw.json"]:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(english, f, indent=2, ensure_ascii=False)

print("Screen texts:", count_texts(UI_EN))
print("Advice texts:", count_texts(advice))
print("Created app/i18n_en.json (reference) and app/i18n_rw.json (to translate)")