"""Simple test runner for extractor functions."""
import json
from types import SimpleNamespace
from extractor import extract_rules, extract_gemini, build_system_prompt

products = {
    "Hillside trail mix tub": {"item_id": "FOODS_3_252", "store_id": "CA_1"}
}

email = (
    "From: Store Manager, CA_1\n"
    "Subject: game day\n"
    "Big game coming - take the Hillside trail mix tub up 40% for that week please. Deb"
)

print("=== extract_rules ===")
print(extract_rules(email, products))

# Fake Gemini client that returns a JSON string in `.text`
class FakeModels:
    def generate_content(self, model, contents, config):
        # We return a JSON object the same shape the real model would produce
        obj = {
            "is_request": True,
            "item_id": "FOODS_3_252",
            "store_id": "CA_1",
            "event_name": "SuperBowl",
            "requested_pct": 0.4,
            "urgency": "normal",
        }
        return SimpleNamespace(text=json.dumps(obj))

class FakeClient:
    def __init__(self):
        self.models = FakeModels()

print("=== extract_gemini (fake client) ===")
print(extract_gemini(email, products, client=FakeClient()))
