import os
import re
import math
import pandas as pd
import requests
from dotenv import load_dotenv
import random

load_dotenv()

HF_API_KEY = os.getenv("HF_API_KEY", "").strip()
HF_MODEL = os.getenv("HF_MODEL", "google/flan-t5-large").strip()
HF_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"

CSV_PATH = os.getenv("CATALOG_PATH", "data/catalog.csv")


# ---------------- CSV / util ----------------
def load_catalog():
    df = pd.read_csv(CSV_PATH)
    df.columns = [c.strip().lower() for c in df.columns]
    if "name" not in df.columns or "description" not in df.columns or "price" not in df.columns:
        raise ValueError("catalog.csv must have columns: name,description,price")

    def to_float(x):
        if pd.isna(x):
            return math.nan
        s = re.sub(r"[^\d.]", "", str(x))
        return float(s) if s else math.nan

    df["price"] = df["price"].apply(to_float)
    df["name"] = df["name"].astype(str)
    df["description"] = df["description"].astype(str)
    return df


# ---------------- filtering ----------------
def filter_catalog(interests: str, budget_min: int, budget_max: int, top_n=25):
    df = load_catalog()
    kws = [k.strip().lower() for k in re.split(r"[,\;/]+", interests) if k.strip()]
    if not kws:
        return df[df["price"].between(budget_min, budget_max, inclusive="both")] \
            .sort_values("price").head(top_n).to_dict(orient="records")

    matches = []
    for kw in kws:
        mask_text = df["name"].str.contains(kw, case=False, na=False) | df["description"].str.contains(kw, case=False, na=False)
        mask_price = df["price"].between(budget_min, budget_max, inclusive="both")
        filtered = df[mask_text & mask_price].copy()
        matches.append(filtered)

    combined = pd.concat(matches).drop_duplicates()
    if combined.empty:
        combined = df[df["price"].between(budget_min, budget_max, inclusive="both")].copy()

    mid = (budget_min + budget_max) / 2
    combined["price_gap"] = (combined["price"] - mid).abs()
    combined = combined.sort_values(by=["price_gap", "price"]).head(top_n)
    return combined.drop(columns=["price_gap"], errors="ignore").to_dict(orient="records")


# ---------------- fallback rules ----------------
_FALLBACK_MAP = {
    "books": [
        "Bestselling novel", "Bookstore gift card", "Handmade bookmark",
        "Book subscription", "Reading lamp", "Collector’s edition set"
    ],
    "music": [
        "Concert ticket voucher", "Wireless headphones", "Bluetooth speaker",
        "Music subscription", "Vinyl record", "Musical instrument accessory"
    ],
    "fitness": [
        "Yoga mat", "Fitness tracker", "Gym class voucher",
        "Resistance bands", "Shaker bottle", "Dumbbell set"
    ],
    "tech": [
        "Power bank", "Wireless earbuds", "Smart LED bulb",
        "Portable speaker", "Gadget gift card", "Smartwatch strap"
    ],
    "art": [
        "Sketchbook", "Painting kit", "Online art class voucher",
        "Framed print", "Custom portrait", "Set of brushes"
    ],
    "home": [
        "Scented candles", "Table lamp", "Succulent plant",
        "Throw blanket", "Decorative frame", "Wall clock"
    ],
    "travel": [
        "Backpack", "Travel pillow", "Trip voucher",
        "Portable charger", "Luggage tag", "Travel organizer kit"
    ],
    "sports": [
        "Football", "Cricket bat", "Sports jersey",
        "Match ticket voucher", "Gym bag", "Sports water bottle"
    ],
    "food": [
        "Chocolate hamper", "Cooking class voucher", "Restaurant gift card",
        "Gourmet coffee set", "Snack basket", "Exotic tea sampler"
    ],
    "gaming": [
        "Gaming mouse", "Mechanical keyboard", "Steam gift card",
        "Headset stand", "Gamepad controller", "LED desk lights"
    ],
    "fashion": [
        "Trendy watch", "Sunglasses", "Wallet",
        "Handbag", "Jewelry piece", "Fashion store voucher",
        "Stylish dress", "Scarf", "Perfume"
    ],
    "photography": [
        "Camera strap", "Tripod", "Photo album",
        "Photography workshop", "Portable light", "Camera cleaning kit"
    ],
    "gardening": [
        "Plant pot set", "Gardening tool kit", "Seeds pack",
        "Hanging planter", "Indoor plant light", "Compost bin"
    ],
    "pets": [
        "Pet toy set", "Pet grooming kit", "Personalized pet collar",
        "Pet bed", "Treats basket", "Pet water fountain"
    ],
    "wellness": [
        "Aroma diffuser", "Massage voucher", "Spa gift set",
        "Meditation app subscription", "Essential oils kit", "Herbal tea set"
    ],
    "handcrafts": [
        "Handmade jewelry", "Clay pottery set", "Hand-painted mug",
        "Embroidery kit", "Crochet set", "Wood carving miniature"
    ],
    "diy": [
        "DIY candle making kit", "DIY soap kit", "DIY robotics kit",
        "DIY painting set", "DIY jewelry kit", "DIY terrarium kit"
    ],
    "kids": [
        "Building blocks set", "Story books", "Educational puzzle",
        "Remote control toy", "Drawing set", "Soft toys"
    ],
    "office": [
        "Desk organizer", "Fancy pen set", "Notebook",
        "Office plant", "Coffee mug", "Ergonomic mousepad"
    ],
    "luxury": [
        "Designer wallet", "Luxury perfume", "Gold-plated pen",
        "Silk scarf", "Premium watch", "Designer handbag"
    ],
    "festival": [
        "Fairy lights", "Festive sweets box", "Decorative lantern",
        "Puja thali", "Gift hamper", "Ethnic dress"
    ]
}


def rule_based_suggestions(interests: str, budget_min: int, budget_max: int, needed: int):
    kws = [k.strip().lower() for k in re.split(r"[,\;/]+", interests) if k.strip()]
    suggestions = []
    for kw in kws:
        for base, items in _FALLBACK_MAP.items():
            if kw in base:
                for it in items:
                    suggestions.append(f"{it.title()} — good for {base} lovers")

    if not suggestions:
        suggestions = [
            "Gift card — flexible choice",
            "Experience voucher — memorable gift",
            "Personalized item — thoughtful",
            "Subscription — long-lasting",
            "Accessory — improves experience"
        ]

    return random.sample(suggestions, min(needed, len(suggestions)))


# ---------------- prompt ----------------
def build_prompt(user_input, filtered_items):
    items_text = ""
    for it in filtered_items[:10]:
        price_str = f"Rs.{int(it['price'])}" if not math.isnan(it.get("price", float("nan"))) else "NA"
        items_text += f"- {it['name']} ({price_str}) — {it['description']}\n"

    prompt = f"""
You are a gift recommendation expert.

User:
- Occasion: {user_input['occasion']}
- Age: {user_input['age']}
- Gender: {user_input['gender']}
- Interests: {user_input['interests']}
- Budget: {user_input['budget_min']} - {user_input['budget_max']}

Catalog items:
{items_text}

Task:
- Suggest EXACTLY 5 gift ideas.
- Use catalog items as inspiration, but also suggest related/experiential gifts.
- Keep each suggestion short: "1. Gift Name — short reason".
- Ensure variety and relevance to interests + budget.
"""
    return prompt.strip()


# ---------------- HF call ----------------
def _hf_headers():
    if not HF_API_KEY or not HF_API_KEY.startswith("hf_"):
        raise RuntimeError("HF API key missing or invalid.")
    return {"Authorization": f"Bearer {HF_API_KEY}"}

def query_llm(prompt: str, timeout=40):
    try:
        headers = _hf_headers()
        payload = {"inputs": prompt, "parameters": {"max_new_tokens": 200, "temperature": 0.7}}
        resp = requests.post(HF_URL, headers=headers, json=payload, timeout=timeout)
        if resp.status_code != 200:
            return {"error": f"{resp.status_code} {resp.text}"}
        data = resp.json()
        if isinstance(data, list) and data and isinstance(data[0], dict):
            return {"text": data[0].get("generated_text", data[0].get("summary_text", ""))}
        return {"text": str(data)}
    except Exception as e:
        return {"error": str(e)}


def _extract_lines(text):
    lines = []
    for ln in re.split(r"[\r\n]+", text):
        ln = ln.strip()
        if not ln:
            continue
        if re.match(r"^\d+\.", ln) or ln.startswith("-"):
            ln = re.sub(r"^[\-\•\s]*\d*\.*\s*", "", ln)
            lines.append(ln)
    return lines


# ---------------- main recommender ----------------
def recommend_gifts(user_input: dict):
    filtered = filter_catalog(user_input.get("interests",""), user_input.get("budget_min",0), user_input.get("budget_max",10**9))
    prompt = build_prompt(user_input, filtered)

    resp = query_llm(prompt)
    suggestions = []

    if "error" in resp or not resp.get("text"):
        picks = [f"{it['name']} — good match" for it in (filtered[:2] if filtered else [])]
        filler = rule_based_suggestions(user_input.get("interests",""), user_input.get("budget_min",0), user_input.get("budget_max",0), 5-len(picks))
        suggestions = picks + filler
    else:
        lines = _extract_lines(resp["text"])
        if not lines:
            filler = rule_based_suggestions(user_input.get("interests",""), user_input.get("budget_min",0), user_input.get("budget_max",0), 5)
            suggestions = filler
        else:
            suggestions = lines[:5]

    final = []
    for i, s in enumerate(suggestions[:5], start=1):
        if "—" not in s:
            s = f"{s} — good choice"
        final.append(f"{i}. {s}")
    return "\n".join(final)
