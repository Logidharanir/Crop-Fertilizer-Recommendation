# ====================================================================================
# Crop & Fertilizer Recommender | Streamlit App (AICTE Internship - Logidharan, 2025)
# ====================================================================================

import streamlit as st, pandas as pd, numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from pathlib import Path

# ----------------------------------------------
# 0. PAGE CONFIGURATION
# ----------------------------------------------
st.set_page_config("Agri Advisor", "🌿", layout="wide")
st.markdown("<style>footer,#MainMenu{visibility:hidden}</style>", unsafe_allow_html=True)

BASE = Path(__file__).resolve().parent

# ----------------------------------------------
# 1. LOAD & CACHE MODELS
# ----------------------------------------------
@st.cache_resource(show_spinner=True)
def train_crop_model():
    df = pd.read_csv(BASE / "Crop_recommendation.csv")
    X = df.drop("label", axis=1)
    y = df["label"]
    scaler = StandardScaler().fit(X)
    model = RandomForestClassifier(n_estimators=300, random_state=42)
    model.fit(scaler.transform(X), y)
    return model, scaler, sorted(y.unique())

@st.cache_resource(show_spinner=True)
def train_fert_model():
    df = pd.read_csv(BASE / "Fertilizer Prediction.csv")
    y = df["Fertilizer Name"]
    X = df.drop("Fertilizer Name", axis=1)

    encoders = {}
    for col in ["Soil Type", "Crop Type"]:
        enc = LabelEncoder().fit(X[col])
        X[col] = enc.transform(X[col])
        encoders[col] = enc

    scaler = StandardScaler()
    num_cols = X.columns.difference(["Soil Type", "Crop Type"])
    X[num_cols] = scaler.fit_transform(X[num_cols])

    model = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42)
    model.fit(X, y)
    return model, scaler, encoders, sorted(y.unique())

crop_model, crop_scaler, crop_labels = train_crop_model()
fert_model, fert_scaler, fert_encoders, fert_labels = train_fert_model()

# ----------------------------------------------
# 2.  IN‑CODE CROP DATABASE  (100 crops)  – PART 1
# ----------------------------------------------
CROP_INFO = {
    # Cereal Grains (10)
    "Rice":      {"Category":"Cereal","Purpose":"Food",
                  "Major Regions":"Asia (India, China, Indonesia)",
                  "Avg Yield":"4.7 t/ha","Value":"≈ $440 / t",
                  "Notes":"Requires flooded fields; staple for half the world."},
    "Wheat":     {"Category":"Cereal","Purpose":"Food",
                  "Major Regions":"China, India, Russia",
                  "Avg Yield":"3.2 t/ha","Value":"≈ $230 / t",
                  "Notes":"Main ingredient for bread, pasta, noodles."},
    "Maize":     {"Category":"Cereal","Purpose":"Food • Fodder • Biofuel",
                  "Major Regions":"USA, China, Brazil",
                  "Avg Yield":"5.5 t/ha","Value":"≈ $180 / t",
                  "Notes":"World’s most‑produced grain; also used for ethanol."},
    "Barley":    {"Category":"Cereal","Purpose":"Food • Malt",
                  "Major Regions":"Russia, France, Germany",
                  "Avg Yield":"3.0 t/ha","Value":"≈ $200 / t",
                  "Notes":"Key grain for beer malt and animal feed."},
    "Sorghum":   {"Category":"Cereal","Purpose":"Food • Fodder",
                  "Major Regions":"Nigeria, USA, India",
                  "Avg Yield":"2.4 t/ha","Value":"≈ $170 / t",
                  "Notes":"Drought‑tolerant cereal for semi‑arid tropics."},
    "Millet":    {"Category":"Cereal","Purpose":"Food",
                  "Major Regions":"India, Niger, China",
                  "Avg Yield":"1.2 t/ha","Value":"≈ $270 / t",
                  "Notes":"Small‑seeded grain popular in dry zones."},
    "Oats":      {"Category":"Cereal","Purpose":"Food • Fodder",
                  "Major Regions":"Russia, Canada, Poland",
                  "Avg Yield":"2.3 t/ha","Value":"≈ $250 / t",
                  "Notes":"Consumed as oatmeal and used in livestock rations."},
    "Rye":       {"Category":"Cereal","Purpose":"Food • Beverage",
                  "Major Regions":"Germany, Poland, Russia",
                  "Avg Yield":"2.6 t/ha","Value":"≈ $210 / t",
                  "Notes":"Used in rye bread and some whiskies."},
    "Quinoa":    {"Category":"Pseudo‑cereal","Purpose":"Health Food",
                  "Major Regions":"Peru, Bolivia",
                  "Avg Yield":"1.0 t/ha","Value":"≈ $2 500 / t",
                  "Notes":"Gluten‑free grain with high protein."},
    "Buckwheat": {"Category":"Pseudo‑cereal","Purpose":"Food",
                  "Major Regions":"Russia, China, Ukraine",
                  "Avg Yield":"1.1 t/ha","Value":"≈ $600 / t",
                  "Notes":"Used in soba noodles and pancakes."},

    # Pulses & Legumes (10)
    "Soybean":   {"Category":"Oilseed","Purpose":"Protein • Oil",
                  "Major Regions":"Brazil, USA, Argentina",
                  "Avg Yield":"2.8 t/ha","Value":"≈ $330 / t",
                  "Notes":"Main global source of plant protein and oil."},
    "Chickpea":  {"Category":"Pulse","Purpose":"Food",
                  "Major Regions":"India, Australia, Turkey",
                  "Avg Yield":"1.0 t/ha","Value":"≈ $550 / t",
                  "Notes":"Used in hummus and Indian dal."},
    "Lentil":    {"Category":"Pulse","Purpose":"Food",
                  "Major Regions":"Canada, India, Turkey",
                  "Avg Yield":"1.2 t/ha","Value":"≈ $600 / t",
                  "Notes":"High‑protein legume; cooks quickly."},
    "Pea":       {"Category":"Pulse","Purpose":"Food • Fodder",
                  "Major Regions":"Russia, China, Canada",
                  "Avg Yield":"2.0 t/ha","Value":"≈ $300 / t",
                  "Notes":"Green peas for veg; dry peas for protein isolates."},
    "Mung Bean": {"Category":"Pulse","Purpose":"Food",
                  "Major Regions":"India, Myanmar, Thailand",
                  "Avg Yield":"0.9 t/ha","Value":"≈ $800 / t",
                  "Notes":"Sprouted for bean‑sprouts; used in sweets."},
    "Kidney Bean":{"Category":"Pulse","Purpose":"Food",
                  "Major Regions":"Mexico, India, Brazil",
                  "Avg Yield":"1.5 t/ha","Value":"≈ $700 / t",
                  "Notes":"Common in chili and rajma curry."},
    "Blackgram": {"Category":"Pulse","Purpose":"Food",
                  "Major Regions":"India, Myanmar",
                  "Avg Yield":"0.8 t/ha","Value":"≈ $900 / t",
                  "Notes":"Key ingredient in idli/dosa batter."},
    "Pigeon Pea":{"Category":"Pulse","Purpose":"Food",
                  "Major Regions":"India, Tanzania",
                  "Avg Yield":"1.0 t/ha","Value":"≈ $650 / t",
                  "Notes":"Long‑duration dal crop tolerant to drought."},
    "Faba Bean": {"Category":"Pulse","Purpose":"Food • Fodder",
                  "Major Regions":"China, Ethiopia",
                  "Avg Yield":"2.0 t/ha","Value":"≈ $450 / t",
                  "Notes":"High protein; fixes nitrogen."},
    "Cowpea":    {"Category":"Pulse","Purpose":"Food • Fodder",
                  "Major Regions":"Nigeria, Niger",
                  "Avg Yield":"0.7 t/ha","Value":"≈ $500 / t",
                  "Notes":"Also called black‑eyed pea; drought‑hardy."},

    # Oilseed & Industrial (10)
    "Groundnut": {"Category":"Oilseed","Purpose":"Oil • Food",
                  "Major Regions":"China, India, Nigeria",
                  "Avg Yield":"1.7 t/ha","Value":"≈ $1 100 / t",
                  "Notes":"Peanut butter and edible oil source."},
    "Sesame":    {"Category":"Oilseed","Purpose":"Oil • Food",
                  "Major Regions":"Sudan, Myanmar, India",
                  "Avg Yield":"0.6 t/ha","Value":"≈ $1 800 / t",
                  "Notes":"High‑value seeds for snacks and tahini."},
    "Mustard":   {"Category":"Oilseed","Purpose":"Oil • Spice",
                  "Major Regions":"Canada, Nepal, India",
                  "Avg Yield":"1.4 t/ha","Value":"≈ $900 / t",
                  "Notes":"Seeds yield pungent oil; leaves as greens."},
    "Sunflower": {"Category":"Oilseed","Purpose":"Oil",
                  "Major Regions":"Russia, Ukraine",
                  "Avg Yield":"1.9 t/ha","Value":"≈ $600 / t",
                  "Notes":"Sunflower oil popular for cooking."},
    "Canola":    {"Category":"Oilseed","Purpose":"Oil",
                  "Major Regions":"Canada, EU",
                  "Avg Yield":"2.1 t/ha","Value":"≈ $550 / t",
                  "Notes":"Low erucic acid rapeseed; healthy cooking oil."},
    "Coconut":   {"Category":"Oilseed","Purpose":"Oil • Food",
                  "Major Regions":"Indonesia, Philippines",
                  "Avg Yield":"7 t nuts/ha","Value":"≈ $275 / t copra",
                  "Notes":"Versatile; water, milk, oil, coir."},
    "Palm Oil":  {"Category":"Oilseed","Purpose":"Oil",
                  "Major Regions":"Indonesia, Malaysia",
                  "Avg Yield":"4.0 t oil/ha","Value":"≈ $1 000 / t",
                  "Notes":"Highest oil yield per hectare crop."},
    "Flax":      {"Category":"Oilseed/Fiber","Purpose":"Linseed Oil • Fiber",
                  "Major Regions":"Kazakhstan, Russia",
                  "Avg Yield":"1.2 t/ha","Value":"≈ $500 / t",
                  "Notes":"Oil for health; fiber for linen."},
    "Castor":    {"Category":"Oilseed","Purpose":"Industrial Oil",
                  "Major Regions":"India, Mozambique",
                  "Avg Yield":"1.3 t/ha","Value":"≈ $900 / t",
                  "Notes":"Ricinoleic acid used in lubricants."},
    "Cotton":    {"Category":"Fiber","Purpose":"Textile",
                  "Major Regions":"India, China, USA",
                  "Avg Yield":"2.2 t lint/ha","Value":"≈ $1 600 / t lint",
                  "Notes":"Primary natural fiber for clothing."},

    # Fruits (10)
    "Banana":    {"Category":"Fruit","Purpose":"Food",
                  "Major Regions":"India, China, Ecuador",
                  "Avg Yield":"4.7 t/ha","Value":"≈ $350 / t",
                  "Notes":"Top exported tropical fruit."},
    "Mango":     {"Category":"Fruit","Purpose":"Food",
                  "Major Regions":"India, Indonesia, Mexico",
                  "Avg Yield":"9.0 t/ha","Value":"≈ $550 / t",
                  "Notes":"Known as ‘king of fruits’ in India."},
    "Apple":     {"Category":"Fruit","Purpose":"Food",
                  "Major Regions":"China, USA, Poland",
                  "Avg Yield":"16 t/ha","Value":"≈ $700 / t",
                  "Notes":"Temperate fruit rich in fiber and vitamins."},
    "Orange":    {"Category":"Fruit","Purpose":"Food",
                  "Major Regions":"Brazil, China, India",
                  "Avg Yield":"12 t/ha","Value":"≈ $450 / t",
                  "Notes":"Citrus high in vitamin C."},
    "Grapes":    {"Category":"Fruit","Purpose":"Food • Wine",
                  "Major Regions":"China, Italy, USA",
                  "Avg Yield":"11 t/ha","Value":"≈ $750 / t",
                  "Notes":"Table grapes & wine industry crop."},
    "Pomegranate":{"Category":"Fruit","Purpose":"Food",
                  "Major Regions":"India, Iran, USA",
                  "Avg Yield":"8 t/ha","Value":"≈ $1 000 / t",
                  "Notes":"Rich in antioxidants; arils consumed fresh."},
    "Papaya":    {"Category":"Fruit","Purpose":"Food",
                  "Major Regions":"India, Brazil, Mexico",
                  "Avg Yield":"40 t/ha","Value":"≈ $280 / t",
                  "Notes":"Fast‑growing tropical fruit; rich in papain."},
    "Watermelon":{"Category":"Fruit","Purpose":"Food",
                  "Major Regions":"China, Turkey",
                  "Avg Yield":"30 t/ha","Value":"≈ $200 / t",
                  "Notes":"Refreshing summer fruit 🍉."},
    "Pineapple": {"Category":"Fruit","Purpose":"Food",
                  "Major Regions":"Costa Rica, Philippines",
                  "Avg Yield":"55 t/ha","Value":"≈ $300 / t",
                  "Notes":"Tropical fruit; canned and fresh markets."},
    "Avocado":   {"Category":"Fruit","Purpose":"Food",
                  "Major Regions":"Mexico, Dominican Rep.",
                  "Avg Yield":"7 t/ha","Value":"≈ $2 300 / t",
                  "Notes":"High healthy fats; export demand rising."},

    # Vegetables (10)
    "Tomato":    {"Category":"Vegetable","Purpose":"Food",
                  "Major Regions":"China, India, USA",
                  "Avg Yield":"38 t/ha","Value":"≈ $170 / t",
                  "Notes":"Used fresh and processed (ketchup, sauces)."},
    "Potato":    {"Category":"Vegetable","Purpose":"Food",
                  "Major Regions":"China, India, Russia",
                  "Avg Yield":"20 t/ha","Value":"≈ $260 / t",
                  "Notes":"Starchy tuber crop; fast‑growing."},
    "Onion":     {"Category":"Vegetable","Purpose":"Food",
                  "Major Regions":"China, India, Egypt",
                  "Avg Yield":"22 t/ha","Value":"≈ $220 / t",
                  "Notes":"Staple aromatic bulb; storage life good."},
    "Garlic":    {"Category":"Vegetable","Purpose":"Food • Medicine",
                  "Major Regions":"China, India",
                  "Avg Yield":"6 t/ha","Value":"≈ $1 000 / t",
                  "Notes":"Strong flavour; natural antibiotic."},
    "Carrot":    {"Category":"Vegetable","Purpose":"Food",
                  "Major Regions":"China, Uzbekistan",
                  "Avg Yield":"30 t/ha","Value":"≈ $240 / t",
                  "Notes":"Rich in beta‑carotene (vitamin A)."},
    "Spinach":   {"Category":"Vegetable","Purpose":"Food",
                  "Major Regions":"China, USA",
                  "Avg Yield":"25 t/ha","Value":"≈ $400 / t",
                  "Notes":"Leafy green high in iron."},
    "Cabbage":   {"Category":"Vegetable","Purpose":"Food",
                  "Major Regions":"China, India",
                  "Avg Yield":"30 t/ha","Value":"≈ $200 / t",
                  "Notes":"Brassica leafy head vegetable."},
    "Brinjal":   {"Category":"Vegetable","Purpose":"Food",
                  "Major Regions":"China, India",
                  "Avg Yield":"27 t/ha","Value":"≈ $180 / t",
                  "Notes":"Also called eggplant; used in curries."},
    "Chilli":    {"Category":"Vegetable","Purpose":"Spice",
                  "Major Regions":"India, Thailand",
                  "Avg Yield":"2.0 t/ha (dry)","Value":"≈ $2 000 / t",
                  "Notes":"Capsaicin gives pungency; used fresh & dried."},
    "Okra":      {"Category":"Vegetable","Purpose":"Food",
                  "Major Regions":"India, Nigeria",
                  "Avg Yield":"10 t/ha","Value":"≈ $350 / t",
                  "Notes":"Also called lady’s finger; mucilaginous pods."},

    # Tubers & Roots (10)
    "Sweet Potato": {"Category":"Root","Purpose":"Food",
                     "Major Regions":"China, Nigeria",
                     "Avg Yield":"15 t/ha","Value":"≈ $220 / t",
                     "Notes":"Rich in beta‑carotene; drought‑tolerant."},
    "Cassava":      {"Category":"Root","Purpose":"Food • Starch",
                     "Major Regions":"Nigeria, Thailand",
                     "Avg Yield":"12 t/ha","Value":"≈ $100 / t",
                     "Notes":"High starch root; key in tropics."},
    "Yam":          {"Category":"Root","Purpose":"Food",
                     "Major Regions":"Nigeria, Ghana",
                     "Avg Yield":"10 t/ha","Value":"≈ $250 / t",
                     "Notes":"Popular in West Africa; long shelf life."},
    "Taro":         {"Category":"Root","Purpose":"Food",
                     "Major Regions":"India, China, Nigeria",
                     "Avg Yield":"6 t/ha","Value":"≈ $300 / t",
                     "Notes":"Edible corms and leaves."},
    "Beetroot":     {"Category":"Root","Purpose":"Food",
                     "Major Regions":"Russia, France",
                     "Avg Yield":"40 t/ha","Value":"≈ $180 / t",
                     "Notes":"Used in salads and juice."},
    "Radish":       {"Category":"Root","Purpose":"Food",
                     "Major Regions":"India, China",
                     "Avg Yield":"25 t/ha","Value":"≈ $160 / t",
                     "Notes":"Fast‑growing; pungent root."},
    "Turnip":       {"Category":"Root","Purpose":"Food",
                     "Major Regions":"France, Germany",
                     "Avg Yield":"30 t/ha","Value":"≈ $180 / t",
                     "Notes":"Cool‑season root vegetable."},
    "Sugar Beet":   {"Category":"Root","Purpose":"Sugar",
                     "Major Regions":"Russia, France, USA",
                     "Avg Yield":"55 t/ha","Value":"≈ $40 / t (raw)",
                     "Notes":"Source of white sugar in Europe."},
    "Horseradish":  {"Category":"Root","Purpose":"Spice • Condiment",
                     "Major Regions":"USA, Germany",
                     "Avg Yield":"18 t/ha","Value":"≈ $500 / t",
                     "Notes":"Strong pungent root used in sauces."},
    "Arrowroot":    {"Category":"Root","Purpose":"Food • Starch",
                     "Major Regions":"India, St. Vincent",
                     "Avg Yield":"7 t/ha","Value":"≈ $350 / t",
                     "Notes":"Used in baby food & gluten‑free baking."},

    # Spices & Medicinal (10)
    "Turmeric":     {"Category":"Spice","Purpose":"Food • Medicine",
                     "Major Regions":"India, Myanmar",
                     "Avg Yield":"7 t/ha","Value":"≈ $1 200 / t",
                     "Notes":"Curcumin‑rich; anti‑inflammatory."},
    "Ginger":       {"Category":"Spice","Purpose":"Food • Medicine",
                     "Major Regions":"India, China",
                     "Avg Yield":"6 t/ha","Value":"≈ $1 000 / t",
                     "Notes":"Used fresh and dry in cooking, ayurveda."},
    "Coriander":    {"Category":"Spice","Purpose":"Food",
                     "Major Regions":"India, Morocco",
                     "Avg Yield":"1.3 t/ha","Value":"≈ $1 000 / t",
                     "Notes":"Leaves used as herb, seeds as spice."},
    "Cumin":        {"Category":"Spice","Purpose":"Food",
                     "Major Regions":"India, Syria",
                     "Avg Yield":"0.9 t/ha","Value":"≈ $2 000 / t",
                     "Notes":"Earthy‑flavored spice; key in curries."},
    "Cardamom":     {"Category":"Spice","Purpose":"Food • Fragrance",
                     "Major Regions":"India, Guatemala",
                     "Avg Yield":"0.3 t/ha","Value":"≈ $10 000 / t",
                     "Notes":"Queen of spices; used in desserts & perfumes."},
    "Black Pepper": {"Category":"Spice","Purpose":"Food",
                     "Major Regions":"India, Vietnam",
                     "Avg Yield":"2.5 t/ha","Value":"≈ $3 500 / t",
                     "Notes":"Most traded spice globally."},
    "Fennel":       {"Category":"Spice","Purpose":"Food • Medicine",
                     "Major Regions":"India, Egypt",
                     "Avg Yield":"2.0 t/ha","Value":"≈ $900 / t",
                     "Notes":"Sweet flavor; used in saunf and digestion."},
    "Fenugreek":    {"Category":"Spice","Purpose":"Food • Medicine",
                     "Major Regions":"India, Pakistan",
                     "Avg Yield":"1.2 t/ha","Value":"≈ $1 100 / t",
                     "Notes":"Used in pickles and diabetes care."},
    "Basil":        {"Category":"Herb","Purpose":"Food • Medicinal",
                     "Major Regions":"India, Italy",
                     "Avg Yield":"2.0 t/ha","Value":"≈ $700 / t (dry)",
                     "Notes":"Used in herbal teas, cooking and skin care."},
    "Mint":         {"Category":"Herb","Purpose":"Food • Medicine",
                     "Major Regions":"India, USA",
                     "Avg Yield":"5.5 t/ha","Value":"≈ $600 / t",
                     "Notes":"Used in chutneys, teas, and flavors."},

    # Nuts & Plantation (10)
    "Almond":       {"Category":"Nut","Purpose":"Food",
                     "Major Regions":"USA, Spain",
                     "Avg Yield":"2.0 t/ha","Value":"≈ $5 000 / t",
                     "Notes":"High protein; used in milk, sweets, oil."},
    "Cashew":       {"Category":"Nut","Purpose":"Food",
                     "Major Regions":"India, Vietnam",
                     "Avg Yield":"1.2 t/ha","Value":"≈ $8 000 / t",
                     "Notes":"Nut and apple both used; export‑driven."},
    "Walnut":       {"Category":"Nut","Purpose":"Food",
                     "Major Regions":"China, USA",
                     "Avg Yield":"2.5 t/ha","Value":"≈ $3 500 / t",
                     "Notes":"Rich in omega‑3 fats."},
    "Hazelnut":     {"Category":"Nut","Purpose":"Food",
                     "Major Regions":"Turkey, Italy",
                     "Avg Yield":"2.3 t/ha","Value":"≈ $4 000 / t",
                     "Notes":"Used in chocolate spreads and desserts."},
    "Tea":          {"Category":"Plantation","Purpose":"Beverage",
                     "Major Regions":"China, India, Kenya",
                     "Avg Yield":"2.0 t/ha","Value":"≈ $1 600 / t (dry)",
                     "Notes":"Leaves processed into green/black tea."},
    "Coffee":       {"Category":"Plantation","Purpose":"Beverage",
                     "Major Regions":"Brazil, Vietnam",
                     "Avg Yield":"1.5 t/ha","Value":"≈ $2 000 / t",
                     "Notes":"Beans roasted for drink; Arabica or Robusta."},
    "Cocoa":        {"Category":"Plantation","Purpose":"Chocolate",
                     "Major Regions":"Ivory Coast, Ghana",
                     "Avg Yield":"0.8 t/ha","Value":"≈ $2 600 / t",
                     "Notes":"Pods used to make cocoa powder and chocolate."},
    "Rubber":       {"Category":"Plantation","Purpose":"Latex",
                     "Major Regions":"Thailand, Indonesia",
                     "Avg Yield":"1.8 t/ha","Value":"≈ $1 500 / t",
                     "Notes":"Latex tapped from bark; used in tires."},
    "Jute":         {"Category":"Fiber","Purpose":"Bags • Ropes",
                     "Major Regions":"India, Bangladesh",
                     "Avg Yield":"2.0 t/ha","Value":"≈ $500 / t",
                     "Notes":"Eco‑friendly fiber crop."},
    "Areca Nut":    {"Category":"Plantation","Purpose":"Chewing",
                     "Major Regions":"India, Bangladesh",
                     "Avg Yield":"1.5 t/ha","Value":"≈ $3 000 / t",
                     "Notes":"Used in pan and rituals; stimulant effects."}
}

# ----------------------------------------------
# 🌳 IN‑CODE PLANT & TREE DATABASE (20 examples)
# ----------------------------------------------
PLANT_INFO = {
    "Mango Tree": {
        "Category": "Fruit Tree",
        "Purpose": "Food",
        "Major Regions": "India, Mexico, Thailand",
        "Avg Height": "10–40 m",
        "Uses": "Fresh fruit, mango pulp, jams",
        "Notes": "Tropical tree; fruiting in 3–5 years."
    },
    "Apple Tree": {
        "Category": "Fruit Tree",
        "Purpose": "Food",
        "Major Regions": "China, USA, Poland",
        "Avg Height": "4–12 m",
        "Uses": "Fresh fruit, cider, cooking",
        "Notes": "Temperate climate; needs chilling hours."
    },
    "Orange Tree": {
        "Category": "Fruit Tree",
        "Purpose": "Food",
        "Major Regions": "Brazil, China, India",
        "Avg Height": "6–15 m",
        "Uses": "Fresh fruit, juice, marmalade",
        "Notes": "Subtropical; sensitive to frost."
    },
    "Banana Plant": {
        "Category": "Herbaceous Plant",
        "Purpose": "Food",
        "Major Regions": "India, Ecuador, Philippines",
        "Avg Height": "3–6 m",
        "Uses": "Fruit, leaves for wrapping food",
        "Notes": "Technically a giant herb."
    },
    "Coconut Tree": {
        "Category": "Palm Tree",
        "Purpose": "Food/Oil",
        "Major Regions": "Indonesia, Philippines, India",
        "Avg Height": "20–30 m",
        "Uses": "Fruit, oil, coir, water",
        "Notes": "Coastal tropical species."
    },
    "Rubber Tree": {
        "Category": "Industrial Tree",
        "Purpose": "Latex",
        "Major Regions": "Thailand, Indonesia, Malaysia",
        "Avg Height": "30–40 m",
        "Uses": "Latex for rubber products",
        "Notes": "Tapped for latex; evergreen."
    },
    "Teak Tree": {
        "Category": "Timber Tree",
        "Purpose": "Wood",
        "Major Regions": "India, Myanmar, Thailand",
        "Avg Height": "20–35 m",
        "Uses": "Furniture, boat building",
        "Notes": "High-quality hardwood with natural oils."
    },
    "Mahogany Tree": {
        "Category": "Timber Tree",
        "Purpose": "Wood",
        "Major Regions": "South America, Africa",
        "Avg Height": "30–40 m",
        "Uses": "Furniture, cabinetry",
        "Notes": "Valuable hardwood; slow-growing."
    },
    "Neem Tree": {
        "Category": "Medicinal Tree",
        "Purpose": "Medicine",
        "Major Regions": "India, Africa",
        "Avg Height": "15–20 m",
        "Uses": "Bark, oil, medicine, pesticides",
        "Notes": "Drought-tolerant; antibacterial."
    },
    "Bamboo": {
        "Category": "Grass Plant",
        "Purpose": "Material",
        "Major Regions": "China, India, SE Asia",
        "Avg Height": "5–30 m",
        "Uses": "Construction, crafts, edible shoots",
        "Notes": "Fastest-growing plant; sustainable."
    },
    "Olive Tree": {
        "Category": "Fruit Tree",
        "Purpose": "Oil/Food",
        "Major Regions": "Mediterranean, Greece, Spain",
        "Avg Height": "8–15 m",
        "Uses": "Olives, olive oil",
        "Notes": "Drought-resistant; long-lived."
    },
    "Coffee Plant": {
        "Category": "Shrub",
        "Purpose": "Beverage",
        "Major Regions": "Brazil, Vietnam, Colombia",
        "Avg Height": "3–7 m",
        "Uses": "Coffee beans",
        "Notes": "Shade-loving understory shrub."
    },
    "Tea Plant": {
        "Category": "Shrub",
        "Purpose": "Beverage",
        "Major Regions": "China, India, Kenya",
        "Avg Height": "2–3 m",
        "Uses": "Tea leaves",
        "Notes": "Plucked every 1–2 weeks after pruning."
    },
    "Avocado Tree": {
        "Category": "Fruit Tree",
        "Purpose": "Food",
        "Major Regions": "Mexico, USA, Peru",
        "Avg Height": "10–20 m",
        "Uses": "Fruit (high in healthy fats)",
        "Notes": "Need well-drained soil; grafted varieties."
    },
    "Banana Passionfruit": {
        "Category": "Vine",
        "Purpose": "Fruit",
        "Major Regions": "Central/South America",
        "Avg Height": "Vines up to 15 m",
        "Uses": "Fruit, ornamental",
        "Notes": "Climbing vine on supports."
    },
    "Lemon Tree": {
        "Category": "Fruit Tree",
        "Purpose": "Food/Flavor",
        "Major Regions": "India, Mexico, Argentina",
        "Avg Height": "7–10 m",
        "Uses": "Fruit, juice, zest",
        "Notes": "Evergreen; sensitive to frost."
    },
    "Maple Tree": {
        "Category": "Shade/Timber Tree",
        "Purpose": "Syrup/Wood",
        "Major Regions": "Canada, USA",
        "Avg Height": "10–45 m",
        "Uses": "Syrup from sap; wood & ornamental",
        "Notes": "Iconic fall foliage."
    },
    "Eucalyptus": {
        "Category": "Timber/Essential Oil",
        "Purpose": "Wood/Oil",
        "Major Regions": "Australia, Brazil, India",
        "Avg Height": "30–60 m",
        "Uses": "Pulp, timber, oil, windbreaks",
        "Notes": "Fast-growing but water-intensive."
    },
    "Pine Tree": {
        "Category": "Conifer",
        "Purpose": "Timber",
        "Major Regions": "North America, Europe",
        "Avg Height": "20–60 m",
        "Uses": "Wood, paper, resin",
        "Notes": "Evergreen coniferous species."
    },
    "Oak Tree": {
        "Category": "Timber/Ornamental",
        "Purpose": "Wood",
        "Major Regions": "Europe, North America",
        "Avg Height": "20–40 m",
        "Uses": "Furniture, flooring, wildlife habitat",
        "Notes": "Long-lived, hardwood with rich ecosystem role."
    },
    # ---- append this to grow PLANT_INFO to 50 entries ----
    # Shade & Timber Trees
    "Birch": {
        "Category": "Timber/Ornamental",
        "Purpose": "Wood",
        "Major Regions": "Russia, Scandinavia, Canada",
        "Avg Height": "15–25 m",
        "Uses": "Furniture, plywood, landscape",
        "Notes": "White bark, fast‑growing in cold climates."
    },
    "Cedar": {
        "Category": "Conifer",
        "Purpose": "Timber/Fragrance",
        "Major Regions": "Lebanon, Himalayas, USA",
        "Avg Height": "20–35 m",
        "Uses": "Aromatic wood, closets, pencils",
        "Notes": "Resin resists insects and decay."
    },
    "Spruce": {
        "Category": "Conifer",
        "Purpose": "Timber/Paper",
        "Major Regions": "Canada, Russia, Nordic countries",
        "Avg Height": "25–50 m",
        "Uses": "Pulp, construction lumber",
        "Notes": "Fast‑growing softwood; musical instruments."
    },
    "Douglas Fir": {
        "Category": "Conifer",
        "Purpose": "Timber",
        "Major Regions": "USA (Pacific NW), New Zealand",
        "Avg Height": "40–60 m",
        "Uses": "Structural beams, plywood",
        "Notes": "High strength‑to‑weight ratio."
    },
    "Redwood": {
        "Category": "Conifer",
        "Purpose": "Timber/Conservation",
        "Major Regions": "California, USA",
        "Avg Height": "70–100 m",
        "Uses": "Durable lumber (limited harvest)",
        "Notes": "Tallest tree species on earth."
    },
    "Poplar": {
        "Category": "Timber",
        "Purpose": "Pulp/Wood",
        "Major Regions": "China, USA",
        "Avg Height": "20–30 m",
        "Uses": "Paper, plywood, biomass",
        "Notes": "Very fast‑growing hardwood."
    },
    "Willow": {
        "Category": "Shade Tree",
        "Purpose": "Ornamental/Bio‑remediation",
        "Major Regions": "China, Europe, USA",
        "Avg Height": "10–25 m",
        "Uses": "Basket weaving, erosion control",
        "Notes": "Weeping habit; roots absorb excess water."
    },
    "Sycamore": {
        "Category": "Shade Tree",
        "Purpose": "Timber",
        "Major Regions": "Europe, North America",
        "Avg Height": "20–40 m",
        "Uses": "Furniture, carving",
        "Notes": "Large lobed leaves; tolerant to pollution."
    },
    "Cherry Blossom": {
        "Category": "Ornamental Tree",
        "Purpose": "Landscape",
        "Major Regions": "Japan, Korea, USA",
        "Avg Height": "5–12 m",
        "Uses": "Aesthetic blooms, festivals",
        "Notes": "Symbol of spring (Sakura)."
    },
    "Jacaranda": {
        "Category": "Ornamental Tree",
        "Purpose": "Landscape",
        "Major Regions": "Brazil, South Africa, India",
        "Avg Height": "10–15 m",
        "Uses": "Purple flower display, shade",
        "Notes": "Deciduous; blooms mid‑summer."
    },

    # Medicinal & Culinary Herbs/Shrubs
    "Aloe Vera": {
        "Category": "Succulent",
        "Purpose": "Medicinal/Cosmetic",
        "Major Regions": "India, Mexico, Egypt",
        "Avg Height": "0.6–1 m",
        "Uses": "Gel for skin care, drinks",
        "Notes": "Drought‑tolerant; easy indoor plant."
    },
    "Basil (Tulsi)": {
        "Category": "Herb",
        "Purpose": "Culinary/Medicinal",
        "Major Regions": "India, Italy",
        "Avg Height": "0.3–0.6 m",
        "Uses": "Flavoring, herbal tea",
        "Notes": "Sacred in Hindu culture."
    },
    "Rosemary": {
        "Category": "Herb",
        "Purpose": "Culinary/Fragrance",
        "Major Regions": "Mediterranean, USA",
        "Avg Height": "0.5–1.5 m",
        "Uses": "Seasoning meat, essential oil",
        "Notes": "Drought‑loving evergreen shrub."
    },
    "Lavender": {
        "Category": "Herb",
        "Purpose": "Aromatherapy",
        "Major Regions": "France, Bulgaria",
        "Avg Height": "0.4–1 m",
        "Uses": "Essential oils, sachets",
        "Notes": "Purple spikes; attracts pollinators."
    },
    "Thyme": {
        "Category": "Herb",
        "Purpose": "Culinary",
        "Major Regions": "Mediterranean",
        "Avg Height": "0.2–0.3 m",
        "Uses": "Seasoning, medicinal tea",
        "Notes": "Low‑growing aromatic ground cover."
    },
    "Oregano": {
        "Category": "Herb",
        "Purpose": "Culinary",
        "Major Regions": "Greece, Turkey",
        "Avg Height": "0.3–0.5 m",
        "Uses": "Pizza herb; dried spice",
        "Notes": "Needs full sun, well‑drained soil."
    },
    "Peppermint": {
        "Category": "Herb",
        "Purpose": "Flavor",
        "Major Regions": "India, USA",
        "Avg Height": "0.5–1 m",
        "Uses": "Tea, confectionery, menthol",
        "Notes": "Hybrid of spearmint & watermint."
    },
    "Chamomile": {
        "Category": "Herb",
        "Purpose": "Tea • Medicinal",
        "Major Regions": "Germany, Egypt",
        "Avg Height": "0.2–0.5 m",
        "Uses": "Calming herbal tea",
        "Notes": "Daisy‑like flowers; mild sedative."
    },
    "Sage": {
        "Category": "Herb",
        "Purpose": "Culinary/Medicinal",
        "Major Regions": "Mediterranean, USA",
        "Avg Height": "0.6 m",
        "Uses": "Stuffing, herbal remedy",
        "Notes": "Silver leaves; antiseptic properties."
    },
    "Bay Laurel": {
        "Category": "Herb/Shrub",
        "Purpose": "Culinary",
        "Major Regions": "Turkey, Greece",
        "Avg Height": "2–10 m",
        "Uses": "Bay leaves; soups & sauces",
        "Notes": "Evergreen; can be pruned into topiary."
    },

    # Indoor & Ornamental Plants
    "Snake Plant": {
        "Category": "Indoor Succulent",
        "Purpose": "Decor/Air Purifier",
        "Major Regions": "Nigeria, Global",
        "Avg Height": "0.3–1 m",
        "Uses": "Indoor decoration",
        "Notes": "Low light tolerant; converts CO₂ at night."
    },
    "Money Plant": {
        "Category": "Indoor Vine",
        "Purpose": "Decor",
        "Major Regions": "SE Asia",
        "Avg Height": "Vine up to 5 m",
        "Uses": "Hanging baskets, hydroponic jars",
        "Notes": "Believed to bring prosperity."
    },
    "Spider Plant": {
        "Category": "Indoor Plant",
        "Purpose": "Decor/Air Purifier",
        "Major Regions": "Global (cultivated)",
        "Avg Height": "0.3 m",
        "Uses": "Indoor pots, hanging",
        "Notes": "Produces plantlets on long stems."
    },
    "Peace Lily": {
        "Category": "Indoor Plant",
        "Purpose": "Decor",
        "Major Regions": "Colombia, Global",
        "Avg Height": "0.4–1 m",
        "Uses": "Indoor shade plant",
        "Notes": "White spathes; tolerates low light."
    },
    "Bougainvillea": {
        "Category": "Ornamental Vine",
        "Purpose": "Landscape",
        "Major Regions": "Brazil, India",
        "Avg Height": "Vine up to 10 m",
        "Uses": "Colorful garden climber",
        "Notes": "Vibrant bracts; drought‑tolerant."
    },
    "Hibiscus": {
        "Category": "Ornamental Shrub",
        "Purpose": "Landscape/Tea",
        "Major Regions": "China, India",
        "Avg Height": "1–3 m",
        "Uses": "Flowers, hibiscus tea",
        "Notes": "Large tropical blooms; attracts butterflies."
    },
    "Jasmine": {
        "Category": "Flowering Vine",
        "Purpose": "Aroma/Ornamental",
        "Major Regions": "India, Egypt",
        "Avg Height": "Vine 2–5 m",
        "Uses": "Perfume, tea scenting",
        "Notes": "Night‑blooming aromatic flowers."
    },
    "Oleander": {
        "Category": "Ornamental Shrub",
        "Purpose": "Landscape",
        "Major Regions": "Mediterranean, India",
        "Avg Height": "2–6 m",
        "Uses": "Roadside hedge",
        "Notes": "Showy flowers; **toxic** if ingested."
    },
    "Dracaena": {
        "Category": "Indoor/Ornamental",
        "Purpose": "Decor",
        "Major Regions": "Africa, Asia",
        "Avg Height": "1–3 m (indoors)",
        "Uses": "Office plant",
        "Notes": "Striped leaves; low maintenance."
    },
    "Rubber Plant": {
        "Category": "Indoor Tree",
        "Purpose": "Decor",
        "Major Regions": "India, Indonesia",
        "Avg Height": "2–3 m (indoors)",
        "Uses": "Indoor focal plant",
        "Notes": "Large glossy leaves; remove dust regularly."
    }
}

CROP_CALENDAR = {
    # ──────────────── CEREALS (10) ────────────────
    "Rice":      {"Category":"Cereal","Sow":"Jun – Jul","Grow":"Aug – Oct","Harvest":"Nov – Dec"},
    "Wheat":     {"Category":"Cereal","Sow":"Nov – Dec","Grow":"Jan – Feb","Harvest":"Mar – Apr"},
    "Maize":     {"Category":"Cereal","Sow":"Jun","Grow":"Jul – Sep","Harvest":"Oct"},
    "Barley":    {"Category":"Cereal","Sow":"Oct – Nov","Grow":"Dec – Jan","Harvest":"Feb – Mar"},
    "Sorghum":   {"Category":"Cereal","Sow":"Jun","Grow":"Jul – Aug","Harvest":"Sep – Oct"},
    "Bajra":     {"Category":"Cereal","Sow":"Jun","Grow":"Jul – Aug","Harvest":"Sep"},
    "Oats":      {"Category":"Cereal","Sow":"Jan","Grow":"Feb – Apr","Harvest":"May"},
    "Rye":       {"Category":"Cereal","Sow":"Oct","Grow":"Nov – Jan","Harvest":"Feb"},
    "Quinoa":    {"Category":"Cereal","Sow":"Apr","Grow":"May – Aug","Harvest":"Sep"},
    "Buckwheat": {"Category":"Cereal","Sow":"Mar","Grow":"Apr – Jun","Harvest":"Jul"},

    # ──────────────── PULSES & LEGUMES (10) ────────────────
    "Chickpea":  {"Category":"Pulse","Sow":"Oct","Grow":"Nov – Jan","Harvest":"Feb – Mar"},
    "Lentil":    {"Category":"Pulse","Sow":"Nov","Grow":"Dec – Feb","Harvest":"Mar"},
    "Pigeon Pea":{"Category":"Pulse","Sow":"Jun","Grow":"Jul – Nov","Harvest":"Dec"},
    "Mung Bean": {"Category":"Pulse","Sow":"Jul","Grow":"Aug – Sep","Harvest":"Oct"},
    "Kidney Bean":{"Category":"Pulse","Sow":"Jun","Grow":"Jul – Sep","Harvest":"Oct"},
    "Blackgram": {"Category":"Pulse","Sow":"Jun","Grow":"Jul – Aug","Harvest":"Sep"},
    "Soybean":   {"Category":"Pulse","Sow":"Jun","Grow":"Jul – Aug","Harvest":"Sep"},
    "Cowpea":    {"Category":"Pulse","Sow":"Jul","Grow":"Aug – Sep","Harvest":"Oct"},
    "Faba Bean": {"Category":"Pulse","Sow":"Nov","Grow":"Dec – Feb","Harvest":"Mar"},
    "Pea":       {"Category":"Pulse","Sow":"Oct","Grow":"Nov – Dec","Harvest":"Jan"},

    # ──────────────── OILSEEDS & FIBER (10) ────────────────
    "Groundnut": {"Category":"Oilseed","Sow":"Jun","Grow":"Jul – Aug","Harvest":"Sep"},
    "Mustard":   {"Category":"Oilseed","Sow":"Oct","Grow":"Nov – Jan","Harvest":"Feb"},
    "Sesame":    {"Category":"Oilseed","Sow":"Jun","Grow":"Jul – Aug","Harvest":"Sep"},
    "Sunflower": {"Category":"Oilseed","Sow":"Jan","Grow":"Feb – Apr","Harvest":"May"},
    "Castor":    {"Category":"Oilseed","Sow":"Jul","Grow":"Aug – Nov","Harvest":"Dec"},
    "Linseed":   {"Category":"Oilseed","Sow":"Oct","Grow":"Nov – Jan","Harvest":"Feb"},
    "Cotton":    {"Category":"Fiber","Sow":"Apr","Grow":"May – Aug","Harvest":"Sep – Oct"},
    "Jute":      {"Category":"Fiber","Sow":"Mar","Grow":"Apr – Jun","Harvest":"Jul – Aug"},
    "Coconut":   {"Category":"Plantation","Sow":"Any (seedlings)","Grow":"Perennial","Harvest":"All year"},
    "Oil Palm":  {"Category":"Plantation","Sow":"Seedlings Feb","Grow":"Perennial","Harvest":"All year"},

    # ──────────────── VEGETABLES (10) ────────────────
    "Tomato":    {"Category":"Vegetable","Sow":"Jan & Jul","Grow":"Feb – Mar / Aug – Sep","Harvest":"Apr – May / Oct – Nov"},
    "Potato":    {"Category":"Vegetable","Sow":"Oct","Grow":"Nov – Jan","Harvest":"Feb"},
    "Onion":     {"Category":"Vegetable","Sow":"Oct","Grow":"Nov – Jan","Harvest":"Mar"},
    "Garlic":    {"Category":"Vegetable","Sow":"Oct","Grow":"Nov – Jan","Harvest":"Mar"},
    "Brinjal":   {"Category":"Vegetable","Sow":"Jun & Jan","Grow":"Jul – Aug / Feb – Mar","Harvest":"Sep / May"},
    "Okra":      {"Category":"Vegetable","Sow":"Feb & Jun","Grow":"Mar – Apr / Jul – Aug","Harvest":"May / Sep"},
    "Cabbage":   {"Category":"Vegetable","Sow":"Aug","Grow":"Sep – Nov","Harvest":"Dec – Jan"},
    "Carrot":    {"Category":"Vegetable","Sow":"Aug","Grow":"Sep – Nov","Harvest":"Dec – Jan"},
    "Chilli":    {"Category":"Vegetable","Sow":"Jun","Grow":"Jul – Aug","Harvest":"Sep – Oct"},
    "Spinach":   {"Category":"Vegetable","Sow":"Oct","Grow":"Nov – Dec","Harvest":"Jan"},

    # ──────────────── FRUITS (10) ────────────────
    "Banana":    {"Category":"Fruit","Sow":"Year‑round (tissue culture)","Grow":"10 months","Harvest":"12 months"},
    "Mango":     {"Category":"Fruit","Sow":"Graft in Jul–Aug","Grow":"3–4 yrs","Harvest":"Mar – May"},
    "Papaya":    {"Category":"Fruit","Sow":"Feb & Jun","Grow":"6 months","Harvest":"8–9 months"},
    "Pomegranate":{"Category":"Fruit","Sow":"Jun","Grow":"18 months","Harvest":"20 months"},
    "Orange":    {"Category":"Fruit","Sow":"Seed/Graft Aug","Grow":"3 yrs","Harvest":"Nov – Jan"},
    "Grapes":    {"Category":"Fruit","Sow":"Cuttings Jan","Grow":"1 yr","Harvest":"Feb – Mar"},
    "Watermelon":{"Category":"Fruit","Sow":"Jan","Grow":"Feb – Apr","Harvest":"May"},
    "Pineapple": {"Category":"Fruit","Sow":"Slips Apr","Grow":"15 months","Harvest":"Jun – Aug"},
    "Guava":     {"Category":"Fruit","Sow":"Jun","Grow":"18 months","Harvest":"Aug – Oct"},
    "Apple":     {"Category":"Fruit","Sow":"Sapling Feb","Grow":"3 yrs","Harvest":"Sep – Oct"},

        # ──────────────── SPICES & HERBS (10) ────────────────
    "Turmeric":   {"Category":"Spice","Sow":"May–Jun","Grow":"Jul–Jan","Harvest":"Feb–Mar"},
    "Ginger":     {"Category":"Spice","Sow":"Apr–May","Grow":"Jun–Dec","Harvest":"Jan–Feb"},
    "Coriander":  {"Category":"Spice","Sow":"Oct","Grow":"Nov–Jan","Harvest":"Feb"},
    "Cumin":      {"Category":"Spice","Sow":"Nov","Grow":"Dec–Feb","Harvest":"Mar"},
    "Fenugreek":  {"Category":"Spice","Sow":"Oct","Grow":"Nov–Jan","Harvest":"Feb"},
    "Cardamom":   {"Category":"Spice","Sow":"Jun","Grow":"Jul–Oct","Harvest":"Nov–Jan"},
    "Clove":      {"Category":"Spice","Sow":"Jul–Aug","Grow":"Perennial","Harvest":"Mar–May"},
    "Black Pepper":{"Category":"Spice","Sow":"Jun","Grow":"Perennial","Harvest":"Jan–Mar"},
    "Bay Leaf":   {"Category":"Spice","Sow":"Monsoon","Grow":"Perennial","Harvest":"All year"},
    "Mustard Leaf":{"Category":"Leafy Green","Sow":"Oct","Grow":"Nov–Dec","Harvest":"Jan"},

    # ──────────────── PLANTATION / EXPORT CROPS (10) ────────────────
    "Tea":        {"Category":"Plantation","Sow":"Seedlings Jun","Grow":"Perennial","Harvest":"Year-round"},
    "Coffee":     {"Category":"Plantation","Sow":"Seedlings Jun","Grow":"Perennial","Harvest":"Nov–Mar"},
    "Rubber":     {"Category":"Plantation","Sow":"Monsoon","Grow":"5–7 years","Harvest":"After 7 years"},
    "Cocoa":      {"Category":"Plantation","Sow":"Jun–Jul","Grow":"Perennial","Harvest":"Oct–Mar"},
    "Arecanut":   {"Category":"Plantation","Sow":"Monsoon","Grow":"5–7 years","Harvest":"After 7 years"},
    "Tobacco":    {"Category":"Commercial","Sow":"Oct–Nov","Grow":"Dec–Feb","Harvest":"Mar–Apr"},
    "Betel Leaf": {"Category":"Plantation","Sow":"Mar–Apr","Grow":"Climber","Harvest":"All year"},
    "Silk Mulberry":{"Category":"Commercial","Sow":"Any","Grow":"Perennial","Harvest":"Every 2 months"},
    "Sugarcane":  {"Category":"Commercial","Sow":"Jan–Mar","Grow":"12–14 months","Harvest":"Feb–Apr"},
    "Vanilla":    {"Category":"Plantation","Sow":"Cuttings Jun","Grow":"Climber","Harvest":"Jan–Mar"},

    # ──────────────── TUBERS & ROOTS (10) ────────────────
    "Sweet Potato":{"Category":"Root","Sow":"Jun–Jul","Grow":"Aug–Nov","Harvest":"Dec"},
    "Tapioca":    {"Category":"Root","Sow":"Apr–May","Grow":"May–Oct","Harvest":"Nov"},
    "Yam":        {"Category":"Root","Sow":"May","Grow":"Jun–Oct","Harvest":"Nov"},
    "Beetroot":   {"Category":"Root","Sow":"Aug","Grow":"Sep–Nov","Harvest":"Dec"},
    "Radish":     {"Category":"Root","Sow":"Sep–Oct","Grow":"Oct–Dec","Harvest":"Jan"},
    "Turnip":     {"Category":"Root","Sow":"Oct","Grow":"Nov–Dec","Harvest":"Jan"},
    "Colocasia":  {"Category":"Root","Sow":"Apr–May","Grow":"Jun–Oct","Harvest":"Nov"},
    "Carrot (Hill)":{"Category":"Root","Sow":"Apr","Grow":"May–Jul","Harvest":"Aug"},
    "Jerusalem Artichoke":{"Category":"Root","Sow":"Feb","Grow":"Mar–Jul","Harvest":"Aug"},
    "Arrowroot":  {"Category":"Root","Sow":"Apr–May","Grow":"Jun–Nov","Harvest":"Dec"},

    # ──────────────── FORAGE & FODDER (10) ────────────────
    "Berseem":    {"Category":"Fodder","Sow":"Oct","Grow":"Nov–Feb","Harvest":"Mar"},
    "Lucerne":    {"Category":"Fodder","Sow":"Sep","Grow":"Oct–Feb","Harvest":"Mar"},
    "Napier Grass":{"Category":"Fodder","Sow":"Mar","Grow":"Apr–Jul","Harvest":"Aug"},
    "Cowpea (Fodder)":{"Category":"Fodder","Sow":"Jun","Grow":"Jul–Aug","Harvest":"Sep"},
    "Sorghum (Fodder)":{"Category":"Fodder","Sow":"Jun","Grow":"Jul–Sep","Harvest":"Oct"},
    "Maize (Fodder)":{"Category":"Fodder","Sow":"Jul","Grow":"Aug–Sep","Harvest":"Oct"},
    "Guar":       {"Category":"Fodder","Sow":"Jul","Grow":"Aug–Sep","Harvest":"Oct"},
    "Sunhemp":    {"Category":"Fodder","Sow":"Jun","Grow":"Jul–Sep","Harvest":"Oct"},
    "Stylo":      {"Category":"Fodder","Sow":"Jul","Grow":"Aug–Oct","Harvest":"Nov"},
    "Dinanath Grass":{"Category":"Fodder","Sow":"Jul","Grow":"Aug–Sep","Harvest":"Oct"},
}



# ----------------------------------------------
# 2. DICTIONARIES & CONSTANTS
# ----------------------------------------------
CROP_LABELS = {i+1: c for i, c in enumerate(crop_labels)}
FERT_LABELS = {k: k for k in fert_labels}
# friendly names & tips + example usage
FERT_INFO = {
    "DAP": {
        "tip": "Rich in phosphorus – ideal for early root development.",
        "example": "Apply as a basal dose before sowing wheat or rice."
    },
    "Urea": {
        "tip": "High nitrogen – split doses & irrigate after application.",
        "example": "Side‑dress leafy vegetables during vegetative stage."
    },
    "17-17-17": {
        "tip": "Balanced mix – safe for any growth stage.",
        "example": "Top‑dress mixed vegetable fields mid‑season."
    },
    "10-26-26": {
        "tip": "High P & K – supports fruit‑set and root health.",
        "example": "Use on tomato plants at flowering stage."
    },
    "14-35-14": {
        "tip": "Phosphate‑rich – boosts flowering.",
        "example": "Apply to sunflower during bud initiation."
    },
    "20-20": {
        "tip": "Starter fertilizer – uniform seedling growth.",
        "example": "Incorporate at sowing for cereals/pulses."
    },
    "28-28": {
        "tip": "High nitrogen – rapid vegetative growth.",
        "example": "Use early on leafy greens; stop before flowering."
    }

}

SOIL_OPTIONS = list(fert_encoders["Soil Type"].classes_)
CROP_OPTIONS = list(fert_encoders["Crop Type"].classes_)

# ----------------------------------------------
# 3. UI
# ----------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["🌱 Crop Recommendation", "🧪 Fertilizer Advisor","🧮 Soil Health Analyzer" , "📅 Crop Calendar", "🌿🌳 Plant Encyclopedia", "📘 Crop Encyclopedia"]
)

with tab1:
    st.header("🌱 Crop Recommendation")
    c1, c2 = st.columns(2)
    with c1:
        N  = st.number_input("Nitrogen (N)", 0, 140, 80)
        P  = st.number_input("Phosphorus (P)", 0, 145, 60)
        K  = st.number_input("Potassium (K)", 0, 205, 60)
        temp = st.slider("Temperature (°C)", 0.0, 50.0, 25.0)
    with c2:
        hum  = st.slider("Humidity (%)", 0.0, 100.0, 65.0)
        ph   = st.slider("Soil pH", 3.5, 9.5, 6.5)
        rain = st.slider("Rainfall (mm)", 0.0, 400.0, 120.0)

    if st.button("🚀 Predict Best Crop"):
        inputs = np.array([[N, P, K, temp, hum, ph, rain]])
        pred = crop_model.predict(crop_scaler.transform(inputs))[0]
        st.success(f"✅ Recommended Crop: **{pred}**")

with tab2:
    st.header("🧪 Fertilizer Advisor")
    f1, f2 = st.columns(2)
    with f1:
        soil = st.selectbox("Soil Type", SOIL_OPTIONS)
        crop = st.selectbox("Crop Type", CROP_OPTIONS)
        N2 = st.slider("Nitrogen (N)", 0, 140, 70)
        P2 = st.slider("Phosphorus (P)", 0, 145, 60)
        K2 = st.slider("Potassium (K)", 0, 205, 60)
    with f2:
        temp2  = st.slider("Temperature (°C)", 5.0, 50.0, 25.0)
        hum2   = st.slider("Humidity (%)", 10.0, 100.0, 65.0)
        moist2 = st.slider("Soil Moisture (%)", 0.0, 100.0, 40.0)

    if st.button("🔬 Suggest Fertilizer"):
        raw = np.array([[temp2, hum2, moist2,
                         fert_encoders["Soil Type"].transform([soil])[0],
                         fert_encoders["Crop Type"].transform([crop])[0],
                         N2, K2, P2]])

        vec_scaled = raw.astype(float)
        vec_scaled[:, [0,1,2,5,6,7]] = fert_scaler.transform(vec_scaled[:, [0,1,2,5,6,7]])

        pred = fert_model.predict(vec_scaled)[0]
        st.success(f"🧪 Recommended Fertilizer: **{pred}**")
        info = FERT_INFO.get(pred)
        if info:
            st.info(f"💡 **Tip:** {info['tip']}")
            st.write(f"📌 **Example:** {info['example']}")

# ------------------- 🧮  SOIL HEALTH ANALYZER TAB --------------------
with tab3:
    st.header("🧮 Soil Health Analyzer")

    s1, s2, s3 = st.columns(3)
    with s1:
        pH = st.number_input("Soil pH", 3.5, 9.5, 6.5, step=0.1)
        ec = st.number_input("Electrical Conductivity (dS/m)", 0.0, 4.0, 0.5, step=0.1)
    with s2:
        N  = st.number_input("Nitrogen (kg/ha)", 0, 300, 80)
        P  = st.number_input("Phosphorus (kg/ha)", 0, 300, 60)
    with s3:
        K  = st.number_input("Potassium (kg/ha)", 0, 300, 80)
        moisture = st.slider("Moisture (%)", 0, 100, 35)

    if st.button("🔍 Analyze Soil"):
        # --- simple rule‑based diagnostics ---
        ph_status = ("Acidic" if pH < 6 else
                     "Neutral" if 6 <= pH <= 7.5 else
                     "Alkaline")

        ec_status = ("Low salts" if ec < 1 else
                     "Medium salts" if ec < 2 else
                     "High salinity")

        def nutrient_status(val):
            return ("Low" if val < 50 else
                    "Adequate" if val < 150 else
                    "High")

        n_stat = nutrient_status(N)
        p_stat = nutrient_status(P)
        k_stat = nutrient_status(K)

        # --- display results ---
        st.subheader("🧾 Soil Health Report")
        st.markdown(f"""
| Parameter | Value | Status |
|-----------|-------|--------|
| **pH** | {pH} | {ph_status} |
| **EC** | {ec} dS/m | {ec_status} |
| **Nitrogen** | {N} kg/ha | {n_stat} |
| **Phosphorus** | {P} kg/ha | {p_stat} |
| **Potassium** | {K} kg/ha | {k_stat} |
| **Moisture** | {moisture}% | {'Dry' if moisture<25 else 'Optimal' if moisture<60 else 'Wet'} |
""")

        # --- brief recommendations ---
        rec = []
        if ph_status == "Acidic":   rec.append("Add lime to raise pH.")
        if ph_status == "Alkaline": rec.append("Incorporate sulfur or organic matter to lower pH.")
        if n_stat == "Low":         rec.append("Apply nitrogenous fertilizer (e.g., Urea).")
        if p_stat == "Low":         rec.append("Add DAP or SSP for phosphorus.")
        if k_stat == "Low":         rec.append("Apply MOP or 17‑17‑17.")
        if ec_status == "High salinity": rec.append("Improve drainage or leach salts with fresh water.")

        if rec:
            st.info("💡 **Recommendations:**\n- " + "\n- ".join(rec))
        else:
            st.success("🎉 Soil is in good condition! Maintain current practices.")

# ----------------  📘  CROP ENCYCLOPEDIA TAB  ----------------
with tab6:
    st.header("📘 Crop Encyclopedia (In‑Code)")

    # pick category first
    categories = sorted({v["Category"] for v in CROP_INFO.values()})
    sel_cat = st.selectbox("Select a category", categories)

    # crops filtered by category
    crop_list = [c for c,v in CROP_INFO.items() if v["Category"] == sel_cat]
    sel_crop = st.selectbox("Select a crop", sorted(crop_list))

    # display details
    data = CROP_INFO[sel_crop]
    yield_text = data.get("Avg Yield", "–")
    value_text = data.get("Value", "–")
    notes_text = data.get("Notes", "–")

    st.subheader(sel_crop)
    st.markdown(f"""
**Category:** {data['Category']}  
**Purpose:** {data['Purpose']}  
**Major Regions:** {data['Major Regions']}  
**Average Yield:** {yield_text}  
**Market Value:** {value_text}  
**Notes:** {notes_text}
""")

with tab5:
    st.header("🌿 Plant & Tree Encyclopedia")
    cats = sorted({v["Category"] for v in PLANT_INFO.values()})
    sel_cat = st.selectbox("Select plant type", cats)
    plants = [p for p, v in PLANT_INFO.items() if v["Category"] == sel_cat]
    sel_plant = st.selectbox("Select a plant/tree", sorted(plants))
    info = PLANT_INFO[sel_plant]
    st.subheader(sel_plant)
    st.markdown(f"""
**Category:** {info['Category']}  
**Purpose:** {info['Purpose']}  
**Major Regions:** {info['Major Regions']}  
**Avg Height:** {info['Avg Height']}  
**Uses:** {info['Uses']}  
**Notes:** {info['Notes']}
""")
    
with tab4:
    st.header("📅 Crop Calendar")
    
    # 1. Choose a category
    cal_categories = sorted({info["Category"] for info in CROP_CALENDAR.values()})
    sel_cat = st.selectbox("Select category", cal_categories)

    # 2. Filter crops by category
    filtered_crops = [c for c, data in CROP_CALENDAR.items() if data["Category"] == sel_cat]
    sel_crop = st.selectbox("Select crop", sorted(filtered_crops))

    # 3. Show calendar details
    data = CROP_CALENDAR[sel_crop]
    st.subheader(sel_crop)
    st.markdown(f"""
**Category:** {data["Category"]}  
**Sowing Period:** {data["Sow"]}  
**Growing Phase:** {data["Grow"]}  
**Harvest Time:** {data["Harvest"]}
""")