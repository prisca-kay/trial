"""
Road Safety Intelligence Dashboard
Built from the supplied STATS19 Streamlit app and notebook.

Prediction is a research demonstration using the UK-trained STATS19 CatBoost
severity model. It is not validated for Kenyan roads or operational decisions.
"""
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin

# The notebook imports these same classes when fitting and exporting the model.
from road_safety_model_support import (
    CategoricalStringTransformer,
    CatBoostClassifierAdapter,
    DecisionAdjustedModel,
)

# ---------------------------------------------------------------------
# Page and visual style
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="UK Road Accident Analysis | STATS19",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root { --navy:#17324D; --teal:#167D7F; --gold:#D5A64A; --ink:#263746; }
    .stApp { background: #F5F7FA; color: var(--ink); }
    [data-testid="stHeader"] { background: rgba(245,247,250,.92); }
    [data-testid="stSidebar"] { background: #EAF0F4; border-right: 1px solid #D9E2EC; }
    h1, h2, h3 { color: #17324D; }
    div[data-testid="stMetric"] {
        background: #FFFFFF; border: 1px solid #E0E7ED; padding: 15px 17px;
        border-radius: 12px; box-shadow: 0 2px 8px rgba(23,50,77,.04);
    }
    div[data-testid="stMetric"] label { color: #526777; }
    .hero {
        background: linear-gradient(110deg,#17324D 0%,#176B70 100%);
        padding: 27px 30px; border-radius: 16px; color: white; margin-bottom: 18px;
    }
    .hero h1 { color: white; margin: 0 0 6px 0; font-size: 2rem; }
    .hero p { color: #E6F2F2; margin: 0; font-size: 1rem; }
    .soft-card {
        background: white; border: 1px solid #E0E7ED; padding: 17px;
        border-radius: 12px; margin: 8px 0;
    }
    .warning-card {
        background: #FFF7E6; border-left: 5px solid #D5A64A;
        padding: 14px 16px; border-radius: 8px; color: #624A1E;
    }
    .small-muted { color: #627789; font-size: .88rem; }
    div.stButton > button[kind="primary"] {
        background: #167D7F; border-color: #167D7F; border-radius: 9px;
    }
    div[data-testid="stTabs"] button[aria-selected="true"] {
        color: #167D7F; border-bottom-color: #167D7F;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"

SEVERITY = {1: "Fatal", 2: "Serious", 3: "Slight"}
URBAN_RURAL = {1: "Urban", 2: "Rural", 3: "Unallocated"}
SEX = {1: "Male", 2: "Female", 3: "Not applicable", 9: "Unknown", -1: "Unknown"}
CASUALTY_CLASS = {1: "Driver or rider", 2: "Passenger", 3: "Pedestrian"}
LIGHT = {
    1: "Daylight", 4: "Darkness - lights lit", 5: "Darkness - lights unlit",
    6: "Darkness - no lighting", 7: "Darkness - lighting unknown", -1: "Unknown",
}
ROAD_TYPE = {
    1: "Roundabout", 2: "One-way street", 3: "Dual carriageway",
    6: "Single carriageway", 7: "Slip road", 9: "Unknown",
}
AGE_BAND = {
    1: "0–5", 2: "6–10", 3: "11–15", 4: "16–20", 5: "21–25",
    6: "26–35", 7: "36–45", 8: "46–55", 9: "56–65",
    10: "66–75", 11: "Over 75", -1: "Unknown",
}


def find_data_file(stem):
    """Find supplied CSV whether stored in data/ or project root, with optional (1) suffix."""
    candidates = [
        DATA_DIR / f"{stem}.csv",
        DATA_DIR / f"{stem}(1).csv",
        ROOT / f"{stem}.csv",
        ROOT / f"{stem}(1).csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = list(DATA_DIR.glob(f"{stem}*.csv")) if DATA_DIR.exists() else []
    matches += list(ROOT.glob(f"{stem}*.csv"))
    return matches[0] if matches else None


@st.cache_data(show_spinner="Loading road safety records...")
def load_data():
    paths = {
        "collision": find_data_file("dft-road-casualty-statistics-collision-2025"),
        "casualty": find_data_file("dft-road-casualty-statistics-casualty-2025"),
        "vehicle": find_data_file("dft-road-casualty-statistics-vehicle-2025"),
    }
    missing = [name for name, path in paths.items() if path is None]
    if missing:
        raise FileNotFoundError(
            "Missing CSV file(s): " + ", ".join(missing)
            + ". Put the three supplied 2025 CSV files in the project's data folder."
        )

    collisions = pd.read_csv(paths["collision"], low_memory=False)
    casualties = pd.read_csv(paths["casualty"], low_memory=False)
    vehicles = pd.read_csv(paths["vehicle"], low_memory=False)

    collisions["date"] = pd.to_datetime(collisions["date"], dayfirst=True, errors="coerce")
    collisions["collision_severity_label"] = (
        pd.to_numeric(collisions["collision_severity"], errors="coerce")
        .map(SEVERITY).fillna("Unknown")
    )
    collisions["area_label"] = (
        pd.to_numeric(collisions["urban_or_rural_area"], errors="coerce")
        .map(URBAN_RURAL).fillna("Unknown")
    )
    collisions["road_type_label"] = (
        pd.to_numeric(collisions["road_type"], errors="coerce")
        .map(ROAD_TYPE).fillna("Other / coded value")
    )
    collisions["light_label"] = (
        pd.to_numeric(collisions["light_conditions"], errors="coerce")
        .map(LIGHT).fillna("Other / coded value")
    )
    casualties["casualty_severity_label"] = (
        pd.to_numeric(casualties["casualty_severity"], errors="coerce")
        .map(SEVERITY).fillna("Unknown")
    )
    casualties["casualty_class_label"] = (
        pd.to_numeric(casualties["casualty_class"], errors="coerce")
        .map(CASUALTY_CLASS).fillna("Unknown")
    )
    casualties["age_band_label"] = (
        pd.to_numeric(casualties["age_band_of_casualty"], errors="coerce")
        .map(AGE_BAND).fillna("Unknown")
    )
    casualties["sex_label"] = (
        pd.to_numeric(casualties["sex_of_casualty"], errors="coerce")
        .map(SEX).fillna("Unknown")
    )
    vehicles["driver_sex_label"] = (
        pd.to_numeric(vehicles["sex_of_driver"], errors="coerce")
        .map(SEX).fillna("Unknown")
    )
    vehicles["driver_age_band_label"] = (
        pd.to_numeric(vehicles["age_band_of_driver"], errors="coerce")
        .map(AGE_BAND).fillna("Unknown")
    )
    return collisions, casualties, vehicles


@st.cache_resource(show_spinner="Loading the CatBoost research model...")
def load_prediction_assets():
    model_path = MODEL_DIR / "stats19_severity_model.joblib"
    contract_path = MODEL_DIR / "model_contract.json"
    metadata_path = MODEL_DIR / "metadata.json"
    missing = [p.name for p in [model_path, contract_path, metadata_path] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Model files not found: " + ", ".join(missing)
            + ". Run the notebook through Sections 9.2 and 9.3, then copy its models folder here."
        )
    with open(contract_path, "r", encoding="utf-8") as f:
        contract = json.load(f)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    model = joblib.load(model_path)
    return model, contract, metadata


def humanize(value):
    labels = {
        "speed_limit": "Speed limit (mph)",
        "engine_cc_mean": "Average engine capacity (cc)",
        "driver_age_mean": "Average driver age (years)",
        "driver_age_min": "Minimum driver age (years)",
        "vehicle_age_mean": "Average vehicle age (years)",
        "vehicle_age_max": "Maximum vehicle age (years)",
        "hour": "Hour of day (0–23)",
        "number_of_vehicles": "Number of vehicles",
        "vehicle_records_count": "Vehicle records count",
        "vulnerable_road_user_count": "Vulnerable road-user count",
    }
    return labels.get(value, re.sub(r"_+", " ", str(value)).strip().capitalize())


def render_prediction_form(model, contract, metadata):
    st.subheader("Collision severity prediction")
    st.markdown(
        '<div class="warning-card"><b>Research demonstration only.</b> '
        'This model was trained on UK STATS19 records. It has not been validated '
        'or calibrated for Kenyan roads. Do not use its output for emergency response, '
        'enforcement, insurance decisions or operational road-safety decisions.</div>',
        unsafe_allow_html=True,
    )
    st.write(
        "Enter the collision and road-context features below. The model estimates "
        "the recorded severity class: Slight, Serious or Fatal."
    )

    feature_specs = contract.get("features", [])
    if not feature_specs:
        st.error("The model contract contains no feature definitions. Regenerate it from the notebook.")
        return

    with st.form("severity_prediction_form"):
        values = {}
        groups = {
            "Road and junction": [],
            "Weather, lighting and time": [],
            "Vehicle and driver": [],
            "Other road-user information": [],
        }
        for spec in feature_specs:
            name = spec["feature"]
            low = name.lower()
            if any(k in low for k in ["road", "junction", "pedestrian_crossing", "carriageway", "trunk"]):
                group = "Road and junction"
            elif any(k in low for k in ["light", "weather", "surface", "month", "hour", "day", "weekend", "time"]):
                group = "Weather, lighting and time"
            elif any(k in low for k in ["vehicle", "driver", "engine", "motorcycle", "goods"]):
                group = "Vehicle and driver"
            else:
                group = "Other road-user information"
            groups[group].append(spec)

        for group_name, specs in groups.items():
            if not specs:
                continue
            with st.expander(group_name, expanded=(group_name == "Road and junction")):
                cols = st.columns(2)
                for i, spec in enumerate(specs):
                    name = spec["feature"]
                    label = humanize(name)
                    with cols[i % 2]:
                        if spec.get("dtype") == "numeric":
                            default = float(spec.get("default", 0))
                            minimum = float(spec.get("min", min(0.0, default)))
                            maximum = float(spec.get("max", max(100.0, default)))
                            if maximum <= minimum:
                                maximum = minimum + 1.0
                            default = min(max(default, minimum), maximum)
                            step = float(spec.get("step", 1 if float(step_is_int(minimum, maximum)) else 0.1))
                            values[name] = st.number_input(
                                label, min_value=minimum, max_value=maximum,
                                value=default, step=step, key=f"pred_{name}",
                                help=f"Training-data input range: {minimum:g} to {maximum:g}. Use the source model's units and coding.",
                            )
                        else:
                            options = [str(x) for x in spec.get("category_examples", [])]
                            if "Missing" not in options:
                                options.append("Missing")
                            if not options:
                                options = ["Missing"]
                            default = str(spec.get("default", options[0]))
                            if default not in options:
                                options.insert(0, default)
                            values[name] = st.selectbox(
                                label, options=options,
                                index=options.index(default), key=f"pred_{name}",
                                help="Choose a category seen in the UK STATS19 training data. These are not Kenyan category mappings.",
                            )

        submitted = st.form_submit_button("Predict collision severity", type="primary", use_container_width=True)

    if submitted:
        input_frame = pd.DataFrame([values], columns=[x["feature"] for x in feature_specs])
        try:
            prediction = str(model.predict(input_frame)[0])
            raw_probabilities = model.predict_proba(input_frame)[0]
            classes = [str(x) for x in model.classes_]
            st.divider()
            st.markdown("### Model result")
            st.metric("Predicted severity", prediction)
            score_table = pd.DataFrame({
                "Severity class": classes,
                "Model score (not calibrated confidence)": raw_probabilities,
            }).sort_values("Model score (not calibrated confidence)", ascending=False)
            st.bar_chart(score_table.set_index("Severity class"))
            st.caption(
                "The displayed scores are the underlying model probabilities. "
                "The notebook's class multipliers adjust the final predicted label; "
                "the scores should not be interpreted as calibrated real-world probabilities."
            )
            st.info(
                f"Model: {metadata.get('model', 'CatBoost depth-7')}. "
                "This is a UK-data research output, not a Kenyan road-risk assessment."
            )
        except Exception as exc:
            st.error(f"Prediction could not be completed: {exc}")
            st.caption("Check that the notebook model, contract and app use the same version.")


def step_is_int(low, high):
    return abs(low - round(low)) < 1e-8 and abs(high - round(high)) < 1e-8


# ---------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
      <h1>🚦 UK Road Accident Analysis &amp; Severity Prediction</h1>
      <p>Explore UK STATS19 collision patterns, understand road-user impacts and demonstrate
      a CatBoost collision-severity model — with a research pathway towards Kenya.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption("Data source: UK Department for Transport STATS19 2025 collision, casualty and vehicle records.")

try:
    collisions, casualties, vehicles = load_data()
except Exception as exc:
    st.error(f"Could not load the road safety data: {exc}")
    st.stop()

# ---------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------
st.sidebar.title("Dashboard filters")
st.sidebar.caption("Choose filters to update the analysis.")
valid_dates = collisions["date"].dropna()
if valid_dates.empty:
    st.error("The collision date field could not be read.")
    st.stop()

min_date, max_date = valid_dates.min().date(), valid_dates.max().date()
date_range = st.sidebar.date_input(
    "Collision date range", value=(min_date, max_date),
    min_value=min_date, max_value=max_date,
)
if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date = end_date = date_range

severity_options = ["Fatal", "Serious", "Slight"]
selected_severity = st.sidebar.multiselect(
    "Recorded severity", severity_options, default=severity_options
)
area_options = sorted(collisions["area_label"].dropna().unique().tolist())
selected_areas = st.sidebar.multiselect("Area type", area_options, default=area_options)
speed_values = pd.to_numeric(collisions["speed_limit"], errors="coerce")
speed_options = sorted(int(x) for x in speed_values.dropna().unique() if x > 0)
selected_speeds = st.sidebar.multiselect("Speed limit (mph)", speed_options, default=speed_options)
road_options = sorted(collisions["road_type_label"].dropna().unique().tolist())
selected_roads = st.sidebar.multiselect("Road type", road_options, default=road_options)

filtered = collisions[
    collisions["date"].dt.date.between(start_date, end_date)
    & collisions["collision_severity_label"].isin(selected_severity)
    & collisions["area_label"].isin(selected_areas)
    & speed_values.isin(selected_speeds)
    & collisions["road_type_label"].isin(selected_roads)
].copy()

collision_ids = set(filtered["collision_index"].dropna().astype(str))
casualty_match = casualties[casualties["collision_index"].astype(str).isin(collision_ids)].copy()
vehicle_match = vehicles[vehicles["collision_index"].astype(str).isin(collision_ids)].copy()

st.sidebar.divider()
st.sidebar.write(f"**Matching collisions:** {len(filtered):,}")
if st.sidebar.button("Reset filters", use_container_width=True):
    st.rerun()

# ---------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------
m1, m2, m3, m4 = st.columns(4)
m1.metric("Collisions", f"{len(filtered):,}")
m2.metric("Casualties", f"{len(casualty_match):,}")
m3.metric("Vehicles involved", f"{len(vehicle_match):,}")
m4.metric("Fatal collisions", f"{int((filtered['collision_severity_label'] == 'Fatal').sum()):,}")

if filtered.empty:
    st.warning("No records match these filters. Widen the date range or select more options.")
    st.stop()

tab_overview, tab_patterns, tab_people, tab_vehicles, tab_predict, tab_kenya, tab_data = st.tabs(
    ["Overview", "Collision patterns", "People", "Vehicles", "Prediction lab", "Kenya road safety", "Data explorer"]
)

with tab_overview:
    st.subheader("Road safety overview")
    left, right = st.columns(2)
    with left:
        st.write("**Collisions by recorded severity**")
        counts = filtered["collision_severity_label"].value_counts().reindex(
            ["Fatal", "Serious", "Slight"], fill_value=0
        )
        st.bar_chart(counts)
    with right:
        st.write("**Collisions by area**")
        st.bar_chart(filtered["area_label"].value_counts())
    st.write("**Collision trend by month**")
    by_month = (
        filtered.assign(month=filtered["date"].dt.to_period("M").astype(str))
        .groupby("month").size().rename("Collisions")
    )
    st.line_chart(by_month)
    st.write("**Recorded collision locations (UK)**")
    map_data = filtered.dropna(subset=["latitude", "longitude"])[["latitude", "longitude"]].copy()
    map_data["latitude"] = pd.to_numeric(map_data["latitude"], errors="coerce")
    map_data["longitude"] = pd.to_numeric(map_data["longitude"], errors="coerce")
    map_data = map_data[
        map_data["latitude"].between(49, 61.5)
        & map_data["longitude"].between(-8.5, 2.5)
    ].dropna()
    if not map_data.empty:
        if len(map_data) > 5000:
            map_data = map_data.sample(5000, random_state=42)
            st.caption("Map shows a reproducible sample of up to 5,000 filtered UK records.")
        st.map(map_data, latitude="latitude", longitude="longitude")
    else:
        st.info("No valid UK map coordinates for these filters.")

with tab_patterns:
    st.subheader("Where and when collisions are recorded")
    c1, c2 = st.columns(2)
    with c1:
        st.write("**Speed limit (mph)**")
        speeds = pd.to_numeric(filtered["speed_limit"], errors="coerce")
        st.bar_chart(speeds[speeds > 0].value_counts().sort_index())
    with c2:
        st.write("**Road type**")
        st.bar_chart(filtered["road_type_label"].value_counts())
    st.write("**Lighting conditions**")
    st.bar_chart(filtered["light_label"].value_counts())
    st.caption("These are counts within the selected records, not exposure-adjusted risk rates.")

with tab_people:
    st.subheader("People affected by collisions")
    c1, c2 = st.columns(2)
    with c1:
        st.write("**Casualties by severity**")
        st.bar_chart(casualty_match["casualty_severity_label"].value_counts())
    with c2:
        st.write("**Casualties by age band**")
        st.bar_chart(casualty_match["age_band_label"].value_counts())
    st.write("**Road-user class**")
    st.bar_chart(casualty_match["casualty_class_label"].value_counts())
    st.write("**Casualty sex (source categories)**")
    st.bar_chart(casualty_match["sex_label"].value_counts())

with tab_vehicles:
    st.subheader("Vehicles and drivers")
    c1, c2 = st.columns(2)
    with c1:
        st.write("**Driver age bands**")
        st.bar_chart(vehicle_match["driver_age_band_label"].value_counts())
    with c2:
        st.write("**Driver sex**")
        st.bar_chart(vehicle_match["driver_sex_label"].value_counts())
    if "vehicle_type" in vehicle_match.columns:
        st.write("**Vehicle type codes**")
        st.bar_chart(vehicle_match["vehicle_type"].value_counts().head(15))
        st.caption("Vehicle type remains in the original STATS19 source coding.")

with tab_predict:
    st.subheader("Interactive prediction")
    try:
        prediction_model, prediction_contract, prediction_metadata = load_prediction_assets()
        render_prediction_form(prediction_model, prediction_contract, prediction_metadata)
    except Exception as exc:
        st.warning("The dashboard works, but the prediction model files have not been installed yet.")
        st.code(
            "models/\n"
            "  stats19_severity_model.joblib\n"
            "  metadata.json\n"
            "  model_contract.json"
        )
        st.caption(str(exc))
        st.info(
            "Run the supplied notebook from the beginning through Sections 9.2 and 9.3. "
            "Then copy the three generated files into the app's models/ folder."
        )

with tab_kenya:
    st.subheader("Applying the research to Kenyan roads")
    st.markdown(
        '<div class="warning-card"><b>Kenya transferability boundary:</b> '
        'The current model was trained on UK STATS19 data. It has not been trained '
        'or tested on Kenyan crash records. Do not interpret its prediction as a '
        'Kenyan severity estimate.</div>',
        unsafe_allow_html=True,
    )
    st.write(
        "The current model can support a research demonstration and help define "
        "a Kenyan data-collection and validation project. A Kenyan model should be "
        "trained or recalibrated using locally collected, consistently coded records."
    )
    st.markdown("**Potential Kenyan road-safety applications after local validation**")
    recommendations = [
        ("High-risk corridor screening", "Combine geocoded crash history, road condition, traffic volume and severity to help road agencies prioritise site inspections."),
        ("Boda boda and vulnerable road users", "Analyse motorcycle, pedestrian and cyclist involvement, helmet use, crossing facilities and injury outcomes."),
        ("Night-time visibility", "Study lighting, weather, road surface, visibility and crash timing to target lighting and reflective-marking improvements."),
        ("Speed management", "Compare posted and observed speeds with crash outcomes, traffic exposure and road design before recommending engineering or enforcement changes."),
        ("County-level planning", "Summarise validated crash patterns by county, road class and road-user group to support county safety action plans."),
        ("Post-crash response", "If linked with ambulance and hospital data, study response time and injury outcomes separately from this current severity model."),
    ]
    for title, description in recommendations:
        st.markdown(f'<div class="soft-card"><b>{title}</b><br>{description}</div>', unsafe_allow_html=True)

    st.markdown("**Suggested Kenyan data partners and sources**")
    st.markdown(
        "- [National Transport and Safety Authority (NTSA)](https://www.ntsa.go.ke/) — police-reported crash records, crash-data system and road-safety reports.\n"
        "- [Ministry of Roads and Transport](https://www.transport.go.ke/) — National Road Safety Action Plan 2024–2028.\n"
        "- [Kenya Roads Board map portal](https://maps.krb.go.ke/) — road network classification and road-condition mapping.\n"
        "- [World Bank Microdata: Kenya Road Traffic Crashes 2012–2023](https://microdata.worldbank.org/catalog/6249) — crowdsourced Ma3Route crash locations, primarily Nairobi; useful for exploratory GIS work, not a substitute for official police records."
    )
    st.caption(
        "Kenya research pathway: obtain permission → define a common crash-data dictionary → "
        "clean and geocode records → map Kenyan variables to model concepts → assess data shift → "
        "train/validate locally → independent county/time holdout → stakeholder review."
    )

with tab_data:
    st.subheader("Explore filtered collision records")
    display_columns = [
        c for c in [
            "date", "time", "collision_severity_label", "number_of_vehicles",
            "number_of_casualties", "speed_limit", "road_type_label",
            "area_label", "latitude", "longitude", "local_authority_district",
        ] if c in filtered.columns
    ]
    st.dataframe(
        filtered[display_columns].sort_values("date", ascending=False),
        use_container_width=True, hide_index=True,
    )
    st.download_button(
        "Download filtered collision data (CSV)",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="filtered_road_collisions_2025.csv",
        mime="text/csv",
    )

st.divider()
st.caption(
    "Research dashboard. UK collision statistics are from the supplied Department for Transport STATS19 files. "
    "Counts depend on selected filters. The CatBoost model predicts recorded collision severity conditional "
    "on a collision; it does not predict whether a crash will occur. Not an official safety assessment."
)
