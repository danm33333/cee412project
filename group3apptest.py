import streamlit as st
import pandas as pd

st.title("Pedestrian Signal Adequacy (Event-Based Analysis)")

# ---------------------------------------------------
# Google Drive CSV link
# Replace FILE_ID with your actual Google Drive file ID
# ---------------------------------------------------
csv_link = "https://drive.google.com/uc?id=1vZhZiUcQNhOfocSmuOG-7Ibh1-0l9das"

st.markdown(f"CSV source: {csv_link}")

# ---------------------------------------------------
# Load data
# ---------------------------------------------------
@st.cache_data
def load_data(url):
    df = pd.read_csv(url)
    return df

df = load_data(csv_link)

st.subheader("Raw Data Preview")
st.dataframe(df.head())

# ---------------------------------------------------
# Clean text columns
# ---------------------------------------------------
df["signal_state"] = df["signal_state"].astype(str).str.strip().str.upper()
df["crosswalk_zone"] = df["crosswalk_zone"].astype(str).str.strip().str.lower()

# Ignore unnecessary columns if present
df = df.drop(columns=["phase", "direction"], errors="ignore")

# ---------------------------------------------------
# Determine whether pedestrian is inside crosswalk
# ---------------------------------------------------
df["in_crosswalk"] = df["crosswalk_zone"] != "outside"

# ---------------------------------------------------
# Keep only rows where pedestrian is inside crosswalk
# ---------------------------------------------------
crossing_df = df[df["in_crosswalk"]]

# ---------------------------------------------------
# Determine crossing events per pedestrian
# ---------------------------------------------------
crossing_events = (
    crossing_df
    .groupby(["station_ID", "ped_ID"])
    .agg(
        start_time=("timestamp", "min"),
        end_time=("timestamp", "max"),
        any_dont_walk=("signal_state", lambda x: (x == "DONT_WALK").any())
    )
    .reset_index()
)

# A violation occurs if any part of crossing overlaps DONT_WALK
crossing_events["violation"] = crossing_events["any_dont_walk"]

st.subheader("Pedestrian Crossing Events")
st.dataframe(crossing_events)

# ---------------------------------------------------
# Overall signal adequacy
# ---------------------------------------------------
total_crossings = len(crossing_events)
violating_crossings = crossing_events["violation"].sum()

adequacy_rate = 1 - (violating_crossings / total_crossings)

st.subheader("Overall Signal Adequacy")

st.metric("Total Crossings", total_crossings)
st.metric("Violating Crossings", int(violating_crossings))
st.metric("Signal Adequacy Rate", f"{adequacy_rate:.2%}")

# ---------------------------------------------------
# Adequacy by station
# ---------------------------------------------------
st.subheader("Signal Adequacy by Crosswalk")

station_summary = (
    crossing_events
    .groupby("station_ID")
    .agg(
        total_crossings=("ped_ID", "count"),
        violations=("violation", "sum")
    )
    .reset_index()
)

station_summary["adequacy_rate"] = (
    1 - station_summary["violations"] / station_summary["total_crossings"]
)

st.dataframe(station_summary)

st.bar_chart(station_summary.set_index("station_ID")[["violations"]])

# ---------------------------------------------------
# Show violating crossings
# ---------------------------------------------------
st.subheader("Violating Pedestrian Crossings")

violations_df = crossing_events[crossing_events["violation"]]

st.dataframe(violations_df)

# ---------------------------------------------------
# Optional station filter
# ---------------------------------------------------
st.subheader("Filter Crossing Events by Station")

station_choice = st.selectbox(
    "Select station",
    sorted(crossing_events["station_ID"].unique())
)

filtered_events = crossing_events[crossing_events["station_ID"] == station_choice]

st.dataframe(filtered_events)
