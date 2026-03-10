import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Group 3 Presentation", layout="wide")

# ---------------------------------------------------
# GOOGLE DRIVE FILE LINKS
# ---------------------------------------------------

CSV_URL = "https://drive.google.com/uc?id=1vZhZiUcQNhOfocSmuOG-7Ibh1-0l9das"

INTRO_IMG1 = "https://drive.google.com/thumbnail?id=1YuHtF8c5f5tZsbs0EDi1I8PQnmHEzFza"
#INTRO_IMG2 = "https://drive.google.com/uc?id=INTRO_IMAGE2_ID"
#INTRO_IMG3 = "https://drive.google.com/uc?id=INTRO_IMAGE3_ID"

DATA_IMG1 = "https://drive.google.com/thumbnail?id=1RxYZuDKilglO9VIexLZvbv2dX5syCESk"
#DATA_IMG2 = "https://drive.google.com/thumbnail?id=1zTu-muFVvxSR9PurEs57NpIbmbbMpS8R"
#DATA_IMG3 = "https://drive.google.com/uc?id=DATA_IMAGE3_ID"

# ---------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    ["Introduction", "Dataset", "Gantt Charts", "Conclusion"]
)

# ---------------------------------------------------
# PAGE 1 — INTRODUCTION
# ---------------------------------------------------

if page == "Introduction":

    st.title("LiDAR-based Pedestrian Safety Analysis")

    st.write("""
    
    Our group evaluated pedestrian signal adequacy at the intersection of
    E MLK Blvd and Georgia Ave in Chattanooga, TN, using LiDAR (light-detection
    and ranging) trajectory data and signal performance logs.

    This analysis focused on determining whether pedestrians remained inside
    the crosswalk during **DONT WALK** signal phases.

    Our working hypothesis was that if over 50% of pedestrians are crossing
    during **DONT WALK** phases, then the signal may be deemed in need of modification.
     
    """)

    #col1, col2, col3 = st.columns(3)
    #col1 = st.columns(1)

    st.image(INTRO_IMG1, caption="Figure 1. Data collection site", width=500)
    #col2.image(INTRO_IMG2, caption="LiDAR Sensor Setup")
    #col3.image(INTRO_IMG3, caption="Pedestrian Detection Example")

# ---------------------------------------------------
# PAGE 2 — DATASET
# ---------------------------------------------------

elif page == "Dataset":

    st.title("Dataset")

    st.write("""
    The first dataset was collected by three LiDAR sensors, each positioned
    at one of three crosswalks at the intersection (white stars in Figure 2).

    Each sensor recorded movement every millisecond over the course of one hour.
    The resulting point clouds were pre-processed by Seoul Robotics software,
    categorized into pedestrians, bicyclists, and vehicles.
    The sensors also assigned each detected entity a unique ID.

    Signal performance logs were provided by traffic control for the same period
    of data collection. These offered a real-time log of traffic signal operations
    such as phase calls, detector status, and signal status.

    """)

    #col1, col2, col3 = st.columns(3)
    col1, col2 = st.columns(2)

    col1.image(DATA_IMG1, caption="Figure 2. Screen record of Lidar Data", use_container_width=True)
    #col2.image(DATA_IMG2, caption="LiDAR Detection Zones", use_container_width=True)
    #col3.image(DATA_IMG3, caption="Crosswalk Coverage")

    st.write("""
    Final variable set used for analysis:

    """)

    #Create the table as a DataFrame
    data = {
        "Variable": [
            "station_ID",
            "timestamp",
            "ped_ID",
            "speed",
            "Signal_phase",
            "phase",
            "direction",
            "Crosswalk_zone"
        ],
        "Description": [
            "Which LIDAR station the reading came from",
            "When the reading occurred",
            "Unique code given to each pedestrian",
            "Speed the pedestrian was walking",
            "What the most recent pedestrian signal was",
            "What area that most recent signal occurred; see PDF for area/phase clarification",
            "Direction the pedestrian was walking",
            "Area the pedestrian was in when reading occurred; 'Outside' = not on crosswalk"
        ]
    }

    df = pd.DataFrame(data)

    #Display the table
    st.table(df)


# ---------------------------------------------------
# PAGE 3 — DASHBOARD
# ---------------------------------------------------

elif page == "Gantt Charts":

    st.title("Crossing-signal Gantt Charts")

    @st.cache_data
    def load_data(url):
        return pd.read_csv(url)

    df = load_data(CSV_URL)

    df["signal_state"] = df["signal_state"].astype(str).str.strip().str.upper()
    df["crosswalk_zone"] = df["crosswalk_zone"].astype(str).str.strip().str.lower()

    df = df.drop(columns=["phase", "direction"], errors="ignore")

    df["timestamp_dt"] = pd.to_datetime(df["timestamp"], unit="ms")

    df["in_crosswalk"] = df["crosswalk_zone"] != "outside"

    crossing_df = df[df["in_crosswalk"]]

    crossing_events = (
        crossing_df
        .groupby(["station_ID","ped_ID"])
        .agg(
            start_time=("timestamp_dt","min"),
            end_time=("timestamp_dt","max"),
            any_dont_walk=("signal_state", lambda x: (x=="DONT_WALK").any())
        )
        .reset_index()
    )

    crossing_events["violation"] = crossing_events["any_dont_walk"]

    # ------------------------------
    # Station filter (NO "All")
    # ------------------------------

    station_filter = st.selectbox(
        "Select Crosswalk Station",
        sorted(df["station_ID"].unique().tolist())
    )

    crossing_events = crossing_events[crossing_events["station_ID"] == station_filter]
    df = df[df["station_ID"] == station_filter]

    # ------------------------------
    # Signal phase timeline
    # ------------------------------

    phase_changes = (
        df.sort_values("timestamp_dt")
        .drop_duplicates("timestamp_dt")[["timestamp_dt","signal_state"]]
    )

    phase_changes["end"] = phase_changes["timestamp_dt"].shift(-1)

    fig, ax = plt.subplots(figsize=(12,6))

    phase_colors = {
        "WALK":"lightgreen",
        "CLEARANCE":"orange",
        "DONT_WALK":"lightcoral"
    }

    for _,row in phase_changes.iterrows():

        if pd.isna(row["end"]):
            continue

        color = phase_colors.get(row["signal_state"], "gray")

        ax.axvspan(
            row["timestamp_dt"],
            row["end"],
            color=color,
            alpha=0.2
        )

    for i,row in crossing_events.iterrows():

        color = "red" if row["violation"] else "blue"

        ax.plot(
            [row["start_time"], row["end_time"]],
            [i,i],
            linewidth=6,
            color=color
        )

    ax.set_xlabel("Time")
    ax.set_ylabel("Crossing Event Index")
    ax.set_title("Pedestrian Crossings vs Signal Phases")

    st.pyplot(fig)

# ---------------------------------------------------
# PAGE 4 — CONCLUSION
# ---------------------------------------------------

elif page == "Conclusion":

    st.title("Conclusion")

    @st.cache_data
    def load_data(url):
        return pd.read_csv(url)

    df = load_data(CSV_URL)

    df["signal_state"] = df["signal_state"].astype(str).str.strip().str.upper()
    df["crosswalk_zone"] = df["crosswalk_zone"].astype(str).str.strip().str.lower()

    df["timestamp_dt"] = pd.to_datetime(df["timestamp"], unit="ms")

    df["in_crosswalk"] = df["crosswalk_zone"] != "outside"

    crossing_df = df[df["in_crosswalk"]]

    crossing_events = (
        crossing_df
        .groupby(["station_ID","ped_ID"])
        .agg(
            start_time=("timestamp_dt","min"),
            end_time=("timestamp_dt","max"),
            any_dont_walk=("signal_state", lambda x: (x=="DONT_WALK").any())
        )
        .reset_index()
    )

    crossing_events["violation"] = crossing_events["any_dont_walk"]

    # ------------------------------
    # Overall metrics
    # ------------------------------

    total_crossings = len(crossing_events)
    violations = crossing_events["violation"].sum()
    adequacy = 1 - violations / total_crossings

    st.subheader("Overall Signal Adequacy")

    col1,col2,col3 = st.columns(3)

    col1.metric("Total Crossings", total_crossings)
    col2.metric("Violating Crossings", int(violations))
    col3.metric("Adequacy Rate", f"{adequacy:.2%}")

    # ------------------------------
    # Station metrics
    # ------------------------------

    station_metrics = (
        crossing_events
        .groupby("station_ID")
        .agg(
            total_crossings=("ped_ID","count"),
            violations=("violation","sum")
        )
        .reset_index()
    )

    station_metrics["adequacy_rate"] = (
        1 - station_metrics["violations"] /
        station_metrics["total_crossings"]
    )

    st.subheader("Station-Level Signal Adequacy")

    st.dataframe(station_metrics)

    # ------------------------------
    # Bar graph
    # ------------------------------

    st.subheader("Violations by Station")

    fig, ax = plt.subplots()

    ax.bar(
        station_metrics["station_ID"],
        station_metrics["violations"]
    )

    ax.set_xlabel("Station ID")
    ax.set_ylabel("Violations")
    ax.set_title("Pedestrian Violations by Crosswalk")

    st.pyplot(fig)

    st.write("""
    As this was a preliminary analysis, it would benefit from verifying
    the accuracy of the crossing classifications, especially considering
    the low adequacy rate at all three crosswalks.

    Performing the same analysis on the detected bicyclists and vehicles
    would further help elucidate the method's efficacy.

    With further refinement this type of analysis could greatly assist in
    improving pedestrian signal timing and designing safer intersections
    for individuals with different mobility needs.

    """)
