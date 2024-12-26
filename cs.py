import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page config
st.set_page_config(page_title="Heart Disease Analysis Dashboard", layout="wide")

# Title
st.title("Heart Disease Analysis Dashboard")


# Read the data
@st.cache_data
def load_data():
    df = pd.read_csv("heart.csv")
    return df


df = load_data()

# Sidebar
st.sidebar.header("Dashboard Controls")
show_raw_data = st.sidebar.checkbox("Show Raw Data")

if show_raw_data:
    st.subheader("Raw Data")
    st.dataframe(df)

# Key Metrics in columns
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Patients", len(df))
with col2:
    st.metric("Heart Disease Cases", f"{(df['target'].mean()*100):.1f}%")
with col3:
    st.metric("Average Age", f"{df['age'].mean():.1f} years")
with col4:
    st.metric(
        "Gender Ratio (M/F)", f"{len(df[df['sex']==1])/len(df[df['sex']==0]):.2f}"
    )

# Main content
tab1, tab2, tab3 = st.tabs(
    ["Distribution Plots", "Relationship Plots", "Statistical Analysis"]
)

with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Age Distribution by Heart Disease Status")
        fig1 = plt.figure(figsize=(10, 6))
        sns.kdeplot(data=df, x="age", hue="target", fill=True)
        st.pyplot(fig1)

    with col2:
        st.subheader("Gender Distribution")
        gender_counts = df["sex"].map({0: "Female", 1: "Male"}).value_counts()
        fig2 = px.pie(
            values=gender_counts.values,
            names=gender_counts.index,
            title="Gender Distribution",
            color_discrete_sequence=["lightpink", "lightblue"],
        )
        st.plotly_chart(fig2)

    st.subheader("Chest Pain Type Distribution")
    fig3 = px.histogram(
        df,
        x="cp",
        color="target",
        barmode="group",
        title="Chest Pain Type by Heart Disease Status",
        labels={"cp": "Chest Pain Type", "count": "Number of Patients"},
    )
    st.plotly_chart(fig3)

with tab2:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Heart Rate vs Age")
        fig4 = px.scatter(
            df,
            x="age",
            y="thalach",
            color="target",
            size="chol",
            hover_data=["sex", "cp"],
            title="Heart Rate vs Age (with Cholesterol as size)",
            labels={
                "age": "Age",
                "thalach": "Maximum Heart Rate",
                "target": "Heart Disease",
                "chol": "Cholesterol",
            },
        )
        st.plotly_chart(fig4)

    with col2:
        st.subheader("Correlation Heatmap")
        # Improve correlation heatmap
        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(df.corr()))
        fig5 = plt.figure(figsize=(10, 8))
        sns.heatmap(
            df.corr(),
            mask=mask,
            annot=True,
            cmap="RdBu_r",
            center=0,
            fmt=".2f",
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.5},
        )
        plt.title("Feature Correlation Matrix")
        st.pyplot(fig5)

    # Add new visualization
    st.subheader("Distribution of Features by Heart Disease Status")
    features_to_plot = ["age", "thalach", "chol", "trestbps"]
    fig6 = make_subplots(
        rows=2, cols=2, subplot_titles=[f.title() for f in features_to_plot]
    )

    for idx, feature in enumerate(features_to_plot):
        row = idx // 2 + 1
        col = idx % 2 + 1

        # Create violin plot for each feature
        fig6.add_trace(
            go.Violin(
                x=df[df["target"] == 0][feature].values,
                name="No Disease",
                side="negative",
                line_color="blue",
            ),
            row=row,
            col=col,
        )
        fig6.add_trace(
            go.Violin(
                x=df[df["target"] == 1][feature].values,
                name="Disease",
                side="positive",
                line_color="red",
            ),
            row=row,
            col=col,
        )

        fig6.update_xaxes(title_text=feature.title(), row=row, col=col)
        fig6.update_yaxes(title_text="Distribution", row=row, col=col)

    fig6.update_layout(
        height=800,
        showlegend=True,
        title_text="Distribution Comparison of Key Features",
    )
    st.plotly_chart(fig6)

with tab3:
    st.subheader("Statistical Summary")

    # Summary statistics by target - with better formatting
    summary_stats = (
        df.groupby("target")
        .agg(
            {
                "age": ["mean", "std", "min", "max"],
                "thalach": ["mean", "std"],
                "chol": ["mean", "std"],
            }
        )
        .round(2)
    )

    # Rename the index for clarity
    summary_stats.index = ["No Heart Disease", "Heart Disease"]

    # Flatten column names and make them more readable
    summary_stats.columns = [
        f"{col[0]}_{col[1]}".replace("_", " ").title() for col in summary_stats.columns
    ]

    st.write("Summary Statistics by Heart Disease Status:")
    st.dataframe(summary_stats)

    # Add filters for detailed analysis
    st.subheader("Filtered Analysis")

    col1, col2 = st.columns(2)
    with col1:
        age_range = st.slider(
            "Select Age Range",
            min_value=int(df["age"].min()),
            max_value=int(df["age"].max()),
            value=(30, 70),
        )

    with col2:
        gender = st.multiselect(
            "Select Gender",
            options=[0, 1],
            default=[0, 1],
            format_func=lambda x: "Female" if x == 0 else "Male",
        )

    # Filter data based on selections
    filtered_df = df[
        (df["age"].between(age_range[0], age_range[1])) & (df["sex"].isin(gender))
    ]

    st.write(f"Filtered Results ({len(filtered_df)} patients):")
    st.dataframe(filtered_df)

# Footer
st.markdown("---")
st.markdown(
    """
    **Data Dictionary:**
    - target: Heart disease presence (1: yes, 0: no)
    - sex: Gender (1: male, 0: female)
    - cp: Chest pain type
    - thalach: Maximum heart rate achieved
    - chol: Cholesterol level
"""
)
