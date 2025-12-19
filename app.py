import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

@st.cache_data
def load_data():
    df = pd.read_csv("data/advertising.csv")


    df = df.rename(columns={
        "Daily Time Spent on Site": "time_on_site",
        "Age": "age",
        "Area Income": "area_income",
        "Daily Internet Usage": "internet_usage",
        "Ad Topic Line": "ad_topic",
        "City": "city",
        "Male": "male",
        "Timestamp": "timestamp",
        "Clicked on Ad": "clicked"
    })

    # Drop city
    df = df.drop(columns=["city"])

    # Convert timestamp and extract hour
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour

    # Simple text features from ad_topic
    df["ad_topic_len"] = df["ad_topic"].str.len()
    df["ad_topic_words"] = df["ad_topic"].str.split().str.len()

    return df

df = load_data()

@st.cache_resource
def train_model(df):
    feature_cols = [
        "age",
        "area_income",
        "time_on_site",
        "internet_usage",
        "hour",
        "male",
        "ad_topic_len",
        "ad_topic_words",
    ]

    X = df[feature_cols]
    y = df["clicked"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    metrics = {
        "accuracy": acc,
        "confusion_matrix": cm,
        "report": report,
        "X_test_scaled": X_test_scaled,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_proba": y_proba,
    }

    return model, scaler, feature_cols, metrics

model, scaler, feature_cols, metrics = train_model(df)


st.set_page_config(
    page_title="Ad Click Prediction",
    layout="wide"
)
st.sidebar.title(" ✨ Ad Click Prediction")
st.sidebar.title(" 📊 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Introduction", "EDA", "Model & Prediction", "Conclusion"]
)


if page == "Introduction":
    st.title("🔎 Online Ad Click Prediction")
    st.subheader("📖 Introduction")

    st.write(
        """
        This app explores the **Advertisement – Click on Ad** dataset from Kaggle and builds a
        machine learning model to predict whether a user will click on an online advertisement. 

        Each row represents one user with information such as:
        - Age
        - Area income
        - Daily time spent on the website
        - Daily internet usage
        - Gender
        - Timestamp (when the ad was shown)
        - Ad topic line (a short headline for the ad)

        The target variable is **`clicked`** (0 = no click, 1 = click), so this is a **binary classification** problem.
        """
    )

    st.markdown("### Dataset Preview")
    st.dataframe(df.head())


elif page == "EDA":
    st.title("Exploratory Data Analysis (EDA)")

    st.markdown("### 1. Summary Statistics")
    st.write(df[["age", "area_income", "time_on_site", "internet_usage",
                 "ad_topic_len", "ad_topic_words", "hour"]].describe())

    st.markdown("### 2. Class Balance")
    col1, col2 = st.columns(2)
    with col1:
        st.write(df["clicked"].value_counts())
    with col2:
        st.write(df["clicked"].value_counts(normalize=True))
    
    st.markdown("### 3. Feature Distributions")
    
    # 1. Create the grid of histograms
    fig_grid, axes = plt.subplots(3, 2, figsize=(14, 10))
    
    # List of columns to plot, matching your screenshot
    grid_cols = [
        "age", "area_income", "time_on_site", 
        "internet_usage", "ad_topic_len", "ad_topic_words"
    ]
    
    # Flatten axes array for easy iteration
    axes_flat = axes.flatten()
    
    for i, col in enumerate(grid_cols):
        # Plot histogram using pandas/matplotlib
        axes_flat[i].hist(df[col], bins=15)
        axes_flat[i].set_title(col)
        axes_flat[i].grid(True)
    
    plt.tight_layout()
    st.pyplot(fig_grid)

    st.markdown("### 4. Feature Distributions By Clicked ")
    numeric_cols = ["age", "area_income", "time_on_site", "internet_usage"]
    feature = st.selectbox("Select feature to plot", numeric_cols)

    fig, ax = plt.subplots()
    sns.histplot(data=df, x=feature, hue="clicked", kde=True, stat="density", common_norm=False, ax=ax)
    ax.set_title(f"Distribution of {feature} by Clicked")
    st.pyplot(fig)

    st.markdown("### 5. Correlation Heatmap")
    corr_cols = ["age", "area_income", "time_on_site", "internet_usage",
                 "ad_topic_len", "ad_topic_words", "hour", "clicked"]
    corr = df[corr_cols].corr()

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax2)
    ax2.set_title("Correlation Heatmap")
    st.pyplot(fig2)

    st.markdown("### 6. Click Rate by Age Group")
    bins = [0, 25, 35, 45, 60, 100]
    labels = ["<=25", "26-35", "36-45", "46-60", "60+"]

    df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels, right=True)
    age_click = df.groupby("age_group", observed=False)["clicked"].mean()

    fig3, ax3 = plt.subplots()
    age_click.plot(kind="bar", ax=ax3)
    ax3.set_ylabel("Average click rate")
    ax3.set_title("Click Rate by Age Group")
    st.pyplot(fig3)

    st.markdown("### 7. Click Rate by Hour")
    hour_click = df.groupby("hour")["clicked"].mean()

    fig4, ax4 = plt.subplots()
    hour_click.plot(kind="line" , marker="o", ax=ax4)
    ax4.set_xlabel("Hour of day ")
    ax4.set_ylabel("Average click rate ")
    ax4.set_title("Click Rate by Hour ")
    ax4.grid(True)
    st.pyplot(fig4)
    

    st.markdown("### 8. Click Rate by Gender ")
    
    # 2. Create the Click Rate by Gender bar chart 
    fig_gender, ax_gender = plt.subplots()
    # Calculate mean click rate by gender 
    df.groupby("male")["clicked"].mean().plot(kind="bar", ax=ax_gender)
    
    ax_gender.set_title("Click Rate by Gender  (male = 1) ")
    ax_gender.set_ylabel("Average click rate ")
    st.pyplot(fig_gender )

elif page == "Model & Prediction":
    # Page title
    st.title("🪄 Predict Ad Click ")

    # Optional short subtitle
    st.markdown("Adjust the user attributes below and click **Predict** to see whether the user is likely to click the ad.")

    # Main two-column layout for inputs
    left_col, right_col = st.columns(2)

    with left_col:
        time_on_site = st.number_input(
            "Daily Time Spent on Site (minutes)",
            float(df["time_on_site"].min()),
            float(df["time_on_site"].max()),
            float(df["time_on_site"].median()),
            step=1.0,
        )

        age = st.slider(
            "Age",
            int(df["age"].min()),
            int(df["age"].max()),
            int(df["age"].median()),
        )

        area_income = st.number_input(
            "Area Income",
            float(df["area_income"].min() ),
            float(df["area_income"].max()),
            float(df["area_income"].median() ),
            step=1000.0,
        )

        male = st.selectbox(
            "Gender",
            options=["Female", "Male"],
            index=1,
        )
        male_value = 1 if male == "Male" else 0

    with right_col:
        internet_usage = st.number_input(
            "Daily Internet Usage (minutes)",
            float(df["internet_usage"].min() ),
            float(df["internet_usage"].max() ),
            float(df["internet_usage"].median() ),
            step=1.0, 
            
        )

        hour = st.slider(
            "Hour of Day (0–23) ",
            0,
            23,
            12,
        )

        ad_topic_len = st.slider(
            "Ad Topic Length (characters)",
            int(df["ad_topic_len"].min()),
            int(df["ad_topic_len"].max()),
            int(df["ad_topic_len"].median()),
        )

        ad_topic_words = st.slider(
            "Ad Topic Word Count",
            int(df["ad_topic_words"].min()) ,
            int(df["ad_topic_words"].max()) ,
            int(df["ad_topic_words"].median()) ,
            
        )

    # Centered predict button
    st.markdown("---")
    btn_col = st.columns([3, 1, 3])[1]
    with btn_col:
        predict_clicked = st.button("🚀 Predict ", use_container_width=True)

    if predict_clicked:
        # Build a one-row DataFrame with user input
        user_data = pd.DataFrame([{
            "age": age,
            "area_income": area_income,
            "time_on_site": time_on_site,
            "internet_usage": internet_usage,
            "hour": hour,
            "male": male_value,
            "ad_topic_len": ad_topic_len,
            "ad_topic_words": ad_topic_words,
        }])

        # Scale using the same scaler as training
        user_scaled = scaler.transform(user_data[feature_cols])

        # Predict
        click_pred = model.predict(user_scaled)[0]
        
        click_proba = model.predict_proba(user_scaled)[0, 1]

        # Result box
        st.markdown("### Prediction Result")
        if click_pred == 1:
            st.success(
                f"✅ Predicted: **User WILL click the ad**\n\n"
                f"Estimated click probability: **{click_proba:.2%}**"
            )
        else:
            st.info(
                f"❌ Predicted: **User will NOT click the ad**\n\n"
                f"Estimated click probability: **{click_proba:.2%}**"
            )


elif page == "Conclusion":
    st.title("🧾 Conclusion ")

    st.write(
        """
        **Key takeaways:**
        - Features such as age, area income, daily time spent on site, and daily internet usage
          show strong relationships with whether a user clicks an ad.
 
        - Simple engineered features from the ad topic line (length and word count) did not show
          strong correlation with clicks, which suggests that the generic buzzword headlines in
          this dataset carry limited predictive signal.
          
        - The interactive app allows users to explore the dataset visually and experiment with
          different feature values to see how they affect the predicted click probability.

        This project demonstrates a complete  workflow: data loading and cleaning,
        EDA, feature engineering, model training and evaluation, and deployment using Streamlit.
        """
    )
