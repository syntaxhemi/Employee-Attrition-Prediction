import streamlit as st
import pandas as pd
import joblib

import plotly.express as px
import matplotlib.pyplot as plt
# -------------------------
# Page Configuration
# -------------------------
st.set_page_config(
    page_title="Employee Attrition Dashboard",
    # page_icon="📊",
    layout="wide"
)

# -------------------------
# Load Dataset
# -------------------------
df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

# -------------------------
# Load Model
# -------------------------
model = joblib.load("trained_model.pkl")

# -------------------------
# Sidebar
# -------------------------
st.sidebar.title("HR Analytics")

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Analytics",
        "Prediction",
        "Model Performance",
        "HR Insights"
    ]
)

# -------------------------
# HOME PAGE
# -------------------------
if page == "Home":

    st.title("Employee Attrition Prediction & Workforce Analytics")

    st.markdown(
        """
        This dashboard helps HR departments analyze employee attrition,
        visualize workforce trends, and predict whether an employee is likely
        to leave the organization using a Machine Learning model.
        """
    )

    total = len(df)
    left = len(df[df["Attrition"] == "Yes"])
    stayed = len(df[df["Attrition"] == "No"])
    rate = left / total * 100

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Employees", total)
    c2.metric("Employees Left", left)
    c3.metric("Employees Stayed", stayed)
    c4.metric("Attrition Rate", f"{rate:.2f}%")

    st.divider()

    st.subheader("Dataset Preview")

    st.dataframe(df.head())

    st.divider()

    st.subheader("Dataset Information")

    col1, col2 = st.columns(2)

    with col1:
        st.write("Rows:", df.shape[0])
        st.write("Columns:", df.shape[1])

    with col2:
        st.write("Numerical Features")
        st.write(df.select_dtypes(include="number").columns.tolist())

    st.success("Dashboard Loaded Successfully")
    

# -------------------------
# ANALYTICS PAGE
# -------------------------
elif page == "Analytics":

    st.title("Workforce Analytics Dashboard")

    st.write("Analyze employee attrition trends using interactive visualizations.")

    # Row 1
    col1, col2 = st.columns(2)

    with col1:
        dept = (
            df.groupby(["Department", "Attrition"])
              .size()
              .reset_index(name="Count")
        )

        fig = px.bar(
            dept,
            x="Department",
            y="Count",
            color="Attrition",
            barmode="group",
            title="Department-wise Employee Attrition"
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        role = (
            df.groupby(["JobRole", "Attrition"])
              .size()
              .reset_index(name="Count")
        )

        fig = px.bar(
            role,
            x="JobRole",
            y="Count",
            color="Attrition",
            title="Job Role-wise Attrition"
        )

        fig.update_layout(xaxis_tickangle=-45)

        st.plotly_chart(fig, use_container_width=True)

    # Row 2
    col3, col4 = st.columns(2)

    with col3:
        fig = px.box(
            df,
            x="Attrition",
            y="MonthlyIncome",
            color="Attrition",
            title="Monthly Income vs Attrition"
        )

        st.plotly_chart(fig, use_container_width=True)

    with col4:
        overtime = (
            df.groupby(["OverTime", "Attrition"])
              .size()
              .reset_index(name="Count")
        )

        fig = px.bar(
            overtime,
            x="OverTime",
            y="Count",
            color="Attrition",
            title="OverTime Analysis"
        )

        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Row 3
    col5, col6 = st.columns(2)

    with col5:
        fig = px.histogram(
            df,
            x="YearsAtCompany",
            color="Attrition",
            nbins=15,
            title="Years at Company Distribution"
        )

        st.plotly_chart(fig, use_container_width=True)

    with col6:
        fig = px.histogram(
            df,
            x="WorkLifeBalance",
            color="Attrition",
            title="Work-Life Balance"
        )

        st.plotly_chart(fig, use_container_width=True)
# -------------------------
# PREDICTION PAGE
# -------------------------

elif page == "Prediction":

    st.title("Employee Attrition Prediction")

    st.write("Fill in the employee details and click **Predict Attrition**.")

    col1, col2 = st.columns(2)

    with col1:

        age = st.number_input("Age", 18, 60, 30)

        business_travel = st.selectbox(
            "Business Travel",
            sorted(df["BusinessTravel"].unique())
        )

        department = st.selectbox(
            "Department",
            sorted(df["Department"].unique())
        )

        distance = st.number_input(
            "Distance From Home",
            1,
            30,
            5
        )

        education = st.selectbox(
            "Education",
            sorted(df["Education"].unique())
        )

        education_field = st.selectbox(
            "Education Field",
            sorted(df["EducationField"].unique())
        )

        gender = st.selectbox(
            "Gender",
            sorted(df["Gender"].unique())
        )

        marital_status = st.selectbox(
            "Marital Status",
            sorted(df["MaritalStatus"].unique())
        )

        job_role = st.selectbox(
            "Job Role",
            sorted(df["JobRole"].unique())
        )

    with col2:

        monthly_income = st.number_input(
            "Monthly Income",
            1000,
            25000,
            5000
        )

        overtime = st.selectbox(
            "OverTime",
            sorted(df["OverTime"].unique())
        )

        job_satisfaction = st.selectbox(
            "Job Satisfaction",
            sorted(df["JobSatisfaction"].unique())
        )

        environment = st.selectbox(
            "Environment Satisfaction",
            sorted(df["EnvironmentSatisfaction"].unique())
        )

        work_life = st.selectbox(
            "Work Life Balance",
            sorted(df["WorkLifeBalance"].unique())
        )

        stock_option = st.selectbox(
            "Stock Option Level",
            sorted(df["StockOptionLevel"].unique())
        )

        years_company = st.number_input(
            "Years At Company",
            0,
            40,
            5
        )

        years_promotion = st.number_input(
            "Years Since Last Promotion",
            0,
            15,
            1
        )

        years_manager = st.number_input(
            "Years With Current Manager",
            0,
            20,
            4
        )

    st.divider()

    if st.button("Predict Attrition", use_container_width=True):

        employee = pd.DataFrame({

            "Age":[age],
            "BusinessTravel":[business_travel],
            "DailyRate":[800],
            "Department":[department],
            "DistanceFromHome":[distance],
            "Education":[education],
            "EducationField":[education_field],
            "EmployeeCount":[1],
            "EmployeeNumber":[9999],
            "EnvironmentSatisfaction":[environment],
            "Gender":[gender],
            "HourlyRate":[60],
            "JobInvolvement":[3],
            "JobLevel":[2],
            "JobRole":[job_role],
            "JobSatisfaction":[job_satisfaction],
            "MaritalStatus":[marital_status],
            "MonthlyIncome":[monthly_income],
            "MonthlyRate":[15000],
            "NumCompaniesWorked":[2],
            "Over18":["Y"],
            "OverTime":[overtime],
            "PercentSalaryHike":[15],
            "PerformanceRating":[3],
            "RelationshipSatisfaction":[3],
            "StandardHours":[80],
            "StockOptionLevel":[stock_option],
            "TotalWorkingYears":[10],
            "TrainingTimesLastYear":[3],
            "WorkLifeBalance":[work_life],
            "YearsAtCompany":[years_company],
            "YearsInCurrentRole":[3],
            "YearsSinceLastPromotion":[years_promotion],
            "YearsWithCurrManager":[years_manager]

        })

        prediction = model.predict(employee)[0]

        probability = model.predict_proba(employee)[0][1]

        st.divider()

        st.subheader("Prediction Result")

        if prediction == 1:

            st.error("⚠️ Employee is likely to leave the organization.")

        else:

            st.success("Employee is likely to stay with the organization.")

        st.metric(
            "Attrition Probability",
            f"{probability*100:.2f}%"
        )

        if probability >= 0.75:

            st.warning("""
### 🔴 High Risk

Recommended HR Actions

- Conduct one-to-one retention meeting

- Review salary and incentives

- Offer promotion/career growth

- Improve work-life balance

- Increase employee engagement
""")

        elif probability >= 0.40:

            st.info("""
### 🟡 Medium Risk

Recommended HR Actions

- Monitor employee satisfaction

- Provide learning opportunities

- Schedule regular feedback sessions
""")

        else:

            st.success("""
### 🟢 Low Risk

Employee appears satisfied.

Continue existing engagement and recognition programs.
""")
elif page == "Model Performance":

    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
        ConfusionMatrixDisplay,
        RocCurveDisplay,
        roc_auc_score
    )

    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split

    st.title("Model Performance")

    st.write("Performance evaluation of the Random Forest model.")

    # Prepare dataset
    temp_df = df.copy()
    temp_df["Attrition"] = temp_df["Attrition"].map({"Yes": 1, "No": 0})

    X = temp_df.drop("Attrition", axis=1)
    y = temp_df["Attrition"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)

    st.subheader("Model Evaluation Metrics")

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("Accuracy", f"{accuracy:.2%}")
    c2.metric("Precision", f"{precision:.2%}")
    c3.metric("Recall", f"{recall:.2%}")
    c4.metric("F1 Score", f"{f1:.2%}")
    c5.metric("ROC AUC", f"{roc:.2%}")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Confusion Matrix")

        fig, ax = plt.subplots()

        ConfusionMatrixDisplay.from_predictions(
            y_test,
            y_pred,
            ax=ax,
            cmap="Blues"
        )

        st.pyplot(fig)

    with col2:
        st.subheader("ROC Curve")

        fig, ax = plt.subplots()

        RocCurveDisplay.from_predictions(
            y_test,
            y_prob,
            ax=ax
        )

        st.pyplot(fig)

    st.success("Model evaluated successfully.")
elif page == "HR Insights":

    st.title("HR Insights Dashboard")

    st.write(
        "Key workforce insights and actionable recommendations based on employee attrition data."
    )

    # -----------------------------
    # KPI Calculations
    # -----------------------------

    highest_attrition_dept = (
        df[df["Attrition"] == "Yes"]["Department"]
        .value_counts()
        .idxmax()
    )

    highest_attrition_role = (
        df[df["Attrition"] == "Yes"]["JobRole"]
        .value_counts()
        .idxmax()
    )

    avg_income = df["MonthlyIncome"].mean()

    avg_years = df["YearsAtCompany"].mean()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Highest Attrition Department",
        highest_attrition_dept
    )

    col2.metric(
        "Highest Risk Job Role",
        highest_attrition_role
    )

    col3.metric(
        "Average Monthly Income",
        f"${avg_income:,.0f}"
    )

    col4.metric(
        "Average Years at Company",
        f"{avg_years:.1f}"
    )

    st.divider()

    # -----------------------------
    # Overtime Analysis
    # -----------------------------

    st.subheader("Employees Working Overtime")

    overtime = (
        df[df["Attrition"] == "Yes"]["OverTime"]
        .value_counts()
        .reset_index()
    )

    overtime.columns = ["OverTime", "Employees"]

    fig = px.bar(
        overtime,
        x="OverTime",
        y="Employees",
        title="Attrition vs Overtime"
    )

    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # Work-Life Balance
    # -----------------------------

    st.subheader("Work-Life Balance")

    wlb = (
        df[df["Attrition"] == "Yes"]
        .groupby("WorkLifeBalance")
        .size()
        .reset_index(name="Employees")
    )

    fig = px.bar(
        wlb,
        x="WorkLifeBalance",
        y="Employees",
        title="Employees Leaving by Work-Life Balance"
    )

    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # Job Satisfaction
    # -----------------------------

    st.subheader("Job Satisfaction")

    js = (
        df[df["Attrition"] == "Yes"]
        .groupby("JobSatisfaction")
        .size()
        .reset_index(name="Employees")
    )

    fig = px.bar(
        js,
        x="JobSatisfaction",
        y="Employees",
        title="Employees Leaving by Job Satisfaction"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # -----------------------------
    # Key Findings
    # -----------------------------

    st.subheader("Findings")

    st.info(f"""
• The **{highest_attrition_dept}** department experiences the highest employee attrition.

• The **{highest_attrition_role}** role records the highest number of employees leaving.

• Employees working **overtime** are more likely to leave the organization.

• Lower **job satisfaction** and **work-life balance** are associated with higher attrition.

• The average employee monthly income is **${avg_income:,.0f}**.

• Employees stay with the company for an average of **{avg_years:.1f} years**.
""")

    st.divider()

    # -----------------------------
    # HR Recommendations
    # -----------------------------

    st.subheader("HR Recommendations")

    st.success("""
### Retention Strategies

• Improve work-life balance through flexible working arrangements.

• Conduct regular employee satisfaction surveys.

• Recognize and reward high-performing employees.

• Provide career growth and promotion opportunities.

• Reduce excessive overtime where possible.

• Organize employee engagement and wellness programs.

• Review compensation for high-risk departments.

• Monitor employees with low job satisfaction scores.
""")        