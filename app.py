import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Employee Attrition Analysis",
    page_icon="👥",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("👥 Employee Attrition Analysis")

# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_data():

    return pd.read_csv("HR_Employee_Attrition.csv")


df = load_data()


# ============================================================
# DATASET INFORMATION
# ============================================================

total_employees = len(df)

employees_left = df["Attrition"].eq("Yes").sum()

employees_stayed = total_employees - employees_left

attrition_rate = (
    employees_left / total_employees
) * 100


# ============================================================
# PREPARE DATA
# ============================================================

X = df.drop(
    "Attrition",
    axis=1
)

y = df["Attrition"].map({
    "No": 0,
    "Yes": 1
})


# Remove unnecessary columns

columns_to_remove = [
    "EmployeeCount",
    "EmployeeNumber",
    "Over18",
    "StandardHours"
]

X = X.drop(
    columns=[
        col
        for col in columns_to_remove
        if col in X.columns
    ],
    errors="ignore"
)


# ============================================================
# IDENTIFY COLUMNS
# ============================================================

categorical_columns = X.select_dtypes(
    include=["object", "string"]
).columns.tolist()

numeric_columns = X.select_dtypes(
    exclude=["object", "string"]
).columns.tolist()


# ============================================================
# PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(

    transformers=[

        (
            "categorical",

            OneHotEncoder(
                handle_unknown="ignore",
                drop="first"
            ),

            categorical_columns
        ),

        (
            "numeric",

            "passthrough",

            numeric_columns
        )
    ]
)


# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


# ============================================================
# ENCODE DATA
# ============================================================

X_train_encoded = preprocessor.fit_transform(
    X_train
)

X_test_encoded = preprocessor.transform(
    X_test
)


# ============================================================
# RANDOM FOREST MODEL
# ============================================================

model = RandomForestClassifier(

    n_estimators=200,

    random_state=42,

    class_weight="balanced"
)


model.fit(
    X_train_encoded,
    y_train
)


# ============================================================
# MODEL PREDICTIONS
# ============================================================

y_pred = model.predict(
    X_test_encoded
)

y_probability = model.predict_proba(
    X_test_encoded
)[:, 1]


# ============================================================
# MODEL EVALUATION
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)
cm = confusion_matrix(
    y_test,
    y_pred
)

classification_rep = classification_report(
    y_test,
    y_pred,
    output_dict=True
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

feature_names = preprocessor.get_feature_names_out()

importance = pd.DataFrame({

    "Feature": feature_names,

    "Importance": model.feature_importances_

})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs([

    "🏠 Overview",

    "🔮 Prediction",

    "📊 Model Analysis",

    "💡 HR Insights"

])


# ============================================================
# TAB 1 — OVERVIEW
# ============================================================

with tab1:

    st.header(
        "🏠 Employee Attrition Overview"
    )


    # ========================================================
    # KPI
    # ========================================================

    st.subheader(
        "📊 HR Key Performance Indicators"
    )

    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "👥 Total Employees",
            total_employees
        )


    with col2:

        st.metric(
            "🚪 Employees Left",
            employees_left
        )


    with col3:

        st.metric(
            "✅ Employees Stayed",
            employees_stayed
        )


    with col4:

        st.metric(
            "📉 Attrition Rate",
            f"{attrition_rate:.2f}%"
        )


    st.divider()


    # ========================================================
    # ATTRITION GRAPH
    # ========================================================

    st.subheader(
        "📊 Employee Attrition"
    )

    fig, ax = plt.subplots(
        figsize=(7, 4)
    )

    sns.countplot(
        data=df,
        x="Attrition",
        ax=ax
    )

    ax.set_title(
        "Employee Attrition Distribution"
    )

    ax.set_xlabel(
        "Attrition"
    )

    ax.set_ylabel(
        "Number of Employees"
    )

    st.pyplot(fig)

    plt.close(fig)


    st.divider()


    # ========================================================
    # GENDER GRAPH
    # ========================================================

    st.subheader(
        "👤 Attrition by Gender"
    )

    fig, ax = plt.subplots(
        figsize=(7, 4)
    )

    sns.countplot(
        data=df,
        x="Gender",
        hue="Attrition",
        ax=ax
    )

    ax.set_title(
        "Attrition by Gender"
    )

    st.pyplot(fig)

    plt.close(fig)


    st.divider()


    # ========================================================
    # DEPARTMENT GRAPH
    # ========================================================

    st.subheader(
        "🏢 Attrition by Department"
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    sns.countplot(
        data=df,
        y="Department",
        hue="Attrition",
        ax=ax
    )

    ax.set_title(
        "Attrition by Department"
    )

    st.pyplot(fig)

    plt.close(fig)


    st.divider()


    # ========================================================
    # OVERTIME GRAPH
    # ========================================================

    st.subheader(
        "⏰ Attrition by Overtime"
    )

    fig, ax = plt.subplots(
        figsize=(7, 4)
    )

    sns.countplot(
        data=df,
        x="OverTime",
        hue="Attrition",
        ax=ax
    )

    ax.set_title(
        "Attrition by Overtime"
    )

    st.pyplot(fig)

    plt.close(fig)


    st.divider()


    # ========================================================
    # INTERACTIVE HR ANALYTICS
    # ========================================================

    st.subheader(
        "🔎 Interactive HR Analytics"
    )

    st.write(
        "Use the filters below to analyze employee attrition."
    )


    filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)


    with filter_col1:

        selected_department = st.selectbox(

            "🏢 Department",

            ["All"] +
            sorted(
                df["Department"].unique().tolist()
            ),

            key="department_filter"

        )


    with filter_col2:

        selected_gender = st.selectbox(

            "👤 Gender",

            ["All"] +
            sorted(
                df["Gender"].unique().tolist()
            ),

            key="gender_filter"

        )


    with filter_col3:

        selected_overtime = st.selectbox(

            "⏰ Overtime",

            ["All"] +
            sorted(
                df["OverTime"].unique().tolist()
            ),

            key="overtime_filter"

        )


    with filter_col4:

        selected_job_role = st.selectbox(

            "💼 Job Role",

            ["All"] +
            sorted(
                df["JobRole"].unique().tolist()
            ),

            key="jobrole_filter"

        )


    # ========================================================
    # APPLY FILTERS
    # ========================================================

    filtered_df = df.copy()


    if selected_department != "All":

        filtered_df = filtered_df[
            filtered_df["Department"]
            == selected_department
        ]


    if selected_gender != "All":

        filtered_df = filtered_df[
            filtered_df["Gender"]
            == selected_gender
        ]


    if selected_overtime != "All":

        filtered_df = filtered_df[
            filtered_df["OverTime"]
            == selected_overtime
        ]


    if selected_job_role != "All":

        filtered_df = filtered_df[
            filtered_df["JobRole"]
            == selected_job_role
        ]


    # ========================================================
    # FILTERED STATISTICS
    # ========================================================

    st.subheader(
        "📊 Filtered Employee Statistics"
    )


    filtered_total = len(filtered_df)

    filtered_left = (
        filtered_df["Attrition"]
        .eq("Yes")
        .sum()
    )

    filtered_stayed = (
        filtered_total -
        filtered_left
    )


    if filtered_total > 0:

        filtered_attrition_rate = (

            filtered_left /
            filtered_total

        ) * 100

    else:

        filtered_attrition_rate = 0


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "👥 Employees",
            filtered_total
        )


    with col2:

        st.metric(
            "🚪 Left",
            filtered_left
        )


    with col3:

        st.metric(
            "✅ Stayed",
            filtered_stayed
        )


    with col4:

        st.metric(
            "📉 Attrition Rate",
            f"{filtered_attrition_rate:.2f}%"
        )


    # ========================================================
    # FILTERED GRAPH
    # ========================================================

    if len(filtered_df) > 0:

        st.subheader(
            "📊 Filtered Attrition Distribution"
        )

        fig, ax = plt.subplots(
            figsize=(7, 4)
        )

        sns.countplot(
            data=filtered_df,
            x="Attrition",
            ax=ax
        )

        ax.set_title(
            "Filtered Attrition Distribution"
        )

        st.pyplot(fig)

        plt.close(fig)


    else:

        st.warning(
            "No employees match the selected filters."
        )


    # ========================================================
    # FILTERED DATA
    # ========================================================

    st.subheader(
        "📋 Filtered Employee Data"
    )

    st.dataframe(
        filtered_df,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # DOWNLOAD FILTERED DATA
    # ========================================================

    filtered_csv = filtered_df.to_csv(
        index=False
    )

    st.download_button(

        label="📥 Download Filtered Employee Data",

        data=filtered_csv,

        file_name="filtered_employee_data.csv",

        mime="text/csv",

        key="download_filtered_data"

    )


# ============================================================
# TAB 2 — PREDICTION
# ============================================================

with tab2:

    st.header(
        "🔮 Employee Attrition Prediction"
    )

    st.write(
        "Enter employee information to estimate "
        "the probability of employee attrition."
    )


    # ========================================================
    # FORM
    # ========================================================

    with st.form(
        "prediction_form"
    ):

        col1, col2, col3 = st.columns(3)


        # ====================================================
        # COLUMN 1
        # ====================================================

        with col1:

            age = st.number_input(
                "Age",
                min_value=18,
                max_value=70,
                value=30
            )

            business_travel = st.selectbox(
                "Business Travel",
                sorted(
                    df["BusinessTravel"].unique()
                )
            )

            department = st.selectbox(
                "Department",
                sorted(
                    df["Department"].unique()
                )
            )

            education_field = st.selectbox(
                "Education Field",
                sorted(
                    df["EducationField"].unique()
                )
            )

            gender = st.selectbox(
                "Gender",
                sorted(
                    df["Gender"].unique()
                )
            )

            job_role = st.selectbox(
                "Job Role",
                sorted(
                    df["JobRole"].unique()
                )
            )

            marital_status = st.selectbox(
                "Marital Status",
                sorted(
                    df["MaritalStatus"].unique()
                )
            )


        # ====================================================
        # COLUMN 2
        # ====================================================

        with col2:

            monthly_income = st.number_input(
                "Monthly Income",
                min_value=1000,
                max_value=50000,
                value=5000
            )

            daily_rate = st.number_input(
                "Daily Rate",
                min_value=100,
                max_value=1500,
                value=800
            )

            hourly_rate = st.number_input(
                "Hourly Rate",
                min_value=20,
                max_value=100,
                value=60
            )

            monthly_rate = st.number_input(
                "Monthly Rate",
                min_value=2000,
                max_value=30000,
                value=15000
            )

            distance = st.number_input(
                "Distance From Home",
                min_value=1,
                max_value=30,
                value=10
            )

            total_years = st.number_input(
                "Total Working Years",
                min_value=0,
                max_value=50,
                value=5
            )

            years_company = st.number_input(
                "Years At Company",
                min_value=0,
                max_value=40,
                value=3
            )


        # ====================================================
        # COLUMN 3
        # ====================================================

        with col3:

            overtime = st.selectbox(
                "OverTime",
                ["Yes", "No"]
            )

            job_satisfaction = st.slider(
                "Job Satisfaction",
                min_value=1,
                max_value=4,
                value=3
            )

            environment_satisfaction = st.slider(
                "Environment Satisfaction",
                min_value=1,
                max_value=4,
                value=3
            )

            job_involvement = st.slider(
                "Job Involvement",
                min_value=1,
                max_value=4,
                value=3
            )

            work_life_balance = st.slider(
                "Work Life Balance",
                min_value=1,
                max_value=4,
                value=3
            )

            stock_option = st.slider(
                "Stock Option Level",
                min_value=0,
                max_value=3,
                value=1
            )

            num_companies = st.number_input(
                "Number of Companies Worked",
                min_value=0,
                max_value=20,
                value=2
            )


        # ====================================================
        # SUBMIT
        # ====================================================

        submit = st.form_submit_button(
            "🔍 Predict Attrition Risk"
        )
            # ========================================================
    # PREDICTION
    # ========================================================

    if submit:

        # ====================================================
        # CREATE INPUT DATA
        # ====================================================

        input_data = pd.DataFrame({

            "Age": [age],

            "BusinessTravel": [
                business_travel
            ],

            "Department": [
                department
            ],

            "EducationField": [
                education_field
            ],

            "Gender": [
                gender
            ],

            "JobRole": [
                job_role
            ],

            "MaritalStatus": [
                marital_status
            ],

            "MonthlyIncome": [
                monthly_income
            ],

            "DailyRate": [
                daily_rate
            ],

            "HourlyRate": [
                hourly_rate
            ],

            "MonthlyRate": [
                monthly_rate
            ],

            "DistanceFromHome": [
                distance
            ],

            "TotalWorkingYears": [
                total_years
            ],

            "YearsAtCompany": [
                years_company
            ],

            "OverTime": [
                overtime
            ],

            "JobSatisfaction": [
                job_satisfaction
            ],

            "EnvironmentSatisfaction": [
                environment_satisfaction
            ],

            "JobInvolvement": [
                job_involvement
            ],

            "WorkLifeBalance": [
                work_life_balance
            ],

            "StockOptionLevel": [
                stock_option
            ],

            "NumCompaniesWorked": [
                num_companies
            ]
        })

        # ====================================================
        # ADD MISSING FEATURES
        # ====================================================

        for column in X.columns:

            if column not in input_data.columns:

                if column in numeric_columns:

                    input_data[column] = 0

                else:

                    input_data[column] = X[column].mode()[0]

        # Keep exactly the same column order as training data

        input_data = input_data[
            X.columns
        ]

        # ====================================================
        # PREPROCESS INPUT
        # ====================================================

        input_encoded = preprocessor.transform(
            input_data
        )

        # ====================================================
        # MAKE PREDICTION
        # ====================================================

        prediction = model.predict(
            input_encoded
        )[0]

        probability = model.predict_proba(
            input_encoded
        )[0][1]

        probability_percent = probability * 100

        # ====================================================
        # PREDICTION RESULT
        # ====================================================

        st.divider()

        st.subheader(
            "🎯 Prediction Result"
        )

        if prediction == 1:

            st.error(
                "🔴 HIGH ATTRITION RISK — "
                "Employee is likely to leave."
            )

        else:

            st.success(
                "🟢 LOW ATTRITION RISK — "
                "Employee is likely to stay."
            )

        # ====================================================
        # PROBABILITY
        # ====================================================

        st.metric(
            "Probability of Employee Leaving",
            f"{probability_percent:.2f}%"
        )

        progress_value = min(
            max(
                int(probability_percent),
                0
            ),
            100
        )

        st.progress(
            progress_value
        )

        # ====================================================
        # RISK LEVEL
        # ====================================================

        if probability_percent >= 70:

            risk_level = "🔴 HIGH RISK"

            st.error(
                f"Risk Level: {risk_level}"
            )

        elif probability_percent >= 40:

            risk_level = "🟠 MEDIUM RISK"

            st.warning(
                f"Risk Level: {risk_level}"
            )

        else:

            risk_level = "🟢 LOW RISK"

            st.success(
                f"Risk Level: {risk_level}"
            )

        # ====================================================
        # PREDICTION REPORT
        # ====================================================

        st.subheader(
            "📋 Prediction Report"
        )

        report_data = pd.DataFrame({

            "Information": [

                "Age",
                "Department",
                "Job Role",
                "Monthly Income",
                "OverTime",
                "Job Satisfaction",
                "Work Life Balance",
                "Distance From Home",
                "Attrition Probability",
                "Risk Level",
                "Prediction"

            ],

            "Value": [

                age,
                department,
                job_role,
                monthly_income,
                overtime,
                job_satisfaction,
                work_life_balance,
                distance,
                f"{probability_percent:.2f}%",
                risk_level,

                (
                    "Likely to Leave"
                    if prediction == 1
                    else "Likely to Stay"
                )

            ]
        })

        st.dataframe(
            report_data,
            use_container_width=True,
            hide_index=True
        )

        # ====================================================
        # DOWNLOAD REPORT
        # ====================================================

        csv_report = report_data.to_csv(
            index=False
        )

        st.download_button(

            label="📥 Download Prediction Report",

            data=csv_report,

            file_name="employee_attrition_prediction.csv",

            mime="text/csv",

            key="download_prediction_report"
        )

        # ====================================================
        # PHASE 14 — AUTOMATIC HR INSIGHTS
        # ====================================================

        st.divider()

        st.header(
            "🤖 Automatic HR Insights"
        )

        # Start HR risk score

        risk_score = 0

        # ====================================================
        # CHECK RISK FACTORS
        # ====================================================

        if overtime == "Yes":

            st.warning(
                "⏰ Employee works overtime frequently."
            )

            risk_score += 2

        if job_satisfaction <= 2:

            st.warning(
                "😟 Low job satisfaction detected."
            )

            risk_score += 2

        if environment_satisfaction <= 2:

            st.warning(
                "🏢 Poor work environment satisfaction."
            )

            risk_score += 1

        if work_life_balance <= 2:

            st.warning(
                "⚖️ Poor work-life balance."
            )

            risk_score += 2

        if distance >= 20:

            st.warning(
                "🚗 Employee lives far from the office."
            )

            risk_score += 1

        if years_company <= 2:

            st.warning(
                "🆕 New employee with short company experience."
            )

            risk_score += 1

        if monthly_income <= 4000:

            st.warning(
                "💰 Monthly income is relatively low."
            )

            risk_score += 1

        if stock_option == 0:

            st.warning(
                "📉 Employee has no stock options."
            )

            risk_score += 1

        # ====================================================
        # OVERALL HR ASSESSMENT
        # ====================================================

        st.divider()

        st.subheader(
            "📊 Overall HR Assessment"
        )

        if risk_score >= 8:

            st.error(
                "🔴 Very High Attrition Risk"
            )

        elif risk_score >= 5:

            st.warning(
                "🟠 Moderate Attrition Risk"
            )

        else:

            st.success(
                "🟢 Low Attrition Risk"
            )

        # ====================================================
        # HR RECOMMENDATIONS
        # ====================================================

        st.divider()

        st.subheader(
            "💡 HR Recommendations"
        )

        if risk_score >= 8:

            st.write(
                "✅ Schedule a one-to-one meeting with the employee."
            )

            st.write(
                "✅ Review salary and benefits."
            )

            st.write(
                "✅ Reduce overtime workload."
            )

            st.write(
                "✅ Offer career growth opportunities."
            )

            st.write(
                "✅ Assign a mentor or manager for regular follow-up."
            )

        elif risk_score >= 5:

            st.write(
                "✅ Monitor employee engagement."
            )

            st.write(
                "✅ Improve work-life balance."
            )

            st.write(
                "✅ Provide training and skill development."
            )

            st.write(
                "✅ Conduct regular feedback sessions."
            )

        else:

            st.write(
                "✅ Employee appears stable."
            )

            st.write(
                "✅ Continue recognition and appreciation."
            )

            st.write(
                "✅ Maintain current engagement strategy."
            )

        # ====================================================
        # POSITIVE EMPLOYEE INDICATORS
        # ====================================================

        st.divider()

        st.subheader(
            "🌟 Positive Employee Indicators"
        )

        if overtime == "No":

            st.success(
                "✔ No overtime workload."
            )

        if job_satisfaction >= 3:

            st.success(
                "✔ Good job satisfaction."
            )

        if work_life_balance >= 3:

            st.success(
                "✔ Healthy work-life balance."
            )

        if years_company >= 5:

            st.success(
                "✔ Strong company experience."
            )

        if stock_option > 0:

            st.success(
                "✔ Employee has stock option benefits."
            )

        # ====================================================
        # AUTOMATIC HR ACTION
        # ====================================================

        st.divider()

        st.subheader(
            "💡 Suggested HR Actions"
        )

        if probability_percent >= 70:

            st.error(
                "🔴 Immediate HR attention recommended."
            )

            st.write(
                "• Schedule a one-to-one discussion with the employee."
            )

            st.write(
                "• Review workload and overtime requirements."
            )

            st.write(
                "• Check job satisfaction and career growth opportunities."
            )

            st.write(
                "• Consider retention strategies."
            )

        elif probability_percent >= 40:

            st.warning(
                "🟠 Preventive HR action recommended."
            )

            st.write(
                "• Monitor employee satisfaction."
            )

            st.write(
                "• Discuss workload and work-life balance."
            )

            st.write(
                "• Provide career development opportunities."
            )

            st.write(
                "• Review compensation and recognition."
            )

        else:

            st.success(
                "🟢 Employee currently shows relatively low attrition risk."
            )

            st.write(
                "• Continue regular employee engagement."
            )

            st.write(
                "• Maintain a healthy work environment."
            )

            st.write(
                "• Continue career development and recognition."
            )
# ============================================================
# TAB 3 — MODEL ANALYSIS
# ============================================================

with tab3:

    st.header(
        "📊 Model Performance Analysis"
    )

    st.write(
        "Evaluation of the Random Forest "
        "employee attrition prediction model."
    )
    st.header("🤖 Machine Learning Model Performance")
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
    "Model Accuracy",
    f"{accuracy * 100:.2f}%"
    )
    

    with col2:
        st.metric(
    "ROC-AUC Score",
    f"{roc_auc:.3f}"
    )
        st.info(
    f"The Random Forest model achieved "
    f"{accuracy * 100:.2f}% accuracy and a "
    f"ROC-AUC score of {roc_auc:.3f} on the test dataset."
    )
        st.divider()

    # ========================================================
    # MODEL METRICS
    # ========================================================

    col1, col2 = st.columns(2)


    with col1:

        st.metric(
            "🎯 Accuracy",
            f"{accuracy * 100:.2f}%"
        )


    with col2:

        st.metric(
            "📈 ROC-AUC",
            f"{roc_auc:.3f}"
        )


    st.divider()


    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    st.subheader(
        "📋 Classification Report"
    )


    report_df = pd.DataFrame(
        classification_rep
    ).transpose()


    st.dataframe(

        report_df.round(3),

        use_container_width=True

    )


    st.divider()


    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    st.subheader(
        "🔲 Confusion Matrix"
    )


    fig, ax = plt.subplots(
        figsize=(6, 4)
    )


    sns.heatmap(

        cm,

        annot=True,

        fmt="d",

        ax=ax

    )


    ax.set_xlabel(
        "Predicted"
    )

    ax.set_ylabel(
        "Actual"
    )

    ax.set_title(
        "Confusion Matrix"
    )


    st.pyplot(fig)

    plt.close(fig)


    st.divider()


    # ========================================================
    # ROC CURVE
    # ========================================================

    st.subheader(
        "📈 ROC Curve"
    )


    fpr, tpr, thresholds = roc_curve(

        y_test,

        y_probability

    )


    fig, ax = plt.subplots(
        figsize=(7, 5)
    )


    ax.plot(

        fpr,

        tpr,

        label=f"ROC-AUC = {roc_auc:.3f}"

    )


    ax.plot(

        [0, 1],

        [0, 1],

        linestyle="--"

    )


    ax.set_xlabel(
        "False Positive Rate"
    )

    ax.set_ylabel(
        "True Positive Rate"
    )

    ax.set_title(
        "ROC Curve"
    )

    ax.legend()


    st.pyplot(fig)

    plt.close(fig)


    st.divider()


    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    st.subheader(
        "⭐ Top 10 Important Features"
    )


    top_features = importance.head(10)


    st.dataframe(

        top_features.round(4),

        use_container_width=True,

        hide_index=True

    )


    fig, ax = plt.subplots(
        figsize=(9, 6)
    )


    sns.barplot(

        data=top_features,

        x="Importance",

        y="Feature",

        ax=ax

    )


    ax.set_title(
        "Top 10 Features Affecting Attrition"
    )


    st.pyplot(fig)

    plt.close(fig)


# ============================================================
# TAB 4 — HR INSIGHTS
# ============================================================

with tab4:

    st.header(
        "💡 HR Insights"
    )


    st.write(
        "Important observations from the "
        "employee attrition dataset."
    )


    # ========================================================
    # TOP FEATURES
    # ========================================================

    st.subheader(
        "⭐ Top Factors Influencing Attrition"
    )


    top_features = importance.head(10)


    for _, row in top_features.iterrows():

        st.write(

            f"• **{row['Feature']}** — "
            f"Importance: "
            f"{row['Importance']:.3f}"

        )


    st.divider()


    # ========================================================
    # DATASET INSIGHTS
    # ========================================================

    st.subheader(
        "📌 Dataset Insights"
    )


    st.info(
        f"👥 Dataset contains "
        f"**{total_employees} employees**."
    )


    st.info(
        f"🚪 **{employees_left} employees** "
        f"have left the organization."
    )


    st.info(
        f"✅ **{employees_stayed} employees** "
        f"are still with the organization."
    )


    st.info(
        f"📉 Overall attrition rate is "
        f"**{attrition_rate:.2f}%**."
    )


    # ========================================================
    # OVERTIME INSIGHT
    # ========================================================

    overtime_employees = (
        df["OverTime"]
        .eq("Yes")
        .sum()
    )


    overtime_left = (

        df[
            df["OverTime"] == "Yes"
        ]["Attrition"]

        .eq("Yes")

        .sum()

    )


    if overtime_employees > 0:

        overtime_rate = (

            overtime_left /
            overtime_employees

        ) * 100


        st.info(

            f"⏰ Employees working overtime "
            f"have an attrition rate of approximately "
            f"**{overtime_rate:.2f}%**."

        )


    st.divider()


    # ========================================================
    # HR RECOMMENDATIONS
    # ========================================================

    st.subheader(
        "💼 General HR Recommendations"
    )


    st.markdown("""

**1. Monitor overtime workload**

Employees working excessive overtime may require
workload review and better work-life balance.


**2. Improve employee satisfaction**

Regular employee feedback and satisfaction surveys
can help identify problems early.


**3. Support career development**

Training, promotions and career growth opportunities
can improve employee retention.


**4. Review compensation**

Salary and compensation should be reviewed for
employees who show higher attrition risk.


**5. Monitor commuting difficulties**

Employees with long distances from home may benefit
from flexible work arrangements where possible.


**6. Improve employee engagement**

Regular communication between managers and employees
can help reduce avoidable attrition.

""")


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Employee Attrition Analysis | "
    "Machine Learning Project"
)
