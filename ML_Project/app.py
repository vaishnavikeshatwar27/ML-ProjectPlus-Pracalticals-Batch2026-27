
import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Student Study Analyzer",
    page_icon="🎓",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0b1020, #151b35);
    color: #f4f6ff;
}
header[data-testid="stHeader"] {
    background: transparent;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1250px;
}
h1, h2, h3, p, label {
    color: #f4f6ff !important;
}
.hero {
    background: linear-gradient(110deg, #302b63, #24243e, #0f3460);
    padding: 30px;
    border-radius: 18px;
    border: 1px solid #555080;
    margin-bottom: 25px;
}
.hero h1 {
    font-size: 34px;
    margin-bottom: 8px;
}
.hero p {
    color: #d0d4ee !important;
    font-size: 15px;
}
.card {
    background: #1c2541;
    border: 1px solid #354265;
    padding: 22px;
    border-radius: 15px;
    min-height: 125px;
    margin-bottom: 12px;
}
.card-title {
    color: #b6c2e1;
    font-size: 13px;
}
.card-value {
    color: #ffffff;
    font-size: 27px;
    font-weight: 700;
    margin-top: 10px;
}
.result {
    background: linear-gradient(120deg, #253b62, #34275b);
    border: 1px solid #7162b5;
    border-radius: 16px;
    padding: 25px;
    margin: 15px 0;
}
.result h2 {
    font-size: 25px;
}
.section {
    background: #171f38;
    border: 1px solid #303d60;
    border-radius: 15px;
    padding: 20px;
    margin-top: 18px;
}
div.stButton > button {
    background: linear-gradient(90deg, #7555e8, #4568d4);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 12px;
    font-weight: 700;
    width: 100%;
}
div.stButton > button:hover {
    background: linear-gradient(90deg, #8c70f4, #6485ec);
    color: white;
    border: none;
}
[data-testid="stSidebar"] {
    background: #111a30;
    border-right: 1px solid #303d60;
}
[data-testid="stMetric"] {
    background: #1c2541;
    padding: 15px;
    border-radius: 12px;
    border: 1px solid #354265;
}
div[data-testid="stProgressBar"] > div > div {
    background-color: #9275ed;
}
footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

# ---------------- DATASET ----------------
data = {
    "study_hours": [1,2,3,4,5,6,7,2,3,8,6,1,4,5,7,2,8,3,6,5],
    "attendance": [40,50,60,65,75,85,95,45,70,98,80,30,70,80,90,50,95,55,85,78],
    "sleep": [5,6,6,7,7,8,8,5,7,8,7,4,6,7,8,5,8,6,7,7],
    "assignments": [30,40,50,60,75,85,95,35,65,100,80,20,65,80,90,40,95,50,85,75],
    "performance": [
        "Needs Improvement","Needs Improvement","Average","Average",
        "Good","Good","Good","Needs Improvement","Average","Good",
        "Good","Needs Improvement","Average","Good","Good",
        "Needs Improvement","Good","Average","Good","Good"
    ]
}

df = pd.DataFrame(data)

X = df[["study_hours", "attendance", "sleep", "assignments"]]
y = df["performance"]

model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X, y)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("## 🎓 Student Analyzer")
    st.caption("STUDENT INPUT PANEL")
    st.divider()

    with st.form("student_form"):
        st.markdown("### 📚 Study Details")

        study = st.slider("Daily Study Hours", 0, 12, 4)
        attendance = st.slider("Attendance (%)", 0, 100, 75)
        sleep = st.slider("Sleep Hours", 0, 12, 7)
        assignments = st.slider("Assignment Completion (%)", 0, 100, 70)

        submitted = st.form_submit_button("🚀 Analyze My Performance")

    st.divider()
    st.caption("Powered by Python & Machine Learning")

# ---------------- HEADER ----------------
st.markdown("""
<div class="hero">
    <h1>🎓 Student Study Habit Analyzer</h1>
    <p>Smart academic performance prediction using Machine Learning</p>
    <p>Analyze your habits • Understand your performance • Improve your future</p>
</div>
""", unsafe_allow_html=True)

# ---------------- PREDICTION ----------------
if "prediction" not in st.session_state:
    st.session_state.prediction = None

if submitted:
    result = model.predict([[
        study, attendance, sleep, assignments
    ]])[0]

    st.session_state.prediction = {
        "result": result,
        "study": study,
        "attendance": attendance,
        "sleep": sleep,
        "assignments": assignments
    }

# ---------------- DASHBOARD ----------------
prediction = st.session_state.prediction

if prediction is None:
    st.info("👈 Enter your study details in the sidebar and click Analyze My Performance.")

    st.markdown("## ✨ Dashboard Overview")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="card">
        <div class="card-title">🤖 ML ALGORITHM</div>
        <div class="card-value">Decision Tree</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="card">
        <div class="card-title">📊 INPUT FEATURES</div>
        <div class="card-value">4 Features</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="card">
        <div class="card-title">🎯 PREDICTION TYPES</div>
        <div class="card-value">3 Classes</div>
        </div>
        """, unsafe_allow_html=True)

else:
    result = prediction["result"]
    study = prediction["study"]
    attendance = prediction["attendance"]
    sleep = prediction["sleep"]
    assignments = prediction["assignments"]

    # Result card
    if result == "Good":
        emoji = "🌟"
        message = "Excellent! Keep maintaining your study habits."
    elif result == "Average":
        emoji = "📘"
        message = "You are doing well. There is room for improvement."
    else:
        emoji = "💡"
        message = "Start improving your daily study routine."

    st.markdown(f"""
    <div class="result">
        <p>YOUR ML PREDICTION</p>
        <h2>{emoji} {result} Performance</h2>
        <p>{message}</p>
        <p>Algorithm: Decision Tree Classifier</p>
    </div>
    """, unsafe_allow_html=True)

    # Metrics
    st.markdown("## 📊 Your Performance Overview")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("📚 Study Hours", f"{study} hrs/day")
    with c2:
        st.metric("🏫 Attendance", f"{attendance}%")
    with c3:
        st.metric("😴 Sleep", f"{sleep} hrs/day")
    with c4:
        st.metric("📝 Assignments", f"{assignments}%")

    # Charts
    st.markdown("## 📈 Study Habit Analysis")

    left, right = st.columns(2)

    with left:
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.markdown("### Your Input Statistics")

        chart_data = pd.DataFrame({
            "Category": [
                "Study Hours (out of 12)",
                "Attendance",
                "Sleep Hours (out of 12)",
                "Assignments"
            ],
            "Score": [
                study / 12 * 100,
                attendance,
                sleep / 12 * 100,
                assignments
            ]
        }).set_index("Category")

        st.bar_chart(chart_data, horizontal=True, height=280)
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.markdown("### Model Information")
        st.write("**Algorithm:** Decision Tree Classifier")
        st.write("**Training records:**", len(df))
        st.write("**Input features:** 4")
        st.write("**Output classes:** 3")
        st.write("**Library:** Scikit-learn")
        st.write("**Learning type:** Supervised Learning")
        st.markdown('</div>', unsafe_allow_html=True)

    # Recommendations
    st.markdown("## 💡 Personalized Study Recommendations")

    suggestions = []

    if study < 3:
        suggestions.append("📚 Increase your daily study time to at least 3 hours.")

    if attendance < 75:
        suggestions.append("🏫 Try to maintain attendance of at least 75%.")

    if sleep < 7:
        suggestions.append("😴 Maintain a regular and healthy sleep schedule.")

    if assignments < 70:
        suggestions.append("📝 Complete pending assignments on time.")

    if not suggestions:
        suggestions.append("🌟 Your entered study habits look balanced. Keep it up!")

    for suggestion in suggestions:
        st.success(suggestion)

    # Performance indicators
    st.markdown("## 🎯 Individual Habit Indicators")

    habits = {
        "Study Hours": (study / 12) * 100,
        "Attendance": attendance,
        "Sleep Hours": (sleep / 12) * 100,
        "Assignments": assignments
    }

    for name, value in habits.items():
        st.write(f"**{name}** — {value:.0f}%")
        st.progress(int(value))

    st.caption("These indicators show input levels, not scientifically validated performance scores.")

# ---------------- ABOUT ----------------
st.divider()

with st.expander("ℹ️ About This Project"):
    st.write("""
    Student Study Habit Analyzer is a beginner-level Machine Learning
    mini project.

    It uses a Decision Tree Classifier to predict an academic
    performance category based on four inputs:

    1. Daily study hours
    2. Attendance percentage
    3. Daily sleep hours
    4. Assignment completion percentage

    The model is trained on a small illustrative sample dataset.
    Its predictions are for demonstration and are not validated
    academic assessments.
    """)

st.markdown("""
<div style="text-align:center; padding:20px; color:#9ca8c9;">
    🎓 Student Study Habit Analyzer | Python | Machine Learning | Streamlit
    <br>College Mini Project
</div>
""", unsafe_allow_html=True)
