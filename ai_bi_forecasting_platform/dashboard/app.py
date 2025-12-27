import streamlit as st
import requests
import pandas as pd
from sqlalchemy import create_engine, text
from prophet import Prophet
import plotly.express as px
from fpdf import FPDF

# =========================
# CONFIG
# =========================
API_URL = "http://127.0.0.1:8000"
DB_URL = "mysql+pymysql://root:root@localhost/bi_platform"

engine = create_engine(DB_URL)

st.set_page_config(
    page_title="AI BI Forecasting Platform",
    page_icon="📊",
    layout="wide"
)

# =========================
# SESSION STATE
# =========================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = None

if "role" not in st.session_state:
    st.session_state.role = None

# =========================
# DARK MODE
# =========================
dark_mode = st.sidebar.toggle("🌙 Dark Mode")

if dark_mode:
    st.markdown("""
        <style>
        body { background-color: #0e1117; color: white; }
        </style>
    """, unsafe_allow_html=True)

# =========================
# DATABASE FUNCTIONS
# =========================
def load_sales():
    query = text("SELECT * FROM sales")
    return pd.read_sql(query, engine)

# =========================
# FORECAST FUNCTION (SAFE)
# =========================
def run_forecast(df):
    df = df.copy()

    df = df.dropna(subset=["sale_date", "revenue"])
    df["sale_date"] = pd.to_datetime(df["sale_date"], errors="coerce")
    df = df.dropna(subset=["sale_date"])

    df = df.rename(columns={
        "sale_date": "ds",
        "revenue": "y"
    })

    if len(df) < 2:
        return None

    model = Prophet()
    model.fit(df)

    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)

    return forecast

# =========================
# PDF REPORT
# =========================
def generate_pdf(df):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, "Sales Report", ln=True)

    for _, row in df.head(20).iterrows():
        pdf.cell(200, 8, f"{row.sale_date} - Rs {row.revenue}", ln=True)

    return pdf.output(dest="S").encode("latin-1")

# =========================
# LOGIN PAGE
# =========================
def login_page():
    st.title("🔐 Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        res = requests.post(
            f"{API_URL}/login",
            params={"username": username, "password": password}
        )

        if res.status_code == 200:
            data = res.json()
            st.session_state.logged_in = True
            st.session_state.username = username
            st.session_state.role = data["role"]
            st.success("Login successful")
            st.rerun()
        else:
            st.error("Invalid credentials")

# =========================
# DASHBOARD PAGE
# =========================
def dashboard_page():
    st.sidebar.write(f"👤 User: {st.session_state.username}")
    st.sidebar.write(f"🔐 Role: {st.session_state.role}")

    if st.sidebar.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.session_state.username = None
        st.session_state.role = None
        st.rerun()

    st.title("🚀 AI-Powered BI & Forecasting Platform")

    tabs = st.tabs([
        "📊 Overview",
        "📈 Analytics",
        "🔮 Forecast",
        "📥 Reports",
        "⚙ Register"
    ])

    # ✅ df IS DEFINED HERE (ONLY PLACE)
    df = load_sales()

    # -------- Overview --------
    with tabs[0]:
        st.subheader("📊 Business Overview")

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Revenue", f"₹{df.revenue.sum():,.0f}")
        col2.metric("Avg Revenue", f"₹{df.revenue.mean():,.0f}")
        col3.metric("Total Records", len(df))

        st.bar_chart(df.groupby("region")["revenue"].sum())

    # -------- Analytics --------
    with tabs[1]:
        st.subheader("📈 Sales Analytics")
        st.dataframe(df)

    # -------- Forecast --------
    with tabs[2]:
        st.subheader("🔮 Revenue Forecast")

        result = run_forecast(df)

        if result is None:
            st.warning("Not enough data to generate forecast")
        else:
            fig = px.line(
                result,
                x="ds",
                y="yhat",
                title="30-Day Revenue Forecast"
            )
            st.plotly_chart(fig, use_container_width=True)

    # -------- Reports --------
    with tabs[3]:
        st.subheader("📥 Download Reports")

        st.download_button(
            "⬇ Download CSV",
            df.to_csv(index=False),
            file_name="sales_report.csv",
            mime="text/csv"
        )

        st.download_button(
            "⬇ Download PDF",
            generate_pdf(df),
            file_name="sales_report.pdf",
            mime="application/pdf"
        )

    # -------- Register --------
    with tabs[4]:
        st.subheader("🆕 Register New User")

        new_user = st.text_input("New Username")
        new_pass = st.text_input("New Password", type="password")

        if st.button("Register"):
            res = requests.post(
                f"{API_URL}/register",
                params={"username": new_user, "password": new_pass}
            )

            if res.status_code == 200:
                st.success("User registered successfully")
            else:
                st.error("User already exists")

# =========================
# ROUTER
# =========================
if st.session_state.logged_in:
    dashboard_page()
else:
    login_page()
