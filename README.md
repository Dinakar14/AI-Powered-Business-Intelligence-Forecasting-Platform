# AI-Powered-Business-Intelligence-Forecasting-Platform
A full-stack Business Intelligence (BI) platform that provides real-time analytics, secure authentication, and AI-based revenue forecasting, inspired by tools like Power BI.  Built using FastAPI, MySQL, Streamlit, and Prophet.

📌 Features

✅ Secure Login & User Registration (JWT Authentication)
✅ Role-based Access Control (Admin / User)
✅ Power BI–style Tabbed Dashboard
✅ Real-time MySQL Analytics
✅ AI-based Revenue Forecasting (Prophet)
✅ Interactive Charts (Plotly)
✅ CSV & PDF Report Downloads
✅ Dark Mode Toggle
✅ Production-level Error Handling

🧠 Tech Stack
Layer	Technology
Frontend	Streamlit
Backend	FastAPI
Database	MySQL
ORM	SQLAlchemy
Authentication	JWT + bcrypt
Forecasting	Facebook Prophet
Visualization	Plotly
Reports	CSV, PDF
Language	Python
🏗️ Architecture
User
 ↓
Streamlit Dashboard (UI)
 ↓ API Calls
FastAPI Backend
 ↓ ORM
MySQL Database
 ↓
Forecasting Engine (Prophet)

🗂️ Project Structure
ai_bi_forecasting_platform/
│
├── api/
│   ├── main.py          # FastAPI backend
│   ├── auth.py          # Authentication & JWT
│   ├── database.py      # MySQL connection
│   ├── models.py        # ORM models
│
├── dashboard/
│   └── app.py           # Streamlit dashboard
│
├── requirements.txt
└── README.md

🛠️ Setup & Installation
1️⃣ Clone Repository
git clone https://github.com/yourusername/ai-bi-forecasting-platform.git
cd ai-bi-forecasting-platform

2️⃣ Create Virtual Environment
python -m venv .venv
.venv\Scripts\activate

3️⃣ Install Dependencies
pip install -r requirements.txt

🗄️ MySQL Setup

Start MySQL Server

Create database:

CREATE DATABASE bi_platform;


Insert sample data:

INSERT INTO sales (sale_date, revenue, region, product)
VALUES
('2024-01-01',1200,'South','A'),
('2024-01-02',1350,'South','A'),
('2024-01-03',1500,'North','B');

▶️ Running the Application
Start Backend (FastAPI)
uvicorn api.main:app --reload


API Docs:

http://127.0.0.1:8000/docs

Start Frontend (Streamlit)
streamlit run dashboard/app.py


Dashboard:

http://localhost:8501

🔐 Authentication
Role	Access
Admin	Full dashboard, analytics, forecasting
User	View analytics & reports

Passwords are bcrypt-hashed, authentication is JWT-based.

📈 Forecasting Logic

Sales data loaded from MySQL

Cleaned & validated

Converted to Prophet format (ds, y)

30-day future forecast generated

Interactive visualization using Plotly

📥 Reports

CSV Export – Raw sales data

PDF Export – Business-ready reports

🧪 Error Handling

Handles empty datasets gracefully

Prevents invalid ML training

Safe PDF encoding

User-friendly UI warnings
