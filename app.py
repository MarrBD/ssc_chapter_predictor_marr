import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from datetime import datetime
import gspread
from google.oauth2.service_account import Credentials

# ================== CONFIG ==================
LEVELS = {
    "SSC": {
        "active": True,
        "model_path": "ssc_chapter_importance_model.pkl",
        "data_path": "ssc_chapters_data.csv",
    },
    "HSC": {"active": False},
    "Admission (ভর্তি পরীক্ষা)": {"active": False},
    "Job (চাকরি পরীক্ষা)": {"active": False},
}

SUBJECT_COLORS = {
    "Math": "#4C9BE8", "Higher Math": "#8E6FE0", "Physics": "#F2A65A",
    "Chemistry": "#5FC98D", "Biology": "#E86C6C",
}

st.set_page_config(page_title="Chapter Importance Predictor", page_icon="📊", layout="wide")

st.markdown("""
<style>
@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
@keyframes floatDots {
    0%, 100% { transform: translateY(0px); opacity: 0.6; }
    50% { transform: translateY(-15px); opacity: 1; }
}
.stApp {
    background-color: #0a0d16;
    background-image:
        radial-gradient(circle at 15% 20%, rgba(76,155,232,0.30) 0%, transparent 32%),
        radial-gradient(circle at 88% 12%, rgba(95,201,141,0.25) 0%, transparent 32%),
        radial-gradient(circle at 25% 85%, rgba(142,111,224,0.28) 0%, transparent 38%),
        radial-gradient(circle at 82% 78%, rgba(242,166,90,0.22) 0%, transparent 32%),
        radial-gradient(circle at 55% 45%, rgba(232,108,108,0.15) 0%, transparent 40%),
        linear-gradient(120deg, #0a0d16 0%, #131a2b 30%, #0d1420 60%, #141020 100%);
    background-size: 200% 200%, 200% 200%, 200% 200%, 200% 200%, 200% 200%, 400% 400%;
    animation: gradientShift 18s ease infinite;
    background-attachment: fixed;
}
.stApp::before {
    content: "";
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background-image:
        radial-gradient(2px 2px at 10% 15%, rgba(255,255,255,0.9) 100%, transparent),
        radial-gradient(2px 2px at 25% 45%, rgba(120,200,255,0.8) 100%, transparent),
        radial-gradient(1.5px 1.5px at 40% 20%, rgba(255,255,255,0.7) 100%, transparent),
        radial-gradient(2px 2px at 60% 70%, rgba(150,255,180,0.8) 100%, transparent),
        radial-gradient(1.5px 1.5px at 75% 30%, rgba(255,255,255,0.7) 100%, transparent),
        radial-gradient(2px 2px at 85% 60%, rgba(255,200,120,0.8) 100%, transparent),
        radial-gradient(1.5px 1.5px at 92% 85%, rgba(255,255,255,0.6) 100%, transparent),
        radial-gradient(2px 2px at 50% 90%, rgba(200,150,255,0.8) 100%, transparent);
    background-size: 100% 100%;
    animation: floatDots 8s ease-in-out infinite;
    pointer-events: none;
    z-index: 0;
}
.block-container { padding-top: 2rem; position: relative; z-index: 1; }

h1 {
    background: linear-gradient(90deg, #4C9BE8, #8E6FE0, #5FC98D);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-weight: 800 !important;
}

.chapter-card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.08);
    border-left: 5px solid var(--accent);
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 12px;
    backdrop-filter: blur(8px);
    box-shadow: 0 4px 20px rgba(0,0,0,0.25);
    transition: transform 0.2s ease;
}
.chapter-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 24px rgba(0,0,0,0.35);
}
.comment-card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 14px 18px;
    margin-bottom: 10px;
    backdrop-filter: blur(8px);
}

section[data-testid="stSidebar"] {
    background: rgba(10,13,22,0.85);
    backdrop-filter: blur(10px);
    border-right: 1px solid rgba(255,255,255,0.06);
}
</style>
""", unsafe_allow_html=True)
# ================== Google Sheets Connection ==================
@st.cache_resource
def get_gsheet():
    scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    creds = Credentials.from_service_account_info(dict(st.secrets["gcp_service_account"]), scopes=scopes)
    client = gspread.authorize(creds)
    return client.open_by_key(st.secrets["SHEET_ID"]).sheet1

def get_all_comments():
    sheet = get_gsheet()
    records = sheet.get_all_records()
    return pd.DataFrame(records) if records else pd.DataFrame(columns=["timestamp","name","rating","comment","status"])

def add_comment(name, rating, comment):
    sheet = get_gsheet()
    sheet.append_row([datetime.now().strftime("%Y-%m-%d %H:%M"), name, rating, comment, "Pending"])

def approve_comment(row_index):
    sheet = get_gsheet()
    sheet.update_cell(row_index + 2, 5, "Approved")  # +2: header row + 1-indexed

def reject_comment(row_index):
    sheet = get_gsheet()
    sheet.delete_rows(row_index + 2)

# ================== Sidebar ==================
st.sidebar.title("⚙️ Settings")
level = st.sidebar.selectbox("শিক্ষাস্তর / ক্যাটেগরি (Level)", list(LEVELS.keys()))

if not LEVELS[level]["active"]:
    st.title("📊 Chapter Importance Predictor")
    st.warning(f"**{level}** — 🚧 শীঘ্রই আসছে (Coming Soon)! আপাতত শুধু **SSC** সচল আছে।")
    st.info("Sidebar থেকে 'SSC' সিলেক্ট করে ব্যবহার শুরু করুন।")
    st.stop()

@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)

@st.cache_data
def load_data(data_path):
    return pd.read_csv(data_path)

bundle = load_model(LEVELS[level]["model_path"])
model, scaler, feature_cols = bundle["model"], bundle["scaler"], bundle["feature_cols"]
df = load_data(LEVELS[level]["data_path"])

subject = st.sidebar.selectbox("বিষয় (Subject)", sorted(df["subject"].unique()))
target_year = st.sidebar.number_input("কোন বছরের জন্য predict করবেন?", min_value=2025, max_value=2035, value=2026, step=1)
top_n = st.sidebar.slider("কতগুলো Chapter দেখাবেন?", 3, 15, 5)

st.sidebar.divider()
page = st.sidebar.radio("📄 Page", ["🏠 Predictor", "⭐ Rate This App", "ℹ️ About", "🔐 Admin"])

# ================== Prediction Logic ==================
def predict_importance(df, model, scaler, feature_cols):
    df_encoded = pd.get_dummies(df, columns=["subject"], prefix="subject")
    for col in feature_cols:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    X = df_encoded[feature_cols]
    return np.clip(model.predict(scaler.transform(X)), 0, 100)

df["predicted_importance"] = predict_importance(df, model, scaler, feature_cols)
is_odd_year = target_year % 2 != 0
adjustment_direction = 1 if is_odd_year else -1
df["adjusted_importance"] = np.clip(df["predicted_importance"] + (adjustment_direction * df["parity_bias"] * 0.5), 0, 100)

# ================== PAGE: Predictor ==================
if page == "🏠 Predictor":
    st.title("📊 SSC Chapter Importance Predictor")
    st.caption("MARR | EUB, CSE, Batch 26 | ML Course Project")

    subject_df = df[df["subject"] == subject].sort_values("adjusted_importance", ascending=False).head(top_n)
    accent = SUBJECT_COLORS.get(subject, "#4C9BE8")

    st.subheader(f"{subject} — {target_year} সালের জন্য সবচেয়ে গুরুত্বপূর্ণ {top_n}টি Chapter")

    for i, row in enumerate(subject_df.itertuples(), 1):
        st.markdown(f"""
        <div class="chapter-card" style="--accent:{accent};">
            <b>{i}. Chapter {row.chapter_no}: {row.chapter_name_en}</b> ({row.chapter_name_bn})<br>
            <span style="color:{accent}; font-size:1.1em; font-weight:bold;">Score: {row.adjusted_importance:.1f}/100</span>
        </div>
        """, unsafe_allow_html=True)
        st.progress(min(int(row.adjusted_importance), 100))

    st.divider()
    st.subheader("বিস্তারিত টেবিল")
    st.dataframe(
        subject_df[["chapter_no","chapter_name_en","chapter_name_bn","predicted_importance","adjusted_importance"]]
        .rename(columns={"chapter_no":"Ch#","chapter_name_en":"Chapter (EN)","chapter_name_bn":"Chapter (BN)",
                          "predicted_importance":"Base Score","adjusted_importance":f"Adjusted Score ({target_year})"}),
        use_container_width=True, hide_index=True
    )

    st.divider()
    st.subheader("📈 Visualization")
    vcol1, vcol2 = st.columns(2)
    with vcol1:
        fig_pie = px.pie(subject_df, values="adjusted_importance", names="chapter_name_en",
                          title=f"{subject} — Importance Share", color_discrete_sequence=px.colors.sequential.Blues_r)
        st.plotly_chart(fig_pie, use_container_width=True)
    with vcol2:
        fig_bar = px.bar(subject_df.sort_values("adjusted_importance"), x="adjusted_importance", y="chapter_name_en",
                          orientation="h", title=f"{subject} — Score Comparison",
                          color="adjusted_importance", color_continuous_scale="Blues")
        fig_bar.update_layout(yaxis_title="", xaxis_title="Importance Score")
        st.plotly_chart(fig_bar, use_container_width=True)

    st.divider()
    st.subheader("💬 User Reviews")
    try:
        comments_df = get_all_comments()
        approved = comments_df[comments_df["status"] == "Approved"] if not comments_df.empty else comments_df
        if approved.empty:
            st.caption("এখনো কোনো approved review নেই — প্রথম review দিতে 'Rate This App' পেজে যান।")
        else:
            for _, r in approved.sort_values("timestamp", ascending=False).iterrows():
                stars = "⭐" * int(r["rating"])
                st.markdown(f"""
                <div class="comment-card">
                    <b>{r['name']}</b> — {stars}<br>
                    <span style="opacity:0.85;">{r['comment']}</span><br>
                    <span style="font-size:0.8em; opacity:0.5;">{r['timestamp']}</span>
                </div>
                """, unsafe_allow_html=True)
    except Exception as e:
        st.caption("Reviews লোড করা যায়নি।")

# ================== PAGE: Rate This App ==================
elif page == "⭐ Rate This App":
    st.title("⭐ App নিয়ে আপনার মতামত দিন")
    with st.form("rating_form"):
        name = st.text_input("আপনার নাম")
        rating = st.slider("Rating (1-5)", 1, 5, 5)
        comment = st.text_area("মন্তব্য")
        submitted = st.form_submit_button("জমা দিন")
        if submitted:
            if name.strip() and comment.strip():
                try:
                    with st.spinner("জমা হচ্ছে..."):
                        add_comment(name.strip(), rating, comment.strip())
                    st.success("✅ ধন্যবাদ! আপনার মতামত জমা হয়েছে, admin approve করলে এটা publicly দেখা যাবে।")
                    st.balloons()
                    st.toast("রিভিউ সফলভাবে জমা হয়েছে!", icon="🎉")
                except Exception as e:
                    st.exception(e)
            else:
                st.warning("নাম ও মন্তব্য দুটোই দিন।")

# ================== PAGE: About ==================
elif page == "ℹ️ About":
    st.title("ℹ️ About This App")
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown("### 👤")
        st.caption("(Photo শীঘ্রই যোগ হবে)")
    with col2:
        st.markdown("""
        **Developer:** Md. Abdur Razzak Roni (MARR)
        **Student ID:** 230321059
        **Department:** CSE, European University of Bangladesh (EUB), Batch 26 — 9th Semester
        **Course:** Machine Learning
        **Course Coordinator:** Mohammad Mehadi Hasan, Assistant Professor, CSE, EUB
        **Data Entry সহায়তা:** Also my teammate: Abidur Rahman, Tuly, Mainuddin
        """)
    st.divider()
    st.subheader("📦 Dataset Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("মোট Chapter", "70")
    c2.metric("বিষয় সংখ্যা", "5")
    c3.metric("বছরের পরিসর", "2015–2024")
    c4.metric("মোট Board", "8")
    st.caption("সব NCTB syllabus অনুযায়ী verified, 8টা board (Dhaka, Rajshahi, Chittagong, Jessore, Comilla, Sylhet, Dinajpur, Barisal) থেকে সংগ্রহ করা।")

    st.subheader("🔍 Data Sample")
    st.dataframe(df[["subject","chapter_no","chapter_name_en","chapter_name_bn"]].head(8), use_container_width=True, hide_index=True)

    st.subheader("🧠 Model")
    st.write("Linear Regression | Test R² = 0.929 | 5-Fold CV R² = 0.954")
    st.caption("App তৈরি হয়েছে: আগস্ট, ২০২৬")

# ================== PAGE: Admin ==================
elif page == "🔐 Admin":
    st.title("🔐 Admin Panel")
    pw = st.text_input("Admin Password", type="password")
    if pw == st.secrets["ADMIN_PASSWORD"]:
        st.success("✅ Login সফল")
        try:
            comments_df = get_all_comments()
            pending = comments_df[comments_df["status"] == "Pending"] if not comments_df.empty else comments_df
            if pending.empty:
                st.info("এখন কোনো pending comment নেই।")
            else:
                for idx, r in pending.iterrows():
                    st.markdown(f"**{r['name']}** — {'⭐'*int(r['rating'])}")
                    st.write(r['comment'])
                    c1, c2 = st.columns(2)
                    if c1.button("✅ Approve", key=f"appr_{idx}"):
                        approve_comment(idx)
                        st.rerun()
                    if c2.button("❌ Reject", key=f"rej_{idx}"):
                        reject_comment(idx)
                        st.rerun()
                    st.divider()
        except Exception as e:
            st.exception(e)
    elif pw:
        st.error("❌ ভুল পাসওয়ার্ড")
