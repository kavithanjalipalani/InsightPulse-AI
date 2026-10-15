
import re
from collections import Counter
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    vader = SentimentIntensityAnalyzer()
except ImportError:
    vader = None


# ==========================================================
# EDUPULSE AI - STUDENT FEEDBACK INTELLIGENCE DASHBOARD
# ==========================================================

st.set_page_config(
    page_title="EduPulse AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------- CUSTOM CSS ----------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #f4f7ff, #f9fbff);
}

.block-container {
    padding-top: 1.5rem;
    max-width: 1450px;
}

[data-testid="stSidebar"] {
    background: #15213b;
}

[data-testid="stSidebar"] * {
    color: white;
}

.hero {
    padding: 32px;
    border-radius: 22px;
    color: white;
    background: linear-gradient(120deg, #172746, #4359b9, #7965dc);
    box-shadow: 0 12px 35px rgba(55, 73, 150, .17);
    margin-bottom: 25px;
}

.hero h1 {
    color: white;
    font-size: 2.35rem;
    font-weight: 800;
    margin-bottom: 8px;
}

.hero p {
    color: #e5eaff;
    font-size: 1rem;
}

.metric {
    padding: 20px;
    background: white;
    border: 1px solid #e6ebf5;
    border-radius: 18px;
    box-shadow: 0 5px 18px rgba(30, 50, 90, .05);
    min-height: 125px;
}

.metric-label {
    color: #75819a;
    font-size: .83rem;
    font-weight: 700;
}

.metric-value {
    color: #1c2946;
    font-size: 2rem;
    font-weight: 800;
    margin: 8px 0;
}

.metric-note {
    color: #8994a9;
    font-size: .78rem;
}

.section-title {
    color: #1c2946;
    font-size: 1.3rem;
    font-weight: 800;
    margin-top: 15px;
}

div[data-testid="stDataFrame"] {
    border-radius: 12px;
}

.stButton button, .stDownloadButton button {
    border-radius: 10px;
    font-weight: 700;
}

</style>
""", unsafe_allow_html=True)


# ------------------------- DEMO DATA -----------------------

DEMO_DATA = [
    ("The faculty explain topics clearly and are very helpful.", "Computer Science", "2026-09-01"),
    ("The lab computers are slow and the software keeps crashing.", "Computer Science", "2026-09-02"),
    ("The library is quiet and has a good collection of books.", "Business Studies", "2026-09-03"),
    ("The canteen food is expensive and the waiting time is too long.", "Electronics", "2026-09-04"),
    ("The course materials are okay, nothing special.", "Business Studies", "2026-09-05"),
    ("The timetable changes at the last minute and it is frustrating.", "Computer Science", "2026-09-06"),
    ("I enjoyed the practical sessions and group activities.", "Electronics", "2026-09-07"),
    ("The Wi-Fi connection is unreliable in classrooms.", "Computer Science", "2026-09-08"),
    ("The staff were friendly and resolved my issue quickly.", "Business Studies", "2026-09-09"),
    ("The assignment instructions were confusing and unclear.", "Electronics", "2026-09-10"),
    ("Good campus environment and friendly students.", "Computer Science", "2026-09-11"),
    ("There are not enough books and seats in the library.", "Business Studies", "2026-09-12"),
    ("The online portal is easy to use and works smoothly.", "Electronics", "2026-09-13"),
    ("The classroom projector is broken again.", "Computer Science", "2026-09-14"),
    ("The workload is manageable and the course is average.", "Business Studies", "2026-09-15"),
    ("Excellent mentoring and supportive faculty.", "Electronics", "2026-09-16"),
    ("The exam schedule was announced too late.", "Computer Science", "2026-09-17"),
    ("The placement training gave me confidence for interviews.", "Business Studies", "2026-09-18"),
    ("The drinking water area is not clean.", "Electronics", "2026-09-19"),
    ("The workshops were informative and well organised.", "Computer Science", "2026-09-20"),
]

DEMO_DF = pd.DataFrame(
    DEMO_DATA,
    columns=["Feedback", "Department", "Date"]
)


# --------------------- SENTIMENT ANALYSIS ------------------

POSITIVE_WORDS = {
    "good", "great", "excellent", "amazing", "helpful", "friendly",
    "clear", "clearly", "enjoyed", "love", "loved", "useful",
    "supportive", "smoothly", "easy", "best", "informative",
    "organised", "organized", "confidence", "resolved", "quiet",
    "happy", "recommend", "impressive", "wonderful", "fantastic",
    "appreciate", "comfortable", "effective", "patiently"
}

NEGATIVE_WORDS = {
    "bad", "poor", "slow", "crashing", "expensive", "frustrating",
    "unreliable", "confusing", "unclear", "broken", "waste",
    "problem", "problems", "dirty", "difficult", "late",
    "terrible", "worst", "disappointed", "annoying", "inadequate",
    "crowded", "unsafe", "uncomfortable", "insufficient"
}


def fallback_sentiment(text):
    """Simple word-list fallback when VADER is unavailable."""

    words = re.findall(r"[a-z']+", str(text).lower())

    positive = sum(word in POSITIVE_WORDS for word in words)
    negative = sum(word in NEGATIVE_WORDS for word in words)

    if positive > negative:
        return "Positive", 0.65, 0.4

    if negative > positive:
        return "Negative", 0.65, -0.4

    return "Neutral", 0.50, 0.0


def analyze_sentiment(text):
    """Return sentiment label, strength indicator and signed score."""

    text = str(text).strip()

    if not text:
        return "Neutral", 0.0, 0.0

    if vader is not None:
        score = vader.polarity_scores(text)["compound"]

        if score >= 0.05:
            label = "Positive"
        elif score <= -0.05:
            label = "Negative"
        else:
            label = "Neutral"

        return label, abs(score), score

    return fallback_sentiment(text)


# --------------------- TOPIC DETECTION ---------------------

TOPICS = {
    "Teaching & Faculty": [
        "faculty", "teacher", "teaching", "lecture", "mentor", "staff"
    ],
    "Labs & Technology": [
        "lab", "computer", "software", "projector", "wifi",
        "wi-fi", "internet", "portal", "system"
    ],
    "Facilities & Cleanliness": [
        "library", "canteen", "food", "water", "clean",
        "seat", "classroom", "campus"
    ],
    "Timetable & Exams": [
        "timetable", "schedule", "exam", "assignment",
        "workload", "deadline"
    ],
    "Placements & Career": [
        "placement", "interview", "career", "training", "workshop"
    ],
    "Course Content": [
        "course", "material", "topic", "subject",
        "curriculum", "practical", "activity"
    ]
}


def detect_topics(feedback_list):
    rows = []

    for topic, keywords in TOPICS.items():
        count = sum(
            any(word in str(text).lower() for word in keywords)
            for text in feedback_list
        )

        if count > 0:
            rows.append({
                "Topic": topic,
                "Mentions": count
            })

    return pd.DataFrame(
        rows, columns=["Topic", "Mentions"]
    ).sort_values("Mentions", ascending=False)


# ------------------------- SIDEBAR -------------------------

with st.sidebar:
    st.markdown("# 🎓 EduPulse AI")
    st.caption("STUDENT FEEDBACK INTELLIGENCE")
    st.divider()

    st.markdown("### 📂 Data source")

    data_source = st.radio(
        "Choose your data",
        [
            "Demo dataset",
            "Upload CSV",
            "Enter feedback"
        ]
    )

    st.divider()

    st.markdown("### 🎛️ Dashboard options")

    show_neutral = st.toggle(
        "Include neutral feedback",
        value=True
    )

    st.caption(
        "Upload a CSV containing student feedback "
        "to analyse your own data."
    )

    st.divider()

    st.markdown("**Technology**")
    st.caption("Python · NLP · Streamlit")
    st.caption("Pandas · Plotly · VADER")


# ------------------------- HEADER --------------------------

st.markdown("""
<div class="hero">
    <p>✦ AI-POWERED CAMPUS INSIGHTS</p>
    <h1>Understand student voices.</h1>
    <p>
        Transform student feedback into sentiment trends,
        discover recurring complaints and explore actionable
        insights in one interactive dashboard.
    </p>
</div>
""", unsafe_allow_html=True)


# ---------------------- LOAD THE DATA -----------------------

if "manual_feedback" not in st.session_state:
    st.session_state.manual_feedback = []

raw_df = pd.DataFrame()

if data_source == "Demo dataset":
    raw_df = DEMO_DF.copy()

elif data_source == "Upload CSV":
    uploaded_file = st.file_uploader(
        "Upload your student feedback CSV",
        type=["csv"]
    )

    if uploaded_file is not None:
        try:
            raw_df = pd.read_csv(uploaded_file)
        except Exception as error:
            st.error(f"Unable to read this CSV: {error}")

    else:
        st.info("Upload a CSV file to start analysing.")
        st.stop()

elif data_source == "Enter feedback":
    with st.form("feedback_form", clear_on_submit=True):
        feedback_text = st.text_area(
            "Enter student feedback",
            placeholder="Example: The lab computers are slow, but the faculty are helpful."
        )

        department = st.selectbox(
            "Department",
            [
                "Computer Science",
                "Business Studies",
                "Electronics",
                "Mechanical",
                "Other"
            ]
        )

        submitted = st.form_submit_button(
            "➕ Add feedback",
            type="primary"
        )

        if submitted:
            if feedback_text.strip():
                st.session_state.manual_feedback.append({
                    "Feedback": feedback_text.strip(),
                    "Department": department,
                    "Date": datetime.now().strftime("%Y-%m-%d")
                })

                st.success("Feedback added successfully!")
            else:
                st.warning("Please enter feedback first.")

    if st.session_state.manual_feedback:
        raw_df = pd.DataFrame(st.session_state.manual_feedback)

        if st.button("Clear entered feedback"):
            st.session_state.manual_feedback = []
            st.rerun()
    else:
        st.info("Add feedback using the form above.")
        st.stop()


if raw_df.empty:
    st.warning("No data available to analyse.")
    st.stop()


# -------------------- SELECT FEEDBACK COLUMN ----------------

column_names = list(raw_df.columns)

default_index = next(
    (
        i for i, column in enumerate(column_names)
        if str(column).strip().lower() in [
            "feedback", "comment", "comments",
            "review", "response", "student_feedback"
        ]
    ),
    0
)

feedback_column = st.selectbox(
    "Select the column containing feedback",
    column_names,
    index=default_index
)

df = raw_df.copy()
df = df.rename(columns={feedback_column: "Feedback"})

df["Feedback"] = (
    df["Feedback"]
    .fillna("")
    .astype(str)
    .str.strip()
)

df = df[df["Feedback"] != ""].reset_index(drop=True)

if df.empty:
    st.warning("No non-empty feedback was found.")
    st.stop()

if "Department" not in df.columns:
    df["Department"] = "Not specified"

if "Date" not in df.columns:
    df["Date"] = None

if "Course" not in df.columns:
    df["Course"] = "Not specified"


# ----------------------- ANALYSE DATA -----------------------

with st.spinner("Analysing student feedback..."):
    sentiment_results = df["Feedback"].apply(analyze_sentiment)

    df["Sentiment"] = sentiment_results.apply(lambda x: x[0])
    df["Strength"] = sentiment_results.apply(lambda x: x[1])
    df["Sentiment Score"] = sentiment_results.apply(lambda x: x[2])
    df["Word Count"] = df["Feedback"].apply(
        lambda text: len(str(text).split())
    )


# --------------------------- FILTERS ------------------------

st.markdown(
    '<p class="section-title">🔎 Explore your feedback</p>',
    unsafe_allow_html=True
)

c1, c2, c3 = st.columns([1, 1, 1.5])

with c1:
    departments = sorted(
        df["Department"].fillna("Not specified").astype(str).unique()
    )

    selected_departments = st.multiselect(
        "Department",
        departments,
        default=departments
    )

with c2:
    sentiments = ["Positive", "Neutral", "Negative"]

    if not show_neutral:
        sentiments.remove("Neutral")

    selected_sentiments = st.multiselect(
        "Sentiment",
        sentiments,
        default=sentiments
    )

with c3:
    search_query = st.text_input(
        "Search feedback",
        placeholder="Search faculty, Wi-Fi, exams..."
    )

filtered = df[
    df["Department"].fillna("Not specified").astype(str).isin(
        selected_departments
    )
    & df["Sentiment"].isin(selected_sentiments)
].copy()

if search_query.strip():
    filtered = filtered[
        filtered["Feedback"].str.contains(
            re.escape(search_query.strip()),
            case=False,
            na=False
        )
    ]

if filtered.empty:
    st.warning("No results match these filters. Change your selection.")
    st.stop()


# ---------------------- DASHBOARD METRICS -------------------

total = len(filtered)

positive = int((filtered["Sentiment"] == "Positive").sum())
neutral = int((filtered["Sentiment"] == "Neutral").sum())
negative = int((filtered["Sentiment"] == "Negative").sum())

positive_pct = positive / total * 100
negative_pct = negative / total * 100
average_words = filtered["Word Count"].mean()

st.markdown(
    '<p class="section-title">📈 Your campus at a glance</p>',
    unsafe_allow_html=True
)

metrics = [
    ("TOTAL RESPONSES", f"{total}", "Feedback entries analysed"),
    ("POSITIVE SENTIMENT", f"{positive_pct:.1f}%", f"{positive} positive responses"),
    ("NEGATIVE SENTIMENT", f"{negative_pct:.1f}%", f"{negative} responses to review"),
    ("AVG. WORD COUNT", f"{average_words:.1f}", "Words per response")
]

metric_columns = st.columns(4)

for column, (label, value, note) in zip(metric_columns, metrics):
    with column:
        st.markdown(f"""
        <div class="metric">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """, unsafe_allow_html=True)

st.write("")


# ----------------------- DASHBOARD TABS ---------------------

overview_tab, explorer_tab, insights_tab, export_tab = st.tabs([
    "📊 Overview",
    "💬 Feedback Explorer",
    "🔍 Complaint Insights",
    "📥 Export Results"
])


# ----------------------- OVERVIEW TAB -----------------------

COLORS = {
    "Positive": "#20b486",
    "Neutral": "#9aa6bb",
    "Negative": "#f06464"
}

with overview_tab:
    left, right = st.columns([1, 1.3])

    with left:
        st.subheader("Sentiment distribution")

        sentiment_counts = (
            filtered["Sentiment"]
            .value_counts()
            .reindex(["Positive", "Neutral", "Negative"], fill_value=0)
            .rename_axis("Sentiment")
            .reset_index(name="Responses")
        )

        fig_pie = px.pie(
            sentiment_counts,
            names="Sentiment",
            values="Responses",
            hole=0.65,
            color="Sentiment",
            color_discrete_map=COLORS
        )

        fig_pie.update_traces(
            textinfo="percent+label",
            marker=dict(line=dict(color="white", width=3))
        )

        fig_pie.update_layout(
            height=360,
            showlegend=False,
            margin=dict(t=20, b=20, l=15, r=15),
            paper_bgcolor="rgba(0,0,0,0)"
        )

        fig_pie.add_annotation(
            text=f"<b>{total}</b><br>responses",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=20, color="#1c2946")
        )

        st.plotly_chart(fig_pie, use_container_width=True)

    with right:
        st.subheader("Sentiment by department")

        dept_sentiment = (
            filtered.groupby(["Department", "Sentiment"])
            .size()
            .reset_index(name="Responses")
        )

        fig_bar = px.bar(
            dept_sentiment,
            x="Department",
            y="Responses",
            color="Sentiment",
            barmode="group",
            color_discrete_map=COLORS
        )

        fig_bar.update_layout(
            height=360,
            margin=dict(t=20, b=20, l=10, r=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend_title_text=""
        )

        st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("Feedback trend over time")

    trend_data = filtered.copy()
    trend_data["_date"] = pd.to_datetime(
        trend_data["Date"], errors="coerce"
    )

    trend_data = trend_data.dropna(subset=["_date"])

    if not trend_data.empty:
        trend_data["_day"] = trend_data["_date"].dt.date

        trend = (
            trend_data.groupby(["_day", "Sentiment"])
            .size()
            .reset_index(name="Responses")
        )

        fig_line = px.line(
            trend,
            x="_day",
            y="Responses",
            color="Sentiment",
            markers=True,
            color_discrete_map=COLORS
        )

        fig_line.update_layout(
            height=320,
            xaxis_title="Date",
            yaxis_title="Responses",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend_title_text=""
        )

        st.plotly_chart(fig_line, use_container_width=True)
    else:
        st.info("A trend chart appears when your CSV has valid dates.")


# -------------------- FEEDBACK EXPLORER TAB -----------------

with explorer_tab:
    st.subheader("Student feedback explorer")
    st.caption("Inspect individual comments and their predicted sentiment.")

    display_columns = [
        "Feedback",
        "Sentiment",
        "Strength",
        "Department",
        "Course",
        "Date",
        "Word Count"
    ]

    display_columns = [
        column for column in display_columns
        if column in filtered.columns
    ]

    st.dataframe(
        filtered[display_columns],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Feedback": st.column_config.TextColumn(
                "Student feedback", width="large"
            ),
            "Strength": st.column_config.ProgressColumn(
                "Sentiment strength indicator",
                min_value=0,
                max_value=1,
                format="%.2f"
            )
        }
    )

    st.subheader("Read example comments")

    chosen_sentiment = st.selectbox(
        "Choose a sentiment",
        ["Negative", "Positive", "Neutral"]
    )

    examples = filtered[
        filtered["Sentiment"] == chosen_sentiment
    ].head(5)

    if examples.empty:
        st.info("No comments in this sentiment group.")
    else:
        for _, row in examples.iterrows():
            with st.container(border=True):
                st.write(str(row["Feedback"]))
                st.caption(
                    f"{row['Department']} · {row['Course']} · "
                    f"{chosen_sentiment}"
                )


# --------------------- COMPLAINT INSIGHTS TAB ---------------

with insights_tab:
    st.subheader("Recurring complaint topics")

    st.caption(
        "Keyword-based analysis of negative feedback. "
        "These are indicators for review, not confirmed root causes."
    )

    negative_feedback = filtered.loc[
        filtered["Sentiment"] == "Negative",
        "Feedback"
    ].astype(str).tolist()

    topic_df = detect_topics(negative_feedback)

    if not topic_df.empty:
        left, right = st.columns([1.2, 1])

        with left:
            fig_topics = px.bar(
                topic_df,
                x="Mentions",
                y="Topic",
                orientation="h",
                text="Mentions",
                color="Mentions",
                color_continuous_scale=["#bfc9ff", "#6659db"]
            )

            fig_topics.update_layout(
                height=350,
                yaxis={"categoryorder": "total ascending"},
                xaxis_title="Feedback entries mentioning topic",
                yaxis_title="",
                coloraxis_showscale=False,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )

            st.plotly_chart(fig_topics, use_container_width=True)

        with right:
            st.markdown("#### Areas to review")

            for _, row in topic_df.head(5).iterrows():
                st.markdown(
                    f"- **{row['Topic']}** — "
                    f"{int(row['Mentions'])} negative feedback "
                    f"entry/entries mention related terms."
                )

        st.info(
            "Suggested next step: manually review comments in the "
            "most frequently mentioned areas before deciding on action."
        )

    else:
        st.success("No negative feedback found in the current filters.")

    st.subheader("Common words in negative feedback")

    if negative_feedback:
        stop_words = {
            "the", "and", "is", "are", "was", "were", "a", "an",
            "to", "of", "in", "it", "for", "on", "with", "our",
            "we", "this", "that", "not", "but", "too", "there",
            "during", "again", "very", "my", "be", "have", "has",
            "at", "as", "by", "or", "from", "needs", "need"
        }

        words = []

        for comment in negative_feedback:
            tokens = re.findall(
                r"[a-zA-Z][a-zA-Z'-]+",
                comment.lower()
            )

            words.extend(
                word for word in tokens
                if word not in stop_words and len(word) > 2
            )

        common_words = Counter(words).most_common(12)

        if common_words:
            word_df = pd.DataFrame(
                common_words,
                columns=["Word", "Occurrences"]
            )

            fig_words = px.bar(
                word_df,
                x="Word",
                y="Occurrences",
                text="Occurrences",
                color="Occurrences",
                color_continuous_scale=["#ffd4c9", "#e87565"]
            )

            fig_words.update_layout(
                height=320,
                coloraxis_showscale=False,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )

            st.plotly_chart(fig_words, use_container_width=True)

    else:
        st.info("Negative comments will appear here when available.")


# ------------------------- EXPORT TAB -----------------------

with export_tab:
    st.subheader("Download your analysis")

    st.write(
        "Export the filtered feedback and sentiment results "
        "for further analysis or your project report."
    )

    export_columns = [
        "Feedback",
        "Sentiment",
        "Strength",
        "Sentiment Score",
        "Department",
        "Course",
        "Date",
        "Word Count"
    ]

    export_columns = [
        column for column in export_columns
        if column in filtered.columns
    ]

    export_df = filtered[export_columns].copy()

    csv_data = export_df.to_csv(
        index=False
    ).encode("utf-8-sig")

    st.download_button(
        label="⬇️ Download analysed feedback (CSV)",
        data=csv_data,
        file_name="edupulse_analysis.csv",
        mime="text/csv",
        type="primary"
    )

    st.subheader("Summary report")

    summary = pd.DataFrame({
        "Metric": [
            "Total responses",
            "Positive responses",
            "Neutral responses",
            "Negative responses",
            "Positive percentage",
            "Negative percentage"
        ],
        "Value": [
            total,
            positive,
            neutral,
            negative,
            f"{positive_pct:.1f}%",
            f"{negative_pct:.1f}%"
        ]
    })

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True
    )


# --------------------------- FOOTER -------------------------

st.divider()

st.markdown("""
<div style="text-align:center; padding:12px; color:#7b879d;">
    <b>🎓 EduPulse AI</b><br>
    Student Feedback Sentiment Analysis & Complaint Insights<br>
    Built with Python · Streamlit · Pandas · Plotly
</div>
""", unsafe_allow_html=True)

st.caption(
    "Note: Sentiment predictions can be incorrect, especially for sarcasm, "
    "mixed-language comments, and context-dependent feedback. "
    "Review important findings manually."
)
