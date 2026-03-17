"""
app.py — Credit Risk Analytics Dashboard
Streamlit app with 3 pages: EDA, Model Performance, Live Risk Scorer
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os

# ─── Page Config ───
st.set_page_config(
    page_title="CreditIQ",
    page_icon="public/logo.svg",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS for Professional Black Look ───
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Global pure black background with subtle gradient */
    .stApp {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #000000 0%, #050510 100%) !important;
    }
    
    [data-testid="stAppViewContainer"] {
        background: transparent !important;
    }
    
    [data-testid="stHeader"] {
        background-color: transparent !important;
    }

    /* Hero Header */
    .hero-header {
        background: linear-gradient(145deg, #0a0a0a, #111116);
        padding: 2rem 2.5rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        border-left: 4px solid #4facfe;
        border-top: 1px solid #222;
        border-right: 1px solid #222;
        border-bottom: 1px solid #222;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.6);
    }
    .hero-header h1 {
        color: #ffffff;
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
        background: linear-gradient(90deg, #ffffff, #a0a0a0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-header p {
        color: #a0a0a0;
        font-size: 1.05rem;
        margin-top: 0.5rem;
        margin-bottom: 0;
    }

    /* Metric Cards */
    .metric-card {
        background: linear-gradient(145deg, #080808, #151515);
        border: 1px solid #2a2a2a;
        border-radius: 12px;
        padding: 1.4rem 1.6rem;
        text-align: center;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.4);
    }
    .metric-card:hover {
        border-color: #4facfe;
        box-shadow: 0 0 20px rgba(79, 172, 254, 0.15);
        transform: translateY(-3px);
    }
    .metric-card .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0.3rem 0;
    }
    .metric-card .metric-label {
        color: #00f2fe;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        font-weight: 600;
    }

    /* Risk Score Output Styles */
    .risk-low {
        background: linear-gradient(135deg, rgba(46, 204, 113, 0.15), rgba(46, 204, 113, 0.05));
        border: 1px solid rgba(46, 204, 113, 0.5);
        padding: 2.5rem;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 0 30px rgba(46, 204, 113, 0.1);
    }
    .risk-medium {
        background: linear-gradient(135deg, rgba(241, 196, 15, 0.15), rgba(241, 196, 15, 0.05));
        border: 1px solid rgba(241, 196, 15, 0.5);
        padding: 2.5rem;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 0 30px rgba(241, 196, 15, 0.1);
    }
    .risk-high {
        background: linear-gradient(135deg, rgba(231, 76, 60, 0.15), rgba(231, 76, 60, 0.05));
        border: 1px solid rgba(231, 76, 60, 0.5);
        padding: 2.5rem;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 0 30px rgba(231, 76, 60, 0.1);
    }
    .risk-label {
        font-size: 1.1rem;
        font-weight: 600;
        color: #a0a0a0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .risk-value {
        font-size: 4rem;
        font-weight: 800;
        color: #ffffff;
        margin: 0.5rem 0;
    }
    .risk-verdict-low { color: #2ecc71; font-size: 1.5rem; font-weight: 700; text-shadow: 0 0 10px rgba(46, 204, 113, 0.5); }
    .risk-verdict-medium { color: #f1c40f; font-size: 1.5rem; font-weight: 700; text-shadow: 0 0 10px rgba(241, 196, 15, 0.5); }
    .risk-verdict-high { color: #e74c3c; font-size: 1.5rem; font-weight: 700; text-shadow: 0 0 10px rgba(231, 76, 60, 0.5); }

    /* Section Headers */
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #ffffff;
        margin: 2rem 0 1.5rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid transparent;
        border-image: linear-gradient(to right, #4facfe, transparent) 1;
    }

    /* Model comparison banner */
    .best-model-banner {
        background: linear-gradient(135deg, rgba(79, 172, 254, 0.15) 0%, rgba(0, 242, 254, 0.05) 100%);
        border: 1px solid rgba(79, 172, 254, 0.4);
        padding: 1.2rem 2rem;
        border-radius: 12px;
        text-align: center;
        margin: 1.5rem 0;
        box-shadow: 0 0 25px rgba(79, 172, 254, 0.15);
    }
    .best-model-banner h3 {
        color: #ffffff;
        margin: 0;
        font-size: 1.3rem;
        font-weight: 600;
        text-shadow: 0 0 10px rgba(79, 172, 254, 0.4);
    }

    /* Hide Streamlit footer and menu */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #050505 0%, #0a0a0f 100%) !important;
        border-right: 1px solid #1a1a1a;
    }
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #ffffff;
    }

    /* Text elements universally white-ish */
    p, span, div, li {
        color: #cccccc;
    }
</style>
<!-- Load Material Icons -->
<link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
""", unsafe_allow_html=True)


# ─── Plotly Theme ───
PLOTLY_TEMPLATE = "plotly_dark"
COLOR_PALETTE = ['#4facfe', '#00f2fe', '#f5576c', '#fa709a']

# ─── Data Loading (cached) ───
@st.cache_data
def load_raw_data():
    return pd.read_csv('data/credit_risk_dataset.csv')


@st.cache_data
def load_clean_data():
    df = load_raw_data()
    df = df[df['person_age'] <= 100].copy()
    df = df[df['person_emp_length'] <= 60].copy()
    df['person_emp_length'].fillna(df['person_emp_length'].median(), inplace=True)
    df['loan_int_rate'].fillna(df['loan_int_rate'].median(), inplace=True)
    return df


@st.cache_resource
def load_models():
    lr = joblib.load('assets/lr_model.pkl')
    rf = joblib.load('assets/rf_model.pkl')
    scaler = joblib.load('assets/scaler.pkl')
    feature_names = joblib.load('assets/feature_names.pkl')
    metrics = joblib.load('assets/metrics_data.pkl')
    return lr, rf, scaler, feature_names, metrics


# ─── Sidebar Navigation ───
st.sidebar.image("public/logo.svg", width=60)
st.sidebar.markdown("<h2>CreditIQ</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<hr style='border-color: #333;'>", unsafe_allow_html=True)
page = st.sidebar.radio(
    "Navigate",
    ["EDA Dashboard", "Model Performance", "Risk Scorer"],
    index=0
)
st.sidebar.markdown("<hr style='border-color: #333;'>", unsafe_allow_html=True)
st.sidebar.markdown(
    """
    <div style='text-align:center; color:#888888; font-size:0.75rem;'>
    Built by <span style="color: #ffffff;">Siddhant Shukla</span><br>
    ML-Powered Credit Risk System<br>
    32,000+ loan records analyzed
    </div>
    """,
    unsafe_allow_html=True
)


# ═══════════════════════════════════════════════════════
#  PAGE 1: EDA DASHBOARD
# ═══════════════════════════════════════════════════════
if page == "EDA Dashboard":
    df = load_clean_data()

    # Hero
    st.markdown("""
    <div class="hero-header">
        <h1>Exploratory Data Analysis</h1>
        <p>Deep dive into 32,000+ loan records — distributions, default patterns, and feature correlations</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Key Metrics Row ──
    total_records = len(df)
    default_rate = df['loan_status'].mean() * 100
    avg_loan = df['loan_amnt'].mean()
    avg_rate = df['loan_int_rate'].mean()
    avg_income = df['person_income'].mean()

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Records</div>
            <div class="metric-value">{total_records:,}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Default Rate</div>
            <div class="metric-value">{default_rate:.1f}%</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Avg Loan Amount</div>
            <div class="metric-value">${avg_loan:,.0f}</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Avg Interest Rate</div>
            <div class="metric-value">{avg_rate:.1f}%</div>
        </div>""", unsafe_allow_html=True)
    with c5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Avg Income</div>
            <div class="metric-value">${avg_income:,.0f}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 1: Loan Amount Distribution + Class Distribution ──
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown('<div class="section-header"><i class="material-icons" style="vertical-align: middle; font-size: 20px;">payments</i> Loan Amount Distribution</div>', unsafe_allow_html=True)
        fig = px.histogram(
            df, x='loan_amnt', nbins=50,
            color='loan_status',
            color_discrete_map={0: '#3498db', 1: '#e74c3c'},
            labels={'loan_amnt': 'Loan Amount ($)', 'loan_status': 'Default Status'},
            template=PLOTLY_TEMPLATE,
            barmode='overlay',
            opacity=0.75,
        )
        fig.update_layout(
            height=380, margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis_title="Loan Amount ($)", yaxis_title="Count",
            font=dict(family="Inter"), paper_bgcolor='#0a0a0a', plot_bgcolor='#0a0a0a'
        )
        fig.for_each_trace(lambda t: t.update(name='No Default' if t.name == '0' else 'Default'))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header"><i class="material-icons" style="vertical-align: middle; font-size: 20px;">pie_chart</i> Class Distribution</div>', unsafe_allow_html=True)
        class_counts = df['loan_status'].value_counts()
        fig = px.pie(
            values=class_counts.values,
            names=['No Default', 'Default'],
            color_discrete_sequence=['#3498db', '#e74c3c'],
            template=PLOTLY_TEMPLATE,
            hole=0.55,
        )
        fig.update_traces(textinfo='percent+label', textfont_size=13, marker=dict(line=dict(color='#0a0a0a', width=2)))
        fig.update_layout(
            height=380, margin=dict(l=20, r=20, t=30, b=20),
            showlegend=False, font=dict(family="Inter"),
            paper_bgcolor='#0a0a0a',
            annotations=[dict(text=f'{default_rate:.1f}%<br>Default', x=0.5, y=0.5,
                             font_size=16, font_color='#e74c3c', showarrow=False)]
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Row 2: Default Rate by Loan Grade + Loan Intent ──
    col3, col4 = st.columns(2)
    with col3:
        st.markdown('<div class="section-header"><i class="material-icons" style="vertical-align: middle; font-size: 20px;">bar_chart</i> Default Rate by Loan Grade</div>', unsafe_allow_html=True)
        grade_default = df.groupby('loan_grade')['loan_status'].mean().reset_index()
        grade_default.columns = ['Loan Grade', 'Default Rate']
        grade_default['Default Rate'] = grade_default['Default Rate'] * 100
        fig = px.bar(
            grade_default, x='Loan Grade', y='Default Rate',
            color='Default Rate',
            color_continuous_scale=['#3498db', '#e74c3c'],
            template=PLOTLY_TEMPLATE,
            text=grade_default['Default Rate'].apply(lambda x: f'{x:.1f}%'),
        )
        fig.update_traces(textposition='outside')
        fig.update_layout(
            height=400, margin=dict(l=20, r=20, t=30, b=20),
            yaxis_title="Default Rate (%)", coloraxis_showscale=False,
            font=dict(family="Inter"), paper_bgcolor='#0a0a0a', plot_bgcolor='#0a0a0a'
        )
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.markdown('<div class="section-header"><i class="material-icons" style="vertical-align: middle; font-size: 20px;">sort</i> Default Rate by Loan Purpose</div>', unsafe_allow_html=True)
        intent_default = df.groupby('loan_intent')['loan_status'].mean().reset_index()
        intent_default.columns = ['Loan Purpose', 'Default Rate']
        intent_default['Default Rate'] = intent_default['Default Rate'] * 100
        intent_default = intent_default.sort_values('Default Rate', ascending=True)
        fig = px.bar(
            intent_default, x='Default Rate', y='Loan Purpose',
            orientation='h',
            color='Default Rate',
            color_continuous_scale=['#2ecc71', '#e74c3c'],
            template=PLOTLY_TEMPLATE,
            text=intent_default['Default Rate'].apply(lambda x: f'{x:.1f}%'),
        )
        fig.update_traces(textposition='outside')
        fig.update_layout(
            height=400, margin=dict(l=20, r=20, t=30, b=20),
            xaxis_title="Default Rate (%)", coloraxis_showscale=False,
            font=dict(family="Inter"), paper_bgcolor='#0a0a0a', plot_bgcolor='#0a0a0a'
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Row 3: Income Distribution + Age Distribution ──
    col5, col6 = st.columns(2)
    with col5:
        st.markdown('<div class="section-header"><i class="material-icons" style="vertical-align: middle; font-size: 20px;">account_balance_wallet</i> Income Distribution by Status</div>', unsafe_allow_html=True)
        fig = px.box(
            df, x='loan_status', y='person_income',
            color='loan_status',
            color_discrete_map={0: '#3498db', 1: '#e74c3c'},
            template=PLOTLY_TEMPLATE,
            labels={'person_income': 'Annual Income ($)', 'loan_status': 'Default Status'}
        )
        fig.update_layout(
            height=380, margin=dict(l=20, r=20, t=30, b=20),
            showlegend=False, font=dict(family="Inter"),
            xaxis=dict(tickvals=[0, 1], ticktext=['No Default', 'Default']),
            paper_bgcolor='#0a0a0a', plot_bgcolor='#0a0a0a'
        )
        st.plotly_chart(fig, use_container_width=True)

    with col6:
        st.markdown('<div class="section-header"><i class="material-icons" style="vertical-align: middle; font-size: 20px;">person</i> Age Distribution</div>', unsafe_allow_html=True)
        fig = px.histogram(
            df, x='person_age', nbins=40,
            color='loan_status',
            color_discrete_map={0: '#3498db', 1: '#e74c3c'},
            template=PLOTLY_TEMPLATE,
            barmode='overlay', opacity=0.7,
            labels={'person_age': 'Age', 'loan_status': 'Default Status'}
        )
        fig.update_layout(
            height=380, margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            font=dict(family="Inter"), paper_bgcolor='#0a0a0a', plot_bgcolor='#0a0a0a'
        )
        fig.for_each_trace(lambda t: t.update(name='No Default' if t.name == '0' else 'Default'))
        st.plotly_chart(fig, use_container_width=True)

    # ── Row 4: Correlation Heatmap ──
    st.markdown('<div class="section-header"><i class="material-icons" style="vertical-align: middle; font-size: 20px;">grid_on</i> Feature Correlation Matrix</div>', unsafe_allow_html=True)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corr_matrix = df[numeric_cols].corr()

    fig = px.imshow(
        corr_matrix,
        text_auto='.2f',
        color_continuous_scale='RdBu_r',
        template=PLOTLY_TEMPLATE,
        aspect='auto',
    )
    fig.update_layout(
        height=500, margin=dict(l=20, r=20, t=30, b=20),
        font=dict(family="Inter", size=11),
        paper_bgcolor='#0a0a0a', plot_bgcolor='#0a0a0a'
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Dataset Preview ──
    with st.expander("View Raw Dataset"):
        st.dataframe(df.head(100), use_container_width=True, height=400)
        st.caption(f"Showing first 100 of {len(df):,} records")


# ═══════════════════════════════════════════════════════
#  PAGE 2: MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════
elif page == "Model Performance":
    # Check if models exist
    if not os.path.exists('assets/rf_model.pkl'):
        st.error("Error: Models not trained yet. Run `python model.py` first.")
        st.stop()

    lr, rf, scaler, feature_names, metrics = load_models()

    # Hero
    st.markdown("""
    <div class="hero-header">
        <h1>Model Performance Comparison</h1>
        <p>Side-by-side evaluation of Logistic Regression vs Random Forest classifiers</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Best Model Banner ──
    lr_auc = metrics['lr']['roc_auc']
    rf_auc = metrics['rf']['roc_auc']
    best_name = 'Random Forest' if rf_auc >= lr_auc else 'Logistic Regression'
    best_auc = max(lr_auc, rf_auc)
    st.markdown(f"""
    <div class="best-model-banner">
        <h3><i class="material-icons" style="vertical-align: middle; font-size: 24px;">verified</i> Superior Model: {best_name} — ROC-AUC: {best_auc:.4f}</h3>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Side-by-side Metrics ──
    col1, col2 = st.columns(2)

    for col, name, m_key in [(col1, 'Logistic Regression', 'lr'), (col2, 'Random Forest', 'rf')]:
        with col:
            st.markdown(f"### {name}")
            mc1, mc2 = st.columns(2)
            mc3, mc4 = st.columns(2)
            mc1.metric("Accuracy", f"{metrics[m_key]['accuracy']:.4f}")
            mc2.metric("Precision", f"{metrics[m_key]['precision']:.4f}")
            mc3.metric("Recall", f"{metrics[m_key]['recall']:.4f}")
            mc4.metric("F1-Score", f"{metrics[m_key]['f1']:.4f}")
            st.metric("ROC-AUC", f"{metrics[m_key]['roc_auc']:.4f}")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── ROC Curves ──
    st.markdown('<div class="section-header"><i class="material-icons" style="vertical-align: middle; font-size: 20px;">insights</i> ROC Curves Comparison</div>', unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=metrics['lr_fpr'], y=metrics['lr_tpr'],
        mode='lines', name=f"Logistic Regression (AUC={lr_auc:.4f})",
        line=dict(color='#3498db', width=3)
    ))
    fig.add_trace(go.Scatter(
        x=metrics['rf_fpr'], y=metrics['rf_tpr'],
        mode='lines', name=f"Random Forest (AUC={rf_auc:.4f})",
        line=dict(color='#e74c3c', width=3)
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode='lines', name='Random (AUC=0.5)',
        line=dict(color='#555555', width=1, dash='dash')
    ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE, height=450,
        xaxis_title='False Positive Rate', yaxis_title='True Positive Rate',
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
        margin=dict(l=20, r=20, t=30, b=20), font=dict(family="Inter"),
        paper_bgcolor='#0a0a0a', plot_bgcolor='#0a0a0a'
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Confusion Matrices ──
    st.markdown('<div class="section-header"><i class="material-icons" style="vertical-align: middle; font-size: 20px;">border_all</i> Confusion Matrices</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    for col, name, cm_key in [(col1, 'Logistic Regression', 'lr_cm'), (col2, 'Random Forest', 'rf_cm')]:
        with col:
            cm = np.array(metrics[cm_key])
            fig = px.imshow(
                cm, text_auto=True,
                labels=dict(x="Predicted", y="Actual"),
                x=['No Default', 'Default'], y=['No Default', 'Default'],
                color_continuous_scale=['#0a0a0a', '#3498db', '#e74c3c'],
                template=PLOTLY_TEMPLATE,
            )
            fig.update_layout(
                title=dict(text=name, font=dict(size=14)),
                height=350, margin=dict(l=20, r=20, t=50, b=20),
                font=dict(family="Inter"), paper_bgcolor='#0a0a0a', plot_bgcolor='#0a0a0a'
            )
            st.plotly_chart(fig, use_container_width=True)

    # ── Feature Importance ──
    st.markdown('<div class="section-header"><i class="material-icons" style="vertical-align: middle; font-size: 20px;">format_list_numbered</i> Feature Importance (Random Forest — Top 10)</div>', unsafe_allow_html=True)

    importances = metrics['rf_feature_importance']
    feat_names = metrics['feature_names']
    feat_df = pd.DataFrame({'Feature': feat_names, 'Importance': importances})
    feat_df = feat_df.sort_values('Importance', ascending=True).tail(10)

    # Color: high importance (risk drivers) in red, lower in green
    colors = ['#e74c3c' if v > feat_df['Importance'].median() else '#2ecc71' for v in feat_df['Importance']]

    fig = go.Figure(go.Bar(
        x=feat_df['Importance'], y=feat_df['Feature'],
        orientation='h', marker_color=colors,
        text=feat_df['Importance'].apply(lambda x: f'{x:.3f}'),
        textposition='outside',
    ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE, height=450,
        margin=dict(l=20, r=60, t=30, b=20),
        xaxis_title='Importance Score', font=dict(family="Inter"),
        yaxis=dict(tickfont=dict(size=12)), paper_bgcolor='#0a0a0a', plot_bgcolor='#0a0a0a'
    )
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════
#  PAGE 3: LIVE RISK SCORER
# ═══════════════════════════════════════════════════════
elif page == "Risk Scorer":
    # Check if models exist
    if not os.path.exists('assets/rf_model.pkl'):
        st.error("Error: Models not trained yet. Run `python model.py` first.")
        st.stop()

    lr, rf, scaler, feature_names, metrics_data = load_models()
    df = load_clean_data()

    # Hero
    st.markdown("""
    <div class="hero-header">
        <h1>Live Borrower Risk Scorer</h1>
        <p>Enter borrower details below and get an instant ML-powered default risk prediction.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar Inputs ──
    st.sidebar.markdown("### Borrower Profile")

    age = st.sidebar.slider("Applicant Age", 18, 80, 30)
    income = st.sidebar.number_input("Annual Income ($)", 5000, 500000, 50000, step=1000)
    emp_length = st.sidebar.slider("Employment Length (years)", 0, 40, 5)
    loan_amnt = st.sidebar.number_input("Loan Amount ($)", 500, 35000, 10000, step=500)
    loan_int_rate = st.sidebar.slider("Interest Rate (%)", 5.0, 24.0, 11.0, step=0.1)
    loan_grade = st.sidebar.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
    loan_intent = st.sidebar.selectbox(
        "Loan Purpose",
        ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"]
    )
    home_ownership = st.sidebar.selectbox(
        "Home Ownership",
        ["RENT", "OWN", "MORTGAGE", "OTHER"]
    )
    default_history = st.sidebar.radio("Prior Default on Record?", ["No", "Yes"])
    cred_hist_length = st.sidebar.slider("Credit History Length (years)", 2, 30, 5)

    # Calculate loan_percent_income
    loan_pct_income = round(loan_amnt / income, 2) if income > 0 else 0

    # Button
    predict_clicked = st.sidebar.button("Assess Risk", use_container_width=True, type="primary")

    if predict_clicked:
        # ── Encode inputs to match training format ──
        grade_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
        default_map = {'No': 0, 'Yes': 1}

        input_dict = {
            'person_age': age,
            'person_income': income,
            'person_emp_length': emp_length,
            'loan_grade': grade_map[loan_grade],
            'loan_amnt': loan_amnt,
            'loan_int_rate': loan_int_rate,
            'loan_percent_income': loan_pct_income,
            'cb_person_default_on_file': default_map[default_history],
            'cb_person_cred_hist_length': cred_hist_length,
        }

        # One-hot encode loan_intent
        intent_categories = ['EDUCATION', 'HOMEIMPROVEMENT', 'MEDICAL', 'PERSONAL', 'VENTURE']
        for cat in intent_categories:
            input_dict[f'loan_intent_{cat}'] = 1 if loan_intent == cat else 0

        # One-hot encode home_ownership
        home_categories = ['MORTGAGE', 'OTHER', 'OWN', 'RENT']
        for cat in home_categories:
            input_dict[f'person_home_ownership_{cat}'] = 1 if home_ownership == cat else 0

        # Build DataFrame in correct feature order
        input_df = pd.DataFrame([input_dict])

        # Ensure all feature columns exist and are in the right order
        for col in feature_names:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[feature_names]

        # Scale
        input_scaled = scaler.transform(input_df)

        # Predict
        default_prob = rf.predict_proba(input_scaled)[0][1] * 100
        lr_prob = lr.predict_proba(input_scaled)[0][1] * 100

        # ── Display Results ──
        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:
            # Determine risk level
            if default_prob < 30:
                risk_class = 'risk-low'
                risk_icon = 'verified_user'
                risk_verdict = 'LOW RISK'
                verdict_class = 'risk-verdict-low'
            elif default_prob < 60:
                risk_class = 'risk-medium'
                risk_icon = 'warning'
                risk_verdict = 'MEDIUM RISK'
                verdict_class = 'risk-verdict-medium'
            else:
                risk_class = 'risk-high'
                risk_icon = 'error'
                risk_verdict = 'HIGH RISK'
                verdict_class = 'risk-verdict-high'

            st.markdown(f"""
            <div class="{risk_class}">
                <div class="risk-label">Default Probability</div>
                <div class="risk-value">{default_prob:.1f}%</div>
                <div class="{verdict_class}"><i class="material-icons" style="vertical-align: middle;">{risk_icon}</i> {risk_verdict}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Risk Progress Bar ──
        st.markdown('<div class="section-header"><i class="material-icons" style="vertical-align: middle; font-size: 20px;">speed</i> Risk Level Gauge</div>', unsafe_allow_html=True)
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=default_prob,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Default Probability (%)", 'font': {'size': 18, 'color': '#ffffff'}},
            number={'suffix': '%', 'font': {'size': 40, 'color': '#ffffff'}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': '#555555'},
                'bar': {'color': '#4facfe'},
                'bgcolor': '#0a0a0a',
                'borderwidth': 2,
                'bordercolor': '#333333',
                'steps': [
                    {'range': [0, 30], 'color': 'rgba(46, 204, 113, 0.4)'},
                    {'range': [30, 60], 'color': 'rgba(241, 196, 15, 0.4)'},
                    {'range': [60, 100], 'color': 'rgba(231, 76, 60, 0.4)'}
                ],
                'threshold': {
                    'line': {'color': 'white', 'width': 4},
                    'thickness': 0.85,
                    'value': default_prob
                }
            }
        ))
        fig.update_layout(
            template=PLOTLY_TEMPLATE, height=300,
            margin=dict(l=30, r=30, t=50, b=20),
            paper_bgcolor='rgba(0,0,0,0)', font=dict(family="Inter")
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Model Comparison for this borrower ──
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Random Forest Prediction</div>
                <div class="metric-value">{default_prob:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        with col_b:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Logistic Regression Prediction</div>
                <div class="metric-value">{lr_prob:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Top Risk Factors ──
        st.markdown('<div class="section-header"><i class="material-icons" style="vertical-align: middle; font-size: 20px;">query_stats</i> Top Risk Factors for This Prediction</div>', unsafe_allow_html=True)

        # Use feature importance * input values to approximate contribution
        importances = rf.feature_importances_
        input_values = input_scaled[0]
        contributions = np.abs(importances * input_values)
        contrib_df = pd.DataFrame({
            'Feature': feature_names,
            'Impact': contributions
        }).sort_values('Impact', ascending=False).head(5)

        # Display clean feature names
        name_map = {
            'loan_percent_income': 'Loan-to-Income Ratio',
            'loan_int_rate': 'Interest Rate',
            'loan_grade': 'Loan Grade',
            'person_income': 'Income',
            'loan_amnt': 'Loan Amount',
            'person_age': 'Applicant Age',
            'cb_person_default_on_file': 'Prior Default History',
            'person_emp_length': 'Employment Length',
            'cb_person_cred_hist_length': 'Credit History Length',
        }

        for i, row in contrib_df.iterrows():
            feat = row['Feature']
            clean_name = name_map.get(feat, feat.replace('_', ' ').title())
            impact_pct = (row['Impact'] / contrib_df['Impact'].sum()) * 100
            color = '#e74c3c' if impact_pct > 20 else '#f1c40f' if impact_pct > 10 else '#2ecc71'
            st.markdown(f"""
            <div style="display:flex; align-items:center; margin-bottom:8px;">
                <div style="width:200px; font-weight:600; color:#dddddd;">{clean_name}</div>
                <div style="flex:1; background:#111111; border-radius:4px; height:20px; overflow:hidden; border: 1px solid #333333;">
                    <div style="width:{impact_pct:.0f}%; background:{color}; height:100%; border-radius:4px; 
                         transition: width 0.5s ease;"></div>
                </div>
                <div style="width:60px; text-align:right; font-weight:600; color:{color};">{impact_pct:.0f}%</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Borrower Summary ──
        st.markdown('<div class="section-header"><i class="material-icons" style="vertical-align: middle; font-size: 20px;">person_outline</i> Borrower Profile Summary</div>', unsafe_allow_html=True)
        summary_col1, summary_col2 = st.columns(2)
        with summary_col1:
            st.markdown(f"""
            | Parameter | Value |
            |---|---|
            | **Age** | {age} years |
            | **Income** | ${income:,} |
            | **Employment** | {emp_length} years |
            | **Home** | {home_ownership} |
            | **Credit History** | {cred_hist_length} years |
            """)
        with summary_col2:
            st.markdown(f"""
            | Parameter | Value |
            |---|---|
            | **Loan Amount** | ${loan_amnt:,} |
            | **Interest Rate** | {loan_int_rate}% |
            | **Loan Grade** | {loan_grade} |
            | **Purpose** | {loan_intent} |
            | **Loan/Income** | {loan_pct_income:.2f} |
            | **Prior Default** | {default_history} |
            """)

    else:
        # Show instructions when nothing is predicted yet
        st.markdown("""
        <div style="text-align:center; padding:4rem 2rem;">
            <i class="material-icons" style="font-size: 4rem; color: #4facfe;">rule</i>
            <h2 style="color:#ffffff; margin:1rem 0;">Ready to Assess Risk</h2>
            <p style="color:#888888; font-size:1.1rem; max-width:500px; margin:0 auto;">
                Fill in the borrower details in the sidebar and click 
                <strong>"Assess Risk"</strong> to get an instant ML-powered prediction.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Show some quick stats
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header"><i class="material-icons" style="vertical-align: middle; font-size: 20px;">info</i> Model Summary Information</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        rf_acc = metrics_data['rf']['accuracy']
        rf_auc = metrics_data['rf']['roc_auc']
        rf_f1 = metrics_data['rf']['f1']
        rf_prec = metrics_data['rf']['precision']
        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">RF Accuracy</div>
                <div class="metric-value">{rf_acc:.1%}</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">RF ROC-AUC</div>
                <div class="metric-value">{rf_auc:.4f}</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">RF F1-Score</div>
                <div class="metric-value">{rf_f1:.4f}</div>
            </div>""", unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">RF Precision</div>
                <div class="metric-value">{rf_prec:.4f}</div>
            </div>""", unsafe_allow_html=True)
