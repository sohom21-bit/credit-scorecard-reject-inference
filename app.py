import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.set_page_config(
    page_title='Credit Scorecard | LendingClub',
    page_icon=' ',
    layout='wide'
)

@st.cache_resource
def load_models():
    lr_model          = joblib.load('saved_objects/lr_model.pkl')
    lr_fuzzy          = joblib.load('saved_objects/lr_fuzzy.pkl')
    woe_tables        = joblib.load('saved_objects/woe_tables.pkl')
    SELECTED_FEATURES = joblib.load('saved_objects/selected_features.pkl')
    metrics           = joblib.load('saved_objects/baseline_metrics.pkl')
    df_woe            = pd.read_csv('saved_objects/df_woe.csv')
    iv_display_df     = joblib.load('saved_objects/iv_display.pkl')
    psi_display_df    = joblib.load('saved_objects/psi_display.pkl')
    return lr_model, lr_fuzzy, woe_tables, SELECTED_FEATURES, metrics, df_woe, iv_display_df, psi_display_df

lr_model, lr_fuzzy, woe_tables, SELECTED_FEATURES, metrics, df_woe, iv_display_df, psi_display_df = load_models()
CAT_FEATURES = ['term','home_ownership','verification_status',
                'purpose','state_risk_tier']

def apply_woe_single(value, feature, woe_tables, cat=False):
    wt = woe_tables[feature]
    if cat:
        woe_map = dict(zip(wt['bin'].astype(str), wt['woe']))
        return woe_map.get(str(value), 0.0)
    else:
        edges = [-np.inf]
        if hasattr(wt['bin'].iloc[0], 'right'):
            for interval in wt['bin']:
                edges.append(interval.right)
        edges[-1] =  np.inf
        edges[0]  = -np.inf
        edges = sorted(set(edges))
        bin_idx = np.digitize([value], edges[1:], right=True)[0]
        bin_idx = np.clip(bin_idx, 0, len(wt)-1)
        return wt['woe'].values[bin_idx]

def score_to_points(prob, pdo=20, base_score=600, base_odds=50):
    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(base_odds)
    odds   = (1 - prob) / max(prob, 1e-6)
    score  = offset + factor * np.log(max(odds, 1e-6))
    return int(np.clip(score, 300, 850))

def get_decision(score):
    if score >= 580:  return 'APPROVE', '#2ecc71'
    if score >= 520:  return 'REFER',   '#f39c12'
    return 'DECLINE', '#e74c3c'

st.sidebar.title('Navigation')
page = st.sidebar.radio('Go to', ['Home','Score Applicant','Model Performance'])

# ── HOME ────────────────────────────────────────────────────
if page == 'Home':
    st.title('Credit Default Prediction with Reject Inference')
    st.markdown('### Production-Grade Credit Scorecard | LendingClub 2007-2018')
    st.markdown('**Author:** MS Statistics | IIT Kanpur')
    st.divider()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric('Modelling Population', '1,345,310')
    col2.metric('Rejected Applications', '27,648,741')
    col3.metric('Baseline Gini', f"{metrics['gini']:.4f}")
    col4.metric('Baseline AUC',  f"{metrics['auc']:.4f}")
    st.divider()
    st.subheader('Key Results')
    results = {
        'Model': ['Baseline','Augmentation','Fuzzy Parcelling','Self Learning'],
        'AUC':   [0.7049, 0.6748, 0.7049, 0.6382],
        'Gini':  [0.4098, 0.3497, 0.4098, 0.2765],
        'KS':    [0.2957, 0.2537, 0.2958, 0.1977],
        'PSI':   [np.float64(0.0038), np.float64(0.0234), np.float64(0.0038), np.float64(0.0141)],
    }
    st.dataframe(pd.DataFrame(results), use_container_width=True)

# ── SCORE APPLICANT ─────────────────────────────────────────
elif page == 'Score Applicant':
    st.title('Score an Applicant')
    st.markdown('Enter applicant details to get a credit score and decision.')
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Loan Details')
        loan_amnt   = st.slider('Loan Amount ($)', 1500, 35000, 10000, step=500)
        term        = st.selectbox('Loan Term (in months)', [36, 60])
        purpose     = st.selectbox('Loan Purpose', [
            'debt_consolidation','credit_card','home_improvement',
            'other','major_purchase','medical','small_business',
            'car','vacation','moving','wedding','house'])
        int_rate    = st.slider('Interest Rate (%)', 5.0, 31.0, 13.0, step=0.1)
        installment = st.slider('(Monthly Installment ($)', 50, 1500, 300, step=10)
    with col2:
        st.subheader('Applicant Details')
        annual_inc  = st.slider('Annual Income ($)', 18000, 250000, 65000, step=1000)
        dti         = st.slider('Debt-to-Income Ratio (%)', 1.0, 40.0, 18.0, step=0.5)
        fico_low    = st.slider('FICO Score', 620, 840, 690, step=5)
        fico_avg    = fico_low + 4
        home_ownership      = st.selectbox('Home Ownership', ['RENT','MORTGAGE','OWN','OTHER'])
        verification_status = st.selectbox('Income Verification', ['Not Verified','Verified','Source Verified'])
        emp_length_num      = st.slider('Employment Length (years)', -1, 10, 3)
        revol_util          = st.slider('Revolving Utilisation (%)', 1.0, 98.0, 50.0, step=1.0)
        state_risk          = st.selectbox('State Risk Tier', ['low_risk','medium_risk','high_risk'])
    st.divider()
    if st.button('Score Applicant', type='primary', use_container_width=True):
        raw = {
            'loan_amnt'          : loan_amnt,
            'int_rate'           : int_rate,
            'installment'        : installment,
            'annual_inc'         : annual_inc,
            'dti'                : dti,
            'fico_avg'           : fico_avg,
            'revol_util'         : revol_util,
            'emp_length_num'     : emp_length_num,
            'term'               : str(float(term)),
            'home_ownership'     : home_ownership,
            'verification_status': verification_status,
            'purpose'            : purpose,
            'state_risk_tier'    : state_risk,
        }
        woe_row = {}
        for feat in SELECTED_FEATURES:
            if feat not in raw:
                woe_row[feat] = 0.0
                continue
            cat_flag = feat in CAT_FEATURES or feat == 'miss_emp_length'
            woe_row[feat] = apply_woe_single(raw[feat], feat, woe_tables, cat=cat_flag)
        X_input  = pd.DataFrame([woe_row])[SELECTED_FEATURES]
        prob     = lr_model.predict_proba(X_input)[0, 1]
        score    = score_to_points(prob)
        decision, dec_color = get_decision(score)
        st.divider()
        r1, r2, r3 = st.columns(3)
        r1.metric('Credit Score', f'{score}', delta='FICO-style 300-850')
        r2.metric('P(Default)', f'{prob:.1%}')
        r3.metric('Decision', decision)
        st.markdown(
            f"<h2 style='text-align:center;color:{dec_color};'>DECISION: {decision}</h2>",
            unsafe_allow_html=True)
        st.divider()
        st.subheader('WoE Feature Contributions')
        contrib = pd.DataFrame({
            'Feature'  : list(woe_row.keys()),
            'WoE Value': list(woe_row.values()),
        }).sort_values('WoE Value')
        fig, ax = plt.subplots(figsize=(8, 4))
        colors  = ['#e74c3c' if v < 0 else '#2ecc71' for v in contrib['WoE Value']]
        ax.barh(contrib['Feature'], contrib['WoE Value'], color=colors, edgecolor='white')
        ax.axvline(0, color='black', linewidth=1)
        ax.set_title('WoE Contributions — Green=Lower Risk, Red=Higher Risk')
        ax.set_xlabel('WoE Value')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ── MODEL PERFORMANCE ───────────────────────────────────────
elif page == 'Model Performance':
    st.title('Model Performance Dashboard')
    st.divider()
    col1, col2, col3 = st.columns(3)
    col1.metric('AUC',  f"{metrics['auc']:.4f}")
    col2.metric('Gini', f"{metrics['gini']:.4f}")
    col3.metric('KS',   f"{metrics['ks']:.4f}")
    st.divider()
    st.subheader('Feature Importance — Information Value')
    st.dataframe(iv_display_df, use_container_width=True, hide_index=True)
    st.divider()
    st.subheader('Default Rate by Score Band')
    band_df = pd.DataFrame({
        'Band'        : ['<10%','10-15%','15-20%','20-25%','25-30%','>30%'],
        'Default Rate': [8.8, 17.6, 24.3, 29.8, 34.8, 46.0],
        'Volume'      : [80467, 64366, 48904, 32375, 22064, 44919],
    })
    st.bar_chart(band_df.set_index('Band')['Default Rate'])
    st.divider()
    st.subheader('PSI — Population Stability')
    st.dataframe(psi_display_df, use_container_width=True, hide_index=True)
