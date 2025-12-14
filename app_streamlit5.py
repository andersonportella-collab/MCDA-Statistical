# app_streamlit.py
"""
MCDA Statistical Analysis — Streamlit bilingual app (Portuguese / English)
Complete app with: descriptive stats, normality tests, correlation, ANOVA/Tukey,
non-parametric tests, regression, Monte Carlo, export (Excel/PDF/JSON) and
manual AI analysis + chat (OpenAI).
Place UFF_UFRN_brasao.png in same folder for logo in UI and PDF.
Adapted for Anderson Portella
"""

import streamlit as st
import pandas as pd
import numpy as np
import io, os, tempfile, datetime, json, base64
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import OLSInfluence, variance_inflation_factor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# OpenAI client — optional
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# optional .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# -----------------------
# Config / Logo path
# -----------------------
LOGO_PATH = "UFF_UFRN_brasao.png"  # coloque o arquivo no mesmo diretório

# CORREÇÃO AQUI: Substituído \n por <br/> para compatibilidade com ReportLab
INSTITUTION_LINE = (
    "Universidade Federal Fluminense – Programa de Pós-Graduação em Engenharia de Produção<br/>"
    "Universidade Federal do Rio Grande do Norte - Departamento de Engenharia Industrial"
)

# -----------------------
# Bilingual text
# -----------------------
TEXT = {
    "app_title": {"Portuguese": "🤖 Análise Estatística MCDA — UFF/UFRN", "English": "🤖 MCDA Statistical Analysis — UFF/UFRN"},
    "upload": {"Portuguese": "Carregue arquivo Excel ou CSV", "English": "Upload Excel or CSV"},
    "download_template": {"Portuguese": "Baixar template Excel", "English": "Download Excel template"},
    "load_example": {"Portuguese": "Carregar exemplo (pequeno)", "English": "Load example (small)"},
    "run_all": {"Portuguese": "Executar todas as análises", "English": "Run all analyses"},
    "export_excel": {"Portuguese": "Gerar Excel", "English": "Generate Excel"},
    "export_pdf": {"Portuguese": "Gerar PDF", "English": "Generate PDF"},
    "data_preview": {"Portuguese": "Visualização dos dados", "English": "Data preview"},
    "results": {"Portuguese": "Resultados", "English": "Results"},
    "desc": {"Portuguese": "Estatística Descritiva", "English": "Descriptive Statistics"},
    "norm": {"Portuguese": "Testes de Normalidade", "English": "Normality Tests"},
    "corr": {"Portuguese": "Correlação", "English": "Correlation"},
    "anova": {"Portuguese": "ANOVA (1-way / 2-way) e Tukey", "English": "ANOVA (1-way / 2-way) and Tukey"},
    "nonparam": {"Portuguese": "Testes Não Paramétricos", "English": "Non-Parametric Tests"},
    "reg": {"Portuguese": "Regressão Linear", "English": "Linear Regression"},
    "mc": {"Portuguese": "Simulação Monte Carlo", "English": "Monte Carlo Simulation"},
    "sensitivity": {"Portuguese": "Análise de Sensibilidade", "English": "Sensitivity Analysis"},
    "export": {"Portuguese": "Exportar resultados", "English": "Export results"},
    "pdf_title": {"Portuguese": "Relatório Estatístico MCDA", "English": "MCDA Statistical Report"},
    "pdf_subtitle": {"Portuguese": "Relatório gerado automaticamente", "English": "Automatically generated report"},
    "no_numeric": {"Portuguese": "Nenhuma coluna numérica encontrada.", "English": "No numeric columns found."},
    "ai_button": {"Portuguese": "Gerar análise com IA", "English": "Generate AI analysis"},
    "ai_help": {"Portuguese": "Clique para gerar uma análise automática (manual) com a IA baseada nos resultados presentes.", "English": "Click to generate an automatic (manual) AI analysis based on the current results."},
    "chat_header": {"Portuguese": "💬 Chat com a IA (após análise)", "English": "💬 Chat with AI (post-analysis)"},
    "ai_not_configured": {"Portuguese": "🔑 Chave da API da OpenAI não configurada. Veja a barra lateral.", "English": "🔑 OpenAI API key not configured. See sidebar."}
}

def t(key, lang):
    return TEXT.get(key, {}).get(lang, key)

# -----------------------
# Template / Read / Validate
# -----------------------
def get_template_bytes(lang="Portuguese"):
    cols = ["Alternative", "Cost", "Efficiency", "Risk", "Group"]
    data = [
        ["A1", 100, 0.8, 2.0, "G1"],
        ["A2", 110, 0.85, 1.8, "G1"],
        ["A3", 95, 0.78, 2.2, "G2"],
        ["A4", 130, 0.9, 1.2, "G2"],
    ]
    df = pd.DataFrame(data, columns=cols)
    buf = io.BytesIO()
    df.to_excel(buf, index=False, engine="xlsxwriter")
    buf.seek(0)
    return buf.getvalue()

def read_upload(uploaded):
    name = getattr(uploaded, "name", "").lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded)
    else:
        return pd.read_excel(uploaded, sheet_name=0)

def validate_matrix(df):
    msgs = []
    if df is None or df.empty:
        msgs.append("Empty or None dataframe")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) == 0:
        msgs.append("No numeric columns")
    total_missing = int(df.isnull().sum().sum())
    if total_missing > 0:
        msgs.append(f"{total_missing} missing values found")
    if df.shape[0] < 2 or df.shape[1] < 2:
        msgs.append("Too few rows/columns for analyses (min 2x2)")
    return {"valid": len(numeric_cols) > 0 and not df.empty, "messages": msgs, "numeric_cols": numeric_cols}

# -----------------------
# Statistical functions (modular)
# -----------------------
def descriptive_stats(df):
    nums = df.select_dtypes(include=[np.number])
    if nums.shape[1] == 0:
        return pd.DataFrame()
    desc = nums.agg(['mean','median','std','var','min','max']).T
    desc['cv'] = desc['std'] / desc['mean'].replace(0, np.nan)
    desc['skew'] = nums.skew()
    desc['kurtosis'] = nums.kurtosis()
    try:
        mode_df = nums.mode().iloc[0]
    except Exception:
        mode_df = pd.Series([np.nan]*nums.shape[1], index=nums.columns)
    desc['mode'] = mode_df
    desc = desc.reset_index().rename(columns={'index':'criterion'})
    return desc

def normality_tests(df):
    nums = df.select_dtypes(include=[np.number])
    rows = []
    for c in nums.columns:
        s = nums[c].dropna()
        if s.shape[0] < 3:
            rows.append({'criterion': c, 'shapiro_stat': np.nan, 'shapiro_p': np.nan})
            continue
        try:
            stat, p = stats.shapiro(s)
        except Exception:
            stat, p = np.nan, np.nan
        rows.append({'criterion': c, 'shapiro_stat': float(stat), 'shapiro_p': float(p)})
    return pd.DataFrame(rows).set_index('criterion')

def correlation_matrix(df, method='pearson'):
    nums = df.select_dtypes(include=[np.number])
    if nums.shape[1] == 0:
        return pd.DataFrame()
    return nums.corr(method=method)

def anova_oneway(df, group_col, value_col):
    formula = f"{value_col} ~ C({group_col})"
    model = smf.ols(formula, data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    return model, anova_table

def tukey_hsd(df, group_col, value_col, alpha=0.05):
    try:
        res = pairwise_tukeyhsd(endog=df[value_col], groups=df[group_col].astype(str), alpha=alpha)
        data = pd.DataFrame(data=res.summary().data[1:], columns=res.summary().data[0])
        return data
    except Exception as e:
        return pd.DataFrame({"error":[str(e)]})

def kruskal_test(df, group_col, value_col):
    groups = [g[value_col].dropna().values for _, g in df.groupby(group_col)]
    if len(groups) < 2:
        return {"stat": np.nan, "pvalue": np.nan, "error":"Less than 2 groups"}
    stat, p = stats.kruskal(*groups)
    return {"stat": float(stat), "pvalue": float(p)}

def mannwhitney_test(df, group_col, value_col):
    groups = [g[value_col].dropna().values for _, g in df.groupby(group_col)]
    if len(groups) != 2:
        return {"stat": np.nan, "pvalue": np.nan, "error":"Requires exactly 2 groups"}
    stat, p = stats.mannwhitneyu(groups[0], groups[1], alternative='two-sided')
    return {"stat": float(stat), "pvalue": float(p)}

def linear_regression(df, y_col, x_cols):
    X = df[x_cols].astype(float)
    Xc = sm.add_constant(X, has_constant='add')
    y = df[y_col].astype(float)
    model = sm.OLS(y, Xc, missing='drop').fit()
    fitted = pd.Series(model.fittedvalues, index=y.dropna().index)
    resid = pd.Series(model.resid, index=fitted.index)
    # diagnostics
    try:
        bp_test = het_breuschpagan(resid, model.model.exog)
        bp_p = float(bp_test[1])
    except Exception:
        bp_p = np.nan
    try:
        dw = float(durbin_watson(resid))
    except Exception:
        dw = np.nan
    infl = OLSInfluence(model)
    cooks = infl.cooks_distance[0]
    mse = float(mean_squared_error(y.dropna(), fitted.loc[y.dropna().index]))
    mae = float(mean_absolute_error(y.dropna(), fitted.loc[y.dropna().index]))
    metrics = {
        "r2": float(model.rsquared),
        "adj_r2": float(model.rsquared_adj),
        "mse": mse,
        "mae": mae,
        "breusch_pagan_p": bp_p,
        "durbin_watson": dw,
        "max_cooks_distance": float(np.nanmax(cooks) if len(cooks)>0 else np.nan)
    }
    # VIF
    vif = {}
    try:
        X_no_const = Xc.loc[:, Xc.columns != 'const']
        for i, col in enumerate(X_no_const.columns):
            vif[col] = float(variance_inflation_factor(X_no_const.values, i))
    except Exception:
        vif = {}
    return model, metrics, vif, resid, fitted, cooks

def monte_carlo(df_numeric, n_iter=1000, noise_frac=0.05):
    arr = []
    df_numeric = df_numeric.copy().astype(float)
    idx = df_numeric.index
    for i in range(int(n_iter)):
        pert = df_numeric.copy()
        for c in pert.columns:
            rng = pert[c].max() - pert[c].min()
            sigma = noise_frac * (rng if rng > 0 else (pert[c].std() if pert[c].std()>0 else 1.0))
            pert[c] = pert[c] + np.random.normal(0, sigma, size=pert.shape[0])
        # normalize by column sums (avoid division by zero)
        denom = pert.sum(axis=0).replace(0, np.nan)
        norm = pert.div(denom, axis=1).fillna(0)
        score = norm.sum(axis=1).values
        arr.append(score)
    arr = np.vstack(arr)
    mean_score = arr.mean(axis=0)
    std_score = arr.std(axis=0)
    df_res = pd.DataFrame({"mean_score": mean_score, "std_score": std_score}, index=idx)
    return df_res, arr

# -----------------------
# Visualization helpers
# -----------------------
def plot_hist_interactive(df, column, nbins=30, title=None):
    title = title or f"Histograma: {column}"
    fig = px.histogram(df, x=column, nbins=nbins, marginal="box", title=title)
    fig.update_layout(template='plotly_white')
    return fig

def plot_box_interactive(df, column, title=None):
    title = title or f"Boxplot: {column}"
    fig = px.box(df, y=column, points="all", title=title)
    fig.update_layout(template='plotly_white')
    return fig

def plot_heatmap_interactive(corr_df, title="Correlation"):
    if corr_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No correlation data")
        return fig
    z = corr_df.values
    text_values = np.round(z, 2).astype(str)
    fig = go.Figure(go.Heatmap(
        z=z,
        x=corr_df.columns,
        y=corr_df.index,
        colorscale='RdBu',
        zmin=-1,
        zmax=1,
        text=text_values,
        texttemplate="%{text}",
        textfont={"size": 12, "color": "white"},
        hovertemplate="%{y} - %{x}: %{text}<extra></extra>"
    ))
    fig.update_layout(title=title, template='plotly_white', xaxis=dict(showgrid=False), yaxis=dict(showgrid=False))
    return fig

def save_static_boxplot(df, column, filename):
    plt.figure(figsize=(5,3))
    plt.boxplot(df[column].dropna())
    plt.title(column)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    return filename

def save_static_hist(df, column, filename):
    plt.figure(figsize=(5,3))
    plt.hist(df[column].dropna(), bins=25)
    plt.title(column)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    return filename

def save_static_heatmap(df_numeric, filename):
    corr = df_numeric.corr()
    plt.figure(figsize=(6,5))
    import seaborn as sns
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='RdBu', vmin=-1, vmax=1)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    return filename

# -----------------------
# Export: Excel / JSON / PDF
# -----------------------
def build_excel_bytes(sheets_dict):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
        for name, obj in sheets_dict.items():
            safe = str(name)[:31]
            try:
                if isinstance(obj, pd.DataFrame):
                    obj.to_excel(writer, sheet_name=safe, index=True)
                else:
                    pd.DataFrame(obj).to_excel(writer, sheet_name=safe, index=False)
            except Exception as e:
                pd.DataFrame({"error":[str(e)]}).to_excel(writer, sheet_name=safe)
    return buf.getvalue()

def build_json(decision_df, desc_df, normal_df, extra):
    out = {
        "decision_matrix": decision_df.reset_index().to_dict(orient="records"),
        "descriptive": desc_df.to_dict(),
        "normality": normal_df.to_dict(),
        "extra": extra
    }
    return json.dumps(out, indent=2, default=str)

def generate_pdf_bytes(lang, decision_df, desc_df, normal_df, corr_df, box_png, hist_png, heat_png, anova_table=None, reg_metrics=None, mc_summary=None):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=36, leftMargin=36, topMargin=72, bottomMargin=36)
    styles = getSampleStyleSheet()
    heading = ParagraphStyle("Heading", parent=styles["Heading1"], alignment=1, spaceAfter=12)
    normal = styles["Normal"]
    small = ParagraphStyle("Small", parent=styles["Normal"], fontSize=9)
    story = []

    if os.path.exists(LOGO_PATH):
        try:
            img = RLImage(LOGO_PATH, width=180, height=180)
            story.append(img)
        except Exception:
            pass

    story.append(Paragraph(f"<b>{t('pdf_title', lang)}</b>", heading))
    
    # CORREÇÃO: Garante que INSTITUTION_LINE não tenha '\n' brutos se usados em Paragraph
    inst_text = INSTITUTION_LINE.replace("\n", "<br/>") 
    story.append(Paragraph(f"{inst_text}", ParagraphStyle("inst", parent=styles["Normal"], alignment=1)))
    
    story.append(Spacer(1, 6))
    story.append(Paragraph(f"{t('pdf_subtitle', lang)} — {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", small))
    story.append(Spacer(1, 12))

    story.append(Paragraph("<b>Data / Dados</b>", styles["Heading2"]))
    story.append(Paragraph(f"Number of alternatives: {len(decision_df.index)}", normal))
    story.append(Paragraph(f"Number of numeric criteria: {len(decision_df.select_dtypes(include=[np.number]).columns)}", normal))
    story.append(Spacer(1,8))

    story.append(Paragraph("<b>Descriptive statistics / Estatística descritiva</b>", styles["Heading3"]))
    try:
        desc_show = desc_df.reset_index().head(10)
        data_table = [desc_show.columns.tolist()] + desc_show.values.tolist()
        tbl = Table(data_table, style=[('GRID',(0,0),(-1,-1),0.25,colors.grey),('BACKGROUND',(0,0),(-1,0),colors.lightblue)])
        story.append(tbl)
    except Exception:
        story.append(Paragraph("No descriptive data", normal))
    story.append(Spacer(1,8))

    story.append(Paragraph("<b>Normality tests / Testes de normalidade</b>", styles["Heading3"]))
    try:
        norm_show = normal_df.reset_index().head(20)
        data_norm = [norm_show.columns.tolist()] + norm_show.values.tolist()
        n_tbl = Table(data_norm, style=[('GRID',(0,0),(-1,-1),0.25,colors.grey)])
        story.append(n_tbl)
    except Exception:
        story.append(Paragraph("No normality results", normal))
    story.append(Spacer(1,12))

    story.append(Paragraph("<b>Plots / Gráficos</b>", styles["Heading3"]))
    for png, caption in [(box_png, "Boxplot"), (hist_png, "Histogram"), (heat_png, "Correlation heatmap")]:
        if png and os.path.exists(png):
            try:
                im = RLImage(png, width=400, height=250)
                story.append(im)
                story.append(Paragraph(caption, small))
                story.append(Spacer(1,6))
            except Exception:
                pass
    story.append(Spacer(1,12))

    story.append(Paragraph("<b>Tests / Testes</b>", styles["Heading3"]))
    if anova_table is not None:
        try:
            a_show = anova_table.round(6).reset_index()
            data = [a_show.columns.tolist()] + a_show.values.tolist()
            story.append(Table(data, style=[('GRID',(0,0),(-1,-1),0.25,colors.grey)]))
            story.append(Spacer(1,8))
        except Exception:
            pass

    story.append(Paragraph("<b>Regression / Regressão</b>", styles["Heading3"]))
    if reg_metrics:
        reg_kv = [[k, str(v)] for k,v in reg_metrics.items()]
        story.append(Table(reg_kv, style=[('GRID',(0,0),(-1,-1),0.25,colors.grey)]))
    else:
        story.append(Paragraph("No regression performed / Nenhuma regressão realizada", normal))
    story.append(Spacer(1,12))

    story.append(Paragraph("<b>Monte Carlo (sensitivity) / Monte Carlo (sensibilidade)</b>", styles["Heading3"]))
    if mc_summary is not None:
        try:
            mc_show = mc_summary.head(20).reset_index()
            data_mc = [mc_show.columns.tolist()] + mc_show.values.tolist()
            story.append(Table(data_mc, style=[('GRID',(0,0),(-1,-1),0.25,colors.grey)]))
        except Exception:
            pass
    else:
        story.append(Paragraph("No Monte Carlo results", normal))

    story.append(Spacer(1,12))
    story.append(Paragraph("Generated by app_streamlit.py", small))
    
    # CORREÇÃO: Usando a variável corrigida inst_text
    story.append(Paragraph(inst_text, ParagraphStyle("inst", parent=styles["Normal"], alignment=1)))
    
    doc.build(story)
    pdf_bytes = buf.getvalue()
    buf.close()
    return pdf_bytes

# -----------------------
# OpenAI: setup, analyze & chat
# -----------------------
import os
from dotenv import load_dotenv

def setup_openai_client():
    """
    Tries to get API key from st.secrets or environment (.env).
    Returns a OpenAI client instance or None.
    """
    if OpenAI is None:
        return None

    # 🔹 Força o carregamento do arquivo "code.env" (no mesmo diretório)
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    dotenv_path = os.path.join(BASE_DIR, "code.env")

    load_dotenv(dotenv_path)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # Tenta pegar dos segredos do Streamlit como fallback
        if "OPENAI_API_KEY" in st.secrets:
             api_key = st.secrets["OPENAI_API_KEY"]
        else:
             st.error(f"🔑 Chave da OpenAI não encontrada no arquivo {dotenv_path} nem nos segredos.")
             return None

    try:
        client = OpenAI(api_key=api_key)
        # st.success("✅ OpenAI configurado com sucesso.") # Comentado para poluir menos a UI
        return client
    except Exception as e:
        st.error(f"❌ Erro ao inicializar cliente OpenAI: {e}")
        return None
    
def build_ai_prompt(lang, results_summary: dict):
    """
    Build a concise prompt for the model using the available results.
    results_summary is a dict with keys like 'descriptive', 'anova', 'regression', 'montecarlo', 'correlation'
    """
    is_pt = (lang == "Portuguese")
    header = "Analise e interpretação concisa dos resultados:" if is_pt else "Concise analysis and interpretation of the results:"
    instructions = (
        ("Responda em Português. Forneça: 1) Interpretação curta; 2) Principais insights; 3) Recomendações práticas.\n"
         if is_pt else
         "Respond in English. Provide: 1) Short interpretation; 2) Key insights; 3) Practical recommendations.\n")
    )
    parts = [header, instructions]
    # Descriptive
    if 'descriptive' in results_summary and not results_summary['descriptive'].empty:
        desc = results_summary['descriptive'].head(6).to_dict(orient='records')
        parts.append(f"Descriptive (first rows): {desc}" if is_pt else f"Descriptive (first rows): {desc}")
    # Correlation: strongest correlations
    if 'correlation' in results_summary and not results_summary['correlation'].empty:
        corr = results_summary['correlation'].copy()
        corr_vals = corr.where(~np.eye(len(corr),dtype=bool)).abs().stack().sort_values(ascending=False).head(6).to_dict()
        parts.append(f"Top correlations (abs): {corr_vals}")
    # ANOVA
    if 'anova' in results_summary and results_summary['anova'] is not None:
        try:
            pval = results_summary['anova']["PR(>F)"].iloc[0]
            parts.append(f"ANOVA p-value: {float(pval)}")
        except Exception:
            parts.append("ANOVA: present")
    # Regression
    if 'regression' in results_summary and isinstance(results_summary['regression'], dict):
        parts.append(f"Regression metrics: {results_summary['regression']}")
    # Monte Carlo
    if 'montecarlo' in results_summary and results_summary['montecarlo'] is not None:
        try:
            mc = results_summary['montecarlo'].head(3).reset_index().to_dict(orient='records')
            parts.append(f"Monte Carlo (top rows): {mc}")
        except Exception:
            parts.append("Monte Carlo: present")
    # small note
    parts.append("Return a concise answer (max ~200-300 words).")
    return "\n\n".join(parts)

def analyze_with_ai(client, lang, results_summary):
    """
    Calls OpenAI to produce a concise analysis based on results_summary.
    Returns text.
    """
    if client is None:
        return None
    prompt = build_ai_prompt(lang, results_summary)
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            temperature=0.5
        )
        text = resp.choices[0].message.content
        return text
    except Exception as e:
        return f"❌ AI analysis error: {e}"

def chat_with_ai(client, lang, user_message):
    """
    Simple chat: includes system message with context = last analysis text (if exists)
    and the last few messages saved in session_state['ai_chat_history'].
    """
    if client is None:
        return {"error": "OpenAI client not available"}
    system_context = ""
    if "ai_last_analysis" in st.session_state and st.session_state.get("ai_last_analysis"):
        system_context = st.session_state["ai_last_analysis"]
    sys_msg = ("Você é um assistente analítico especializado. Use o contexto a seguir para responder: \n"
               if lang=="Portuguese" else
               "You are an analytical assistant. Use the following context to answer:\n")
    sys_content = sys_msg + system_context
    messages = [{"role":"system", "content": sys_content}]
    # include recent chat history (user & assistant pairs)
    history = st.session_state.get("ai_chat_history", [])
    # keep up to last 6 messages
    for m in history[-6:]:
        messages.append(m)
    messages.append({"role":"user", "content": user_message})
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=300,
            temperature=0.7
        )
        return {"answer": resp.choices[0].message.content}
    except Exception as e:
        return {"error": str(e)}

# -----------------------
# Streamlit UI
# -----------------------
def main():
    st.set_page_config(page_title="🤖MCDA Stats — UFF/UFRN", layout="wide")
    # initialize session state variables for AI
    if "ai_chat_history" not in st.session_state:
        st.session_state["ai_chat_history"] = []
    if "ai_last_analysis" not in st.session_state:
        st.session_state["ai_last_analysis"] = ""
    if "ai_enabled" not in st.session_state:
        st.session_state["ai_enabled"] = False
    if "uploaded_df" not in st.session_state:
        st.session_state["uploaded_df"] = None

    # sidebar: logo small + language + API help
    with st.sidebar:
    #    if os.path.exists(LOGO_PATH):
    #        st.image(LOGO_PATH, width=120)
        lang = st.selectbox("Idioma / Language", options=["Portuguese", "English"], index=0, help="Escolha o idioma")
        is_pt = (lang == "Portuguese")

        # template / example / run all
        if st.button(t("download_template", lang)):
            template_bytes = get_template_bytes(lang)
            st.download_button(label=t("download_template", lang), data=template_bytes, file_name="mcda_template.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        if st.button(t("load_example", lang)):
            df_example = pd.DataFrame({
                "Alternative": ["A1","A2","A3","A4"],
                "Cost": [120, 95, 110, 130],
                "Efficiency": [0.8, 0.9, 0.85, 0.75],
                "Risk": [2.0, 1.5, 1.8, 2.2],
                "Group": ["G1","G1","G2","G2"]
            })
            st.session_state['uploaded_df'] = df_example
            st.success(t("load_example", lang) + " — OK")

        run_all = st.button(t("run_all", lang))

        st.markdown("---")
        # st.header("OpenAI / API")
        st.markdown("""
        **Developers / Desenvolvedores**
        - [Anderson Portella](https://www.linkedin.com/in/andersonportella/)
        - [Prof. Dra. Miriam Rocha](https://www.linkedin.com/in/miriam-rocha-39686833/)
        - [Prof. Dr. Marcos dos Santos](https://www.linkedin.com/in/profmarcosdossantos/)
        - [Prof. Dr. Carlos Francisco Simões Gomes](https://www.linkedin.com/in/carlos-francisco-sim%C3%B5es-gomes-7284a3b/)        
        """)
        client_test = setup_openai_client()
        if client_test is None:
            st.error("OpenAI API key not found.")
            st.markdown("Add `OPENAI_API_KEY` to `.env` or to Streamlit secrets.")
        else:
            st.success("OpenAI key detected — AI enabled")

    # top title with central logo (rendered above via sidebar small logo, main center logo too)
    st.markdown("<br>", unsafe_allow_html=True)
    if os.path.exists(LOGO_PATH):
        try:
            with open(LOGO_PATH, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode()
            st.markdown(f"<div style='text-align:center;'><img src='data:image/png;base64,{img_b64}' width='160'></div>", unsafe_allow_html=True)
        except Exception:
            pass

    st.markdown(f"<h1 style='text-align:center;'>{t('app_title', lang)}</h1>", unsafe_allow_html=True)
    # CORREÇÃO NA EXIBIÇÃO HTML: Aqui usamos <br> normal pois é HTML do Streamlit
    inst_html = INSTITUTION_LINE.replace("<br/>", "<br>")
    st.markdown(f"<p style='text-align:center; font-weight: bold;'>{inst_html}</p>", unsafe_allow_html=True)

    # single file uploader (no duplication)
    uploaded = st.file_uploader(t("upload", lang), type=["xlsx","xls","csv"])
    if uploaded:
        try:
            df_loaded = read_upload(uploaded)
            st.session_state['uploaded_df'] = df_loaded
            st.success("File loaded" if not is_pt else "Arquivo carregado")
        except Exception as e:
            st.error(f"Error reading file: {e}")

    df = st.session_state.get('uploaded_df', None)
    if df is None:
        st.info("Please upload a spreadsheet or load the sample (sidebar). / Faça upload de um arquivo ou carregue o exemplo (barra lateral).")
        return

    # preview
    st.subheader(t("data_preview", lang))
    st.dataframe(df.head())

    # validation
    val = validate_matrix(df)
    if val['messages']:
        for m in val['messages']:
            st.warning(m)
    if not val['valid']:
        st.error(t("no_numeric", lang))
        return

    # detect numeric & categorical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in numeric_cols]

    # choose which analyses to run
    st.sidebar.header("Analyses / Análises")
    options = st.sidebar.multiselect("Select analyses / Selecione análises", options=[
        "desc","norm","corr","anova","nonparam","reg","mc"
    ], default=["desc","norm","corr"])

    # run all option expands selection
    if run_all:
        options = ["desc","norm","corr","anova","nonparam","reg","mc"]

    results = {}

    st.header(t("results", lang))

    # 1 Descriptive
    if "desc" in options:
        st.subheader(t("desc", lang))
        desc_df = descriptive_stats(df)
        st.dataframe(desc_df)
        results['descriptive'] = desc_df

    # 2 Normality tests
    if "norm" in options:
        st.subheader(t("norm", lang))
        norm_df = normality_tests(df)
        st.dataframe(norm_df)
        results['normality'] = norm_df

    # 3 Correlation
    if "corr" in options:
        st.subheader(t("corr", lang))
        corr_method = st.selectbox("Method / Método", ["pearson","spearman"])
        corr_df = correlation_matrix(df, method=corr_method)
        st.dataframe(corr_df)
        st.plotly_chart(plot_heatmap_interactive(corr_df, title=("Correlation" if lang=="English" else "Correlação")), use_container_width=True)
        results['correlation'] = corr_df

    # 4 ANOVA / Levene
    if "anova" in options:
        st.subheader(t("anova", lang))
        if len(cat_cols) == 0:
            st.info("No categorical columns for ANOVA / Não há colunas categóricas para ANOVA")
        else:
            group_col = st.selectbox("Group (categorical) / Coluna grupo", options=[None] + cat_cols)
            value_col = st.selectbox("Value (numeric) / Variável numérica", options=numeric_cols)
            if st.button("Run ANOVA / Executar ANOVA"):
                if group_col is None:
                    st.warning("Select a group column")
                else:
                    try:
                        # Levene
                        lev_res = None
                        try:
                            groups = [g[value_col].dropna().values for _, g in df.groupby(group_col)]
                            lev_stat, lev_p = stats.levene(*groups, center='median')
                            lev_res = {'stat': float(lev_stat), 'pvalue': float(lev_p)}
                            st.write("Levene:", lev_res)
                        except Exception as e:
                            st.warning(f"Levene error: {e}")
                        # ANOVA or Kruskal depending on assumptions
                        normal_by_group = {}
                        for name, group in df.groupby(group_col):
                            s = group[value_col].dropna()
                            if len(s) >= 3:
                                try:
                                    _, p = stats.shapiro(s)
                                except Exception:
                                    p = np.nan
                            else:
                                p = np.nan
                            normal_by_group[name] = p
                        st.write("Normality by group (p-values):", normal_by_group)
                        all_normal = all([p is not None and not np.isnan(p) and p>0.05 for p in normal_by_group.values()])
                        levene_ok = (lev_res is not None and lev_res.get('pvalue',0)>0.05)
                        if all_normal and levene_ok:
                            st.success("Parametric assumptions OK — ANOVA recommended")
                            model, anova_tbl = anova_oneway(df, group_col, value_col)
                            st.dataframe(anova_tbl)
                            results['anova'] = anova_tbl
                            # Tukey if significant
                            p_anova = anova_tbl["PR(>F)"].iloc[0] if "PR(>F)" in anova_tbl.columns else None
                            if p_anova is not None and p_anova < 0.05:
                                st.write("ANOVA significant. Running Tukey HSD")
                                tuk = tukey_hsd(df, group_col, value_col)
                                st.dataframe(tuk)
                                results['tukey'] = tuk
                        else:
                            st.info("Parametric assumptions not met — running Kruskal–Wallis")
                            kw = kruskal_test(df, group_col, value_col)
                            st.write("Kruskal:", kw)
                            results['kruskal'] = kw
                    except Exception as e:
                        st.error(f"ANOVA error: {e}")

    # 5 Non-parametric (Mann-Whitney)
    if "nonparam" in options:
        st.subheader(t("nonparam", lang))
        if len(cat_cols) and len(numeric_cols):
            gcol = st.selectbox("Group / Grupo (non-param)", cat_cols, key="np_group")
            vcol = st.selectbox("Variable / Variável (non-param)", numeric_cols, key="np_val")
            if st.button("Run Kruskal / Executar Kruskal"):
                try:
                    kr = kruskal_test(df, gcol, vcol)
                    st.write("Kruskal:", kr)
                except Exception as e:
                    st.error(f"Kruskal error: {e}")

    # 6 Regression
    if "reg" in options:
        st.subheader(t("reg", lang))
        if len(numeric_cols) >= 2:
            y_col = st.selectbox("Dependent Y / Variável dependente (Y)", numeric_cols, key="reg_y")
            x_choices = [c for c in numeric_cols if c != y_col]
            x_cols = st.multiselect("Independent X / Preditores (X)", options=x_choices, key="reg_x")
            if st.button("Run Regression / Executar Regressão"):
                if not x_cols:
                    st.warning("Select at least one predictor")
                else:
                    try:
                        model, metrics, vif, resid, fitted, cooks = linear_regression(df, y_col, x_cols)
                        st.text(model.summary().as_text())
                        st.json(metrics)
                        if vif:
                            st.write("VIF:", vif)
                        fig = px.scatter(x=fitted, y=resid, labels={'x':'Fitted','y':'Residuals'}, title="Residuals vs Fitted")
                        fig.add_hline(y=0, line_dash="dash")
                        st.plotly_chart(fig, use_container_width=True)
                        cooks_ser = pd.Series(cooks, index=df.index)
                        st.write("Top Cook's distances")
                        st.dataframe(cooks_ser.sort_values(ascending=False).head(5).to_frame("cooks_d"))
                        results['regression'] = metrics
                    except Exception as e:
                        st.error(f"Regression error: {e}")
        else:
            st.info("At least 2 numeric columns required for regression")

    # 7 Monte Carlo
    if "mc" in options:
        st.subheader(t("mc", lang))
        mc_iters = st.number_input("Iterations / Iterações", min_value=100, max_value=20000, value=1000, step=100)
        noise_frac = st.slider("Noise fraction / Fração de ruído", 0.0, 0.5, 0.05, 0.01)
        if st.button("Run Monte Carlo / Executar MC"):
            df_numeric = df.select_dtypes(include=[np.number])
            if df_numeric.empty:
                st.warning(t("no_numeric", lang))
            else:
                mc_summary, mc_arr = monte_carlo(df_numeric, n_iter=mc_iters, noise_frac=noise_frac)
                st.dataframe(mc_summary)
                fig_mc = px.histogram(mc_arr[:,0], nbins=50, title="MC distribution (first alternative)")
                st.plotly_chart(fig_mc, use_container_width=True)
                results['montecarlo'] = mc_summary

    # Export buttons
    st.header(t("export", lang))
    col1, col2, col3 = st.columns(3)

    decision_df = df.copy()
    desc_df = results.get('descriptive', pd.DataFrame())
    normal_df = results.get('normality', pd.DataFrame())
    corr_df = results.get('correlation', pd.DataFrame())
    anova_tbl = results.get('anova', None)
    reg_metrics = results.get('regression', None)
    mc_summary = results.get('montecarlo', None)

    with col1:
        if st.button(t("export_excel", lang)):
            sheets = {"decision_matrix": decision_df.reset_index()}
            if not desc_df.empty:
                sheets["descriptive"] = desc_df
            if not normal_df.empty:
                sheets["normality"] = normal_df.reset_index()
            if not corr_df.empty:
                sheets["correlation"] = corr_df
            if anova_tbl is not None:
                sheets["anova"] = anova_tbl.reset_index()
            xbytes = build_excel_bytes(sheets)
            st.download_button("Download .xlsx", xbytes, file_name="mcda_results.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    with col2:
        if st.button(t("export", lang) + " (JSON)"):
            extra = {"generated_at": str(datetime.datetime.now())}
            json_str = build_json(decision_df, desc_df, normal_df, extra)
            st.download_button("Download .json", json_str, file_name="mcda_results.json", mime="application/json")

    with col3:
        if st.button(t("export_pdf", lang)):
            tmpdir = tempfile.mkdtemp()
            box_png = hist_png = heat_png = None
            try:
                sel_col = numeric_cols[0] if numeric_cols else None
                if sel_col:
                    box_png = os.path.join(tmpdir, "box.png")
                    hist_png = os.path.join(tmpdir, "hist.png")
                    heat_png = os.path.join(tmpdir, "heat.png")
                    save_static_boxplot(df, sel_col, box_png)
                    save_static_hist(df, sel_col, hist_png)
                    save_static_heatmap(df.select_dtypes(include=[np.number]), heat_png)
            except Exception:
                box_png = hist_png = heat_png = None
            pdf_bytes = generate_pdf_bytes(lang, decision_df, desc_df, normal_df, corr_df, box_png, hist_png, heat_png, anova_table=anova_tbl, reg_metrics=reg_metrics, mc_summary=mc_summary)
            st.download_button("Download PDF", pdf_bytes, file_name="mcda_report_uff_ufrn.pdf", mime="application/pdf")

    # -----------------------
    # AI manual analysis trigger (user selected manual mode)
    # -----------------------
    st.markdown("---")
    st.subheader(t("ai_button", lang))
    st.write(t("ai_help", lang))

    client = setup_openai_client()
    if client is None:
        st.warning(t("ai_not_configured", lang))
    else:
        # prepare a concise results summary to send to the AI when asked
        results_summary = {}
        # include selected result objects (small)
        if 'descriptive' in results:
            results_summary['descriptive'] = results['descriptive']
        if 'correlation' in results:
            results_summary['correlation'] = results['correlation']
        if 'anova' in results:
            results_summary['anova'] = results['anova']
        if 'regression' in results:
            results_summary['regression'] = results['regression']
        if 'montecarlo' in results:
            results_summary['montecarlo'] = results['montecarlo']

        if st.button(t("ai_button", lang)):
            with st.spinner("Generating AI analysis..." if not is_pt else "Gerando análise com IA..."):
                ai_text = analyze_with_ai(client, lang, results_summary)
                if ai_text is None:
                    st.error("AI not available")
                elif ai_text.startswith("❌"):
                    st.error(ai_text)
                else:
                    st.session_state["ai_last_analysis"] = ai_text
                    st.session_state["ai_enabled"] = True
                    st.success("AI analysis generated" if not is_pt else "Análise com IA gerada")
                    st.subheader("📊 AI Analysis / Análise da IA")
                    st.info(ai_text)
                    # reset chat history
                    st.session_state["ai_chat_history"] = []

    # Chat: only when ai_enabled
    if st.session_state.get("ai_enabled", False) and client is not None:
        st.markdown("---")
        st.header(t("chat_header", lang))
        # show history
        for msg in st.session_state["ai_chat_history"]:
            try:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
            except Exception:
                # fallback plain write
                st.write(f"{msg['role']}: {msg['content']}")
        # user input
        prompt = st.chat_input("Pergunte sobre a análise..." if is_pt else "Ask about the analysis...")
        if prompt:
            # append user message to history
            st.session_state["ai_chat_history"].append({"role":"user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.spinner("IA pensando..." if is_pt else "AI thinking..."):
                resp = chat_with_ai(client, lang, prompt)
            if resp.get("error"):
                assistant_text = f"❌ Error: {resp.get('error')}"
                st.session_state["ai_chat_history"].append({"role":"assistant", "content": assistant_text})
                with st.chat_message("assistant"):
                    st.markdown(assistant_text)
            else:
                assistant_text = resp.get("answer", "")
                st.session_state["ai_chat_history"].append({"role":"assistant", "content": assistant_text})
                with st.chat_message("assistant"):
                    st.markdown(assistant_text)

    elif not st.session_state.get("ai_enabled", False):
        st.info("💡 " + ("Execute a análise com IA (botão) para abrir o chat." if is_pt else "Run the AI analysis (button) to enable the chat."))

if __name__ == "__main__":
    main()