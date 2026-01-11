# 🤖 MCDA Statistical Analysis Tool (UFF/UFRN)

## Overview

This application is a comprehensive tool designed to perform **Statistical Analysis on Multi-Criteria Decision Analysis (MCDA) Matrices**. Developed in Python using Streamlit, it serves researchers and decision-makers by validating data, identifying redundancies, and testing the robustness of decision matrices before applying ranking methods (like AHP, TOPSIS, PROMETHEE, etc.).

It allows for a bilingual interface (English/Portuguese) and integrates with **OpenAI** to provide automated, AI-driven insights into the statistical results.

## Key Features

* **Descriptive Statistics**: Calculates Mean, Median, Std Dev, Variance, Skewness, Kurtosis, and Coefficient of Variation (CV) to understand data distribution.
* **Normality Tests**: Shapiro-Wilk test to determine if criteria follow a normal distribution.
* **Correlation Analysis**: Pearson and Spearman heatmaps to identify redundant criteria (multicollinearity).
* **Hypothesis Testing**:
    * **Parametric**: One-way ANOVA and Tukey HSD Post-hoc.
    * **Non-Parametric**: Kruskal-Wallis and Mann-Whitney U tests.
* **Regression Analysis**: OLS Regression with VIF (Variance Inflation Factor) and Cook's Distance to analyze relationships between criteria.
* **Monte Carlo Simulation**: Sensitivity analysis to test how noise/uncertainty affects alternative scoring.
* **AI Integration**: Connects to OpenAI (GPT-4o-mini) to generate a qualitative interpretation of the statistical data.
* **Reporting**: Exports full results to Excel, JSON, and PDF formats.

## Installation

1.  **Clone the repository** or download the source code.
2.  **Create a virtual environment** (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Configuration

### OpenAI API (Optional)
To enable the AI Chat and Analysis features:
1.  Create a file named `code.env` in the root directory.
2.  Add your API key:
    ```env
    OPENAI_API_KEY=sk-proj-your-api-key-here
    ```

### Logo
Place the image file `UFF_UFRN_brasao.png` in the root directory to display the institution logos in the app and PDF reports.

## Usage

Run the application using Streamlit:

```bash
streamlit run app_streamlit5.py

The application will open in your default web browser (usually at http://localhost:8501).

Upload Data: Upload a .csv or .xlsx file.

Format: Rows = Alternatives, Columns = Criteria.

Note: Ensure there is at least one categorical column if you wish to run Group Analysis (ANOVA).

Select Analyses: Use the sidebar to toggle specific statistical modules.

Run AI Analysis: Click "Generate AI analysis" at the bottom to get a text summary.

Export: Download the final report in your preferred format.

Credits
Developed by:

Anderson Portella (Production Engineer, Transpetro / Ph.D. Candidate, UFF)

Prof. Dra. Miriam Rocha

Eng. Robson Souza

Prof. Dr. Marcos dos Santos

Prof. Dr. Carlos Francisco Simões Gomes

Institutions:

Universidade Federal Fluminense (UFF)
Universidade Federal do Rio Grande do Norte (UFRN)
Instituto Militar de Engenharia (IME)
