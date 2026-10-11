"""
The Newsroom | Verification Desk
Professional editorial interface for AI-assisted news text analysis.
Provides text classification, probability metrics, and preprocessing transparency.
"""
from pathlib import Path
import html
import os
import sys

# Ensure project root is available on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import config
from frontend.api_client import FakeNewsAPIClient

# Benchmark demo excerpts (strictly synthetic examples for UI validation)
DEMO_SAMPLES = {
    "economic_report": {
        "label": "Verified Report Example",
        "description": "Standard central bank policy report style with formal tone and attributed data.",
        "title": "Fed Holds Benchmark Interest Rates Steady, Citing Measured Progress on Inflation",
        "text": (
            "The Federal Reserve decided on Wednesday to keep its benchmark interest rate unchanged, "
            "pointing to steady employment numbers and moderating inflation metrics over recent months. "
            "In its official monetary policy statement, the central bank noted that while economic expansion "
            "remains resilient, committee members will continue assessing incoming labor market data and "
            "quarterly price indices before deciding on future adjustments to borrowing costs."
        ),
    },
    "conspiracy_claim": {
        "label": "Sensational Claim Example",
        "description": "Fabricated leak narrative using hyperbolic rhetoric and anonymous whistleblower claims.",
        "title": "Shocking Whistleblower Leak: Secret Subterranean Complex Found Beneath Capitol",
        "text": (
            "SHOCKING INVESTIGATION: Construction crews working overnight have reportedly discovered an enormous "
            "subterranean metropolis hidden miles beneath the Capitol building foundation! Anonymous insiders claim "
            "that a secretive global syndicate has been operating classified advanced energy machines in deep tunnels "
            "for decades while mainstream media and government officials deliberately suppressed the evidence from the public!"
        ),
    },
}

# Page Configuration - Editorial Tab & Responsive Layout
st.set_page_config(
    page_title="The Newsroom | Verification Desk",
    page_icon="🗞️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Refined Editorial Stylesheet
st.markdown(
    """
    <style>
    /* Base typography & palette */
    :root {
        --text-headline: #0f172a;
        --text-body: #334155;
        --text-muted: #64748b;
        --bg-surface: #ffffff;
        --bg-warm: #fcfbf9;
        --border-subtle: #e2e8f0;
        --border-mid: #cbd5e1;
        --accent-real: #15803d;
        --accent-real-bg: #f0fdf4;
        --accent-real-border: #bbf7d0;
        --accent-fake: #b91c1c;
        --accent-fake-bg: #fef2f2;
        --accent-fake-border: #fecaca;
    }

    /* Container constraints & typography */
    .block-container {
        max-width: 1040px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    /* Masthead Styles */
    .masthead-container {
        border-bottom: 2px solid #0f172a;
        padding-bottom: 0.75rem;
        margin-bottom: 1.5rem;
    }
    .masthead-kicker {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #475569;
        margin-bottom: 0.25rem;
    }
    .masthead-title {
        font-family: Georgia, "Times New Roman", Times, serif;
        font-size: 2.1rem;
        font-weight: 700;
        color: #0f172a;
        line-height: 1.15;
        letter-spacing: -0.01em;
        margin: 0;
    }
    .masthead-subline {
        font-family: Georgia, "Times New Roman", Times, serif;
        font-style: italic;
        font-size: 1.05rem;
        color: #475569;
        margin-top: 0.25rem;
        margin-bottom: 0.5rem;
    }
    .masthead-meta-bar {
        display: flex;
        flex-wrap: wrap;
        justify-content: space-between;
        align-items: center;
        border-top: 1px solid #e2e8f0;
        padding-top: 0.5rem;
        margin-top: 0.5rem;
        font-size: 0.8rem;
        color: #64748b;
    }

    /* Editorial Headline */
    .headline-section {
        margin-top: 1rem;
        margin-bottom: 2rem;
    }
    .editorial-h1 {
        font-family: Georgia, "Times New Roman", Times, serif;
        font-size: 1.9rem;
        font-weight: 600;
        color: #0f172a;
        line-height: 1.25;
        margin-bottom: 0.4rem;
    }
    .editorial-subtitle {
        font-size: 1.02rem;
        color: #475569;
        line-height: 1.5;
        max-width: 820px;
        margin: 0;
    }

    /* Section Subheadings */
    .section-kicker {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #64748b;
        margin-bottom: 0.25rem;
    }
    .section-title {
        font-family: Georgia, "Times New Roman", Times, serif;
        font-size: 1.35rem;
        font-weight: 600;
        color: #0f172a;
        margin-bottom: 1rem;
    }

    /* Workspace Card Container */
    .editorial-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 1.5rem;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
        margin-bottom: 1.75rem;
    }

    /* Example Story Box */
    .sample-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 4px;
        padding: 0.9rem;
        margin-bottom: 0.75rem;
        transition: border-color 0.15s ease;
    }
    .sample-tag {
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: #475569;
        margin-bottom: 0.2rem;
    }
    .sample-title {
        font-family: Georgia, "Times New Roman", Times, serif;
        font-size: 0.95rem;
        font-weight: 600;
        color: #0f172a;
        line-height: 1.3;
        margin-bottom: 0.3rem;
    }
    .sample-desc {
        font-size: 0.8rem;
        color: #64748b;
        line-height: 1.4;
        margin: 0;
    }

    /* Verification Report Card */
    .report-card-real {
        background-color: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-left: 5px solid #15803d;
        border-radius: 4px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1.5rem;
    }
    .report-card-fake {
        background-color: #fef2f2;
        border: 1px solid #fecaca;
        border-left: 5px solid #b91c1c;
        border-radius: 4px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1.5rem;
    }
    .report-kicker {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #475569;
        margin-bottom: 0.25rem;
    }
    .report-heading-real {
        font-family: Georgia, "Times New Roman", Times, serif;
        font-size: 1.6rem;
        font-weight: 700;
        color: #15803d;
        margin: 0 0 0.4rem 0;
    }
    .report-heading-fake {
        font-family: Georgia, "Times New Roman", Times, serif;
        font-size: 1.6rem;
        font-weight: 700;
        color: #b91c1c;
        margin: 0 0 0.4rem 0;
    }
    .report-summary-text {
        font-size: 0.95rem;
        color: #334155;
        line-height: 1.5;
        margin: 0;
    }

    /* Metric & Progress Display */
    .metric-value-large {
        font-family: Georgia, "Times New Roman", Times, serif;
        font-size: 2.2rem;
        font-weight: 700;
        color: #0f172a;
        line-height: 1;
    }
    .metric-label-sub {
        font-size: 0.8rem;
        color: #64748b;
        margin-top: 0.25rem;
    }

    /* Editorial Colophon / Footer */
    .editorial-colophon {
        border-top: 1px solid #e2e8f0;
        padding-top: 1.5rem;
        margin-top: 3rem;
        font-size: 0.82rem;
        color: #64748b;
        line-height: 1.55;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def init_session_state():
    """Initialize persistent session parameters."""
    if "news_text" not in st.session_state:
        st.session_state["news_text"] = ""
    if "news_title" not in st.session_state:
        st.session_state["news_title"] = ""
    if "prediction_result" not in st.session_state:
        st.session_state["prediction_result"] = None
    if "backend_url" not in st.session_state:
        st.session_state["backend_url"] = getattr(config, "BACKEND_URL", "http://127.0.0.1:8000")


def populate_sample(sample_id: str):
    """Populate workspace fields with selected editorial demo sample."""
    sample = DEMO_SAMPLES.get(sample_id)
    if sample:
        st.session_state["news_title"] = sample["title"]
        st.session_state["news_text"] = sample["text"]
        st.session_state["prediction_result"] = None


def reset_workspace():
    """Clear article inputs and active verification analysis."""
    st.session_state["news_title"] = ""
    st.session_state["news_text"] = ""
    st.session_state["prediction_result"] = None


def main():
    init_session_state()

    # Client connectivity probe
    client = FakeNewsAPIClient(base_url=st.session_state["backend_url"])
    health = client.check_health()

    # =========================================================================
    # Secondary Control Panel (Sidebar)
    # =========================================================================
    with st.sidebar:
        st.markdown('<div class="section-kicker">System Configuration</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title" style="font-size: 1.15rem; margin-bottom: 0.5rem;">Desk & Inference Node</div>', unsafe_allow_html=True)

        # Service Connectivity Indicator
        if health["online"]:
            st.success("API Service Online")
            if health["model_loaded"]:
                st.caption(f"Active Model: `{Path(health['model_path']).name if health.get('model_path') else 'Loaded'}`")
                st.caption(f"Computation Device: `{health.get('device', 'cpu')}`")
            else:
                st.warning("Model Not Loaded")
                st.caption("Backend is running without trained weights.")
        else:
            st.error("API Service Offline")
            st.caption("FastAPI server unavailable at configured URL.")

        st.divider()

        # Configurable API Host
        new_url = st.text_input(
            "Service Endpoint URL",
            value=st.session_state["backend_url"],
            help="Target URL for FastAPI inference server.",
        )
        if new_url.strip() != st.session_state["backend_url"]:
            st.session_state["backend_url"] = new_url.strip()
            st.rerun()

        st.divider()

        # Architecture Specifications
        st.markdown('<div class="section-kicker">Model Specifications</div>', unsafe_allow_html=True)
        st.markdown(
            """
            - **Backbone**: DistilBERT (`distilbert-base-uncased`)
            - **Task**: Binary Sequence Classification
            - **Input**: Headline + Article Text
            - **Sequence Cutoff**: 128 / 256 tokens
            - **Label Index**: `0 = Real News`, `1 = Fake News`
            """
        )

    # =========================================================================
    # A. Editorial Masthead
    # =========================================================================
    status_text = "Operational" if health["online"] and health["model_loaded"] else "Offline"
    status_color = "#15803d" if health["online"] and health["model_loaded"] else "#b91c1c"

    st.markdown(
        f"""
        <header class="masthead-container">
            <div class="masthead-kicker">Automated Editorial Screening &middot; Verification Desk</div>
            <h1 class="masthead-title">The Newsroom</h1>
            <div class="masthead-subline">AI-assisted news text analysis &amp; stylistic verification</div>
            <div class="masthead-meta-bar">
                <span>Inference Service: <strong style="color: {status_color};">&bull; {status_text}</strong></span>
                <span>Notice: Classifies stylistic and textual patterns; does not independently verify factual claims.</span>
            </div>
        </header>
        """,
        unsafe_allow_html=True,
    )

    # =========================================================================
    # B. Page Headline & Editorial Scope
    # =========================================================================
    st.markdown(
        """
        <section class="headline-section">
            <div class="editorial-h1">Examine the story. Understand the signal.</div>
            <p class="editorial-subtitle">
                Submit an article headline and text for transformer-based sequence analysis. 
                The model compares stylistic framing, lexical choices, and reporting conventions 
                against established newsroom benchmarks to calculate class probabilities.
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    # =========================================================================
    # D. Benchmark Demo Examples (Placed above workspace for immediate testing)
    # =========================================================================
    st.markdown('<div class="section-kicker">Benchmark Reference Samples</div>', unsafe_allow_html=True)
    st.markdown('<p style="font-size: 0.88rem; color: #64748b; margin-bottom: 0.75rem;">Load synthetic excerpts formatted according to standard reporting patterns to observe model evaluation:</p>', unsafe_allow_html=True)

    col_sample_a, col_sample_b = st.columns(2)
    with col_sample_a:
        sample_real = DEMO_SAMPLES["economic_report"]
        st.markdown(
            f"""
            <div class="sample-box">
                <div class="sample-tag">{html.escape(sample_real['label'])}</div>
                <div class="sample-title">{html.escape(sample_real['title'])}</div>
                <p class="sample-desc">{html.escape(sample_real['description'])}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Load verified report example", key="btn_load_real", use_container_width=True):
            populate_sample("economic_report")
            st.rerun()

    with col_sample_b:
        sample_fake = DEMO_SAMPLES["conspiracy_claim"]
        st.markdown(
            f"""
            <div class="sample-box">
                <div class="sample-tag">{html.escape(sample_fake['label'])}</div>
                <div class="sample-title">{html.escape(sample_fake['title'])}</div>
                <p class="sample-desc">{html.escape(sample_fake['description'])}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Load sensational claim example", key="btn_load_fake", use_container_width=True):
            populate_sample("conspiracy_claim")
            st.rerun()

    st.markdown("<div style='margin-bottom: 1.75rem;'></div>", unsafe_allow_html=True)

    # =========================================================================
    # C. Article Submission Workspace
    # =========================================================================
    st.markdown('<div class="section-kicker">Investigation Workspace</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Article analysis</div>', unsafe_allow_html=True)

    headline_input = st.text_input(
        "Article Headline (Optional)",
        value=st.session_state["news_title"],
        placeholder="Enter or paste the article headline...",
        help="Providing a headline gives the model essential context for sequence classification.",
        key="headline_input_field",
    )
    st.session_state["news_title"] = headline_input

    article_input = st.text_area(
        "Article Body Text (Required)",
        value=st.session_state["news_text"],
        placeholder="Paste article body text, lead paragraphs, or transcript to evaluate...",
        height=200,
        help="Input text is preprocessed to decode HTML, strip wire bylines, and preserve crucial negations.",
        key="article_input_field",
    )
    st.session_state["news_text"] = article_input

    # Real-time Metrics on Input Content
    raw_text = article_input.strip()
    words_in_text = len(raw_text.split()) if raw_text else 0
    chars_in_text = len(article_input)
    st.caption(f"Input Scope: **{words_in_text:,} words** | **{chars_in_text:,} characters** (Max allowable: 50,000 characters)")

    # Action Controls
    col_act1, col_act2, _ = st.columns([2, 1.5, 5])
    with col_act1:
        analyze_clicked = st.button("Analyze article", type="primary", use_container_width=True)
    with col_act2:
        clear_clicked = st.button("Clear input", use_container_width=True)

    if clear_clicked:
        reset_workspace()
        st.rerun()

    # Submission and Processing
    if analyze_clicked:
        if not raw_text:
            st.warning("Please enter or paste news article text before requesting analysis.")
        elif not health["online"]:
            st.error("Cannot perform analysis: The FastAPI backend service is offline. Start the service via `uvicorn backend.main:app`.")
        elif not health["model_loaded"]:
            st.warning("Inference unavailable: Model weights are not loaded into memory. Ensure model weights exist in `saved_models/distilbert_fake_news`.")
        else:
            with st.spinner("Processing article text through DistilBERT inference pipeline..."):
                response = client.predict(text=raw_text, title=headline_input.strip())

            if response["success"]:
                st.session_state["prediction_result"] = response["data"]
            else:
                st.session_state["prediction_result"] = None
                err_type = response.get("error_type", "SERVER_ERROR")
                err_msg = response.get("message", "An unexpected error occurred.")
                if err_type == "CONNECTION_ERROR":
                    st.error(f"Connection failure: {err_msg}")
                elif err_type == "VALIDATION_ERROR":
                    st.warning(f"Input validation notice: {err_msg}")
                elif err_type == "MODEL_UNAVAILABLE":
                    st.warning(f"Model unavailable: {err_msg}")
                else:
                    st.error(f"Analysis error ({err_type}): {err_msg}")

    # =========================================================================
    # E. Verification Report Card (Analysis Results)
    # =========================================================================
    result_data = st.session_state["prediction_result"]
    if result_data:
        st.markdown("<hr style='margin: 2rem 0; border: 0; border-top: 1px solid #e2e8f0;'>", unsafe_allow_html=True)
        st.markdown('<div class="section-kicker">Evaluation Report</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Model classification summary</div>', unsafe_allow_html=True)

        predicted_label = result_data.get("label", 0)
        label_title = result_data.get("label_name", "Real News")
        confidence_val = float(result_data.get("confidence", 0.0))
        probability_map = result_data.get("probabilities", {})
        cleaned_body = result_data.get("cleaned_text", "")

        prob_real = float(probability_map.get("Real News", 0.0))
        prob_fake = float(probability_map.get("Fake News", 0.0))

        # Editorial Verdict Card
        if predicted_label == 1:
            verdict_class = "report-card-fake"
            heading_class = "report-heading-fake"
            display_verdict = "Likely Fake News"
            explanation = (
                "The classifier detected rhetorical framing, lexical choices, and stylistic markers "
                "strongly correlated with fabricated claims or manipulative clickbait in the training benchmark. "
                "This indicates stylistic divergence from standard journalistic reporting conventions."
            )
        else:
            verdict_class = "report-card-real"
            heading_class = "report-heading-real"
            display_verdict = "Likely Real News"
            explanation = (
                "The article text exhibits lexical balance, neutral phrasing, and framing patterns typical "
                "of standard verified journalism. The model found high contextual alignment with verified newsroom reports."
            )

        st.markdown(
            f"""
            <div class="{verdict_class}">
                <div class="report-kicker">Model Classification</div>
                <div class="{heading_class}">{html.escape(display_verdict)}</div>
                <p class="report-summary-text">{html.escape(explanation)}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Statistical Metrics Layout
        col_metric, col_distribution = st.columns([1, 2])

        with col_metric:
            st.markdown('<div class="section-kicker">Primary Confidence</div>', unsafe_allow_html=True)
            st.markdown(
                f"""
                <div class="metric-value-large">{confidence_val * 100:.1f}%</div>
                <div class="metric-label-sub">Decision confidence for assigned class: <strong>{html.escape(label_title)}</strong></div>
                """,
                unsafe_allow_html=True,
            )

        with col_distribution:
            st.markdown('<div class="section-kicker">Class Probability Distribution</div>', unsafe_allow_html=True)
            
            # Probability Gauge: Real News
            st.markdown(
                f"<div style='display: flex; justify-content: space-between; font-size: 0.88rem; margin-bottom: 0.2rem;'>"
                f"<span>Verified Journalism Alignment (Real News)</span><strong>{prob_real * 100:.1f}%</strong></div>",
                unsafe_allow_html=True,
            )
            st.progress(min(max(prob_real, 0.0), 1.0))

            st.markdown("<div style='margin-bottom: 0.6rem;'></div>", unsafe_allow_html=True)

            # Probability Gauge: Fake News
            st.markdown(
                f"<div style='display: flex; justify-content: space-between; font-size: 0.88rem; margin-bottom: 0.2rem;'>"
                f"<span>Deceptive / Fabricated Alignment (Fake News)</span><strong>{prob_fake * 100:.1f}%</strong></div>",
                unsafe_allow_html=True,
            )
            st.progress(min(max(prob_fake, 0.0), 1.0))

        # =====================================================================
        # F. Preprocessing Transparency Section
        # =====================================================================
        st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
        with st.expander("View text preprocessing (Pipeline transparency)"):
            st.caption(
                "Normalized text passed to the DistilBERT sequence tokenizer after headline integration, "
                "wire service byline elimination, URL stripping, and negation-preserving filtering:"
            )
            st.code(cleaned_body if cleaned_body else "(No normalized content produced)", language="text")

    # =========================================================================
    # H. Editorial Colophon & Research Disclaimer
    # =========================================================================
    st.markdown(
        """
        <footer class="editorial-colophon">
            <strong>Academic &amp; Editorial Disclaimer:</strong> This system applies deep learning sequence classification 
            (fine-tuned DistilBERT) to analyze linguistic style, framing, and lexical choices. Output probabilities reflect 
            statistical alignment with training distributions and do <strong>not</strong> constitute independent investigative 
            fact-checking or absolute factual verification. Always corroborate critical claims with primary reporting and domain sources.
        </footer>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()