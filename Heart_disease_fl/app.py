# app.py
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from fpdf import FPDF
from fl_bqc_simulator import HeartDiseaseFLBQC
import base64

# --- Custom CSS for Styling ---
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
        font-family: 'Segoe UI', sans-serif;
    }
    .stButton>button {
        background-color: #007BFF;
        color: white;
        border-radius: 10px;
        height: 50px;
        font-size: 18px;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 18px;
        font-weight: bold;
    }
    .metric-card {
        border-radius: 10px;
        padding: 15px;
        background: white;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
        margin-bottom: 15px;
        text-align: center;
    }
    .header {
        color: #003366;
        text-align: center;
        font-size: 32px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown('<div class="header">🫀 HeartGuard AI: FL vs FL+BQC Dashboard</div>', unsafe_allow_html=True)
st.markdown("### 🔐 Privacy-Preserving Heart Disease Prediction with Quantum-Inspired Security")
st.markdown("---")

# --- Sidebar ---
st.sidebar.image("https://www.cdc.gov/heartdisease/images/heart-disease-graph.jpg", width=280)
st.sidebar.title("⚙️ Simulation Settings")
n_clients = st.sidebar.slider("Number of Hospitals (Clients)", 2, 4, 3)
epochs = st.sidebar.slider("Training Rounds", 5, 20, 10)
run_simulation = st.sidebar.button("🚀 Run Simulation")

# --- Main Content ---
if run_simulation:
    with st.spinner("🏥 Loading and Cleaning Heart Disease Data..."):
        from load_data import *
        simulator = HeartDiseaseFLBQC(n_clients=n_clients)

    with st.spinner("🔁 Running Classical FL..."):
        df_fl = simulator.run_fl_only(epochs=epochs)

    with st.spinner("🔮 Running FL + BQC (Quantum-Blind)..."):
        df_bqc = simulator.run_efl_bqc(epochs=epochs)

    df = pd.concat([df_fl, df_bqc], ignore_index=True)
    df.to_csv("results.csv", index=False)
    st.success("✅ Simulation Complete!")

    # Combine results for plotting
    df_combined = df.pivot(index='round', columns='method', values=['accuracy', 'privacy_leakage'])

    # --- Tabs ---
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "📈 Accuracy", "🛡️ Privacy", "📤 Export"])

    with tab1:
        st.subheader("🎯 Performance Overview")

        col1, col2, col3 = st.columns(3)
        final_fl_acc = df_fl['accuracy'].iloc[-1]
        final_bqc_acc = df_bqc['accuracy'].iloc[-1]
        acc_diff = final_fl_acc - final_bqc_acc

        col1.markdown('<div class="metric-card">', unsafe_allow_html=True)
        col1.metric("FL Only Accuracy", f"{final_fl_acc:.3f}", delta=f"{acc_diff:+.3f}")
        col1.markdown('</div>', unsafe_allow_html=True)

        col2.markdown('<div class="metric-card">', unsafe_allow_html=True)
        col2.metric("FL+BQC Accuracy", f"{final_bqc_acc:.3f}", delta=f"{-acc_diff:.3f}")
        col2.markdown('</div>', unsafe_allow_html=True)

        priv_fl = df_fl['privacy_leakage'].mean()
        priv_bqc = df_bqc['privacy_leakage'].mean()
        improvement = (priv_fl - priv_bqc) / priv_fl * 100

        col3.markdown('<div class="metric-card">', unsafe_allow_html=True)
        col3.metric("Privacy Improvement", "✅ Yes" if priv_bqc < priv_fl else "❌ No")
        col3.markdown(f"<small>Leakage ↓ {improvement:.1f}%</small>", unsafe_allow_html=True)
        col3.markdown('</div>', unsafe_allow_html=True)

        # Summary Insights
        st.markdown("### 💡 Key Insights")
        if priv_bqc < priv_fl:
            st.success("🔐 **FL+BQC significantly reduces privacy leakage** — quantum-inspired blindness works!")
        else:
            st.warning("⚠️ Privacy not improved — consider adjusting simulation parameters.")

        if acc_diff < 0.05:
            st.success("📊 Accuracy remains stable — EFL-BQC is viable!")
        else:
            st.info("📉 Slight accuracy drop — trade-off for enhanced privacy.")

    with tab2:
        st.subheader("📈 Model Accuracy Over Rounds")
        fig_acc = go.Figure()
        fig_acc.add_trace(go.Scatter(x=df_fl['round'], y=df_fl['accuracy'],
                                     mode='lines+markers', name='FL Only',
                                     line=dict(color='#1f77b4', width=3), marker=dict(size=6)))
        fig_acc.add_trace(go.Scatter(x=df_bqc['round'], y=df_bqc['accuracy'],
                                     mode='lines+markers', name='FL + BQC',
                                     line=dict(color='#d62728', width=3), marker=dict(size=6)))
        fig_acc.update_layout(
            xaxis_title="Round",
            yaxis_title="Accuracy",
            hovermode='x unified',
            template='plotly_white',
            legend=dict(orientation="h", y=1.1)
        )
        st.plotly_chart(fig_acc, use_container_width=True)

    with tab3:
        st.subheader("🛡️ Privacy Leakage (Gradient Exposure)")
        fig_priv = go.Figure()
        fig_priv.add_trace(go.Scatter(x=df_fl['round'], y=df_fl['privacy_leakage'],
                                      mode='lines+markers', name='FL Only',
                                      line=dict(color='red', width=3), marker=dict(size=6)))
        fig_priv.add_trace(go.Scatter(x=df_bqc['round'], y=df_bqc['privacy_leakage'],
                                      mode='lines+markers', name='FL + BQC',
                                      line=dict(color='green', width=3), marker=dict(size=6)))
        fig_priv.update_layout(
            xaxis_title="Round",
            yaxis_title="Leakage Score",
            hovermode='x unified',
            template='plotly_white',
            legend=dict(orientation="h", y=1.1)
        )
        st.plotly_chart(fig_priv, use_container_width=True)

    with tab4:
        st.subheader("📤 Export Results")

        # CSV
        csv = df.to_csv(index=False).encode()
        st.download_button("💾 Download CSV Results", csv, "federated_results.csv", "text/csv")

        # PDF Report
        if st.button("🖨️ Generate Styled PDF Report"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", 'B', 16)
            pdf.set_fill_color(200, 220, 255)
            pdf.cell(0, 10, "HeartGuard AI: FL vs FL+BQC Report", ln=True, align='C', fill=True)
            pdf.ln(10)

            pdf.set_font("Arial", size=12)
            pdf.cell(0, 8, f"• FL Accuracy: {final_fl_acc:.3f}", ln=True)
            pdf.cell(0, 8, f"• FL+BQC Accuracy: {final_bqc_acc:.3f}", ln=True)
            pdf.cell(0, 8, f"• Privacy Improved: {'Yes' if priv_bqc < priv_fl else 'No'}", ln=True)
            pdf.cell(0, 8, f"• Leakage Reduction: {improvement:.1f}%", ln=True)
            pdf.ln(10)

            # Add results table (simplified)
            pdf.set_font("Courier", size=10)
            for _, row in df.tail(10).iterrows():
                pdf.cell(30, 6, str(row['round']), border=1)
                pdf.cell(40, 6, row['method'], border=1)
                pdf.cell(40, 6, f"{row['accuracy']:.3f}", border=1)
                pdf.cell(50, 6, f"{row['privacy_leakage']:.2e}", border=1)
                pdf.ln(6)

            pdf.output("HeartGuard_Report.pdf")
            st.success("📄 PDF report saved as `HeartGuard_Report.pdf`")

        # Show DataFrame
        st.dataframe(df, use_container_width=True)

else:
    st.markdown("## 👋 Welcome to HeartGuard AI")
    st.markdown("""
    This dashboard simulates **Federated Learning (FL)** and **Enhanced FL with Blind Quantum Computing (EFL-BQC)**  
    on real **heart disease patient data** — all without sharing sensitive records.

    ### 🔍 What You Can Do:
    - Simulate secure AI collaboration between hospitals
    - Compare accuracy and privacy of FL vs FL+BQC
    - Export results for research or reporting

    💡 **Click 'Run Simulation' in the sidebar to begin!**
    """)
    st.image("https://www.cdc.gov/heartdisease/images/heart-disease-graph.jpg", width=700)
    st.caption("Heart Disease affects 1 in 4 deaths in the US (CDC)")