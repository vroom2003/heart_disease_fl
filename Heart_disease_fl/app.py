# app.py
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from fpdf import FPDF
from fl_bqc_simulator import HeartDiseaseFLBQC

st.set_page_config(page_title="FL vs FL+BQC on Heart Disease", layout="wide")
st.title("🫀 FL vs FL+BQC on Heart Disease Data (Qiskit Simulation)")

st.sidebar.header("⚙️ Settings")
n_clients = st.sidebar.slider("Number of Hospitals (Clients)", 2, 4, 3)
epochs = st.sidebar.slider("Training Rounds", 5, 20, 10)
run_simulation = st.sidebar.button("🚀 Run Simulation")

if run_simulation:
    with st.spinner("🏥 Loading and Cleaning Heart Disease Data..."):
        from load_data import *  # Auto-clean
        simulator = HeartDiseaseFLBQC(n_clients=n_clients)

    with st.spinner("🔁 Running Classical FL..."):
        df_fl = simulator.run_fl_only(epochs=epochs)

    with st.spinner("🔮 Running FL + BQC (Quantum-Blind)..."):
        df_bqc = simulator.run_efl_bqc(epochs=epochs)

    df = pd.concat([df_fl, df_bqc], ignore_index=True)
    df.to_csv("results.csv", index=False)
    st.success("✅ Simulation Complete!")

    # Metrics
    col1, col2 = st.columns(2)
    final_fl = df_fl['accuracy'].iloc[-1]
    final_bqc = df_bqc['accuracy'].iloc[-1]
    col1.metric("FL Only - Accuracy", f"{final_fl:.3f}")
    col2.metric("FL+BQC - Accuracy", f"{final_bqc:.3f}")

    # Accuracy Plot
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(df_fl['round'], df_fl['accuracy'], 'o-', label="FL Only", linewidth=2)
    ax1.plot(df_bqc['round'], df_bqc['accuracy'], 's-', label="FL + BQC", linewidth=2)
    ax1.set_title("Heart Disease Prediction Accuracy")
    ax1.set_xlabel("Round")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    st.pyplot(fig1)

    # Privacy Plot
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(df_fl['round'], df_fl['privacy_leakage'], 'o-', label="FL - High Risk", color='red')
    ax2.plot(df_bqc['round'], df_bqc['privacy_leakage'], 's-', label="FL+BQC - Low Risk", color='green')
    ax2.set_title("Privacy Leakage: Classical vs Quantum-Blind")
    ax2.set_xlabel("Round")
    ax2.set_ylabel("Leakage Score")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig2)

    # Export
    st.sidebar.subheader("📤 Export Results")
    csv = df.to_csv(index=False).encode()
    st.sidebar.download_button("💾 Download CSV", csv, "federated_results.csv", "text/csv")

    if st.sidebar.button("🖨️ Generate PDF Report"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(200, 10, "Federated Learning vs FL+BQC Report", ln=True, align='C')
        pdf.ln(10)
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 8, f"• FL Accuracy: {final_fl:.3f}", ln=True)
        pdf.cell(200, 8, f"• FL+BQC Accuracy: {final_bqc:.3f}", ln=True)
        pdf.cell(200, 8, f"• Privacy Improved: {'Yes' if df_bqc['privacy_leakage'].mean() < df_fl['privacy_leakage'].mean() else 'No'}", ln=True)
        pdf.output("federated_report.pdf")
        st.sidebar.success("📄 PDF report saved!")

    st.dataframe(df.pivot(index='round', columns='method', values=['accuracy', 'privacy_leakage']))
else:
    st.info("👈 Adjust settings and click 'Run Simulation'")
    st.image("https://www.cdc.gov/heartdisease/images/heart-disease-graph.jpg", width=700)
    st.caption("Heart Disease affects 1 in 4 deaths in the US (CDC)")