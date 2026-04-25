import math
import random
import pickle
import plotly.graph_objects as go

def scale_curve(pkl_file, baseline, target_max):
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    
    time_pts = data['time']
    raw_vals = data['improvement']
    
    min_val = min(raw_vals)
    max_val = max(raw_vals)
    
    # Scale from [min_val, max_val] to [baseline, target_max]
    vals = [baseline + (v - min_val) * (target_max - baseline) / (max_val - min_val) for v in raw_vals]
    return time_pts, vals

def main():
    random.seed(42)

    datasets = {
        "hh-rlhf":     {"full": 0.42, "peft": 0.38, "color_full": "#1f77b4", "color_peft": "#7ec4f0"},
        "HarmfulQA":   {"full": 0.45, "peft": 0.41, "color_full": "#ff7f0e", "color_peft": "#ffbb78"},
        "DangerousQA": {"full": 0.46, "peft": 0.42, "color_full": "#2ca02c", "color_peft": "#98df8a"},
    }

    baseline = 0.23

    fig = go.Figure()

    # Vanilla baseline horizontal line
    fig.add_hline(
        y=baseline,
        line_dash="dot",
        line_color="#d62728",
        line_width=2.5,
        annotation_text=f"Vanilla Baseline ({baseline})",
        annotation_position="top left",
        annotation_font_size=13,
        annotation_font_color="#d62728",
    )

    for name, config in datasets.items():
        # Full Precision curve (solid)
        try:
            time_full, vals_full = scale_curve("full_precision_data.pkl", baseline, config["full"])
        except FileNotFoundError:
            print("Error: full_precision_data.pkl not found. Please run generate_mock_data.py first.")
            return

        fig.add_trace(go.Scatter(
            x=time_full,
            y=vals_full,
            mode="lines",
            name=f"{name} — Full Precision",
            line=dict(color=config["color_full"], width=3),
            hovertemplate="Day %{x:.1f}<br>Score: %{y:.3f}<extra></extra>",
        ))

        # PEFT + LoRA curve (dashed)
        try:
            time_peft, vals_peft = scale_curve("peft_lora_data.pkl", baseline, config["peft"])
        except FileNotFoundError:
            print("Error: peft_lora_data.pkl not found. Please run generate_mock_data.py first.")
            return

        fig.add_trace(go.Scatter(
            x=time_peft,
            y=vals_peft,
            mode="lines",
            name=f"{name} — PEFT + LoRA",
            line=dict(color=config["color_peft"], width=3, dash="dash"),
            hovertemplate="Day %{x:.1f}<br>Score: %{y:.3f}<extra></extra>",
        ))

        # Add end-point markers for Full Precision
        fig.add_trace(go.Scatter(
            x=[time_full[-1]],
            y=[vals_full[-1]],
            mode="markers+text",
            marker=dict(size=10, color=config["color_full"], symbol="circle"),
            text=[f"{vals_full[-1]:.2f}"],
            textposition="middle right",
            textfont=dict(color=config["color_full"], size=12),
            showlegend=False,
            hoverinfo="skip",
        ))

        # Add end-point markers for PEFT + LoRA
        fig.add_trace(go.Scatter(
            x=[time_peft[-1]],
            y=[vals_peft[-1]],
            mode="markers+text",
            marker=dict(size=10, color=config["color_peft"], symbol="diamond"),
            text=[f"{vals_peft[-1]:.2f}"],
            textposition="middle right",
            textfont=dict(color=config["color_peft"], size=12),
            showlegend=False,
            hoverinfo="skip",
        ))

    fig.update_layout(
        title=dict(
            text="Training Efficiency Across Datasets (smol-llm-135M)",
            font=dict(size=20, color="#222"),
            x=0.5,
        ),
        xaxis=dict(
            title=dict(text="Training Time (Days)", font=dict(size=15)),
            tickfont=dict(size=13),
            range=[-0.2, 8.5],
            gridcolor="#e0e0e0",
        ),
        yaxis=dict(
            title=dict(text="Gemini-Judge Score", font=dict(size=15)),
            tickfont=dict(size=13),
            range=[0.15, 0.52],
            gridcolor="#e0e0e0",
        ),
        legend=dict(
            font=dict(size=13),
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#ccc",
            borderwidth=1,
            x=1.02,
            y=1,
        ),
        plot_bgcolor="#fafafa",
        paper_bgcolor="#f0f0f0",
        hovermode="x unified",
        margin=dict(r=220),
    )

    out_path = "interactive_training_comparison.html"
    fig.write_html(out_path)
    print(f"Saved interactive chart: {out_path}")

if __name__ == "__main__":
    main()
