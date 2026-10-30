import math
import os
import tempfile
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr

MATERIALS = {
    "Structural Steel": (12e-6, 200.0),
    "Concrete (illustrative)": (10e-6, 30.0),
    "Aluminum": (23e-6, 69.0),
    "Stainless Steel": (17e-6, 193.0),
    "Copper": (17e-6, 110.0),
    "Custom": (12e-6, 200.0),
}
MATERIAL_NAMES = list(MATERIALS.keys())
MATERIAL_SET = set(MATERIAL_NAMES)

@dataclass
class ThermalMember:
    material: str
    length_m: float
    alpha: float
    elastic_modulus_gpa: float

    def calculate(self, initial_c, final_c, restrained=False):
        delta_t = final_c - initial_c
        delta_length_m = self.alpha * self.length_m * delta_t
        strain = self.alpha * delta_t
        stress_mpa = -self.elastic_modulus_gpa * 1000 * strain if restrained else 0.0
        return {
            "Temperature change (°C)": delta_t,
            "Length change (mm)": delta_length_m * 1000,
            "Final length (m)": self.length_m + delta_length_m,
            "Thermal strain (µε)": strain * 1e6,
            "Idealized stress (MPa)": stress_mpa,
            "Movement magnitude (mm)": abs(delta_length_m * 1000),
        }

def validate_number(value, name, positive=False):
    if value is None or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite number.")
    value = float(value)
    if positive and value <= 0:
        raise ValueError(f"{name} must be greater than zero.")
    return value

def make_member(material, length, alpha_micro, e_gpa):
    if material not in MATERIAL_SET:
        raise ValueError("Please select a valid material.")
    length = validate_number(length, "Length", positive=True)
    alpha_micro = validate_number(alpha_micro, "Thermal expansion coefficient", positive=True)
    e_gpa = validate_number(e_gpa, "Elastic modulus", positive=True)
    return ThermalMember(material, length, alpha_micro * 1e-6, e_gpa)

def defaults(material):
    alpha, e = MATERIALS[material]
    return alpha * 1e6, e

def temperature_curve(member, t_initial, t_final, restrained):
    temperatures = np.linspace(min(t_initial, t_final) - 20, max(t_initial, t_final) + 20, 121)
    delta_lengths = member.alpha * member.length_m * (temperatures - t_initial) * 1000
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(temperatures, delta_lengths, linewidth=2.5, label=member.material)
    ax.scatter([t_final], [member.calculate(t_initial, t_final, restrained)["Length change (mm)"]], s=80, zorder=3, label="Selected final temperature")
    ax.axhline(0, color="gray", linewidth=1)
    ax.set(xlabel="Final temperature (°C)", ylabel="Length change (mm)", title="Thermal expansion / contraction curve")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig

def compare_materials(member, t_initial, t_final, view_mode):
    names = [name for name in MATERIAL_NAMES if name != "Custom"]
    if member.material == "Custom":
        names.append("Custom")
    rows = []
    for name in names:
        if name == member.material:
            alpha, e = member.alpha, member.elastic_modulus_gpa
        else:
            alpha, e = MATERIALS[name]
        value_mm = alpha * member.length_m * (t_final - t_initial) * 1000
        rows.append({"Material": name,
                     "Expansion coefficient (×10⁻⁶/°C)": round(alpha * 1e6, 4),
                     "Original length (m)": member.length_m,
                     "Initial temperature (°C)": t_initial,
                     "Final temperature (°C)": t_final,
                     "Temperature change (°C)": t_final - t_initial,
                     "Length change (mm)": round(value_mm, 5),
                     "Category": "SELECTED / EDITED" if name == member.material else "PRESET"})
    df = pd.DataFrame(rows)
    display_df = df if view_mode == "Compare all materials" else df[df["Material"] == member.material]
    fig, ax = plt.subplots(figsize=(9, 4.3))
    colors = ["#f97316" if n == member.material else "#94a3b8" for n in display_df["Material"]]
    values = display_df["Length change (mm)"].to_numpy()
    bars = ax.barh(display_df["Material"], values, color=colors, height=0.6)
    margin = max(0.7, float(np.max(np.abs(values))) * 0.22) if len(values) else 0.7
    ax.set_xlim(min(0, float(np.min(values))) - margin, max(0, float(np.max(values))) + margin)
    for bar, value in zip(bars, values):
        offset = margin * 0.12
        ax.text(value + (offset if value >= 0 else -offset),
                bar.get_y() + bar.get_height()/2, f"{value:+.2f} mm",
                ha="left" if value >= 0 else "right", va="center", fontsize=10)
    ax.axvline(0, color="#334155", linewidth=1)
    ax.set(xlabel="Length change (mm)", title=f"Same length {member.length_m:g} m | ΔT = {t_final-t_initial:+g}°C")
    ax.grid(axis="x", alpha=0.16)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return df, fig

def calculate_dashboard(material, length, t_initial, t_final, alpha_micro, e_gpa, restrained, view_mode):
    try:
        t_initial = validate_number(t_initial, "Initial temperature")
        t_final = validate_number(t_final, "Final temperature")
        member = make_member(material, length, alpha_micro, e_gpa)
        result = member.calculate(t_initial, t_final, restrained)
        change = result["Length change (mm)"]
        state = "Expansion" if change > 0 else "Contraction" if change < 0 else "No change"
        stress_note = "Idealized fully restrained stress" if restrained else "Free movement: no restraint stress modeled"
        summary = (f"### {state}: {change:+.3f} mm\n"
                   f"**Final length:** {result['Final length (m)']:.6f} m  |  "
                   f"**Thermal strain:** {result['Thermal strain (µε)']:+.2f} µε\n\n"
                   f"**{stress_note}:** {result['Idealized stress (MPa)']:+.3f} MPa  |  "
                   f"**Absolute movement:** {result['Movement magnitude (mm)']:.3f} mm\n\n"
                   "*Positive length change means expansion; negative means contraction. "
                   "Negative idealized stress means compression.*")
        df = pd.DataFrame([{"Material": material, "Length (m)": length, "Initial T (°C)": t_initial,
                            "Final T (°C)": t_final, "Alpha (µ/°C)": alpha_micro,
                            "E (GPa)": e_gpa, "Fully restrained": restrained, **result}])
        comparison, bar_fig = compare_materials(member, t_initial, t_final, view_mode)
        curve_fig = temperature_curve(member, t_initial, t_final, restrained)
        path = tempfile.NamedTemporaryFile(suffix=".csv", prefix="thermal_expansion_", delete=False).name
        df.to_csv(path, index=False)
        return summary, df.round(6), curve_fig, comparison.round(5), bar_fig, comparison.round(5), path
    except (ValueError, TypeError, OverflowError) as error:
        raise gr.Error(str(error))

check = ThermalMember("Structural Steel", 10, 12e-6, 200).calculate(20, 60)
assert np.isclose(check["Length change (mm)"], 4.8)
assert np.isclose(check["Final length (m)"], 10.0048)
print("Verification passed: 10 m steel member +40°C => +4.8 mm")
assert np.isclose(ThermalMember("Concrete (illustrative)", 15, 10e-6, 30).calculate(35, 10)["Length change (mm)"], -3.75)
assert np.isclose(ThermalMember("Aluminum", 8, 23e-6, 69).calculate(20, 70)["Length change (mm)"], 9.2)
_test_member = ThermalMember("Structural Steel", 10, 20e-6, 200)
_test_compare, _test_plot = compare_materials(_test_member, 20, 60, "Compare all materials")
assert np.isclose(_test_compare.loc[_test_compare["Material"] == "Structural Steel", "Length change (mm)"].iloc[0], 8.0)
assert (_test_compare["Initial temperature (°C)"] == 20).all()
assert (_test_compare["Final temperature (°C)"] == 60).all()
_test_compare2, _test_plot2 = compare_materials(_test_member, 20, 80, "Compare all materials")
assert np.isclose(_test_compare2.loc[_test_compare2["Material"] == "Aluminum", "Length change (mm)"].iloc[0], 13.8)
plt.close(_test_plot2)
plt.close(_test_plot)
print("Additional verification passed: contraction, aluminum expansion, edited coefficient comparison")


with gr.Blocks(title="Thermal Expansion Calculator · CE 206") as demo:
    gr.Markdown("# 🌡️ Thermal Expansion Calculator")
    gr.Markdown("**CE 206 | Civil Engineering** — Calculate thermal movement, explore temperatures, and compare material coefficients.")
    gr.Markdown("**How it works:** Choose a material → enter length and temperatures → click **Calculate & update charts**. The comparison table and chart update together for every material, using the SAME temperatures.")
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1 · Your member")
            material = gr.Dropdown(MATERIAL_NAMES, value="Structural Steel", label="Material")
            length = gr.Number(value=10, label="Original length (m)")
            with gr.Row():
                initial = gr.Number(value=20, label="Initial temperature (°C)")
                final = gr.Number(value=60, label="Final temperature (°C)")
            alpha = gr.Number(value=12, label="Expansion coefficient α (×10⁻⁶/°C)")
            elastic = gr.Number(value=200, label="Young's modulus E (GPa)")
            restrained = gr.Checkbox(value=False, label="Estimate fully restrained thermal stress")
            run = gr.Button("Calculate & update charts", variant="primary")
            gr.Markdown("**Coefficient α:** Determines thermal length change. You may edit it, including for a preset material. **Young's modulus E:** Used only for the optional fully restrained stress estimate, **not** the expansion or bar chart.")
        with gr.Column(scale=2):
            gr.Markdown("### 2 · Calculation result")
            summary = gr.Markdown("Click **Calculate & update charts** to see the result.")
            result_table = gr.Dataframe(label="Detailed results — selected member", interactive=False)
            chart = gr.Plot(label="Temperature response — selected member")
    with gr.Tab("Bar chart comparison"):
        gr.Markdown("### What is this comparison doing?")
        gr.Markdown("Each material is evaluated at the **same length, initial temperature, and final temperature**. This shows how material coefficients affect expansion. **Orange = your selected material**, using your edited coefficient; gray = other materials with preset illustrative coefficients. Changing the selected name alone does not change other materials' numerical values.")
        view_mode = gr.Radio(["Compare all materials", "Selected material only"], value="Compare all materials", label="What should the bar chart show?")
        comparison_table = gr.Dataframe(label="Comparison data (selected/edited vs presets)", interactive=False)
        comparison_chart = gr.Plot(label="Length change comparison")
        gr.Markdown("**Important:** Switching the chosen material only highlights its bar; it will not alter the other materials’ coefficients. To change ALL bars, change the original length or temperatures, then click the Calculate button. To change only your material, edit its coefficient α.")
    with gr.Tab("All materials · same conditions"):
        gr.Markdown("### All materials at your chosen length and temperatures")
        gr.Markdown("This is a **calculated comparison**, not a list of fixed examples. Every row uses the same original length, initial temperature, and final temperature from the form. Each material has its own coefficient. Click **Calculate & update charts** whenever you change the inputs. The selected material uses your edited coefficient.")
        scenario_table = gr.Dataframe(label="Live same-condition results for every material", interactive=False)
        gr.Markdown("For instance, 10 m from 20°C to 60°C gives steel +4.8 mm, concrete +4.0 mm and aluminum +9.2 mm. With a higher final temperature, all these results increase proportionally.")
        csv_file = gr.File(label="Download CSV for your latest calculation")
        gr.Markdown("""### Equations, sign convention & limitations
- **Temperature change:** ΔT = Tf − Ti (°C)
- **Free length change:** ΔL = α × L₀ × ΔT (m), multiplied by 1000 for millimeters
- **Thermal strain:** ε = α × ΔT (dimensionless)
- **Ideal fully restrained elastic stress:** σ = −E × ε (MPa after conversion)
- **Movement magnitude:** |ΔL| (not an automatically approved expansion-joint gap)

Positive ΔL means **expansion**; negative ΔL means **contraction**. Negative stress means **compression**. All coefficients are illustrative constants. This educational tool does not model partial restraint, cracking, creep, safety factors, or nonlinear temperature dependence.""")
    material.change(fn=defaults, inputs=material, outputs=[alpha, elastic])
    run.click(fn=calculate_dashboard,
              inputs=[material, length, initial, final, alpha, elastic, restrained, view_mode],
              outputs=[summary, result_table, chart, comparison_table, comparison_chart, scenario_table, csv_file])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "10000")), show_error=True)
