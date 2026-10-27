"""Smoke test for the prediction pipeline post-Phase-2 training."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from prediction import load_pipeline, get_pipeline_info, predict

info = get_pipeline_info()
print("Pipeline steps   :", info["steps"])
print("Model type       :", info["model_type"])
print("n_estimators     :", info["n_estimators"])
print("Features         :", info["features"])
print()

for temp, ph, sal, turb in [(22.5, 7.8, 35.0, 10.0), (28.0, 8.2, 38.0, 40.0)]:
    r = predict(temp, ph, sal, turb)
    print(f"  T={temp} pH={ph} Sal={sal} Turb={turb} => {r['label']} (P_safe={r['prob_safe']:.3f})")

r_bad = predict(5.0, 7.8, 35.0, 10.0)
print()
print("Invalid guard    :", r_bad["valid"], "|", r_bad["error"][:70])
print()
print("OK - pipeline smoke test passed.")
