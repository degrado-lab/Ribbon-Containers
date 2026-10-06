"""Apply _COMPILE_TARGETS patch to rfd3.engine.RFD3InferenceEngine for UBM fast mode compatibility."""
import sys

engine_file = "/app/foundry/.venv/lib/python3.12/site-packages/rfd3/engine.py"
with open(engine_file, "r") as f:
    content = f.read()

target = "class RFD3InferenceEngine(BaseInferenceEngine):"
replacement = """class RFD3InferenceEngine(BaseInferenceEngine):
    _COMPILE_TARGETS = (
        "encoder",
        "diffusion_token_encoder",
        "diffusion_transformer",
        "decoder",
    )"""

if "_COMPILE_TARGETS" not in content:
    if target not in content:
        print(f"Error: Could not find target class in {engine_file}", file=sys.stderr)
        sys.exit(1)
    content = content.replace(target, replacement, 1)
    with open(engine_file, "w") as f:
        f.write(content)
    print("Successfully patched RFD3InferenceEngine with _COMPILE_TARGETS")
else:
    print("RFD3InferenceEngine already contains _COMPILE_TARGETS")
