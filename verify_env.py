import sys

print(f"Python: {sys.version}")

modules = [
    ("torch", "PyTorch"),
    ("torchaudio", "torchaudio"),
    ("df", "DeepFilterNet"),
    ("df.enhance", "DeepFilterNet enhance"),
    ("gradio", "Gradio"),
    ("numpy", "NumPy"),
    ("scipy", "SciPy"),
    ("matplotlib", "Matplotlib"),
    ("soundfile", "SoundFile"),
    ("pesq", "PESQ"),
    ("pystoi", "PySTOI"),
]

all_ok = True
for mod_name, label in modules:
    try:
        mod = __import__(mod_name, fromlist=["*"])
        ver = getattr(mod, "__version__", "ok")
        print(f"  [OK]   {label:25s} {ver}")
    except Exception as e:
        print(f"  [FAIL] {label:25s} FAILED: {e}")
        all_ok = False

import torch
print(f"\nCUDA available: {torch.cuda.is_available()}")
status = "ALL IMPORTS OK" if all_ok else "SOME IMPORTS FAILED"
print(f"Result: {status}")
