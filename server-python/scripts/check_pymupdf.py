# 作者：yangkunpeng1
# 日期：2026-07-21
"""诊断当前解释器与 PyMuPDF 是否一致。"""
import sys
import subprocess

print("executable:", sys.executable)
print("version   :", sys.version)
print("prefix    :", sys.prefix)

r = subprocess.run(
    [sys.executable, "-m", "pip", "show", "PyMuPDF"],
    capture_output=True,
    text=True,
)
print("--- pip show PyMuPDF ---")
print(r.stdout or r.stderr or "(empty)")

print("--- import check ---")
try:
    import pymupdf
    print("import pymupdf OK", pymupdf.__version__)
except Exception as e:
    print("import pymupdf FAIL:", e)

try:
    import fitz
    print("import fitz OK", getattr(fitz, "__doc__", "")[:40])
except Exception as e:
    print("import fitz FAIL:", e)
