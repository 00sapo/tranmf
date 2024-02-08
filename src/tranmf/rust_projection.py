from ctypes import cdll
from pathlib import Path

# Load the Rust library

parent_dir = Path(__file__).resolve().parents[2]
rust_lib = cdll.LoadLibrary(parent_dir / "target" / "release" / "libtranmf.so")

# Call the exported function
rust_lib.run_genetic_algorithm()
