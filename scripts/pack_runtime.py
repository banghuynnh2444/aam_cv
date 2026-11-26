"""Build the model bundle to transfer privately alongside a source checkout."""
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
FILES = [
    "checkpoints/scrfd_2.5g_bnkps.onnx",
    "checkpoints/w600k_r50.onnx",
    "checkpoints/classical_aam_model.npz",
    "checkpoints/aam_cascade_sdm_v2.npz",
    "checkpoints/dan_rafdb_best.pth",
    "checkpoints/effnet_b0_fairface_v3.pth",
    "identity_embeddings.npz",
]


def main():
    missing = [name for name in FILES if not (ROOT / name).is_file()]
    if missing:
        raise FileNotFoundError("Missing runtime files: " + ", ".join(missing))
    target = ROOT / "exports" / "AAM_cv_runtime.zip"
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as bundle:
        for name in FILES:
            bundle.write(ROOT / name, name)
    print(f"Created: {target} ({target.stat().st_size / 1024**2:.1f} MiB)")
    print("Contains enrolled face embeddings. Share privately with your teammate.")


if __name__ == "__main__":
    main()
