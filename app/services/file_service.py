import hashlib
import os
import re
from pathlib import Path

UPLOAD_DIR = "app/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


def save_uploaded_files(uploaded_files):
    """Save uploaded PDFs using safe, content-addressed filenames."""
    saved_files = []
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    for uploaded_file in uploaded_files:
        original_name = Path(uploaded_file.name).name
        safe_stem = Path(original_name).stem
        safe_stem = re.sub(r"[^A-Za-z0-9_-]+", "_", safe_stem).strip("_")
        safe_stem = (safe_stem or "document")[:60]
        content = uploaded_file.getvalue()
        digest = hashlib.sha256(content).hexdigest()[:12]
        file_path = os.path.join(UPLOAD_DIR, f"{safe_stem}_{digest}.pdf")

        with open(file_path, "wb") as output_file:
            output_file.write(content)

        saved_files.append(file_path)

    return saved_files
