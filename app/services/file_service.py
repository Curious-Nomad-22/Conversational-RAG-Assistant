import os

UPLOAD_DIR = "app/uploads"

os.makedirs(UPLOAD_DIR, exist_ok=True)

def save_uploaded_files(uploaded_files):
    saved_files = []

    for file in uploaded_files:
        file_path = os.path.join(UPLOAD_DIR, file.name)

        with open(file_path, "wb") as f:
            f.write(file.getbuffer())

        saved_files.append(file_path)

    return saved_files