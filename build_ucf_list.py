import os

# הנתיב שבו שמורים קבצי הפיצ'רים – גרסה של WSL (Ubuntu)
feature_dir = "/mnt/d/UR-DMU/i3d-features/i3d-features/rgb/"

# איפה לשמור את קובץ הרשימה
output_file = "/mnt/d/UR-DMU/i3d-features/UCF_Test_full.list"

# מוצא את כל קבצי .npy בתיקייה
npy_files = sorted([f for f in os.listdir(feature_dir) if f.endswith(".npy")])

# כותב את הנתיבים לקובץ
with open(output_file, "w") as f:
    for fname in npy_files:
        full_path = os.path.join(feature_dir, fname)
        f.write(full_path + "\n")