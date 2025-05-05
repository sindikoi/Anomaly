הקובץ הזה יכיל את כל ההסברים על הפרויקט, איך להריץ אותו, וההוראות הבסיסיות.

README.md:

markdown
Copy
# UR-DMU Project

This is a project for weakly supervised video anomaly detection using I3D features. The project uses models trained on UCF-Crime and XD-Violence datasets for video anomaly detection.

## Requirements

To run this project, you need to install the following dependencies:

- **Python 3.x**
- **Torch**: The version should match the one specified in `requirements.txt`.
- **Other dependencies**: Listed in `requirements.txt`.

You can install them by running:

```bash
pip install -r requirements.txt
Project Structure
The project includes the following components:

ucf_infer.py: Code to run inference on the UCF-Crime dataset.

xd_infer.py: Code to run inference on the XD-Violence dataset.

model/: Directory containing the trained models.

frame_label/: Directory for storing the output of predictions and ground truth.

How to Run
Running UCF-Crime Inference:
bash
Copy
python ucf_infer.py
Running XD-Violence Inference:
bash
Copy
python xd_infer.py
Results
The results will be saved in the frame_label/ directory, including:

ucf_pre.npy: Predictions for the UCF-Crime dataset.

xd_pre.npy: Predictions for the XD-Violence dataset.

Metrics
The following metrics will be displayed after running inference:

AUC (Area Under the Curve)

AP Score (Average Precision)

Classification Accuracy

markdown
Copy

### **שלב 2: הוספת הקבצים ל-Git**

1. פתחי טרמינל במחשב (באיבונטו).
2. גשי לתיקיית הפרויקט שלך:
   ```bash
   cd /mnt/d/UR-DMU
