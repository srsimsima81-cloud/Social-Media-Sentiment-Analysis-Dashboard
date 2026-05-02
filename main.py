import os
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT_DIR)

def banner():
    print("\n==============================")
    print("📡 REPUTATION RADAR SYSTEM")
    print("==============================\n")

def menu():
    print("1. Run Streamlit Dashboard (UI)")
    print("2. Test Sentiment Prediction (CLI)")
    print("3. Run Dataset Evaluation (BERT)")
    print("0. Exit\n")

def run_dashboard():
    os.system("streamlit run app/dashboard.py")

def run_predict():
    os.system("python src/predict.py")

def run_evaluation():
    os.system("python src/train.py")

# -----------------------------
# MAIN
# -----------------------------
banner()

while True:
    menu()
    choice = input("Enter choice: ")

    if choice == "1":
        run_dashboard()

    elif choice == "2":
        run_predict()

    elif choice == "3":
        run_evaluation()

    elif choice == "0":
        print("Exiting...")
        break

    else:
        print("Invalid choice. Try again.")