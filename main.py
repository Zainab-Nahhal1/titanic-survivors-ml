from src.data_cleaning import load_and_clean_data
from src.model_selection import compare_models
from src.final_model import train_final_and_save_submission

def main():
    print("🚀 Titanic ML Project Starting...")
    X, y, X_test, test_df = load_and_clean_data()
    compare_models(X, y)
    train_final_and_save_submission(X, y, X_test, test_df)
    print("✅ Pipeline finished. Check results/ for the submission file.")

if __name__ == "__main__":
    main()
