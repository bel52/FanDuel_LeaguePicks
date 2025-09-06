from app.data_ingestion import load_weekly_data_with_warnings

def main():
    df, warns = load_weekly_data_with_warnings()
    print("\n=== Diagnostics ===")
    for w in warns:
        print(w)
    if df is None:
        print("\nResult: df=None")
    else:
        print(f"\nResult: df shape = {df.shape}")
        print(df.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
