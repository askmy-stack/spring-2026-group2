import sys
from pathlib import Path
from pipeline import SeizurePipeline

# --- SMART PATH SETUP ---
BASE_DIR = Path(__file__).parent.resolve()
CONFIG_PATH = BASE_DIR / "config.yaml"


def main():
    print(" SEIZURE SYSTEM MANAGER")
    print("--------------------------------")
    print("1. Level 1: Whole Dataset (CHB-MIT / Siena)")
    print("2. Level 2: Dynamic Subject (Local File)")
    print("3. Level 3: Dynamic Subject (Download from URL)")
    print("--------------------------------")

    choice = input("Select Data Level [1-3]: ").strip()

    if not CONFIG_PATH.exists():
        return print(f" Config not found at {CONFIG_PATH}")

    pipe = SeizurePipeline(str(CONFIG_PATH))
    index_path = None
    df_clean = None

    if choice == "1":
        # --- LEVEL 1: FULL DATASET ---
        print("\n Starting Full Data Pipeline...")
        pipe.repair_edf_headers()
        pipe.fetch_missing_summaries()
        pipe.parse_demographics()  # NEW: Parses Age/Sex for report

        df = pipe.discover_files()
        if df.empty: return print(" No files found.")

        labels_csv = pipe.parse_labels()
        df_clean = pipe.preprocess(df)
        splits = pipe.split_data(df_clean)

        # Build Index
        index_path = pipe.build_index(splits['train'], labels_csv, "train")
        pipe.build_index(splits['test'], labels_csv, "test")

    elif choice == "2":
        # --- LEVEL 2: LOCAL SINGLE FILE ---
        path_str = input("\n Enter path to EEG file: ").strip().strip("'")
        f_path = Path(path_str)
        if not f_path.exists(): return print(" File not found.")

        print(f" Processing: {f_path.name}")
        df = pipe.discover_files(custom_path=str(f_path.parent))
        df = df[df['path'].apply(lambda x: Path(x).name == f_path.name)]
        df_clean = pipe.preprocess(df)
        index_path = pipe.build_index(df_clean, None, "dynamic")

    elif choice == "3":
        # --- LEVEL 3: URL DOWNLOAD ---
        url = input("\n Enter Dataset URL (.edf): ").strip()
        downloaded_path = pipe.download_external_data(url)

        if downloaded_path:
            print(f" Processing Downloaded Data...")
            df = pipe.discover_files(custom_path=str(downloaded_path.parent))
            df = df[df['path'].apply(lambda x: Path(x).name == downloaded_path.name)]
            df_clean = pipe.preprocess(df)
            index_path = pipe.build_index(df_clean, None, "dynamic")
        else:
            return

    else:
        print("Invalid choice.")
        return

    # --- FINAL REPORT (Metadata, Labels, Age/Sex) ---
    if df_clean is not None and not df_clean.empty and index_path:
        pipe.generate_report(df_clean, index_path)


if __name__ == "__main__":
    main()
