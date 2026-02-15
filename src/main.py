from dataloader import SeizurePipeline
from src.eda.eda_pipeline import BIDSEDAPreprocessPipeline
import yaml
from pathlib import Path


def run_eda_module():
    print("\n STARTING eda & PREPROCESSING SUITE")
    print("-" * 32)

    config_path = Path("eda/preprocess_eda_config.yaml")
    if not config_path.exists():
        print(" Config not found: preprocess_eda_config.yaml")
        return

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Run the eda Pipeline
    try:
        runner = BIDSEDAPreprocessPipeline(cfg)
        runner.run()
        print("\n eda Complete! Check 'src/results/figs' folder.")
    except Exception as e:
        print(f"\n eda Failed: {e}")


def main():
    print(" SEIZURE SYSTEM MASTER")
    print("-" * 32)
    print("1. Run Dataloader (Raw -> BIDS -> Split -> Index)")
    print("2. Test BIDS Integrity")
    print("3. Run eda & Visualization (Generate Plots)")
    print("-" * 32)

    choice = input("Select [1-3]: ").strip()

    if choice == '1':
        pipe = SeizurePipeline()
        pipe.run_pipeline_level_1()
    elif choice == '2':
        from mne_bids import make_report
        pipe = SeizurePipeline()
        print(make_report(pipe.bids_root))
    elif choice == '3':
        run_eda_module()
    else:
        print(" Invalid selection.")


if __name__ == "__main__":
    main()