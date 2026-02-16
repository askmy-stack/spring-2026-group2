import sys
import yaml
from pathlib import Path
from dataloader import EnhancedEEGLoader as UniversalEEGLoader


def quick_verify(loader, num_samples=10):
    """Quick verification function for backward compatibility"""
    for i in range(min(num_samples, len(loader))):
        data, label = loader[i]
        assert data.shape == (16, 256)
        assert label.item() in [0, 1]
    print(f"PASS Verified {num_samples} samples")


def clear_screen():
    """Clear terminal screen"""
    print("\n" * 2)


def show_menu():
    """Display main menu"""
    print("=" * 70)
    print("  EEG DATA LOADER")
    print("=" * 70)
    print("\nMAIN MENU:")
    print("=" * 50)
    print("1.  Run Full Pipeline (Ingest -> BIDS -> Index -> Balance)")
    print("2.  Generate BIDS Report")
    print("3.  Quick Verification (Check Dataset Integrity)")
    print("4.  Show Dataset Statistics")
    print("5.  Export Metadata Report")
    print("6.  Clear Cache")
    print("0.  Exit")
    print("=" * 50)


def run_full_pipeline():
    """Run complete data processing pipeline"""
    print("\nStarting Full Pipeline...")
    print("This will:")
    print("  1. Scan for raw EEG files")
    print("  2. Convert to BIDS format")
    print("  3. Apply preprocessing & resampling")
    print("  4. Generate augmented versions (if applicable)")
    print("  5. Create train/val/test splits")
    print("  6. Generate windowed indices")
    print("  7. Balance datasets")
    print("\nThis may take several minutes depending on data size.")

    confirm = input("\nContinue? [y/N]: ").strip().lower()

    if confirm != 'y':
        print("Pipeline cancelled.")
        return

    try:
        loader = UniversalEEGLoader(config_path="config.yaml", mode="train")

        # Check if method exists (for backward compatibility)
        if hasattr(loader, 'run_pipeline_level_1'):
            loader.run_pipeline_level_1(augment=False)
        else:
            print("Error: Pipeline method not available in this loader.")
            print("Please use StandardEEGLoader or add pipeline methods.")
            return

        print("\nPipeline completed successfully!")

        print("\nGenerating summary...")
        for mode in ['train', 'val', 'test']:
            try:
                dataset = UniversalEEGLoader(config_path="config.yaml", mode=mode)
                print(f"\n{mode.upper()}: {len(dataset)} windows")
            except:
                continue

    except Exception as e:
        print(f"\nPipeline failed: {e}")
        import traceback
        traceback.print_exc()


def generate_bids_report():
    """Generate BIDS validation report"""
    print("\nGenerating BIDS Report...")

    try:
        loader = UniversalEEGLoader(config_path="config.yaml", mode="train")

        try:
            from mne_bids import make_report
            bids_root = loader.bids_root

            report = make_report(bids_root)

            # FIXED: Use output_dir instead of results_dir
            report_path = loader.output_dir / "bids_validation_report.txt"

            with open(report_path, 'w') as f:
                f.write(str(report))

            print(f"Report saved to: {report_path}")
            print("\nReport Summary:")
            print(report)

        except ImportError:
            print("mne-bids make_report not available")
            print("Generating basic BIDS structure report...")

            subjects = list(loader.bids_root.glob("sub-*"))
            print(f"\nBIDS Structure:")
            print(f"  Root: {loader.bids_root}")
            print(f"  Subjects: {len(subjects)}")

            for subj in subjects[:5]:
                eeg_files = list((subj / "eeg").glob("*.edf"))
                print(f"  {subj.name}: {len(eeg_files)} recordings")

            if len(subjects) > 5:
                print(f"  ... and {len(subjects) - 5} more subjects")

    except Exception as e:
        print(f"Report generation failed: {e}")
        import traceback
        traceback.print_exc()


def run_quick_verification():
    """Run quick dataset integrity check"""
    print("\nRunning Quick Verification...")
    print("=" * 50)

    try:
        # FIXED: Create loader first, then pass to quick_verify
        loader = UniversalEEGLoader(config_path="config.yaml", mode="train")

        if len(loader) == 0:
            print("Warning: No data loaded. Run pipeline first (Option 1).")
            return

        # Now call quick_verify with the loader object
        quick_verify(loader, num_samples=10)

        print("\nAll checks passed!")

    except Exception as e:
        print(f"\nVerification failed: {e}")
        import traceback
        traceback.print_exc()


def show_dataset_statistics():
    """Display dataset statistics"""
    print("\nDataset Statistics")
    print("=" * 50)

    for mode in ['train', 'val', 'test']:
        try:
            dataset = UniversalEEGLoader(config_path="config.yaml", mode=mode)
            summary = dataset.get_dataset_summary()

            print(f"\n{mode.upper()} SET:")
            print(f"  Total windows: {summary['total_windows']}")
            print(f"  Window duration: {summary['window_duration_sec']} sec")
            print(f"  Sampling rate: {summary['target_sfreq']} Hz")

            # FIXED: Check if label_mode exists
            if 'label_mode' in summary:
                print(f"  Label mode: {summary['label_mode']}")

            print(f"  Label distribution:")
            for label, count in summary['labels'].items():
                pct = (count / summary['total_windows'] * 100) if summary['total_windows'] > 0 else 0
                print(f"    {label}: {count} ({pct:.2f}%)")

            # FIXED: Check cache stats format
            if 'cache_stats' in summary:
                print(f"  Cache status:")
                cache_stats = summary['cache_stats']

                # Handle different field names
                if 'memory_usage_mb' in cache_stats:
                    print(f"    Memory usage: {cache_stats['memory_usage_mb'] / 1024:.2f} GB")
                elif 'memory_usage_gb' in cache_stats:
                    print(f"    Memory usage: {cache_stats['memory_usage_gb']:.2f} GB")

                if 'memory_items' in cache_stats:
                    print(f"    Cached items: {cache_stats['memory_items']} items")
                elif 'cached_items' in cache_stats:
                    print(f"    Cached items: {cache_stats['cached_items']} items")

        except Exception as e:
            print(f"\n{mode.upper()} SET: Error loading - {e}")
            continue

    print("\n" + "=" * 50)


def export_metadata_report():
    """Export comprehensive metadata report"""
    print("\nExporting Metadata Report...")

    try:
        import json
        from datetime import datetime

        loader = UniversalEEGLoader(config_path="config.yaml", mode="train")

        report = {
            'generated_at': datetime.now().isoformat(),
            'configuration': loader.cfg,
            'datasets': {}
        }

        for mode in ['train', 'val', 'test']:
            try:
                dataset = UniversalEEGLoader(config_path="config.yaml", mode=mode)
                report['datasets'][mode] = dataset.get_dataset_summary()
            except:
                continue

        # FIXED: Use output_dir
        report_path = loader.output_dir / "metadata_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"Report exported to: {report_path}")

        print("\nSummary:")
        for mode, data in report['datasets'].items():
            print(f"  {mode.upper()}: {data['total_windows']} windows")

    except Exception as e:
        print(f"Export failed: {e}")
        import traceback
        traceback.print_exc()


def clear_cache():
    """Clear all cache levels"""
    print("\nClearing Cache...")

    try:
        loader = UniversalEEGLoader(config_path="config.yaml", mode="train")

        print("Warning: This will clear all cached data.")
        print("You will need to re-run the pipeline to regenerate tensors.")

        confirm = input("\nContinue? [y/N]: ").strip().lower()

        if confirm != 'y':
            print("Cache clear cancelled.")
            return

        # FIXED: Handle different cache implementations
        if hasattr(loader, 'cacher'):
            # New refactored loader with cacher
            loader.cacher.clear()
            print("All caches cleared successfully!")
        elif hasattr(loader, 'clear_cache'):
            # Loader with clear_cache method
            loader.clear_cache()
            print("All caches cleared successfully!")
        elif hasattr(loader, 'cache_manager'):
            # Old loader with cache_manager
            loader.cache_manager.clear_all()
            print("All caches cleared successfully!")
        else:
            print("No cache to clear (using StandardEEGLoader)")

        # FIXED: Handle tensor directory
        import shutil
        tensor_dir = loader.bids_root / "derivatives" / "tensors"

        if tensor_dir.exists():
            confirm_tensors = input("\nAlso delete pre-computed tensors? [y/N]: ").strip().lower()
            if confirm_tensors == 'y':
                shutil.rmtree(tensor_dir)
                tensor_dir.mkdir(parents=True, exist_ok=True)
                print("Tensors deleted.")

    except Exception as e:
        print(f"Cache clear failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main program loop"""
    while True:
        clear_screen()
        show_menu()

        try:
            choice = input("\nSelect option: ").strip()

            if choice == '0':
                print("\nExiting...")
                sys.exit(0)

            elif choice == '1':
                run_full_pipeline()

            elif choice == '2':
                generate_bids_report()

            elif choice == '3':
                run_quick_verification()

            elif choice == '4':
                show_dataset_statistics()

            elif choice == '5':
                export_metadata_report()

            elif choice == '6':
                clear_cache()

            else:
                print("\nInvalid option. Please try again.")

            input("\nPress Enter to continue...")

        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Exiting...")
            sys.exit(0)

        except Exception as e:
            print(f"\nUnexpected error: {e}")
            import traceback
            traceback.print_exc()
            input("\nPress Enter to continue...")


if __name__ == "__main__":
    main()
