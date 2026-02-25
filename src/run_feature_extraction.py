"""
Main script to run feature extraction
"""
import argparse
from pathlib import Path
import yaml
from features.extractor import FeatureExtractor


def main():
    parser = argparse.ArgumentParser(
        description='Extract features from EEG windows'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='fe_config.yaml',
        help='Feature extraction config file'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='all',
        choices=['train', 'val', 'test', 'all'],
        help='Which split to process'
    )

    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        return

    with open(config_path) as f:
        config = yaml.safe_load(f)

    print("=" * 70)
    print("EEG FEATURE EXTRACTION PIPELINE")
    print("=" * 70)
    print(f"\nConfig: {config_path}")
    print(f"Features enabled:")
    print(f"  - Time domain: {config.get('features', {}).get('time_domain', {}).get('enable', True)}")
    print(f"  - Frequency domain: {config.get('features', {}).get('frequency_domain', {}).get('enable', True)}")
    print(f"  - Nonlinear: {config.get('features', {}).get('nonlinear', {}).get('enable', True)}")
    print(f"  - Connectivity: {config.get('features', {}).get('connectivity', {}).get('enable', True)}")

    # Initialize extractor
    extractor = FeatureExtractor(config)

    # Define paths
    dataloader_dir = Path(config.get('paths', {}).get('dataloader_dir', 'results/dataloader'))
    features_dir = Path(config.get('paths', {}).get('features_dir', 'results/features'))

    # Determine splits to process
    splits = ['train', 'val', 'test'] if args.split == 'all' else [args.split]

    # Process each split
    for split in splits:
        index_csv = dataloader_dir / f'window_index_{split}.csv'
        output_csv = features_dir / f'features_{split}.csv'

        if not index_csv.exists():
            print(f"\n  Skipping {split}: {index_csv} not found")
            continue

        print(f"\n{'=' * 70}")
        print(f"PROCESSING: {split.upper()}")
        print('=' * 70)

        extractor.process_window_index(index_csv, output_csv)

    print("\n" + "=" * 70)
    print("✓ FEATURE EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"\nOutput directory: {features_dir}")
    print("Files created:")
    for split in splits:
        output_csv = features_dir / f'features_{split}.csv'
        if output_csv.exists():
            print(f"  ✓ {output_csv}")


if __name__ == '__main__':
    main()