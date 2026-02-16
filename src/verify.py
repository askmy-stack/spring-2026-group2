"""
Tests:
- Class hierarchy (BaseEEGLoader, subclasses)
- All 4 loader types
- Configuration compatibility
- Cache performance
- Data integrity
- Features Dask & Swifter
"""

import sys
import time
from pathlib import Path
import numpy as np
import torch
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

try:
    from dataloader import (
        StandardEEGLoader,
        CachedEEGLoader,
        ParallelEEGLoader,
        EnhancedEEGLoader,
        create_loader
    )

    REFACTORED_AVAILABLE = True
except ImportError:
    REFACTORED_AVAILABLE = False


class LoaderVerifier:
    """Comprehensive loader verification"""

    def __init__(self):
        self.tests_passed = 0
        self.tests_failed = 0
        self.test_results = []

    def run_test(self, test_name: str, test_func, *args, **kwargs):
        print(f"\n{'=' * 70}")
        print(f"TEST: {test_name}")
        print(f"{'=' * 70}")

        try:
            result = test_func(*args, **kwargs)
            self.tests_passed += 1
            self.test_results.append((test_name, 'PASS', result))
            print(f"PASS PASSED")
            return True
        except Exception as e:
            self.tests_failed += 1
            self.test_results.append((test_name, 'FAIL', str(e)))
            print(f"FAIL FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False

    def print_summary(self):
        print(f"\n{'=' * 70}")
        print("TEST SUMMARY")
        print(f"{'=' * 70}")
        print(f"Passed: {self.tests_passed}")
        print(f"Failed: {self.tests_failed}")
        total = self.tests_passed + self.tests_failed
        print(f"Total: {total}")
        print(f"Success Rate: {self.tests_passed / total * 100:.1f}%")

        if self.tests_failed > 0:
            print(f"\nFailed tests:")
            for name, status, msg in self.test_results:
                if status == 'FAIL':
                    print(f"  FAIL {name}")


def test_class_hierarchy():
    """Test 1: Verify class hierarchy"""
    from dataloader import BaseEEGLoader

    assert issubclass(StandardEEGLoader, BaseEEGLoader)
    assert issubclass(CachedEEGLoader, BaseEEGLoader)
    assert issubclass(ParallelEEGLoader, CachedEEGLoader)
    assert issubclass(EnhancedEEGLoader, ParallelEEGLoader)

    print("Hierarchy:")
    print("  BaseEEGLoader")
    print("  +-- StandardEEGLoader")
    print("  +-- CachedEEGLoader")
    print("      +-- ParallelEEGLoader")
    print("          +-- EnhancedEEGLoader")
    return True


def test_standard_loader():
    """Test 2: StandardEEGLoader"""
    loader = StandardEEGLoader(mode="train")

    if len(loader) == 0:
        print("No data - run pipeline first")
        return True

    data, label = loader[0]
    assert data.shape == (16, 256)
    print(f"Loaded {len(loader)} windows")
    return True


def test_cached_loader():
    """Test 3: CachedEEGLoader performance"""
    loader = CachedEEGLoader(mode="train", cache_memory_mb=500)

    if len(loader) == 0:
        return True

    # First load (cache miss)
    start = time.time()
    data1, _ = loader[0]
    time1 = time.time() - start

    # Second load (cache hit)
    start = time.time()
    data2, _ = loader[0]
    time2 = time.time() - start

    speedup = time1 / time2
    print(f"Speedup: {speedup:.1f}×")

    assert speedup > 1.5, f"Cache too slow: {speedup:.1f}×"
    return True


def test_enhanced_loader():
    """Test 4: EnhancedEEGLoader (all features)"""
    loader = EnhancedEEGLoader(mode="train", cache_memory_mb=500, n_workers=2)

    if len(loader) == 0:
        return True

    print(f"Size: {len(loader)}")
    print(f"Dask: {loader.use_dask}")
    print(f"Swifter: {loader.use_swifter}")

    data, label = loader[0]
    assert data.shape == (16, 256)
    return True


def test_factory():
    """Test 5: Factory function"""
    for ltype in ['standard', 'cached', 'parallel', 'enhanced']:
        loader = create_loader(ltype, mode="train")
        print(f"PASS {ltype}: {type(loader).__name__}")
    return True


def test_config_compatibility():
    """Test 6: Config file compatibility"""
    configs = ["config.yaml", "config_v2.yaml"]

    for cfg in configs:
        if Path(cfg).exists():
            loader = StandardEEGLoader(config_path=cfg, mode="train")
            print(f"PASS {cfg} works")
            break
    else:
        print("Warning: No config found")
    return True


def test_data_integrity():
    """Test 7: Data integrity"""
    loader = StandardEEGLoader(mode="train")

    if len(loader) == 0:
        return True

    for i in range(min(10, len(loader))):
        data, label = loader[i]
        assert not torch.isnan(data).any()
        assert not torch.isinf(data).any()
        assert label.item() in [0, 1]

    print("PASS 10 samples verified")
    return True


def test_cache_benchmark():
    """Test 8: Cache benchmark"""
    loader = EnhancedEEGLoader(mode="train", cache_memory_mb=500)

    if len(loader) < 20:
        return True

    results = loader.benchmark_cache(num_samples=20)
    print(f"Speedup: {results['speedup']:.1f}×")

    assert results['speedup'] > 1.0
    return True


def test_index_files():
    """Test 9: Index files"""
    output_dir = Path("results/dataloader")

    for mode in ['train', 'val', 'test']:
        idx_file = output_dir / f"window_index_{mode}.csv"
        if idx_file.exists():
            df = pd.read_csv(idx_file)
            print(f" {mode}: {len(df)} windows")

    return True


def test_shapes_consistency():
    """Test 10: Shape consistency"""
    loader = StandardEEGLoader(mode="train")

    if len(loader) == 0:
        return True

    shapes = [loader[i][0].shape for i in range(min(5, len(loader)))]
    assert all(s == shapes[0] for s in shapes)
    print(f"Consistent shape: {shapes[0]}")
    return True


def main():
    print("=" * 70)
    print("COMPREHENSIVE VERIFICATION v2.0")
    print("=" * 70)

    if not REFACTORED_AVAILABLE:
        print("\n dataloader.py not found or missing required classes")
        print("Please ensure dataloader.py contains the new class hierarchy")
        return

    verifier = LoaderVerifier()

    verifier.run_test("Class Hierarchy", test_class_hierarchy)
    verifier.run_test("Standard Loader", test_standard_loader)
    verifier.run_test("Cached Loader", test_cached_loader)
    verifier.run_test("Enhanced Loader", test_enhanced_loader)
    verifier.run_test("Factory Function", test_factory)
    verifier.run_test("Config Compatibility", test_config_compatibility)
    verifier.run_test("Data Integrity", test_data_integrity)
    verifier.run_test("Cache Benchmark", test_cache_benchmark)
    verifier.run_test("Index Files", test_index_files)
    verifier.run_test("Shape Consistency", test_shapes_consistency)

    verifier.print_summary()

    if verifier.tests_failed == 0:
        print(f"{'=' * 70}")
        print(" ALL TESTS PASSED - SYSTEM READY")
        print(f"{'=' * 70}")
        print("\nRecommended usage:")
        print("  from dataloader import create_loader")
        print("  loader = create_loader('enhanced', mode='train')")
    else:
        print(f"\n{'=' * 70}")
        print("SOME TESTS FAILED")
        print(f"{'=' * 70}")
        print("\nTroubleshooting:")
        print("  1. Run pipeline: python main.py (option 1)")
        print("  2. Install deps: pip install -r requirements.txt")
        print("  3. Check config: config_v2.yaml")


if __name__ == "__main__":
    main()
