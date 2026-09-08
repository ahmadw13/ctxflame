"""
Cross-platform test runner for ctxflame.
Discovers and executes all unit tests using the standard library unittest runner.
"""

import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
TESTS_DIR = ROOT_DIR / "tests"


def main() -> int:
    print("[INFO] Discovering and running ctxflame test suite...")
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=str(TESTS_DIR), pattern="test_*.py")

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    if result.wasSuccessful():
        print("\n[SUCCESS] All ctxflame unit tests passed cleanly.")
        return 0
    else:
        print(f"\n[FAILURE] Test suite failed with {len(result.failures)} failure(s) and {len(result.errors)} error(s).")
        return 1


if __name__ == "__main__":
    sys.exit(main())
