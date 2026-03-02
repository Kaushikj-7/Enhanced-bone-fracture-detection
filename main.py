#!/usr/bin/env python
import sys
import os

# Append project root to sys.path so we can import src/models
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from src.run_full_pipeline import main
except ImportError as e:
    print(f"Error importing pipeline: {e}")
    sys.exit(1)

if __name__ == "__main__":
    print("================================================================")
    print("Local Training & Evaluation Pipeline (RTX Optimized)")
    print("================================================================")
    # Default arguments can be passed via command line, run_full_pipeline uses argparse
    main()
