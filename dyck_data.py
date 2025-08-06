#!/usr/bin/env python3
"""
Generate a data/dyck_data_{n}.pkl file
"""

import sys
import os
from src.data.dyck_generator import generate_dyck_data
import argparse

def main(n):
    print(f"Generating Dyck data for n={n}...")
    data_path = f"data/dyck_data_{n}.pkl"
    try:
        generate_dyck_data(n, data_path=data_path, force_regenerate=False)
        print(f"Generated: {data_path}")
    except Exception as e:
        print(f"Error generating n={n}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Dyck data files.")
    parser.add_argument("--n", type=int, help="The value of n for Dyck data generation.")
    args = parser.parse_args()
    main(args.n)