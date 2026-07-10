#!/usr/bin/env python3

import argparse
import random
import struct
import time


def main():
    parser = argparse.ArgumentParser(
        description="Generate a binary float32 vector dataset for ApexVec."
    )

    parser.add_argument(
        "-o",
        "--output",
        default="vectors.bin",
        help="Output binary file (default: vectors.bin)",
    )

    parser.add_argument(
        "-n",
        "--num-vectors",
        type=int,
        default=10000,
        help="Number of vectors (default: 10000)",
    )

    parser.add_argument(
        "-d",
        "--dimensions",
        type=int,
        default=1024,
        help="Vector dimensions (default: 1024)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    args = parser.parse_args()

    random.seed(args.seed)

    total_bytes = (
        args.num_vectors *
        args.dimensions *
        4
    )

    print("Generating dataset...")
    print(f"Output File : {args.output}")
    print(f"Vectors     : {args.num_vectors:,}")
    print(f"Dimensions  : {args.dimensions}")
    print(f"Dataset Size: {total_bytes / (1024 * 1024):.2f} MB")

    start = time.perf_counter()

    with open(args.output, "wb") as f:
        for _ in range(args.num_vectors):
            vector = [
                random.random()
                for _ in range(args.dimensions)
            ]
            f.write(
                struct.pack(
                    f"{args.dimensions}f",
                    *vector,
                )
            )

    elapsed = time.perf_counter() - start

    print("\nDone!")
    print(f"Saved to: {args.output}")
    print(f"Generation Time: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()