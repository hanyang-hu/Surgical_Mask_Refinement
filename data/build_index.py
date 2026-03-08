"""Build metadata index for paired RGB images and masks.

This script scans the dataset directory and creates a JSON index file
containing paths to all paired samples (RGB, coarse_mask, refined_mask).

The index enables efficient dataset creation and splitting.
"""

from pathlib import Path
import json
import argparse
from typing import Dict, List, Tuple
from tqdm import tqdm


def scan_source_directory(
    source_dir: Path,
    source_name: str
) -> Tuple[List[Dict[str, str]], List[str]]:
    """Scan a single source directory (real_world or synthetic) for paired samples.
    
    Args:
        source_dir: Path to source directory (contains RGB/, coarse_mask/, refined_mask/)
        source_name: Name of the source ("real_world" or "synthetic")
        
    Returns:
        Tuple of (valid_samples, incomplete_samples)
        - valid_samples: List of sample dictionaries with complete triplets
        - incomplete_samples: List of filenames with missing files
    """
    rgb_dir = source_dir / "RGB"
    coarse_dir = source_dir / "coarse_mask"
    refined_dir = source_dir / "refined_mask"
    
    # Check that all subdirectories exist
    if not rgb_dir.exists():
        print(f"Warning: {rgb_dir} does not exist")
        return [], []
    if not coarse_dir.exists():
        print(f"Warning: {coarse_dir} does not exist")
        return [], []
    if not refined_dir.exists():
        print(f"Warning: {refined_dir} does not exist")
        return [], []
    
    # Get all RGB files as the reference
    rgb_files = sorted(rgb_dir.glob("*.png"))
    
    valid_samples = []
    incomplete_samples = []
    
    for rgb_file in rgb_files:
        file_stem = rgb_file.stem
        
        # Check for corresponding mask files
        coarse_file = coarse_dir / f"{file_stem}.png"
        refined_file = refined_dir / f"{file_stem}.png"
        
        if coarse_file.exists() and refined_file.exists():
            # Valid triplet found
            sample = {
                "id": f"{source_name}_{file_stem}",
                "file_stem": file_stem,
                "source": source_name,
                "rgb_path": str(rgb_file),
                "coarse_mask_path": str(coarse_file),
                "refined_mask_path": str(refined_file),
            }
            valid_samples.append(sample)
        else:
            # Incomplete triplet
            missing = []
            if not coarse_file.exists():
                missing.append("coarse_mask")
            if not refined_file.exists():
                missing.append("refined_mask")
            incomplete_samples.append(f"{source_name}/{file_stem} (missing: {', '.join(missing)})")
    
    return valid_samples, incomplete_samples


def scan_dataset_directory(
    data_root: Path,
    sources: List[str] = ["real_world", "synthetic"]
) -> Tuple[List[Dict[str, str]], Dict[str, int], List[str]]:
    """Scan dataset directory and collect paired samples from all sources.
    
    Args:
        data_root: Root directory containing real_world/ and synthetic/ folders
        sources: List of source names to scan
        
    Returns:
        Tuple of (all_samples, source_counts, all_incomplete)
        - all_samples: List of all valid sample dictionaries
        - source_counts: Dictionary mapping source name to count
        - all_incomplete: List of all incomplete sample identifiers
    """
    all_samples = []
    source_counts = {}
    all_incomplete = []
    
    for source in sources:
        source_dir = data_root / source
        
        if not source_dir.exists():
            print(f"Warning: Source directory {source_dir} does not exist, skipping")
            source_counts[source] = 0
            continue
        
        print(f"Scanning {source}...")
        valid_samples, incomplete_samples = scan_source_directory(source_dir, source)
        
        all_samples.extend(valid_samples)
        source_counts[source] = len(valid_samples)
        all_incomplete.extend(incomplete_samples)
    
    return all_samples, source_counts, all_incomplete


def build_index(
    data_root: str,
    output_dir: str,
    sources: List[str] = ["real_world", "synthetic"]
) -> None:
    """Build and save dataset index.
    
    Args:
        data_root: Root directory of dataset
        output_dir: Directory to save index file
        sources: List of source types to include
    """
    data_root_path = Path(data_root)
    output_dir_path = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_dir_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Building dataset index from: {data_root}")
    print(f"Sources: {', '.join(sources)}")
    print("=" * 60)
    
    # Scan all sources
    all_samples, source_counts, incomplete_samples = scan_dataset_directory(
        data_root_path, sources
    )
    
    # Save index
    output_path = output_dir_path / "all_samples.json"
    with open(output_path, 'w') as f:
        json.dump(all_samples, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("INDEX BUILD COMPLETE")
    print("=" * 60)
    print(f"Total valid samples: {len(all_samples)}")
    for source, count in source_counts.items():
        print(f"  - {source}: {count}")
    print(f"Incomplete samples: {len(incomplete_samples)}")
    if incomplete_samples and len(incomplete_samples) <= 10:
        for incomplete in incomplete_samples:
            print(f"  - {incomplete}")
    elif len(incomplete_samples) > 10:
        print(f"  (showing first 10 of {len(incomplete_samples)})")
        for incomplete in incomplete_samples[:10]:
            print(f"  - {incomplete}")
    print(f"\nIndex saved to: {output_path}")


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Build dataset index from RGB and mask triplets"
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help="Root directory containing real_world/ and synthetic/ folders"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/metadata",
        help="Output directory for index file (default: data/metadata)"
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["real_world", "synthetic"],
        help="Source types to include (default: real_world synthetic)"
    )
    
    args = parser.parse_args()
    
    build_index(
        data_root=args.dataset_root,
        output_dir=args.output_dir,
        sources=args.sources
    )


if __name__ == "__main__":
    main()
