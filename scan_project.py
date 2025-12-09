#!/usr/bin/env python3
"""
BlueDepth-Crescent Project Structure Scanner
Scans and analyzes the entire project structure
Intelligently skips large media folders while tracking them
"""

import os
import sys
from pathlib import Path
from collections import defaultdict

def scan_project(root_path='.', show_tree=True, show_stats=True):
    """Main scanning function"""

    print("=" * 80)
    print("BlueDepth-Crescent Project Structure Scanner")
    print("=" * 80)
    print(f"\nScanning from: {Path(root_path).resolve()}")
    print()

    # Directories to ignore
    ignore_dirs = {
        '__pycache__', '.git', 'venv', '.venv', 'env',
        'node_modules', '.pytest_cache', '.mypy_cache',
        'dist', 'build', '.egg-info', '.idea', '.vscode'
    }

    # Media extensions to track but not enumerate deeply
    media_extensions = {'.jpg', '.jpeg', '.png', '.mp4', '.avi', '.mov', '.gif', '.bmp'}

    # Statistics
    stats = {
        'total_files': 0,
        'total_dirs': 0,
        'by_extension': defaultdict(int),
        'python_modules': [],
        'key_files': {},
        'data_dirs': [],
        'media_folders': [],  # Track folders with many media files
        'skipped_media_count': 0
    }

    # Key file categories
    key_categories = {
        'Entry Points': ['main.py', 'setup.py', 'app.py'],
        'Configuration': ['config.py', 'requirements.txt', '.env', '.gitignore'],
        'Training': ['train_unet.py', 'trainer.py', 'train.py'],
        'Models': ['base_model.py', 'unet_standard.py', 'unet_light.py', 'unet_attention.py'],
        'Data': ['dataset.py', 'dataloader.py', 'organize_data.py'],
        'Inference': ['enhancer.py', 'batch_processor.py', 'video_processor.py'],
        'Documentation': ['README.md', 'ARCHITECTURE.md', 'TRAINING_GUIDE.md']
    }

    def should_ignore(path):
        return any(ignore in str(path) for ignore in ignore_dirs)

    def is_media_heavy_folder(directory, sample_size=20):
        """Check if folder contains many media files"""
        try:
            items = list(os.listdir(directory))[:sample_size]
            media_count = sum(1 for item in items 
                            if Path(item).suffix.lower() in media_extensions)
            # If more than 50% of sampled files are media
            return media_count > len(items) * 0.5 and len(items) > 10
        except:
            return False

    def count_media_in_folder(directory):
        """Count media files in a folder without deep recursion"""
        try:
            all_items = os.listdir(directory)
            media_count = sum(1 for item in all_items 
                            if Path(item).suffix.lower() in media_extensions)
            return media_count, len(all_items)
        except:
            return 0, 0

    def scan_dir(directory, depth=0, max_depth=10):
        """Recursively scan directory"""
        if depth > max_depth or should_ignore(directory):
            return None

        tree = {}

        try:
            items = sorted(os.listdir(directory))
        except PermissionError:
            return None

        # Check if this is a media-heavy folder
        if is_media_heavy_folder(directory):
            media_count, total_count = count_media_in_folder(directory)
            stats['media_folders'].append({
                'path': str(directory),
                'media_files': media_count,
                'total_files': total_count
            })
            stats['skipped_media_count'] += media_count

            # Still count these files in statistics
            for item in items:
                item_path = Path(directory) / item
                if item_path.is_file():
                    stats['total_files'] += 1
                    ext = item_path.suffix.lower()
                    stats['by_extension'][ext] += 1

            # Return a marker instead of full tree
            return {'[MEDIA_FOLDER]': f'{media_count} media files'}

        for item in items:
            item_path = Path(directory) / item

            if should_ignore(item_path):
                continue

            if item_path.is_file():
                stats['total_files'] += 1
                ext = item_path.suffix.lower()
                stats['by_extension'][ext] += 1

                # Track key files
                for category, files in key_categories.items():
                    if item in files:
                        if category not in stats['key_files']:
                            stats['key_files'][category] = []
                        stats['key_files'][category].append(str(item_path))

                # Track Python modules
                if item == '__init__.py':
                    stats['python_modules'].append(str(directory))

                tree[item] = 'file'

            elif item_path.is_dir():
                stats['total_dirs'] += 1

                # Track data directories
                if item in ['data', 'train', 'test', 'val', 'frames', 'videos', 'checkpoints']:
                    stats['data_dirs'].append(str(item_path))

                subtree = scan_dir(item_path, depth + 1, max_depth)
                if subtree:
                    tree[item] = subtree

        return tree

    # Scan the project
    print("[SCANNING] Analyzing project structure...\n")
    project_tree = scan_dir(root_path)

    # Print tree structure
    if show_tree:
        print("=" * 80)
        print("PROJECT STRUCTURE")
        print("=" * 80)
        print()
        print_tree(project_tree, root_path)
        print()

    # Print statistics
    if show_stats:
        print("=" * 80)
        print("STATISTICS")
        print("=" * 80)
        print(f"\nTotal Directories: {stats['total_dirs']}")
        print(f"Total Files: {stats['total_files']}")

        # Media folder summary
        if stats['media_folders']:
            print(f"\n[MEDIA FOLDERS DETECTED]")
            print(f"  Intelligently skipped {len(stats['media_folders'])} media-heavy folders")
            print(f"  Total media files: {stats['skipped_media_count']}")
            print(f"\n  Details:")
            for mf in stats['media_folders'][:10]:
                rel_path = Path(mf['path']).relative_to(root_path) if Path(mf['path']).is_relative_to(Path(root_path).resolve()) else mf['path']
                print(f"    {str(rel_path):40} {mf['media_files']:5} media files ({mf['total_files']} total)")
            if len(stats['media_folders']) > 10:
                print(f"    ... and {len(stats['media_folders']) - 10} more")

        print(f"\n[FILES BY TYPE]")
        sorted_exts = sorted(stats['by_extension'].items(), key=lambda x: x[1], reverse=True)
        for ext, count in sorted_exts[:15]:
            ext_name = ext if ext else '(no extension)'
            print(f"  {ext_name:20} {count:4} files")

        print(f"\n[PYTHON MODULES] ({len(stats['python_modules'])} packages)")
        for module in sorted(stats['python_modules'])[:20]:
            rel_module = Path(module).relative_to(root_path) if Path(module).is_relative_to(Path(root_path).resolve()) else module
            print(f"  {rel_module}")
        if len(stats['python_modules']) > 20:
            print(f"  ... and {len(stats['python_modules']) - 20} more")

        print(f"\n[KEY FILES BY CATEGORY]")
        for category, files in stats['key_files'].items():
            print(f"\n  {category}:")
            for f in files:
                try:
                    rel_f = Path(f).relative_to(root_path) if Path(f).is_relative_to(Path(root_path).resolve()) else f
                    print(f"    - {rel_f}")
                except:
                    print(f"    - {f}")

        print(f"\n[DATA DIRECTORIES] ({len(stats['data_dirs'])})")
        for data_dir in sorted(stats['data_dirs'])[:10]:
            try:
                count = len(list(Path(data_dir).glob('*')))
                rel_dir = Path(data_dir).relative_to(root_path) if Path(data_dir).is_relative_to(Path(root_path).resolve()) else data_dir
                print(f"  {str(rel_dir):40} ({count} items)")
            except:
                print(f"  {data_dir}")

    print()
    print("=" * 80)
    print("SCAN COMPLETE")
    print("=" * 80)

def print_tree(tree, path, prefix="", is_last=True, max_files_shown=10):
    """Print directory tree"""
    if isinstance(tree, dict):
        items = list(tree.items())

        # Separate media folders, directories, and files
        media_folders = [(k, v) for k, v in items if isinstance(v, dict) and '[MEDIA_FOLDER]' in v]
        directories = [(k, v) for k, v in items if isinstance(v, dict) and '[MEDIA_FOLDER]' not in v]
        files = [(k, v) for k, v in items if v == 'file']

        # Show directories first
        for i, (name, subtree) in enumerate(directories):
            is_last_item = (i == len(directories) - 1) and len(media_folders) == 0 and len(files) == 0
            connector = " " if is_last_item else " "

            # Check if it's a Python package
            has_init = '__init__.py' in subtree
            marker = " [PACKAGE]" if has_init else ""

            # Count Python files
            py_count = sum(1 for k, v in subtree.items() if isinstance(k, str) and k.endswith('.py'))
            py_info = f" ({py_count} .py)" if py_count > 0 else ""

            print(f"{prefix}{connector}{name}/{marker}{py_info}")

            extension = "    " if is_last_item else "   "
            print_tree(subtree, f"{path}/{name}", prefix + extension, is_last_item, max_files_shown)

        # Show media folders with summary
        for i, (name, subtree) in enumerate(media_folders):
            is_last_item = (i == len(media_folders) - 1) and len(files) == 0
            connector = " " if is_last_item else " "

            media_info = subtree.get('[MEDIA_FOLDER]', 'media files')
            print(f"{prefix}{connector}{name}/ [MEDIA: {media_info}]")

        # Show important files (limit to avoid clutter)
        important_exts = {'.py', '.yaml', '.yml', '.txt', '.md', '.sh', '.bat', '.json'}
        important_files = [k for k, v in files if Path(k).suffix.lower() in important_exts]
        other_files = [k for k, v in files if Path(k).suffix.lower() not in important_exts]

        for i, name in enumerate(important_files[:max_files_shown]):
            is_last_file = (i == len(important_files) - 1) and len(other_files) == 0
            connector = " " if is_last_file else " "
            print(f"{prefix}{connector}{name}")

        if len(important_files) > max_files_shown:
            remaining = len(important_files) - max_files_shown
            connector = " " if len(other_files) == 0 else " "
            print(f"{prefix}{connector}[{remaining} more important files...]")

        if len(other_files) > 0:
            print(f"{prefix} [{len(other_files)} other files]")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Scan BlueDepth-Crescent project structure")
    parser.add_argument('--path', type=str, default='.', help='Root path to scan')
    parser.add_argument('--no-tree', action='store_true', help='Hide tree structure')
    parser.add_argument('--no-stats', action='store_true', help='Hide statistics')

    args = parser.parse_args()

    scan_project(
        root_path=args.path,
        show_tree=not args.no_tree,
        show_stats=not args.no_stats
    )
