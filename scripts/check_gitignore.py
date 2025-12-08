#!/usr/bin/env python3
"""
Check if sensitive files are properly ignored
Run: python scripts/check_gitignore.py
"""

import subprocess
from pathlib import Path

def check_tracked_files():
    """Check for files that shouldn't be tracked"""
    
    print(" Checking for improperly tracked files...")
    print("=" * 70)
    
    # Patterns that should NOT be tracked
    bad_patterns = [
        '*.pth', '*.pt', '*.ckpt',  # Model files
        '*.log',                     # Logs
        'venv/', 'env/',            # Virtual environments
        'data/', 'dataset/',        # Datasets
        '__pycache__/',             # Python cache
        '.env',                     # Environment files
    ]
    
    issues_found = False
    
    for pattern in bad_patterns:
        try:
            # Check if any files matching pattern are tracked
            result = subprocess.run(
                ['git', 'ls-files', pattern],
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.stdout.strip():
                issues_found = True
                print(f" Found tracked files matching: {pattern}")
                print(f"   Files: {result.stdout.strip()}")
                print()
        except:
            pass
    
    if not issues_found:
        print(" No improperly tracked files found!")
    else:
        print("\n  Some files should not be tracked!")
        print("\nTo fix:")
        print("  git rm --cached <file>")
        print("  git commit -m 'Remove tracked files'")
    
    print("=" * 70)


def check_untracked_important():
    """Check if important files are untracked"""
    
    print("\n Checking important files...")
    print("=" * 70)
    
    important_files = [
        'README.md',
        'requirements.txt',
        'main.py',
        '.gitignore',
        'setup.py',
        'LICENSE',
    ]
    
    for file in important_files:
        if Path(file).exists():
            try:
                result = subprocess.run(
                    ['git', 'ls-files', file],
                    capture_output=True,
                    text=True,
                    check=False
                )
                
                if result.stdout.strip():
                    print(f" {file}")
                else:
                    print(f"  {file} (exists but not tracked)")
            except:
                print(f"? {file} (git not available)")
        else:
            print(f" {file} (missing)")
    
    print("=" * 70)


def main():
    print("\nBlueDepth-Crescent .gitignore Verification")
    print("=" * 70)
    
    # Check if .gitignore exists
    if not Path('.gitignore').exists():
        print(" .gitignore not found!")
        return 1
    
    check_tracked_files()
    check_untracked_important()
    
    print("\n Verification complete!")
    return 0


if __name__ == "__main__":
    exit(main())
