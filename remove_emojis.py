"""
Remove all emojis from Python and Markdown files in the project
Simple and focused - just removes emojis, nothing else
"""

import re
from pathlib import Path
from typing import List
import logging


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# EMOJI REMOVAL
# ============================================================================
def remove_emojis(text: str) -> str:
    """
    Remove emoji characters from text
    
    Args:
        text: Input text with potential emojis
        
    Returns:
        Text with emojis removed
    """
    # Comprehensive emoji regex pattern
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002500-\U00002BEF"  # chinese char
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001F900-\U0001F9FF"  # supplemental symbols
        "\U0001FA00-\U0001FA6F"  # chess symbols
        "\U0001FA70-\U0001FAFF"  # symbols and pictographs extended-a
        "\u200d"                  # zero width joiner
        "\u2640-\u2642"          # gender signs
        "\u2600-\u26FF"          # miscellaneous symbols
        "\u2700-\u27BF"          # dingbats
        "]+",
        flags=re.UNICODE
    )
    
    return emoji_pattern.sub('', text)


def process_file(file_path: Path) -> bool:
    """
    Remove emojis from a single file
    
    Args:
        file_path: Path to file
        
    Returns:
        True if file was modified, False otherwise
    """
    try:
        # Read file
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Remove emojis
        cleaned_content = remove_emojis(content)
        
        # Check if file was modified
        if content != cleaned_content:
            # Write back
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_content)
            
            logger.info(f"Cleaned: {file_path.relative_to(Path.cwd())}")
            return True
        
        return False
        
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        return False


def process_directory(
    root_dir: str,
    extensions: List[str] = ['.py', '.md', '.txt', '.yaml', '.yml', '.sh', '.bat']
) -> dict:
    """
    Recursively process all files in directory
    
    Args:
        root_dir: Root directory path
        extensions: List of file extensions to process
        
    Returns:
        Dictionary with processing statistics
    """
    root_dir = Path(root_dir)
    
    if not root_dir.exists():
        logger.error(f"Directory not found: {root_dir}")
        return {'processed': 0, 'modified': 0, 'failed': 0}
    
    # Directories to skip
    skip_dirs = {'venv', 'env', 'ENV', '.venv', '__pycache__', '.git', 
                 '.idea', '.vscode', 'node_modules', 'build', 'dist'}
    
    processed = 0
    modified = 0
    failed = 0
    
    logger.info(f"Processing directory: {root_dir}")
    logger.info(f"File extensions: {extensions}\n")
    
    # Find all matching files
    for ext in extensions:
        files = root_dir.rglob(f"*{ext}")
        
        for file_path in files:
            # Skip virtual environment and cache directories
            if any(skip in file_path.parts for skip in skip_dirs):
                continue
            
            processed += 1
            
            try:
                if process_file(file_path):
                    modified += 1
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                failed += 1
    
    # Summary
    stats = {
        'processed': processed,
        'modified': modified,
        'failed': failed
    }
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Files processed:  {processed}")
    print(f"Files modified:   {modified}")
    print(f"Files failed:     {failed}")
    print("=" * 70)
    
    return stats


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Remove emojis from Python and Markdown files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process current directory (default: .py and .md files)
  python remove_emojis.py
  
  # Process specific directory
  python remove_emojis.py path/to/directory
  
  # Process only Python files
  python remove_emojis.py --extensions .py
  
  # Process multiple file types
  python remove_emojis.py --extensions .py .md .txt .yaml
        """
    )
    
    parser.add_argument(
        'directory',
        nargs='?',
        default='.',
        help='Root directory to process (default: current directory)'
    )
    
    parser.add_argument(
        '--extensions',
        nargs='+',
        default=['.py', '.md', '.txt', '.yaml', '.yml', '.sh', '.bat'],
        help='File extensions to process (default: .py .md .txt .yaml .yml .sh .bat)'
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("BlueDepth-Crescent: Emoji Remover")
    print("=" * 70)
    print(f"Directory:   {Path(args.directory).resolve()}")
    print(f"Extensions:  {args.extensions}")
    print("=" * 70 + "\n")
    
    # Confirm
    response = input("Remove emojis from files? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("Cancelled.")
        exit(0)
    
    print()
    
    # Process
    stats = process_directory(args.directory, args.extensions)
    
    # Done
    if stats['modified'] > 0:
        print(f"\n Done! Modified {stats['modified']} files.")
    else:
        print(f"\n Done! No emojis found.")
