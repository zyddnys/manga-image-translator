#!/usr/bin/env python3
"""
Rename translated manga files to sequential page numbers.
Format: page-001.jpg, page-002.jpg, etc.
"""

import os
import re
from pathlib import Path

# Configuration
TRANSLATED_DIR = "data/translated"
BACKUP_LIST = "rename_backup_list.txt"
DRY_RUN = True  # Set to False to actually rename files

def extract_page_number(filename):
    """
    Extract page number from filename like:
    'Solo Leveling - c001 (v01) - p019 [dig] [Yen Press] [LuCaZ].jpg'
    Returns tuple: (chapter, page_number)
    """
    # Match pattern: c001 (v01) - p019
    match = re.search(r'c(\d+).*?p(\d+)', filename)
    if match:
        chapter = int(match.group(1))
        page = int(match.group(2))
        return (chapter, page)
    return None

def get_file_list(directory):
    """Get all image files and sort them by chapter and page number."""
    files = []
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            filepath = os.path.join(directory, filename)
            page_info = extract_page_number(filename)
            if page_info:
                files.append({
                    'original': filename,
                    'path': filepath,
                    'chapter': page_info[0],
                    'page': page_info[1]
                })
    
    # Sort by chapter, then by page number
    files.sort(key=lambda x: (x['chapter'], x['page']))
    return files

def generate_new_names(files):
    """Generate sequential page numbers for all files."""
    renamed_files = []
    for idx, file_info in enumerate(files, start=1):
        ext = os.path.splitext(file_info['original'])[1]
        new_name = f"page-{idx:03d}{ext}"
        renamed_files.append({
            **file_info,
            'new_name': new_name,
            'new_path': os.path.join(TRANSLATED_DIR, new_name)
        })
    return renamed_files

def create_backup_list(renamed_files):
    """Create a backup list for reverting if needed."""
    with open(BACKUP_LIST, 'w', encoding='utf-8') as f:
        f.write("# Backup list for reverting renames\n")
        f.write("# Format: new_name -> original_name\n\n")
        for file_info in renamed_files:
            f.write(f"{file_info['new_name']} -> {file_info['original']}\n")
    print(f"✓ Backup list saved to: {BACKUP_LIST}")

def rename_files(renamed_files, dry_run=True):
    """Rename the files."""
    if dry_run:
        print("\n" + "="*70)
        print("DRY RUN MODE - No files will be renamed")
        print("="*70 + "\n")
    else:
        print("\n" + "="*70)
        print("RENAMING FILES...")
        print("="*70 + "\n")
    
    for file_info in renamed_files:
        old_path = file_info['path']
        new_path = file_info['new_path']
        
        print(f"Chapter {file_info['chapter']:02d}, Page {file_info['page']:03d}:")
        print(f"  FROM: {file_info['original']}")
        print(f"  TO:   {file_info['new_name']}")
        
        if not dry_run:
            try:
                os.rename(old_path, new_path)
                print("  ✓ Renamed")
            except Exception as e:
                print(f"  ✗ Error: {e}")
        print()

def main():
    # Check if directory exists
    if not os.path.exists(TRANSLATED_DIR):
        print(f"Error: Directory '{TRANSLATED_DIR}' not found!")
        return
    
    print(f"Scanning directory: {TRANSLATED_DIR}\n")
    
    # Get and sort files
    files = get_file_list(TRANSLATED_DIR)
    
    if not files:
        print("No manga files found!")
        return
    
    print(f"Found {len(files)} files to rename\n")
    
    # Generate new names
    renamed_files = generate_new_names(files)
    
    # Show preview of first 20 and last 5
    print("Preview of renaming (first 20 files):")
    print("-" * 70)
    for file_info in renamed_files[:20]:
        print(f"  {file_info['original'][:50]:50s} -> {file_info['new_name']}")
    
    if len(renamed_files) > 20:
        print(f"\n  ... ({len(renamed_files) - 25} more files) ...\n")
        print("Last 5 files:")
        for file_info in renamed_files[-5:]:
            print(f"  {file_info['original'][:50]:50s} -> {file_info['new_name']}")
    
    print()
    
    # Create backup list
    create_backup_list(renamed_files)
    
    # Rename files
    rename_files(renamed_files, dry_run=DRY_RUN)
    
    if DRY_RUN:
        print("\n" + "="*70)
        print("TO ACTUALLY RENAME FILES:")
        print("  1. Review the preview above")
        print("  2. Open this script and change: DRY_RUN = False")
        print("  3. Run the script again")
        print("="*70 + "\n")
    else:
        print("\n" + "="*70)
        print("✓ All files renamed successfully!")
        print(f"✓ Backup list saved to: {BACKUP_LIST}")
        print("="*70 + "\n")

if __name__ == "__main__":
    main()

