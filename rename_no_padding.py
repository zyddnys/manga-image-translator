#!/usr/bin/env python3
"""
Rename page-001.jpg to page-1.jpg (remove zero padding)
"""

import os
import re

TRANSLATED_DIR = "data/translated"
DRY_RUN = False # Set to False to actually rename

def main():
    if not os.path.exists(TRANSLATED_DIR):
        print(f"Error: Directory '{TRANSLATED_DIR}' not found!")
        return
    
    files_to_rename = []
    
    # Find all page-XXX.jpg files
    for filename in os.listdir(TRANSLATED_DIR):
        match = re.match(r'page-(\d+)(\.jpg|\.jpeg|\.png)$', filename, re.IGNORECASE)
        if match:
            page_num = int(match.group(1))
            extension = match.group(2)
            new_name = f"page-{page_num}{extension}"
            
            if filename != new_name:
                files_to_rename.append({
                    'old': filename,
                    'new': new_name,
                    'old_path': os.path.join(TRANSLATED_DIR, filename),
                    'new_path': os.path.join(TRANSLATED_DIR, new_name)
                })
    
    if not files_to_rename:
        print("No files need renaming (already without padding or no page-XXX files found)")
        return
    
    print(f"Found {len(files_to_rename)} files to rename")
    print("\nPreview (first 20):")
    print("-" * 60)
    
    for item in files_to_rename[:20]:
        print(f"  {item['old']:20s} -> {item['new']}")
    
    if len(files_to_rename) > 20:
        print(f"\n  ... and {len(files_to_rename) - 20} more files")
    
    print()
    
    if DRY_RUN:
        print("="*60)
        print("DRY RUN MODE - No files will be renamed")
        print("To actually rename, set DRY_RUN = False in the script")
        print("="*60)
    else:
        print("="*60)
        print("RENAMING FILES...")
        print("="*60)
        
        for item in files_to_rename:
            try:
                os.rename(item['old_path'], item['new_path'])
                print(f"✓ {item['old']} -> {item['new']}")
            except Exception as e:
                print(f"✗ Error renaming {item['old']}: {e}")
        
        print("\n✓ Done!")

if __name__ == "__main__":
    main()

