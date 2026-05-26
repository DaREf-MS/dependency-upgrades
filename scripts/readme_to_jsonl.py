import json
import os
from pathlib import Path

# --- Configuration ---
ROOT_PATH = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_PATH / 'data'
TOPICGPT_PATH = DATA_PATH / 'input'
INPUT_DIR = DATA_PATH / "downloaded_readmes"
OUTPUT_JSONL = TOPICGPT_PATH / "readmes_structured.jsonl"

def transform_readmes_to_jsonl(subset_limit=None):
    """
    Reads markdown files from INPUT_DIR and saves them as a JSONL file.
    :param subset_limit: Integer to limit the number of files processed.
    """
    # Get all markdown files in the directory
    all_files = list(INPUT_DIR.glob("*.md"))
    
    # Apply subset selection if specified
    if subset_limit:
        all_files = all_files[:subset_limit]
        print(f"🔍 Processing a subset of {subset_limit} files...")
    else:
        print(f"🚀 Processing all {len(all_files)} files...")

    processed_count = 0

    # Open the file in write mode
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f_out:
        for file_path in all_files:
            try:
                # 1. Extract the repo name (ID)
                # Reversing the &SEP& transformation and removing suffix
                repo_id = file_path.name.replace("_README.md", "").replace("&SEP&", "/")
                
                # 2. Read the markdown content
                with open(file_path, "r", encoding="utf-8") as f_in:
                    content = f_in.read()
                
                # 3. Create the dictionary structure
                record = {
                    "id": repo_id,
                    "text": content
                }
                
                # 4. Write as a single JSON line
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                processed_count += 1
                
            except Exception as e:
                print(f"⚠️ Error processing {file_path.name}: {e}")

    print(f"✅ Successfully created: {OUTPUT_JSONL}")
    print(f"📊 Total records written: {processed_count}")

# --- Execution ---
# Set to None to process everything, or a number (e.g., 10) for a subset
transform_readmes_to_jsonl(subset_limit=10)