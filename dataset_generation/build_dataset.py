import pandas as pd
import re
import os
from datasets import load_dataset
from damage_generator import generate_synthetic_pair

# --- CONFIGURATION ---
TARGET_SENTENCE_COUNT = 500000  # Reduce to 10000 for testing, 500000 for production
OUTPUT_FILE = "datasets/synthetic_training_data.csv"

print("Connecting to Hugging Face to stream Wikipedia text...")

try:
    # We use wikitext which is standard, high-quality Wikipedia articles
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
except Exception as e:
    print(f"ERROR: Failed to load dataset from Hugging Face: {e}")
    print("Please check your internet connection and try again.")
    exit(1)

def clean_and_split_text(raw_text):
    """Removes weird Wikipedia formatting and splits into sentences."""
    # Remove titles and empty lines
    if raw_text.startswith(" = ") or len(raw_text.strip()) < 10:
        return []
    
    # Split paragraph into sentences by looking for periods followed by a space
    sentences = re.split(r'(?<=[.!?]) +', raw_text.strip())
    
    valid_sentences = []
    for s in sentences:
        words = s.split()
        # Only keep sentences that are a reasonable length (5 to 30 words)
        if 5 <= len(words) <= 30 and not re.search(r'[@#^&*<>\\/|]', s):
            valid_sentences.append(s)
            
    return valid_sentences

data_rows = []
print(f"Mining and corrupting {TARGET_SENTENCE_COUNT} sentences. Please wait...")

# Stream through the dataset
for item in dataset:
    sentences = clean_and_split_text(item['text'])
    
    for clean_sentence in sentences:
        # 1. Run our damage math
        clean, damaged = generate_synthetic_pair(clean_sentence)
        
        # 2. Save the pair
        data_rows.append({
            "input": damaged,
            "target": clean
        })
        
        if len(data_rows) % 2000 == 0:
            print(f"Processed {len(data_rows)} / {TARGET_SENTENCE_COUNT} sentences...")
            
        if len(data_rows) >= TARGET_SENTENCE_COUNT:
            break
            
    if len(data_rows) >= TARGET_SENTENCE_COUNT:
        break

# Validate we actually got data
if len(data_rows) == 0:
    print("ERROR: No valid sentences were extracted. Check your dataset or filtering rules.")
    exit(1)

# 3. Ensure output directory exists
output_dir = os.path.dirname(OUTPUT_FILE)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created output directory: {output_dir}")

# 4. Save to a Pandas DataFrame and export to CSV
print("\nCreating Pandas DataFrame...")
df = pd.DataFrame(data_rows)

try:
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    print(f"\nSuccess! Saved {len(df)} training pairs to {OUTPUT_FILE}.")
except Exception as e:
    print(f"ERROR: Failed to save CSV file: {e}")
    exit(1)

print("Here is a sneak peek at your new dataset:")
print(df.head())