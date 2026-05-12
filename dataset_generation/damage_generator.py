import random

# --- CORRUPTION PARAMETERS ---
# These control how aggressively we damage the text
CHAR_CORRUPTION_RATE = 0.15  # 15% chance to damage a character in a word
WORD_MASK_RATE = 0.20         # 20% chance to mask an entire word
MIN_WORD_LENGTH = 3           # Only corrupt words longer than this
SPAN_MIN_LENGTH = 2           # Minimum words to drop in a phrase span
SPAN_MAX_LENGTH = 4           # Maximum words to drop in a phrase span
MIN_SENTENCE_LENGTH = 5       # Minimum sentence length for span drops

def remove_random_chars(text, corruption_rate=CHAR_CORRUPTION_RATE):
    """
    Replaces random characters in words with underscores to simulate faded ink.
    
    Args:
        text: Input sentence
        corruption_rate: Probability of corrupting each word (0.0 to 1.0)
    
    Returns:
        Damaged text with some characters replaced by '_'
    """
    if not text or not text.strip():
        return text
        
    words = text.split()
    if len(words) == 0:
        return text
        
    damaged_words = []
    
    for word in words:
        if len(word) > MIN_WORD_LENGTH and random.random() < corruption_rate:
            char_idx = random.randint(1, len(word) - 2) 
            word = word[:char_idx] + "_" + word[char_idx+1:]
        damaged_words.append(word)
        
    # Fallback: if random chance missed everything, force at least one character to break
    result = " ".join(damaged_words)
    if result == text and len(words) > 0:
        # Find a word that's long enough to corrupt
        valid_indices = [i for i, w in enumerate(words) if len(w) > MIN_WORD_LENGTH]
        if valid_indices:
            idx = random.choice(valid_indices)
            char_idx = random.randint(1, len(words[idx]) - 2)
            words[idx] = words[idx][:char_idx] + "_" + words[idx][char_idx+1:]
            return " ".join(words)
            
    return result

def mask_random_words(text, corruption_rate=WORD_MASK_RATE):
    """
    Replaces random words with [WORD_GAP] to simulate completely illegible words.
    
    Args:
        text: Input sentence
        corruption_rate: Probability of masking each word (0.0 to 1.0)
    
    Returns:
        Text with some words replaced by [WORD_GAP]
    """
    if not text or not text.strip():
        return text
        
    words = text.split()
    if len(words) == 0:
        return text
    
    for i in range(len(words)):
        if random.random() < corruption_rate:
            words[i] = "[WORD_GAP]"
            
    # Fallback: guarantee at least one gap
    if "[WORD_GAP]" not in words and len(words) > 0:
        words[random.randint(0, len(words)-1)] = "[WORD_GAP]"
            
    return " ".join(words)

def drop_phrase_spans(text):
    """
    Replaces a continuous chunk of 2-4 words with [LINE_GAP] to simulate 
    torn pages or heavy water damage.
    
    Args:
        text: Input sentence
    
    Returns:
        Text with a phrase span replaced by [LINE_GAP]
    """
    if not text or not text.strip():
        return text
        
    words = text.split()
    
    # Only drop spans if sentence is long enough
    if len(words) < MIN_SENTENCE_LENGTH:
        return text 
        
    # Drop a random span of words
    start_idx = random.randint(0, len(words) - SPAN_MAX_LENGTH)
    span_length = random.randint(SPAN_MIN_LENGTH, SPAN_MAX_LENGTH)
    
    words[start_idx : start_idx + span_length] = ["[LINE_GAP]"]
    return " ".join(words)

def generate_synthetic_pair(clean_text):
    """
    Takes a clean sentence and applies realistic manuscript damage.
    
    Damage types:
    - 'char': Faded letters (k_ng instead of king)
    - 'word': Missing words ([WORD_GAP])
    - 'span': Torn sections ([LINE_GAP] for multiple words)
    - 'mixed': Combination of character and word damage
    
    Args:
        clean_text: Original undamaged sentence
    
    Returns:
        Tuple of (clean_text, damaged_text)
    """
    if not clean_text or not clean_text.strip():
        return clean_text, clean_text
    
    damage_type = random.choice(['char', 'word', 'span', 'mixed'])
    
    if damage_type == 'char':
        damaged = remove_random_chars(clean_text)
    elif damage_type == 'word':
        damaged = mask_random_words(clean_text)
    elif damage_type == 'span':
        damaged = drop_phrase_spans(clean_text)
    else:  # mixed
        damaged = mask_random_words(clean_text, corruption_rate=0.1)
        damaged = remove_random_chars(damaged, corruption_rate=0.1)
        
    return clean_text, damaged

if __name__ == "__main__":
    # Let's test it with a clean sentence
    sample_sentence = "The king of England ruled wisely for many years before the great war."
    
    print("--- SYNTHETIC DAMAGE GENERATOR ---")
    print(f"ORIGINAL: {sample_sentence}\n")
    
    for i in range(5):
        clean, damaged = generate_synthetic_pair(sample_sentence)
        print(f"Target : {clean}")
        print(f"Input  : {damaged}")
        print("-" * 50)