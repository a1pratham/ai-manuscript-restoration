from typing import List, Tuple, Dict

# --- GAP DETECTION THRESHOLDS ---
# These determine when OCR text is too unreliable and should be masked for BERT

ABSOLUTE_GARBAGE_THRESHOLD = 45.0  
# Below 45%: OCR is completely confused (heavy smudge, torn page)
# Example: "k7@#" with 23% confidence → definitely mask it

SUSPICIOUS_SHORT_WORD_THRESHOLD = 75.0
# Below 75% AND word length ≤ 3: Likely hallucination
# Example: "th" with 68% confidence → probably a guess, mask it
# But "the" with 68% might be real → keep it (handled by length check)

SHORT_WORD_MAX_LENGTH = 3
# Words this length or shorter are masked more aggressively
# Rationale: Short words are easier to hallucinate ("a", "in", "of")

def format_for_bert(words: List[str], confidences: List[float]) -> Tuple[str, Dict[int, float]]:
    """
    Convert OCR output into BERT-ready format by masking low-confidence words.
    
    Masking Strategy:
    1. Confidence < 45%: Always mask (absolute garbage)
    2. Confidence < 75% AND length ≤ 3: Mask (suspicious short word)
    3. Otherwise: Keep original OCR text
    
    Args:
        words: List of words extracted by TrOCR
        confidences: List of confidence scores (0-100) for each word
    
    Returns:
        Tuple of:
        - masked_text: String with [MASK] tokens replacing unreliable words
        - confidence_map: Dict mapping mask_position → original_OCR_confidence
          Example: {0: 35.2, 2: 42.8} means first and third masks had those confidences
    
    Raises:
        ValueError: If words and confidences lists have different lengths
    
    Example:
        >>> words = ["The", "k_ng", "of", "England"]
        >>> confidences = [95.0, 23.5, 88.0, 91.2]
        >>> format_for_bert(words, confidences)
        ("The [MASK] of England", {0: 23.5})
    """
    # Validation
    if not words:
        return "", {}
    
    if not confidences:
        # If no confidences provided, assume all words are good
        return " ".join(words), {}
    
    if len(words) != len(confidences):
        raise ValueError(
            f"Mismatched lengths: {len(words)} words but {len(confidences)} confidence scores.\n"
            f"Words: {words}\n"
            f"Confidences: {confidences}"
        )
    
    masked_words = []
    confidence_map = {}
    mask_counter = 0
    
    for word, conf in zip(words, confidences):
        # Validate confidence is in reasonable range
        if conf < 0 or conf > 100:
            print(f"WARNING: Confidence {conf} out of range [0, 100] for word '{word}'. Treating as 0.")
            conf = max(0, min(100, conf))
        
        should_mask = False
        
        # Rule 1: Absolute garbage (heavy damage)
        if conf < ABSOLUTE_GARBAGE_THRESHOLD:
            should_mask = True
            reason = f"below {ABSOLUTE_GARBAGE_THRESHOLD}% threshold"
        
        # Rule 2: Suspicious short word (likely hallucination)
        elif conf < SUSPICIOUS_SHORT_WORD_THRESHOLD and len(word) <= SHORT_WORD_MAX_LENGTH:
            should_mask = True
            reason = f"short word (<={SHORT_WORD_MAX_LENGTH} chars) with <{SUSPICIOUS_SHORT_WORD_THRESHOLD}% confidence"
        
        if should_mask:
            masked_words.append("[MASK]")
            confidence_map[mask_counter] = conf
            mask_counter += 1
            print(f"  Masking '{word}' (conf: {conf:.1f}%) - {reason}")
        else:
            masked_words.append(word)
    
    masked_text = " ".join(masked_words)
    
    return masked_text, confidence_map


def analyze_gap_distribution(confidences: List[float]) -> Dict[str, any]:
    """
    Analyze the distribution of OCR confidence scores to assess document quality.
    
    Args:
        confidences: List of confidence scores (0-100)
    
    Returns:
        Dictionary with statistics:
        - avg_confidence: Mean confidence
        - min_confidence: Lowest confidence
        - max_confidence: Highest confidence
        - garbage_count: Words below ABSOLUTE_GARBAGE_THRESHOLD
        - suspicious_count: Words below SUSPICIOUS_SHORT_WORD_THRESHOLD
        - quality_rating: "Excellent"/"Good"/"Fair"/"Poor"/"Very Poor"
    """
    if not confidences:
        return {
            'avg_confidence': 0.0,
            'min_confidence': 0.0,
            'max_confidence': 0.0,
            'garbage_count': 0,
            'suspicious_count': 0,
            'quality_rating': 'No Data'
        }
    
    avg_conf = sum(confidences) / len(confidences)
    min_conf = min(confidences)
    max_conf = max(confidences)
    
    garbage_count = sum(1 for c in confidences if c < ABSOLUTE_GARBAGE_THRESHOLD)
    suspicious_count = sum(1 for c in confidences if c < SUSPICIOUS_SHORT_WORD_THRESHOLD)
    
    # Quality rating based on average confidence
    if avg_conf >= 90:
        quality_rating = "Excellent"
    elif avg_conf >= 75:
        quality_rating = "Good"
    elif avg_conf >= 60:
        quality_rating = "Fair"
    elif avg_conf >= 45:
        quality_rating = "Poor"
    else:
        quality_rating = "Very Poor"
    
    return {
        'avg_confidence': round(avg_conf, 2),
        'min_confidence': round(min_conf, 2),
        'max_confidence': round(max_conf, 2),
        'garbage_count': garbage_count,
        'suspicious_count': suspicious_count,
        'quality_rating': quality_rating
    }


if __name__ == "__main__":
    print("="*60)
    print("GAP DETECTION - TEST")
    print("="*60 + "\n")
    
    # Test case 1: Mixed quality manuscript line
    print("Test 1: Mixed quality line")
    print("-" * 60)
    words1 = ["The", "k_ng", "of", "En", "ruled", "wisely"]
    confidences1 = [95.2, 23.5, 88.3, 68.1, 91.7, 94.3]
    
    print(f"Input words: {words1}")
    print(f"Confidences: {confidences1}\n")
    
    masked_text1, conf_map1 = format_for_bert(words1, confidences1)
    
    print(f"\nResult: '{masked_text1}'")
    print(f"Confidence map: {conf_map1}")
    
    stats1 = analyze_gap_distribution(confidences1)
    print(f"\nQuality Analysis: {stats1['quality_rating']}")
    print(f"  Average confidence: {stats1['avg_confidence']}%")
    print(f"  Garbage words: {stats1['garbage_count']}")
    print(f"  Suspicious words: {stats1['suspicious_count']}")
    
    # Test case 2: Heavily damaged line
    print("\n" + "="*60)
    print("Test 2: Heavily damaged line")
    print("-" * 60)
    words2 = ["Th_", "m_n", "w__", "b_rn", "in"]
    confidences2 = [42.1, 18.3, 31.2, 39.8, 67.4]
    
    print(f"Input words: {words2}")
    print(f"Confidences: {confidences2}\n")
    
    masked_text2, conf_map2 = format_for_bert(words2, confidences2)
    
    print(f"\nResult: '{masked_text2}'")
    print(f"Confidence map: {conf_map2}")
    
    stats2 = analyze_gap_distribution(confidences2)
    print(f"\nQuality Analysis: {stats2['quality_rating']}")
    print(f"  Average confidence: {stats2['avg_confidence']}%")
    
    # Test case 3: Error handling - mismatched lengths
    print("\n" + "="*60)
    print("Test 3: Error handling")
    print("-" * 60)
    try:
        words3 = ["The", "king"]
        confidences3 = [95.0]  # Mismatch!
        masked_text3, conf_map3 = format_for_bert(words3, confidences3)
    except ValueError as e:
        print(f"✓ Caught expected error:\n  {e}")