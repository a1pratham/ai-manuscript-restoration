import numpy as np
from typing import List, Tuple

# --- EVALUATION PARAMETERS ---
DEFAULT_TOP_K = 3  # Number of top predictions to consider for Top-K accuracy

def calculate_cer(reference: str, hypothesis: str) -> float:
    """
    Calculate Character Error Rate (CER) between reference and hypothesis.
    
    CER measures character-level accuracy:
    CER = (substitutions + deletions + insertions) / total_reference_chars
    
    Args:
        reference: Ground truth text
        hypothesis: Predicted text
    
    Returns:
        CER as a float (0.0 = perfect, 1.0 = completely wrong)
    """
    if not reference:
        return 0.0 if not hypothesis else 1.0
    
    if not hypothesis:
        return 1.0
    
    # Levenshtein distance at character level
    ref_chars = list(reference)
    hyp_chars = list(hypothesis)
    
    # Create DP matrix
    d = np.zeros((len(ref_chars) + 1, len(hyp_chars) + 1), dtype=int)
    
    # Initialize first row and column
    for i in range(len(ref_chars) + 1):
        d[i][0] = i
    for j in range(len(hyp_chars) + 1):
        d[0][j] = j
    
    # Fill the matrix
    for i in range(1, len(ref_chars) + 1):
        for j in range(1, len(hyp_chars) + 1):
            if ref_chars[i-1] == hyp_chars[j-1]:
                cost = 0
            else:
                cost = 1
            
            d[i][j] = min(
                d[i-1][j] + 1,      # deletion
                d[i][j-1] + 1,      # insertion
                d[i-1][j-1] + cost  # substitution
            )
    
    # CER = edit distance / reference length
    edit_distance = d[len(ref_chars)][len(hyp_chars)]
    cer = edit_distance / len(ref_chars)
    
    return float(cer)

def calculate_wer(reference: str, hypothesis: str) -> float:
    """
    Calculate Word Error Rate (WER) between reference and hypothesis.
    
    WER measures word-level accuracy:
    WER = (substitutions + deletions + insertions) / total_reference_words
    
    Args:
        reference: Ground truth text
        hypothesis: Predicted text
    
    Returns:
        WER as a float (0.0 = perfect, 1.0 = completely wrong)
    """
    if not reference or not reference.strip():
        return 0.0 if not hypothesis or not hypothesis.strip() else 1.0
    
    if not hypothesis or not hypothesis.strip():
        return 1.0
    
    # Split into words
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    
    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0
    
    # Levenshtein distance at word level
    d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1), dtype=int)
    
    # Initialize first row and column
    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j
    
    # Fill the matrix
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                cost = 0
            else:
                cost = 1
            
            d[i][j] = min(
                d[i-1][j] + 1,      # deletion
                d[i][j-1] + 1,      # insertion
                d[i-1][j-1] + cost  # substitution
            )
    
    # WER = edit distance / reference length
    edit_distance = d[len(ref_words)][len(hyp_words)]
    wer = edit_distance / len(ref_words)
    
    return float(wer)

def calculate_top_k_accuracy(
    references: List[str], 
    predictions_list: List[List[str]], 
    k: int = DEFAULT_TOP_K
) -> float:
    """
    Calculate Top-K accuracy: percentage of times the correct word appears
    in the top K predictions.
    
    Args:
        references: List of ground truth words
        predictions_list: List of prediction lists (each inner list has K predictions)
        k: Number of top predictions to consider
    
    Returns:
        Top-K accuracy as percentage (0.0 to 100.0)
    """
    if not references or not predictions_list:
        return 0.0
    
    if len(references) != len(predictions_list):
        print(f"WARNING: Mismatched lengths - references: {len(references)}, predictions: {len(predictions_list)}")
        min_len = min(len(references), len(predictions_list))
        references = references[:min_len]
        predictions_list = predictions_list[:min_len]
    
    if len(references) == 0:
        return 0.0
    
    correct = 0
    
    for ref, preds in zip(references, predictions_list):
        if not preds:
            continue
            
        # Take top k predictions
        top_k_preds = preds[:k]
        
        # Check if reference appears in top k
        if ref in top_k_preds:
            correct += 1
    
    accuracy = (correct / len(references)) * 100.0
    return float(accuracy)

def evaluate_reconstruction(
    reference_text: str, 
    reconstructed_text: str,
    reference_words: List[str] = None,
    predicted_word_lists: List[List[str]] = None,
    k: int = DEFAULT_TOP_K
) -> dict:
    """
    Comprehensive evaluation of reconstruction quality.
    
    Args:
        reference_text: Ground truth full text
        reconstructed_text: Model's reconstructed text
        reference_words: Optional list of ground truth words for Top-K
        predicted_word_lists: Optional list of prediction lists for Top-K
        k: Number of top predictions to consider
    
    Returns:
        Dictionary with CER, WER, and optionally Top-K accuracy
    """
    results = {
        'cer': calculate_cer(reference_text, reconstructed_text),
        'wer': calculate_wer(reference_text, reconstructed_text)
    }
    
    # Add Top-K accuracy if word-level predictions provided
    if reference_words is not None and predicted_word_lists is not None:
        results['top_k_accuracy'] = calculate_top_k_accuracy(
            reference_words, 
            predicted_word_lists, 
            k
        )
    
    return results

if __name__ == "__main__":
    # Test the metrics
    print("--- EVALUATION METRICS TEST ---\n")
    
    reference = "The king ruled wisely"
    hypothesis1 = "The king ruled wisely"  # Perfect
    hypothesis2 = "The king ruled wise"    # Missing one char
    hypothesis3 = "The king rule wisely"   # Different word form
    
    print(f"Reference: '{reference}'")
    print(f"\nTest 1 - Perfect match: '{hypothesis1}'")
    print(f"  CER: {calculate_cer(reference, hypothesis1):.3f}")
    print(f"  WER: {calculate_wer(reference, hypothesis1):.3f}")
    
    print(f"\nTest 2 - Character error: '{hypothesis2}'")
    print(f"  CER: {calculate_cer(reference, hypothesis2):.3f}")
    print(f"  WER: {calculate_wer(reference, hypothesis2):.3f}")
    
    print(f"\nTest 3 - Word error: '{hypothesis3}'")
    print(f"  CER: {calculate_cer(reference, hypothesis3):.3f}")
    print(f"  WER: {calculate_wer(reference, hypothesis3):.3f}")
    
    print("\n--- Top-K Accuracy Test ---")
    refs = ["king", "ruled", "wisely"]
    preds = [
        ["king", "kong", "wing"],
        ["ruled", "rules", "ruler"],
        ["wise", "wisely", "wisdom"]
    ]
    
    print(f"References: {refs}")
    print(f"Predictions: {preds}")
    print(f"Top-3 Accuracy: {calculate_top_k_accuracy(refs, preds, k=3):.1f}%")