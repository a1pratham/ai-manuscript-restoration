"""
End-to-End Manuscript Reconstruction Test

Tests the complete pipeline:
1. Simulate OCR output (words + confidence scores)
2. Detect gaps (format for BERT with masks)
3. BERT reconstructs missing words
4. Display results

This is the integration test for the entire system.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ocr.gap_detection import format_for_bert, analyze_gap_distribution
from ml_model.reconstruction_engine import ReconstructionEngine

# --- TEST CONFIGURATION ---
TEST_REQUIRES_TRAINED_MODEL = True


def test_gap_detection_and_reconstruction():
    """
    Test the gap detection and reconstruction pipeline with sample data.
    
    Simulates a damaged manuscript line with varying OCR confidence.
    """
    print("="*60)
    print("END-TO-END RECONSTRUCTION TEST")
    print("="*60 + "\n")
    
    # Simulate OCR output from a damaged manuscript
    print("--- Simulated OCR Output ---")
    words = ["The", "k_ng", "of", "En", "ruled", "wisely"]
    confidences = [95.2, 23.5, 88.3, 68.1, 91.7, 94.3]
    
    print(f"Words: {words}")
    print(f"Confidences: {confidences}\n")
    
    # Analyze document quality
    stats = analyze_gap_distribution(confidences)
    print(f"Document Quality: {stats['quality_rating']}")
    print(f"  Average confidence: {stats['avg_confidence']}%")
    print(f"  Garbage words: {stats['garbage_count']}")
    print(f"  Suspicious words: {stats['suspicious_count']}\n")
    
    # Step 1: Format for BERT (convert low-confidence words to [MASK])
    print("--- Step 1: Gap Detection ---")
    masked_text, confidence_map = format_for_bert(words, confidences)
    
    print(f"Masked text: '{masked_text}'")
    print(f"Confidence map: {confidence_map}\n")
    
    if '[MASK]' not in masked_text:
        print("⚠️  No gaps detected - all words have sufficient confidence")
        print("Test completed successfully (no reconstruction needed)\n")
        return
    
    # Step 2: Reconstruct with BERT
    print("--- Step 2: BERT Reconstruction ---")
    
    try:
        print("Loading reconstruction engine...")
        engine = ReconstructionEngine()
        
        print(f"Input to BERT: '{masked_text}'")
        print(f"OCR confidence map: {confidence_map}\n")
        
        result = engine.reconstruct_for_api(masked_text, confidence_map)
        
        print("\n" + "="*60)
        print("RECONSTRUCTION RESULTS")
        print("="*60)
        print(f"Original (damaged): {' '.join(words)}")
        print(f"Masked for BERT:    {masked_text}")
        print(f"Reconstructed:      {result['reconstructed_text']}")
        print("="*60 + "\n")
        
        # Show detailed predictions for each gap
        print("Detailed predictions per gap:")
        for pred in result['all_predictions']:
            print(f"\n  Gap {pred['position']}:")
            print(f"    Chosen: '{pred['chosen_word']}' (score: {pred['final_score']:.3f})")
            print(f"    Alternatives:")
            for alt in pred['alternatives']:
                print(f"      - '{alt['word']}' "
                      f"(BERT: {alt['bert_score']:.3f}, "
                      f"OCR: {alt['ocr_confidence']:.3f}, "
                      f"Final: {alt['final_score']:.3f})")
        
        print("\n✓ Test completed successfully!\n")
        
    except FileNotFoundError as e:
        print("\n" + "="*60)
        print("⚠️  MODEL NOT FOUND")
        print("="*60)
        print(str(e))
        print("\nThis test requires a trained BERT model.")
        print("Please run: python reconstruction/train_model.py")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


def test_multiple_scenarios():
    """
    Test reconstruction with different damage scenarios.
    """
    print("="*60)
    print("MULTIPLE SCENARIO TEST")
    print("="*60 + "\n")
    
    test_cases = [
        {
            "name": "Heavily Damaged",
            "words": ["Th_", "k__g", "r_led", "w__ely"],
            "confidences": [42.0, 18.0, 35.0, 28.0]
        },
        {
            "name": "Moderately Damaged",
            "words": ["The", "king", "of", "En_land", "ruled"],
            "confidences": [95.0, 88.0, 92.0, 65.0, 90.0]
        },
        {
            "name": "Minimal Damage",
            "words": ["The", "king", "ruled", "wisely"],
            "confidences": [98.0, 96.0, 94.0, 95.0]
        }
    ]
    
    try:
        engine = ReconstructionEngine()
        
        for test in test_cases:
            print(f"\n--- {test['name']} ---")
            print(f"Words: {test['words']}")
            print(f"Confidences: {test['confidences']}")
            
            masked_text, conf_map = format_for_bert(test['words'], test['confidences'])
            print(f"Masked: '{masked_text}'")
            
            if '[MASK]' in masked_text:
                result = engine.reconstruct_for_api(masked_text, conf_map)
                print(f"Result: '{result['reconstructed_text']}'")
            else:
                print("Result: No reconstruction needed (high confidence)")
            
            print("-" * 60)
        
        print("\n✓ All scenarios tested successfully!\n")
        
    except FileNotFoundError as e:
        print(f"\n⚠️  Skipping scenario tests - model not found")
        print("Run: python reconstruction/train_model.py\n")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("GAP DETECTION + RECONSTRUCTION TEST SUITE")
    print("="*60 + "\n")
    
    # Test 1: Basic end-to-end test
    test_gap_detection_and_reconstruction()
    
    # Test 2: Multiple scenarios
    print("\n" + "="*60 + "\n")
    test_multiple_scenarios()
    
    print("="*60)
    print("TEST SUITE COMPLETE")
    print("="*60 + "\n")