"""
TrOCR Line-by-Line OCR Test

Tests handwriting recognition on extracted line images:
1. Loads line images from line_segmentation_output/ (created by line_test.py)
2. Runs TrOCR on each line individually
3. Displays extracted text and confidence scores
4. Saves results to a text file

This verifies the OCR pipeline is working correctly on properly
segmented single-line images.
"""

import sys
from pathlib import Path
from PIL import Image

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ocr.htr_engine import HandwritingRecognitionEngine
from ocr.gap_detection import format_for_bert, analyze_gap_distribution

# --- TEST CONFIGURATION ---
LINE_IMAGES_DIR = "line_segmentation_output"  # Output from line_test.py
RESULTS_FILE = "ocr_results.txt"


def find_line_images():
    """
    Find line images from line_test.py output.
    
    Returns:
        List of Path objects to line images, sorted by line number
    """
    line_dir = project_root / LINE_IMAGES_DIR
    
    if not line_dir.exists():
        return None, f"Directory not found: {line_dir}"
    
    # Find all line_XX.png files
    line_files = sorted(line_dir.glob("line_*.png"))
    
    if not line_files:
        return None, f"No line images found in: {line_dir}"
    
    return line_files, None


def test_ocr_on_lines():
    """
    Run TrOCR on each extracted line and display results.
    """
    print("="*60)
    print("TrOCR LINE-BY-LINE OCR TEST")
    print("="*60 + "\n")
    
    # Check for line images
    line_files, error = find_line_images()
    
    if error:
        print(f"❌ {error}")
        print("\nThis test requires line images from line_test.py")
        print("\nTo run this test:")
        print("1. First run: python test/line_test.py")
        print("2. This creates individual line images")
        print("3. Then run this test to OCR each line\n")
        return
    
    print(f"✓ Found {len(line_files)} line images")
    print(f"  Location: {project_root / LINE_IMAGES_DIR}\n")
    
    try:
        # Initialize OCR engine
        print("Initializing TrOCR handwriting recognition engine...")
        engine = HandwritingRecognitionEngine()
        print()
        
    except Exception as e:
        print(f"❌ Failed to load TrOCR engine: {e}")
        print("\nPlease ensure transformers and torch are installed:")
        print("  pip install transformers torch pillow\n")
        return
    
    # Process each line
    results = []
    all_confidences = []
    
    print("="*60)
    print("PROCESSING LINES")
    print("="*60 + "\n")
    
    for idx, line_path in enumerate(line_files, 1):
        print(f"--- Line {idx}: {line_path.name} ---")
        
        try:
            # Load line image
            line_image = Image.open(line_path).convert("RGB")
            print(f"  Image size: {line_image.size}")
            
            # Run OCR
            words, confidences = engine.read_handwriting(line_image)
            
            if not words:
                print(f"  ⚠️  No text extracted")
                results.append({
                    'line_num': idx,
                    'filename': line_path.name,
                    'text': "",
                    'words': [],
                    'confidences': [],
                    'avg_confidence': 0.0
                })
                print()
                continue
            
            # Calculate statistics
            avg_conf = sum(confidences) / len(confidences)
            min_conf = min(confidences)
            max_conf = max(confidences)
            all_confidences.extend(confidences)
            
            # Display results
            extracted_text = " ".join(words)
            print(f"  Extracted: '{extracted_text}'")
            print(f"  Words: {len(words)}")
            print(f"  Confidence: avg={avg_conf:.1f}%, min={min_conf:.1f}%, max={max_conf:.1f}%")
            
            # Show word-level details
            print(f"  Word details:")
            for word, conf in zip(words, confidences):
                status = "✓" if conf >= 75 else "⚠" if conf >= 45 else "✗"
                print(f"    {status} '{word}' - {conf:.1f}%")
            
            # Test gap detection on this line
            masked_text, conf_map = format_for_bert(words, confidences)
            if '[MASK]' in masked_text:
                print(f"  Gap detection: '{masked_text}'")
                print(f"  Masks needed: {len(conf_map)}")
            else:
                print(f"  Gap detection: No masks needed (high confidence)")
            
            results.append({
                'line_num': idx,
                'filename': line_path.name,
                'text': extracted_text,
                'words': words,
                'confidences': confidences,
                'avg_confidence': avg_conf,
                'masked_text': masked_text,
                'mask_count': len(conf_map)
            })
            
            print()
            
        except Exception as e:
            print(f"  ❌ ERROR processing line: {e}")
            results.append({
                'line_num': idx,
                'filename': line_path.name,
                'text': f"ERROR: {e}",
                'words': [],
                'confidences': [],
                'avg_confidence': 0.0
            })
            print()
    
    # Overall statistics
    print("="*60)
    print("OVERALL STATISTICS")
    print("="*60)
    print(f"Total lines processed: {len(results)}")
    
    successful_lines = [r for r in results if r['words']]
    print(f"Successful extractions: {len(successful_lines)}")
    
    if all_confidences:
        overall_avg = sum(all_confidences) / len(all_confidences)
        overall_stats = analyze_gap_distribution(all_confidences)
        
        print(f"\nOverall OCR Quality: {overall_stats['quality_rating']}")
        print(f"  Average confidence: {overall_avg:.1f}%")
        print(f"  Minimum confidence: {min(all_confidences):.1f}%")
        print(f"  Maximum confidence: {max(all_confidences):.1f}%")
        print(f"  Low confidence words: {overall_stats['garbage_count']}")
        print(f"  Suspicious words: {overall_stats['suspicious_count']}")
        
        total_masks = sum(r.get('mask_count', 0) for r in results)
        total_words = sum(len(r['words']) for r in results)
        if total_words > 0:
            mask_percentage = (total_masks / total_words) * 100
            print(f"\nReconstruction needed:")
            print(f"  Total words: {total_words}")
            print(f"  Words requiring BERT: {total_masks} ({mask_percentage:.1f}%)")
    
    print("="*60 + "\n")
    
    # Save results to file
    output_dir = project_root / LINE_IMAGES_DIR
    results_path = output_dir / RESULTS_FILE
    
    print(f"Saving results to: {results_path}")
    
    try:
        with open(results_path, 'w', encoding='utf-8') as f:
            f.write("TrOCR LINE-BY-LINE OCR RESULTS\n")
            f.write("="*60 + "\n\n")
            
            for result in results:
                f.write(f"Line {result['line_num']}: {result['filename']}\n")
                f.write(f"Text: {result['text']}\n")
                
                if result['words']:
                    f.write(f"Words: {len(result['words'])}\n")
                    f.write(f"Average Confidence: {result['avg_confidence']:.1f}%\n")
                    f.write("Word Details:\n")
                    for word, conf in zip(result['words'], result['confidences']):
                        f.write(f"  '{word}' - {conf:.1f}%\n")
                    
                    if 'masked_text' in result:
                        f.write(f"Gap Detection: {result['masked_text']}\n")
                
                f.write("\n" + "-"*60 + "\n\n")
            
            if all_confidences:
                f.write("OVERALL STATISTICS\n")
                f.write("="*60 + "\n")
                f.write(f"Total lines: {len(results)}\n")
                f.write(f"Successful extractions: {len(successful_lines)}\n")
                f.write(f"Overall average confidence: {sum(all_confidences)/len(all_confidences):.1f}%\n")
                f.write(f"Quality rating: {overall_stats['quality_rating']}\n")
        
        print(f"  ✓ Results saved\n")
        
    except Exception as e:
        print(f"  ⚠️  Failed to save results file: {e}\n")
    
    # Display final summary
    print("✓ Test completed successfully!")
    print("\nNext steps:")
    print("1. Review the extracted text above")
    print("2. Check the saved results file for detailed breakdown")
    print("3. If quality is good, test end-to-end reconstruction:")
    print("   python test/gap_test.py\n")


if __name__ == "__main__":
    test_ocr_on_lines()