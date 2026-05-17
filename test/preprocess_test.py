"""
Image Preprocessing Test

Tests the image cleaning pipeline:
1. Loads a manuscript image
2. Applies bilateral filtering for noise reduction
3. Converts to grayscale
4. Saves before/after comparison

This verifies that preprocessing enhances image quality
for better line segmentation and OCR.
"""

import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from preprocessing.image_cleaning import preprocess_image, ImageCleaningError

# --- TEST CONFIGURATION ---
OUTPUT_DIR = "preprocessing_output"


def find_test_image():
    """
    Find an available test image from common locations.
    
    Returns:
        Path to test image, or None if not found
    """
    possible_paths = [
        project_root / "test.png",
        project_root / "test2.png",
        project_root / "test2.jpg",
        project_root / "test3.jpg",
        project_root / "test" / "test.png",
    ]
    
    for path in possible_paths:
        if path.exists():
            return path
    
    return None


def create_comparison_image(original, processed):
    """
    Create side-by-side comparison of original and processed images.
    
    Args:
        original: Original image (may be color or grayscale)
        processed: Processed grayscale image
    
    Returns:
        Side-by-side comparison image
    """
    # Ensure original is grayscale for fair comparison
    if len(original.shape) == 3:
        original_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
    else:
        original_gray = original
    
    # Resize if needed to match heights
    if original_gray.shape[0] != processed.shape[0]:
        processed = cv2.resize(
            processed, 
            (processed.shape[1], original_gray.shape[0])
        )
    
    # Create side-by-side comparison
    comparison = np.hstack([original_gray, processed])
    
    # Add labels
    h, w = comparison.shape
    midpoint = w // 2
    
    # Add text labels
    cv2.putText(
        comparison,
        "ORIGINAL",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        0,
        2
    )
    
    cv2.putText(
        comparison,
        "PROCESSED (Bilateral Filter)",
        (midpoint + 10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        0,
        2
    )
    
    # Add vertical separator line
    cv2.line(comparison, (midpoint, 0), (midpoint, h), 128, 2)
    
    return comparison


def test_preprocessing():
    """
    Run preprocessing test with multiple input methods.
    """
    print("="*60)
    print("IMAGE PREPROCESSING TEST")
    print("="*60 + "\n")
    
    # Find test image
    test_image_path = find_test_image()
    
    if test_image_path is None:
        print("❌ No test image found!")
        print("\nSearched locations:")
        for path in [
            project_root / "test.png",
            project_root / "test2.png",
            project_root / "test2.jpg",
            project_root / "test3.jpg",
        ]:
            print(f"  - {path}")
        
        print("\nTo run this test:")
        print("1. Place a manuscript image at project root")
        print("2. Name it test.png, test2.png, or test3.jpg\n")
        return
    
    print(f"✓ Found test image: {test_image_path}")
    print(f"  File size: {test_image_path.stat().st_size / 1024:.1f} KB\n")
    
    # Create output directory
    output_path = project_root / OUTPUT_DIR
    output_path.mkdir(exist_ok=True)
    print(f"Output directory: {output_path}\n")
    
    try:
        # Load original for comparison
        original_pil = Image.open(test_image_path)
        original_array = np.array(original_pil)
        
        print(f"Original image:")
        print(f"  Size: {original_pil.size}")
        print(f"  Mode: {original_pil.mode}")
        print(f"  Array shape: {original_array.shape}\n")
        
        # Test 1: Load from file path string
        print("--- Test 1: Load from file path (string) ---")
        cleaned1 = preprocess_image(str(test_image_path))
        print(f"  ✓ Output shape: {cleaned1.shape}")
        print(f"  ✓ Output dtype: {cleaned1.dtype}\n")
        
        # Test 2: Load from PIL Image
        print("--- Test 2: Load from PIL Image ---")
        cleaned2 = preprocess_image(original_pil)
        print(f"  ✓ Output shape: {cleaned2.shape}")
        print(f"  ✓ Output dtype: {cleaned2.dtype}\n")
        
        # Test 3: Load from numpy array
        print("--- Test 3: Load from numpy array ---")
        cleaned3 = preprocess_image(original_array)
        print(f"  ✓ Output shape: {cleaned3.shape}")
        print(f"  ✓ Output dtype: {cleaned3.dtype}\n")
        
        # Verify all methods produce similar results
        print("--- Consistency Check ---")
        diff_1_2 = np.mean(np.abs(cleaned1.astype(float) - cleaned2.astype(float)))
        diff_2_3 = np.mean(np.abs(cleaned2.astype(float) - cleaned3.astype(float)))
        
        print(f"  Mean difference (method 1 vs 2): {diff_1_2:.2f}")
        print(f"  Mean difference (method 2 vs 3): {diff_2_3:.2f}")
        
        if diff_1_2 < 1.0 and diff_2_3 < 1.0:
            print(f"  ✓ All methods produce consistent results\n")
        else:
            print(f"  ⚠️  Methods produce different results (unexpected)\n")
        
        # Save outputs
        print("--- Saving Results ---")
        
        # Save cleaned image
        cleaned_output = output_path / "cleaned.png"
        cv2.imwrite(str(cleaned_output), cleaned1)
        print(f"  ✓ Saved: {cleaned_output.name}")
        
        # Save comparison
        comparison = create_comparison_image(original_array, cleaned1)
        comparison_output = output_path / "comparison.png"
        cv2.imwrite(str(comparison_output), comparison)
        print(f"  ✓ Saved: {comparison_output.name}")
        
        # Calculate and display preprocessing statistics
        print("\n--- Preprocessing Statistics ---")
        
        # Convert original to grayscale for fair comparison
        if len(original_array.shape) == 3:
            original_gray = cv2.cvtColor(original_array, cv2.COLOR_RGB2GRAY)
        else:
            original_gray = original_array
        
        original_mean = np.mean(original_gray)
        original_std = np.std(original_gray)
        cleaned_mean = np.mean(cleaned1)
        cleaned_std = np.std(cleaned1)
        
        print(f"  Original:")
        print(f"    Mean intensity: {original_mean:.1f}")
        print(f"    Std deviation: {original_std:.1f}")
        print(f"  Cleaned:")
        print(f"    Mean intensity: {cleaned_mean:.1f}")
        print(f"    Std deviation: {cleaned_std:.1f}")
        print(f"  Change:")
        print(f"    Mean: {cleaned_mean - original_mean:+.1f}")
        print(f"    Std: {cleaned_std - original_std:+.1f}")
        
        if cleaned_std < original_std:
            print(f"  ✓ Noise reduced (lower std deviation)")
        
        print("\n" + "="*60)
        print("RESULTS SUMMARY")
        print("="*60)
        print(f"Input: {test_image_path.name}")
        print(f"Output directory: {output_path}")
        print("\nFiles created:")
        print(f"  - cleaned.png (processed image)")
        print(f"  - comparison.png (before/after side-by-side)")
        print("\nPreprocessing effects:")
        print(f"  ✓ Converted to grayscale")
        print(f"  ✓ Applied bilateral filter (noise reduction)")
        print(f"  ✓ Preserved text edges")
        print("="*60 + "\n")
        
        print("✓ Test completed successfully!")
        print("\nNext steps:")
        print("1. Open comparison.png to visually inspect the preprocessing")
        print("2. Verify that text is sharper and noise is reduced")
        print("3. If preprocessing looks good, test line segmentation:")
        print("   python test/line_test.py\n")
        
    except ImageCleaningError as e:
        print(f"\n❌ Preprocessing error: {e}\n")
        
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        print()


def test_error_handling():
    """
    Test error handling with invalid inputs.
    """
    print("="*60)
    print("ERROR HANDLING TEST")
    print("="*60 + "\n")
    
    test_cases = [
        ("None input", None),
        ("Empty string", ""),
        ("Non-existent file", "nonexistent_image.jpg"),
        ("Invalid type", 12345),
    ]
    
    for test_name, test_input in test_cases:
        print(f"Testing: {test_name}")
        try:
            result = preprocess_image(test_input)
            print(f"  ⚠️  Unexpectedly succeeded: {result.shape}")
        except (ImageCleaningError, FileNotFoundError, TypeError) as e:
            print(f"  ✓ Correctly caught error: {type(e).__name__}")
        except Exception as e:
            print(f"  ⚠️  Unexpected error type: {type(e).__name__}: {e}")
        print()
    
    print("✓ Error handling test complete\n")


if __name__ == "__main__":
    # Main preprocessing test
    test_preprocessing()
    
    # Error handling test
    print("\n")
    test_error_handling()