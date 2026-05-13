"""
Line Segmentation Visual Test

Tests the line segmentation algorithm and saves visual results:
1. Loads a manuscript image
2. Preprocesses it (cleaning)
3. Segments into individual lines
4. Saves each line as a separate image for inspection

This helps verify that line extraction is working correctly
before running OCR on the lines.
"""

import sys
from pathlib import Path
import cv2
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from preprocessing.image_cleaning import preprocess_image
from preprocessing.line_segmentation import segment_lines

# --- TEST CONFIGURATION ---
OUTPUT_DIR = "line_segmentation_output"  # Will be created in project root


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


def visualize_segmentation_overlay(original_image, line_images_info):
    """
    Create a visualization showing detected line boundaries on original image.
    
    Args:
        original_image: Original grayscale image
        line_images_info: List of (line_image, y_min, y_max) tuples
    
    Returns:
        Visualization image with colored line boundaries
    """
    # Convert to color for visualization
    viz = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
    
    # Draw rectangles around each detected line
    colors = [
        (255, 0, 0),    # Blue
        (0, 255, 0),    # Green
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
    ]
    
    for idx, (line_img, y_min, y_max) in enumerate(line_images_info):
        color = colors[idx % len(colors)]
        # Draw horizontal lines at top and bottom of detected line
        cv2.line(viz, (0, y_min), (viz.shape[1], y_min), color, 2)
        cv2.line(viz, (0, y_max), (viz.shape[1], y_max), color, 2)
        
        # Add line number label
        cv2.putText(
            viz, 
            f"Line {idx + 1}", 
            (10, y_min - 5), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6, 
            color, 
            2
        )
    
    return viz


def test_line_segmentation():
    """
    Run line segmentation test and save visual results.
    """
    print("="*60)
    print("LINE SEGMENTATION VISUAL TEST")
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
        print("2. Name it test.png, test2.png, or test3.jpg")
        print("3. Image should contain multiple lines of handwritten text\n")
        return
    
    print(f"✓ Found test image: {test_image_path}")
    print(f"  File size: {test_image_path.stat().st_size / 1024:.1f} KB\n")
    
    try:
        # Step 1: Load and preprocess
        print("Step 1: Loading and preprocessing image...")
        cleaned_image = preprocess_image(str(test_image_path))
        print(f"  ✓ Cleaned image: {cleaned_image.shape}\n")
        
        # Step 2: Segment lines
        print("Step 2: Segmenting lines...")
        line_images = segment_lines(cleaned_image)
        
        if not line_images:
            print("\n⚠️  No text lines detected!")
            print("Possible reasons:")
            print("  - Image is blank or has no text")
            print("  - Text is too faint (try increasing contrast)")
            print("  - Image resolution too low")
            print("  - Text lines too thin (adjust MIN_LINE_HEIGHT in line_segmentation.py)\n")
            return
        
        print(f"\n✓ Extracted {len(line_images)} text lines\n")
        
        # Step 3: Create output directory
        output_path = project_root / OUTPUT_DIR
        output_path.mkdir(exist_ok=True)
        print(f"Step 3: Saving results to: {output_path}")
        
        # Save original cleaned image
        original_output = output_path / "0_original_cleaned.png"
        cv2.imwrite(str(original_output), cleaned_image)
        print(f"  ✓ Saved: {original_output.name}")
        
        # Save each line with metadata
        line_info_list = []
        for idx, line_img in enumerate(line_images):
            line_filename = f"line_{idx + 1:02d}.png"
            line_output = output_path / line_filename
            cv2.imwrite(str(line_output), line_img)
            
            # Store info for visualization (approximate y position)
            # In real segmentation, we'd track exact positions - here we estimate
            line_info_list.append((line_img, 0, 0))  # Placeholder
            
            print(f"  ✓ Saved: {line_filename} ({line_img.shape[1]}x{line_img.shape[0]} px)")
        
        # Create summary visualization
        print("\nStep 4: Creating visualization...")
        
        # Create a montage of all lines
        if line_images:
            # Calculate dimensions for montage
            max_width = max(img.shape[1] for img in line_images)
            total_height = sum(img.shape[0] + 10 for img in line_images)  # 10px padding
            
            # Create white canvas
            montage = np.ones((total_height, max_width), dtype=np.uint8) * 255
            
            # Place each line
            y_offset = 0
            for idx, line_img in enumerate(line_images):
                h, w = line_img.shape
                # Center the line horizontally
                x_offset = (max_width - w) // 2
                montage[y_offset:y_offset+h, x_offset:x_offset+w] = line_img
                y_offset += h + 10  # Add padding
            
            montage_output = output_path / "lines_montage.png"
            cv2.imwrite(str(montage_output), montage)
            print(f"  ✓ Saved: {montage_output.name}")
        
        # Create summary text file
        summary_output = output_path / "segmentation_summary.txt"
        with open(summary_output, 'w', encoding='utf-8') as f:
            f.write("LINE SEGMENTATION TEST RESULTS\n")
            f.write("="*60 + "\n\n")
            f.write(f"Input Image: {test_image_path.name}\n")
            f.write(f"Original Size: {cleaned_image.shape[1]}x{cleaned_image.shape[0]} px\n\n")
            f.write(f"Lines Detected: {len(line_images)}\n\n")
            f.write("Line Details:\n")
            f.write("-"*60 + "\n")
            for idx, line_img in enumerate(line_images):
                f.write(f"Line {idx+1}: {line_img.shape[1]}x{line_img.shape[0]} px\n")
        
        print(f"  ✓ Saved: {summary_output.name}")
        
        print("\n" + "="*60)
        print("RESULTS SUMMARY")
        print("="*60)
        print(f"Input: {test_image_path.name}")
        print(f"Lines detected: {len(line_images)}")
        print(f"Output directory: {output_path}")
        print("\nFiles created:")
        print(f"  - 0_original_cleaned.png (preprocessed input)")
        print(f"  - line_01.png through line_{len(line_images):02d}.png (individual lines)")
        print(f"  - lines_montage.png (all lines stacked)")
        print(f"  - segmentation_summary.txt (metadata)")
        print("="*60 + "\n")
        
        print("✓ Test completed successfully!")
        print("\nNext steps:")
        print("1. Open the output directory to visually inspect the lines")
        print("2. Verify each line contains a single row of text")
        print("3. If segmentation looks good, proceed to OCR testing")
        print("4. Run: python test/ocr_test_lines.py\n")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        print("\nTest failed. Check the error above for details.\n")


if __name__ == "__main__":
    test_line_segmentation()