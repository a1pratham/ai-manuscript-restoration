import cv2
import numpy as np
from typing import List, Tuple
from PIL import Image

# --- LINE SEGMENTATION PARAMETERS ---

MIN_LINE_HEIGHT = 20  
# Minimum pixel height for a text line
# Filters out noise and tiny artifacts
# Too low: captures noise specks
# Too high: misses small handwriting

MIN_LINE_WIDTH = 5
# Minimum pixel width for a valid line region
# Filters out vertical artifacts (margin lines, creases)
# Too low: captures noise
# Too high: misses short words or fragments

WORD_CHAIN_MAX_DISTANCE = 50
# Maximum horizontal gap (pixels) to chain words into same line
# Handwriting is often slanted - words on same baseline may have vertical offset
# This allows "chaining" words that are slightly misaligned vertically
# Too low: breaks cursive lines into fragments
# Too high: merges adjacent lines

VERTICAL_OVERLAP_THRESHOLD = 0.3
# How much vertical overlap (0.0-1.0) needed to consider words on same line
# 0.3 means if 30% of word heights overlap, they're probably same line
# Accounts for ascenders/descenders (tall letters like 'h', 'g')

class LineSegmentationError(Exception):
    """Custom exception for line segmentation failures."""
    pass

def segment_lines(image: np.ndarray) -> List[np.ndarray]:
    """
    Extract individual text lines from a full manuscript page image.
    
    Algorithm:
    1. Binarize image (black text on white background)
    2. Find connected components (word-level blobs)
    3. Chain nearby words into lines using spatial proximity
    4. Extract each line as a separate image
    
    Why chaining?
    - Simple horizontal projection fails on slanted cursive writing
    - Words on same line may have different y-coordinates
    - We need to group words that "flow together" horizontally
    
    Args:
        image: Grayscale numpy array (H x W) from preprocess_image()
    
    Returns:
        List of line images (each is a numpy array)
        Ordered top-to-bottom as they appear in original
    
    Raises:
        LineSegmentationError: If image is invalid or processing fails
    
    Example:
        >>> cleaned = preprocess_image("manuscript.jpg")
        >>> lines = segment_lines(cleaned)
        >>> print(f"Found {len(lines)} lines")
        Found 12 lines
    """
    # Input validation
    if image is None:
        raise LineSegmentationError("Image is None")
    
    if not isinstance(image, np.ndarray):
        raise LineSegmentationError(
            f"Expected numpy array, got {type(image)}"
        )
    
    if image.size == 0:
        raise LineSegmentationError("Image is empty")
    
    if len(image.shape) != 2:
        raise LineSegmentationError(
            f"Expected grayscale image (2D), got shape {image.shape}"
        )
    
    print(f"Segmenting lines from image: {image.shape}")
    
    try:
        # Step 1: Binarize the image (Otsu's method auto-finds threshold)
        # Result: Pure black text on pure white background
        _, binary = cv2.threshold(
            image, 
            0, 
            255, 
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        print(f"  Binarized image using Otsu's threshold")
        
    except Exception as e:
        raise LineSegmentationError(f"Binarization failed: {e}")
    
    try:
        # Step 2: Find connected components (individual "blobs" of text)
        # Each blob is typically a word or character cluster
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, 
            connectivity=8
        )
        
        print(f"  Found {num_labels - 1} connected components")
        
        if num_labels <= 1:  # Only background, no text
            print("  WARNING: No text detected in image")
            return []
        
    except Exception as e:
        raise LineSegmentationError(f"Connected components analysis failed: {e}")
    
    # Step 3: Extract valid word boxes (filter noise)
    word_boxes = []
    
    for i in range(1, num_labels):  # Skip label 0 (background)
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        
        # Filter by size thresholds
        if h >= MIN_LINE_HEIGHT and w >= MIN_LINE_WIDTH:
            word_boxes.append({
                'x': x,
                'y': y,
                'width': w,
                'height': h,
                'center_x': x + w // 2,
                'center_y': y + h // 2
            })
    
    print(f"  Filtered to {len(word_boxes)} valid word boxes")
    
    if not word_boxes:
        print("  WARNING: No valid text regions found after filtering")
        return []
    
    # Step 4: Chain words into lines
    # Sort by vertical position first
    word_boxes.sort(key=lambda b: b['y'])
    
    lines = []
    used = set()  # Track which words have been assigned to a line
    
    for i, box in enumerate(word_boxes):
        if i in used:
            continue
        
        # Start a new line with this word
        current_line = [box]
        used.add(i)
        
        # Try to chain other words to this line
        # Keep extending rightward, allowing for slant
        changed = True
        while changed:
            changed = False
            
            # Find the rightmost word in current line
            rightmost = max(current_line, key=lambda b: b['x'] + b['width'])
            
            # Look for nearby words to the right
            for j, candidate in enumerate(word_boxes):
                if j in used:
                    continue
                
                # Check horizontal proximity
                horizontal_gap = candidate['x'] - (rightmost['x'] + rightmost['width'])
                
                if horizontal_gap > WORD_CHAIN_MAX_DISTANCE:
                    continue  # Too far away
                
                # Check vertical alignment (accounting for slant)
                # Do their vertical ranges overlap?
                line_top = min(b['y'] for b in current_line)
                line_bottom = max(b['y'] + b['height'] for b in current_line)
                line_height = line_bottom - line_top
                
                candidate_top = candidate['y']
                candidate_bottom = candidate['y'] + candidate['height']
                
                # Calculate overlap
                overlap_top = max(line_top, candidate_top)
                overlap_bottom = min(line_bottom, candidate_bottom)
                overlap = max(0, overlap_bottom - overlap_top)
                
                overlap_ratio = overlap / line_height if line_height > 0 else 0
                
                if overlap_ratio >= VERTICAL_OVERLAP_THRESHOLD:
                    # This word belongs to the same line!
                    current_line.append(candidate)
                    used.add(j)
                    changed = True
                    break  # Restart search from new rightmost word
        
        lines.append(current_line)
    
    print(f"  Grouped words into {len(lines)} lines")
    
    # Step 5: Extract line images
    line_images = []
    
    for line_idx, line_boxes in enumerate(lines):
        # Get bounding box for entire line
        x_min = min(b['x'] for b in line_boxes)
        y_min = min(b['y'] for b in line_boxes)
        x_max = max(b['x'] + b['width'] for b in line_boxes)
        y_max = max(b['y'] + b['height'] for b in line_boxes)
        
        # Add small padding
        padding = 5
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(image.shape[1], x_max + padding)
        y_max = min(image.shape[0], y_max + padding)
        
        # Extract line image
        line_img = image[y_min:y_max, x_min:x_max]
        line_images.append(line_img)
        
        print(f"  Line {line_idx + 1}: {len(line_boxes)} words, "
              f"position y={y_min}-{y_max}, size {line_img.shape}")
    
    print(f"✓ Segmentation complete: {len(line_images)} lines extracted\n")
    
    return line_images


if __name__ == "__main__":
    print("="*60)
    print("LINE SEGMENTATION - TEST")
    print("="*60 + "\n")
    
    from pathlib import Path
    import sys
    
    # Add parent directory to path for imports
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    sys.path.insert(0, str(project_root))
    
    from preprocessing.image_cleaning import preprocess_image
    
    # Try to find test images
    test_images = [
        project_root / "test.png",
        project_root / "test2.png",
        project_root / "test2.jpg",
        project_root / "test3.jpg",
    ]
    
    found_image = None
    for img_path in test_images:
        if img_path.exists():
            found_image = img_path
            break
    
    if found_image is None:
        print("❌ No test images found. Tested locations:")
        for path in test_images:
            print(f"   - {path}")
        print("\nTo test properly, place a manuscript image at:")
        print(f"   {project_root / 'test.png'}")
        sys.exit(1)
    
    print(f"Testing with: {found_image}\n")
    
    try:
        # Step 1: Clean the image
        print("Step 1: Preprocessing image...")
        cleaned = preprocess_image(str(found_image))
        print(f"✓ Cleaned image: {cleaned.shape}\n")
        
        # Step 2: Segment lines
        print("Step 2: Segmenting lines...")
        line_images = segment_lines(cleaned)
        
        print("\n" + "="*60)
        print("RESULTS:")
        print("="*60)
        print(f"Extracted {len(line_images)} text lines\n")
        
        for i, line_img in enumerate(line_images):
            print(f"Line {i+1}: {line_img.shape[1]}x{line_img.shape[0]} pixels")
        
        # Optional: Save line images for visual inspection
        print("\nSaving line images for inspection...")
        output_dir = project_root / "line_outputs"
        output_dir.mkdir(exist_ok=True)
        
        for i, line_img in enumerate(line_images):
            output_path = output_dir / f"line_{i+1}.png"
            cv2.imwrite(str(output_path), line_img)
        
        print(f"✓ Saved {len(line_images)} line images to: {output_dir}")
        print("\nYou can now visually inspect the extracted lines!")
        
    except LineSegmentationError as e:
        print(f"ERROR: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()