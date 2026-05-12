import cv2
import numpy as np
from PIL import Image
from typing import Union

# --- IMAGE CLEANING PARAMETERS ---
BILATERAL_DIAMETER = 9      # Diameter of pixel neighborhood (higher = slower but smoother)
BILATERAL_SIGMA_COLOR = 75  # Filter sigma in color space (higher = more colors mixed)
BILATERAL_SIGMA_SPACE = 75  # Filter sigma in coordinate space (higher = farther pixels influence)

# Bilateral filter: Smooths image while preserving edges
# Perfect for manuscript images - removes noise but keeps text sharp

MIN_IMAGE_SIZE = 10  # Minimum width/height in pixels to be considered valid

class ImageCleaningError(Exception):
    """Custom exception for image cleaning failures."""
    pass

def preprocess_image(image: Union[Image.Image, np.ndarray, str]) -> np.ndarray:
    """
    Clean and prepare a manuscript image for OCR processing.
    
    Processing steps:
    1. Convert to numpy array if needed
    2. Convert to grayscale (if color)
    3. Apply bilateral filter (removes noise, preserves edges)
    4. Return as numpy array ready for line segmentation
    
    Args:
        image: Input image in one of these formats:
               - PIL.Image object
               - numpy array (grayscale or color)
               - File path string (will be loaded)
    
    Returns:
        Cleaned grayscale image as numpy array (H x W)
    
    Raises:
        ImageCleaningError: If image is invalid or processing fails
        FileNotFoundError: If path provided but file doesn't exist
    
    Example:
        >>> from PIL import Image
        >>> img = Image.open("manuscript.jpg")
        >>> cleaned = preprocess_image(img)
        >>> print(cleaned.shape)
        (800, 600)
    """
    # Handle different input types
    try:
        if image is None:
            raise ImageCleaningError("Image is None")
        
        # Handle string path
        if isinstance(image, str):
            import os
            if not os.path.exists(image):
                raise FileNotFoundError(f"Image file not found: {image}")
            
            # Load image using PIL for better compatibility
            pil_image = Image.open(image)
            image_array = np.array(pil_image)
            print(f"Loaded image from path: {image}")
        
        # Handle PIL Image
        elif isinstance(image, Image.Image):
            image_array = np.array(image)
            print(f"Converted PIL Image to array")
        
        # Handle numpy array
        elif isinstance(image, np.ndarray):
            image_array = image
            print(f"Using numpy array directly")
        
        else:
            raise ImageCleaningError(
                f"Unsupported image type: {type(image)}. "
                f"Expected PIL.Image, numpy.ndarray, or file path string."
            )
        
    except ImageCleaningError:
        raise
    except FileNotFoundError:
        raise
    except Exception as e:
        raise ImageCleaningError(f"Failed to load image: {str(e)}")
    
    # Validate image is not empty
    if image_array.size == 0:
        raise ImageCleaningError("Image is empty (size = 0)")
    
    if image_array.shape[0] < MIN_IMAGE_SIZE or image_array.shape[1] < MIN_IMAGE_SIZE:
        raise ImageCleaningError(
            f"Image too small: {image_array.shape}. "
            f"Minimum size: {MIN_IMAGE_SIZE}x{MIN_IMAGE_SIZE} pixels."
        )
    
    print(f"Image shape: {image_array.shape}")
    
    try:
        # Convert to grayscale if needed
        if len(image_array.shape) == 3:
            if image_array.shape[2] == 3:  # RGB
                gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
                print("Converted RGB to grayscale")
            elif image_array.shape[2] == 4:  # RGBA
                # Remove alpha channel first
                rgb = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
                gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                print("Converted RGBA to grayscale")
            else:
                raise ImageCleaningError(f"Unexpected number of channels: {image_array.shape[2]}")
        
        elif len(image_array.shape) == 2:
            # Already grayscale
            gray = image_array
            print("Image already grayscale")
        
        else:
            raise ImageCleaningError(f"Unexpected image dimensions: {image_array.shape}")
        
    except ImageCleaningError:
        raise
    except Exception as e:
        raise ImageCleaningError(f"Grayscale conversion failed: {str(e)}")
    
    try:
        # Apply bilateral filter for noise reduction while preserving edges
        # This is crucial for manuscript images:
        # - Removes paper texture and scan artifacts
        # - Preserves sharp text edges for better OCR
        # - Smooths ink inconsistencies (faded areas, smudges)
        
        print(f"Applying bilateral filter (d={BILATERAL_DIAMETER}, "
              f"sigmaColor={BILATERAL_SIGMA_COLOR}, sigmaSpace={BILATERAL_SIGMA_SPACE})...")
        
        filtered = cv2.bilateralFilter(
            gray,
            BILATERAL_DIAMETER,
            BILATERAL_SIGMA_COLOR,
            BILATERAL_SIGMA_SPACE
        )
        
        print("✓ Image preprocessing complete")
        return filtered
        
    except Exception as e:
        # If bilateral filter fails, return the grayscale image
        print(f"WARNING: Bilateral filter failed: {e}")
        print("Returning unfiltered grayscale image")
        return gray


if __name__ == "__main__":
    print("="*60)
    print("IMAGE CLEANING - TEST")
    print("="*60 + "\n")
    
    from pathlib import Path
    
    # Try to find test images
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
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
        print("\nDemonstrating with synthetic image instead...\n")
        
        # Create a synthetic test image
        print("Creating synthetic manuscript-like image...")
        synthetic = np.random.randint(200, 255, (400, 600), dtype=np.uint8)
        # Add some "text-like" dark regions
        synthetic[100:120, 50:200] = 50
        synthetic[200:220, 100:400] = 50
        
        print(f"Input: Synthetic image {synthetic.shape}")
        cleaned = preprocess_image(synthetic)
        print(f"Output: Cleaned image {cleaned.shape}")
        print(f"✓ Processing successful!")
        
    else:
        print(f"Testing with: {found_image}\n")
        
        try:
            # Test 1: Load from path string
            print("--- Test 1: Load from file path ---")
            cleaned1 = preprocess_image(str(found_image))
            print(f"✓ Result shape: {cleaned1.shape}\n")
            
            # Test 2: Load as PIL Image
            print("--- Test 2: Load as PIL Image ---")
            pil_img = Image.open(found_image)
            cleaned2 = preprocess_image(pil_img)
            print(f"✓ Result shape: {cleaned2.shape}\n")
            
            # Test 3: Load as numpy array
            print("--- Test 3: Load as numpy array ---")
            np_img = np.array(Image.open(found_image))
            cleaned3 = preprocess_image(np_img)
            print(f"✓ Result shape: {cleaned3.shape}\n")
            
            print("="*60)
            print("All tests passed! ✓")
            print("="*60)
            
        except ImageCleaningError as e:
            print(f"ERROR: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")
            import traceback
            traceback.print_exc()
    
    # Test error handling
    print("\n--- Test 4: Error handling ---")
    try:
        preprocess_image(None)
    except ImageCleaningError as e:
        print(f"✓ Correctly caught None input: {e}")
    
    try:
        preprocess_image("nonexistent_file.jpg")
    except FileNotFoundError as e:
        print(f"✓ Correctly caught missing file: {e}")