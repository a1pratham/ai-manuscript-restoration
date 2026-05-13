"""
DEPRECATED: Legacy Tesseract OCR Test

⚠️  WARNING: This test uses Tesseract OCR which is NOT used in the actual pipeline.

The manuscript reconstruction system uses TrOCR (Microsoft's Transformer-based OCR)
which is much better for handwriting recognition.

This file is kept for reference only.

--------------------------------------------------
FOR ACTUAL OCR TESTING, USE:
--------------------------------------------------
  python test/ocr_test_lines.py
--------------------------------------------------

This will use the correct TrOCR engine that the system actually uses.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def show_deprecation_notice():
    """Display deprecation notice and redirect users."""
    print("\n" + "="*60)
    print("⚠️  DEPRECATED TEST FILE")
    print("="*60)
    print("\nThis test uses Tesseract OCR, which is NOT part of")
    print("the manuscript reconstruction pipeline.")
    print("\nThe actual system uses:")
    print("  • TrOCR (Microsoft's Transformer-based OCR)")
    print("  • Much better for handwriting recognition")
    print("  • Already integrated in ocr/htr_engine.py")
    print("\n" + "="*60)
    print("TO TEST THE ACTUAL OCR PIPELINE:")
    print("="*60)
    print("\n1. Extract lines from manuscript:")
    print("   python test/line_test.py")
    print("\n2. Run TrOCR on extracted lines:")
    print("   python test/ocr_test_lines.py")
    print("\n3. Test full reconstruction:")
    print("   python test/gap_test.py")
    print("\n" + "="*60)
    print("\nIf you REALLY need Tesseract for some reason,")
    print("you can enable it by editing this file and")
    print("uncommenting the legacy code below.")
    print("="*60 + "\n")


# LEGACY TESSERACT CODE (COMMENTED OUT)
"""
import pytesseract
from PIL import Image

# Hardcoded Windows path (won't work on Mac/Linux)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def test_tesseract_ocr():
    '''
    Legacy Tesseract OCR test.
    
    NOTE: Tesseract is designed for PRINTED text, not handwriting.
    It will perform poorly on manuscript images.
    '''
    test_image_path = project_root / "test3.jpg"
    
    if not test_image_path.exists():
        print(f"Test image not found: {test_image_path}")
        return
    
    print(f"Testing Tesseract on: {test_image_path}")
    
    try:
        img = Image.open(test_image_path)
        text = pytesseract.image_to_string(img)
        
        print("Extracted text:")
        print("-" * 60)
        print(text)
        print("-" * 60)
        
    except Exception as e:
        print(f"ERROR: {e}")
        print("\nMake sure Tesseract is installed:")
        print("  pip install pytesseract")
        print("  Download from: https://github.com/tesseract-ocr/tesseract")

if __name__ == "__main__":
    show_deprecation_notice()
    
    # Uncomment to run legacy test
    # test_tesseract_ocr()
"""


if __name__ == "__main__":
    show_deprecation_notice()
    
    # Ask user if they want to see the legacy code
    print("\nWould you like to view the legacy Tesseract code? (y/n): ", end="")
    try:
        response = input().strip().lower()
        if response == 'y':
            print("\n" + "="*60)
            print("LEGACY TESSERACT CODE")
            print("="*60)
            print("""
# This code is NOT maintained and may not work

import pytesseract
from PIL import Image

# Platform-specific path (Windows only)
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

# For Mac: /usr/local/bin/tesseract
# For Linux: /usr/bin/tesseract

test_image = project_root / "test3.jpg"
if test_image.exists():
    img = Image.open(test_image)
    text = pytesseract.image_to_string(img)
    print(text)
else:
    print("Test image not found")

# NOTE: Tesseract performs poorly on handwriting!
# Use TrOCR instead (see ocr_test_lines.py)
""")
            print("="*60 + "\n")
    except (KeyboardInterrupt, EOFError):
        print("\n") 