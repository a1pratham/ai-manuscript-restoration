"""
Manuscript Reconstruction API Server

FastAPI backend that provides endpoints for:
1. Health check
2. Upload and reconstruct manuscript images
3. Full pipeline: preprocessing → line segmentation → OCR → gap detection → BERT reconstruction

Usage:
    uvicorn server:app --reload
    
Then access:
    http://localhost:8000/docs - Interactive API documentation
    http://localhost:8000/health - Health check
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import io
import sys
from pathlib import Path
import logging
from typing import List, Dict, Any
import traceback

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from preprocessing.line_segmentation import segment_lines, LineSegmentationError
from preprocessing.image_cleaning import preprocess_image, ImageCleaningError
from ocr.htr_engine import HandwritingRecognitionEngine
from ocr.gap_detection import format_for_bert, analyze_gap_distribution
from ml_model.reconstruction_engine import ReconstructionEngine

# --- CONFIGURATION ---
MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
MIN_IMAGE_DIMENSION = 100  # Minimum width or height in pixels
MAX_IMAGE_DIMENSION = 4000  # Maximum to prevent memory issues

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- FASTAPI APP ---
app = FastAPI(
    title="Manuscript Reconstruction API",
    description="AI-powered system for reconstructing damaged historical manuscripts",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# --- CORS MIDDLEWARE ---
# Allow requests from web frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- GLOBAL STATE ---
# Load models once at startup for efficiency
ocr_engine = None
reconstruction_engine = None
models_loaded = False
model_load_error = None


@app.on_event("startup")
async def startup_event():
    """
    Initialize models when server starts.
    """
    global ocr_engine, reconstruction_engine, models_loaded, model_load_error
    
    logger.info("="*60)
    logger.info("MANUSCRIPT RECONSTRUCTION API - STARTING")
    logger.info("="*60)
    
    try:
        # Load TrOCR engine
        logger.info("Loading TrOCR handwriting recognition engine...")
        ocr_engine = HandwritingRecognitionEngine()
        logger.info("✓ TrOCR engine loaded")
        
        # Load BERT reconstruction engine
        logger.info("Loading BERT reconstruction engine...")
        reconstruction_engine = ReconstructionEngine()
        logger.info("✓ BERT engine loaded")
        
        models_loaded = True
        logger.info("="*60)
        logger.info("✓ ALL MODELS LOADED - API READY")
        logger.info("="*60)
        
    except FileNotFoundError as e:
        model_load_error = str(e)
        logger.error("="*60)
        logger.error("❌ MODEL LOADING FAILED")
        logger.error("="*60)
        logger.error(model_load_error)
        logger.error("="*60)
        logger.warning("API will start but /api/upload-and-reconstruct will fail")
        logger.warning("Please train the BERT model first: python reconstruction/train_model.py")
        
    except Exception as e:
        model_load_error = f"Unexpected error: {str(e)}"
        logger.error("="*60)
        logger.error("❌ MODEL LOADING FAILED")
        logger.error("="*60)
        logger.error(model_load_error)
        logger.error(traceback.format_exc())
        logger.error("="*60)


@app.get("/")
async def root():
    """
    Root endpoint with API information.
    """
    return {
        "message": "Manuscript Reconstruction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "reconstruct": "/api/upload-and-reconstruct"
        }
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    
    Returns system status and model availability.
    """
    return {
        "status": "healthy" if models_loaded else "degraded",
        "models_loaded": models_loaded,
        "ocr_engine": "loaded" if ocr_engine is not None else "not loaded",
        "reconstruction_engine": "loaded" if reconstruction_engine is not None else "not loaded",
        "error": model_load_error if model_load_error else None
    }


def validate_image(image: Image.Image, filename: str):
    """
    Validate uploaded image meets requirements.
    
    Args:
        image: PIL Image object
        filename: Original filename
    
    Raises:
        HTTPException: If image is invalid
    """
    # Check file extension
    file_ext = Path(filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}. "
                   f"Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    # Check dimensions
    width, height = image.size
    
    if width < MIN_IMAGE_DIMENSION or height < MIN_IMAGE_DIMENSION:
        raise HTTPException(
            status_code=400,
            detail=f"Image too small: {width}x{height}px. "
                   f"Minimum: {MIN_IMAGE_DIMENSION}x{MIN_IMAGE_DIMENSION}px"
        )
    
    if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
        raise HTTPException(
            status_code=400,
            detail=f"Image too large: {width}x{height}px. "
                   f"Maximum: {MAX_IMAGE_DIMENSION}x{MAX_IMAGE_DIMENSION}px. "
                   f"Please resize before uploading."
        )
    
    logger.info(f"Image validated: {filename} ({width}x{height}px)")


@app.post("/api/upload-and-reconstruct")
async def upload_and_reconstruct(file: UploadFile = File(...)):
    """
    Upload a manuscript image and reconstruct damaged text.
    
    Pipeline:
    1. Preprocess image (cleaning, grayscale conversion)
    2. Segment into individual text lines
    3. Run TrOCR on each line to extract words + confidence
    4. Detect gaps (low confidence words → [MASK])
    5. BERT reconstructs missing words
    
    Args:
        file: Uploaded image file (JPEG, PNG, etc.)
    
    Returns:
        JSON with:
        - lines: List of reconstructed lines with predictions
        - statistics: Overall quality metrics
        - status: Success/error status
    
    Example Response:
    {
        "status": "success",
        "lines": [
            {
                "line_number": 1,
                "original_ocr": "The k_ng ruled wisely",
                "masked_text": "The [MASK] ruled wisely",
                "reconstructed": "The king ruled wisely",
                "predictions": [...]
            }
        ],
        "statistics": {
            "total_lines": 3,
            "total_words": 24,
            "words_reconstructed": 5,
            "avg_confidence": 78.5
        }
    }
    """
    # Check if models are loaded
    if not models_loaded:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. " + (model_load_error or "Unknown error")
        )
    
    logger.info(f"Received upload: {file.filename}")
    
    try:
        # Read uploaded file
        contents = await file.read()
        
        # Check file size
        if len(contents) > MAX_UPLOAD_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large: {len(contents)/1024/1024:.1f}MB. "
                       f"Maximum: {MAX_UPLOAD_SIZE/1024/1024:.0f}MB"
            )
        
        logger.info(f"File size: {len(contents)/1024:.1f} KB")
        
        # Load image
        try:
            image = Image.open(io.BytesIO(contents))
            validate_image(image, file.filename)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to load image: {str(e)}"
            )
        
        # Step 1: Preprocess image
        logger.info("Step 1: Preprocessing image...")
        try:
            cleaned_image = preprocess_image(image)
            logger.info(f"  ✓ Preprocessed: {cleaned_image.shape}")
        except ImageCleaningError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Image preprocessing failed: {str(e)}"
            )
        
        # Step 2: Segment lines
        logger.info("Step 2: Segmenting lines...")
        try:
            line_images = segment_lines(cleaned_image)
            
            if not line_images:
                logger.warning("No text lines detected")
                return JSONResponse(
                    status_code=200,
                    content={
                        "status": "success",
                        "message": "No text detected in image",
                        "lines": [],
                        "statistics": {
                            "total_lines": 0,
                            "total_words": 0,
                            "words_reconstructed": 0
                        }
                    }
                )
            
            logger.info(f"  ✓ Extracted {len(line_images)} lines")
            
        except LineSegmentationError as e:
            raise HTTPException(
                status_code=500,
                detail=f"Line segmentation failed: {str(e)}"
            )
        
        # Process each line
        results = []
        all_confidences = []
        total_words = 0
        total_reconstructed = 0
        
        logger.info("Step 3: Processing each line...")
        
        for line_idx, line_img in enumerate(line_images):
            logger.info(f"  Processing line {line_idx + 1}/{len(line_images)}...")
            
            try:
                # Step 3: Run TrOCR
                pil_line = Image.fromarray(line_img)
                words, confidences = ocr_engine.read_handwriting(pil_line)
                
                if not words:
                    logger.warning(f"    No words extracted from line {line_idx + 1}")
                    results.append({
                        "line_number": line_idx + 1,
                        "status": "no_text",
                        "original_ocr": "",
                        "masked_text": "",
                        "reconstructed": "",
                        "predictions": []
                    })
                    continue
                
                total_words += len(words)
                all_confidences.extend(confidences)
                
                original_text = " ".join(words)
                logger.info(f"    OCR: '{original_text}'")
                
                # Step 4: Detect gaps
                masked_text, confidence_map = format_for_bert(words, confidences)
                logger.info(f"    Masked: '{masked_text}'")
                
                # Step 5: Reconstruct with BERT (if needed)
                if '[MASK]' in masked_text:
                    reconstruction_result = reconstruction_engine.reconstruct_for_api(
                        masked_text,
                        confidence_map
                    )
                    
                    reconstructed_text = reconstruction_result['reconstructed_text']
                    predictions = reconstruction_result['all_predictions']
                    total_reconstructed += len(predictions)
                    
                    logger.info(f"    Reconstructed: '{reconstructed_text}'")
                else:
                    reconstructed_text = original_text
                    predictions = []
                    logger.info(f"    No reconstruction needed (high confidence)")
                
                results.append({
                    "line_number": line_idx + 1,
                    "status": "success",
                    "original_ocr": original_text,
                    "word_confidences": [
                        {"word": w, "confidence": c}
                        for w, c in zip(words, confidences)
                    ],
                    "masked_text": masked_text,
                    "reconstructed": reconstructed_text,
                    "predictions": predictions
                })
                
            except Exception as e:
                logger.error(f"    Error processing line {line_idx + 1}: {e}")
                results.append({
                    "line_number": line_idx + 1,
                    "status": "error",
                    "error": str(e),
                    "original_ocr": "",
                    "masked_text": "",
                    "reconstructed": "",
                    "predictions": []
                })
        
        # Calculate overall statistics
        quality_stats = analyze_gap_distribution(all_confidences) if all_confidences else {}
        
        statistics = {
            "total_lines": len(line_images),
            "successful_lines": sum(1 for r in results if r['status'] == 'success'),
            "total_words": total_words,
            "words_reconstructed": total_reconstructed,
            "avg_confidence": quality_stats.get('avg_confidence', 0),
            "quality_rating": quality_stats.get('quality_rating', 'Unknown')
        }
        
        logger.info("="*60)
        logger.info("✓ RECONSTRUCTION COMPLETE")
        logger.info(f"  Lines: {statistics['total_lines']}")
        logger.info(f"  Words: {statistics['total_words']}")
        logger.info(f"  Reconstructed: {statistics['words_reconstructed']}")
        logger.info(f"  Quality: {statistics['quality_rating']}")
        logger.info("="*60)
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "lines": results,
                "statistics": statistics
            }
        )
        
    except HTTPException:
        raise
        
    except Exception as e:
        logger.error("="*60)
        logger.error("❌ RECONSTRUCTION FAILED")
        logger.error("="*60)
        logger.error(traceback.format_exc())
        
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*60)
    print("STARTING MANUSCRIPT RECONSTRUCTION API")
    print("="*60)
    print("\nServer will start at: http://localhost:8000")
    print("API Documentation: http://localhost:8000/docs")
    print("Health Check: http://localhost:8000/health")
    print("\nPress Ctrl+C to stop")
    print("="*60 + "\n")
    
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )