import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from typing import List, Tuple
import numpy as np

# --- MODEL CONFIGURATION ---
TROCR_MODEL_NAME = "microsoft/trocr-base-handwritten"
# This is Microsoft's TrOCR model fine-tuned on handwritten text
# Alternatives: "microsoft/trocr-large-handwritten" (better but slower)

MIN_WORD_LENGTH = 1  # Minimum characters to consider as a valid word
MIN_CONFIDENCE = 0.0  # Minimum confidence to include word (0 = include all)

class HandwritingRecognitionEngine:
    """
    TrOCR-based handwriting recognition engine for manuscript images.
    
    Uses Microsoft's TrOCR (Transformer-based OCR) which combines:
    - Vision Transformer (ViT) encoder for image understanding
    - GPT-2 decoder for text generation
    
    The model is specifically trained on handwritten text datasets.
    """
    
    def __init__(self, model_name: str = TROCR_MODEL_NAME):
        """
        Initialize the handwriting recognition engine.
        
        Args:
            model_name: Hugging Face model identifier
        
        Raises:
            RuntimeError: If model fails to load
        """
        print(f"Initializing Handwriting Recognition Engine...")
        print(f"Model: {model_name}")
        
        # Auto-detect device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device: {self.device.upper()}")
        
        try:
            print("Loading TrOCR processor...")
            self.processor = TrOCRProcessor.from_pretrained(model_name)
            
            print("Loading TrOCR model...")
            self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            
            print("✓ Model loaded successfully\n")
            
        except Exception as e:
            raise RuntimeError(
                f"\n{'='*60}\n"
                f"ERROR: Failed to load TrOCR model\n"
                f"{'='*60}\n"
                f"Details: {str(e)}\n\n"
                f"Possible causes:\n"
                f"1. No internet connection (model needs to download first time)\n"
                f"2. Insufficient disk space (~1GB needed)\n"
                f"3. Missing transformers or torch dependencies\n\n"
                f"To fix:\n"
                f"1. Check internet connection\n"
                f"2. Run: pip install transformers torch pillow\n"
                f"3. Ensure ~1GB free disk space\n"
                f"{'='*60}\n"
            )
    
    def read_handwriting(
        self, 
        image: Image.Image, 
        return_confidence: bool = True
    ) -> Tuple[List[str], List[float]]:
        """
        Extract text and confidence scores from a handwritten line image.
        
        Process:
        1. Image → TrOCR Processor (resize, normalize)
        2. Vision Encoder → Image embeddings
        3. Text Decoder → Generated text + token probabilities
        4. Extract words and calculate per-word confidence
        
        Args:
            image: PIL Image of a single handwritten line
            return_confidence: If True, calculate confidence scores per word
        
        Returns:
            Tuple of:
            - words: List of extracted words
            - confidences: List of confidence scores (0-100) per word
        
        Example:
            >>> from PIL import Image
            >>> engine = HandwritingRecognitionEngine()
            >>> img = Image.open("manuscript_line.jpg")
            >>> words, conf = engine.read_handwriting(img)
            >>> print(words)
            ['The', 'king', 'ruled', 'wisely']
            >>> print(conf)
            [95.2, 87.3, 91.8, 88.5]
        """
        # Input validation
        if image is None:
            print("ERROR: Image is None")
            return [], []
        
        if not isinstance(image, Image.Image):
            print(f"ERROR: Expected PIL.Image, got {type(image)}")
            return [], []
        
        # Check if image is empty or too small
        if image.size[0] < 10 or image.size[1] < 10:
            print(f"WARNING: Image too small ({image.size}), likely empty")
            return [], []
        
        try:
            # Preprocess image for TrOCR
            pixel_values = self.processor(
                images=image, 
                return_tensors="pt"
            ).pixel_values.to(self.device)
            
            # Generate text with confidence scores
            with torch.no_grad():
                outputs = self.model.generate(
                    pixel_values,
                    output_scores=True,
                    return_dict_in_generate=True
                )
            
            # Decode the generated text
            generated_text = self.processor.batch_decode(
                outputs.sequences, 
                skip_special_tokens=True
            )[0]
            
            print(f"  Raw OCR output: '{generated_text}'")
            
            # Split into words
            words = generated_text.split()
            
            # Filter out very short words (likely noise)
            words = [w for w in words if len(w) >= MIN_WORD_LENGTH]
            
            if not words:
                print("  No valid words extracted")
                return [], []
            
            # Calculate per-word confidence if requested
            if return_confidence:
                confidences = self._calculate_word_confidences(
                    outputs, 
                    words,
                    generated_text
                )
            else:
                confidences = [100.0] * len(words)  # Default to perfect confidence
            
            return words, confidences
            
        except Exception as e:
            print(f"ERROR during handwriting recognition: {e}")
            return [], []
    
    def _calculate_word_confidences(
        self, 
        outputs, 
        words: List[str],
        full_text: str
    ) -> List[float]:
        """
        Calculate confidence score for each word based on token probabilities.
        
        Confidence Calculation:
        - Each word is made of multiple tokens (subwords)
        - We average the probability of all tokens in a word
        - Convert to percentage (0-100)
        
        Example:
        Word "wisely" might be tokenized as ["wise", "ly"]
        If "wise" has 0.92 probability and "ly" has 0.85 probability
        Word confidence = (0.92 + 0.85) / 2 * 100 = 88.5%
        
        Args:
            outputs: Model generation outputs with scores
            words: List of words extracted
            full_text: Complete generated text
        
        Returns:
            List of confidence scores (0-100) per word
        """
        if not hasattr(outputs, 'scores') or not outputs.scores:
            # No confidence scores available, return default
            print("  WARNING: No confidence scores available, using 100%")
            return [100.0] * len(words)
        
        try:
            # Get token probabilities
            # scores is a tuple of tensors, one per generation step
            token_probs = []
            for score_tensor in outputs.scores:
                # Apply softmax to get probabilities
                probs = torch.softmax(score_tensor, dim=-1)
                # Get max probability (the chosen token's probability)
                max_prob = probs.max().item()
                token_probs.append(max_prob)
            
            # Decode each token to map to words
            tokens = self.processor.tokenizer.convert_ids_to_tokens(
                outputs.sequences[0]
            )
            
            # Group token probabilities by word
            word_confidences = []
            current_word_idx = 0
            current_word_probs = []
            
            for token, prob in zip(tokens, token_probs):
                # Skip special tokens
                if token in ['<s>', '</s>', '<pad>']:
                    continue
                
                current_word_probs.append(prob)
                
                # Check if we've finished a word (space or end of tokens)
                if current_word_idx < len(words):
                    # Simple heuristic: if accumulated token length matches word
                    accumulated_text = self.processor.tokenizer.convert_tokens_to_string(
                        tokens[len(word_confidences):len(word_confidences) + len(current_word_probs)]
                    ).strip()
                    
                    if accumulated_text.endswith(words[current_word_idx]) or \
                       len(current_word_probs) > len(words[current_word_idx]):
                        # Calculate average confidence for this word
                        avg_prob = sum(current_word_probs) / len(current_word_probs)
                        word_confidences.append(avg_prob * 100.0)
                        current_word_probs = []
                        current_word_idx += 1
            
            # Handle any remaining words (fallback)
            while len(word_confidences) < len(words):
                if current_word_probs:
                    avg_prob = sum(current_word_probs) / len(current_word_probs)
                    word_confidences.append(avg_prob * 100.0)
                else:
                    word_confidences.append(50.0)  # Default moderate confidence
            
            return word_confidences[:len(words)]
            
        except Exception as e:
            print(f"  WARNING: Confidence calculation failed: {e}")
            print("  Using default confidence scores")
            return [75.0] * len(words)  # Fallback to moderate confidence


if __name__ == "__main__":
    print("="*60)
    print("HANDWRITING RECOGNITION ENGINE - TEST")
    print("="*60 + "\n")
    
    try:
        # Initialize engine
        engine = HandwritingRecognitionEngine()
        
        # Test with sample image - try multiple possible locations
        import os
        from pathlib import Path
        
        # Get the directory where this script is located
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
        
        # Try multiple possible locations
        possible_paths = [
            project_root / "test.png",           # Root level
            project_root / "test2.png",          # Alternative
            project_root / "test3.jpg",          # Alternative
            script_dir / "test.png",             # OCR folder
            Path("test.png"),                    # Current directory
            Path("../test.png"),                 # Up one level
        ]
        
        test_image_path = None
        for path in possible_paths:
            if path.exists():
                test_image_path = path
                print(f"✓ Found test image: {path}")
                break
        
        if test_image_path is None:
            print("❌ No test image found in expected locations:")
            for path in possible_paths:
                print(f"   - {path.absolute()}")
            print("\nTo test properly, place a handwritten text image at:")
            print(f"   {project_root / 'test.png'}")
            print("\nEngine initialized successfully!")
            
        else:
            print(f"\nLoading test image: {test_image_path}")
            test_image = Image.open(test_image_path).convert("RGB")
            
            print(f"Image size: {test_image.size}")
            print("\nRunning OCR...\n")
            
            words, confidences = engine.read_handwriting(test_image)
            
            print("\n" + "="*60)
            print("RESULTS:")
            print("="*60)
            print(f"Extracted {len(words)} words:\n")
            
            for word, conf in zip(words, confidences):
                print(f"  '{word}' - {conf:.1f}% confidence")
        
    except RuntimeError as e:
        print(str(e))
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()