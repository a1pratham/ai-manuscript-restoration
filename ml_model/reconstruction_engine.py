import os
import torch
from transformers import pipeline

# --- MODEL CONFIGURATION ---
DEFAULT_MODEL_PATH = "../models/manuscript-bert-final"
BERT_WEIGHT = 0.6  # Weight for BERT's linguistic confidence
OCR_WEIGHT = 0.4   # Weight for OCR's physical confidence
TOP_K_PREDICTIONS = 3  # Number of alternative predictions to return

class ReconstructionEngine:
    """
    BERT-based reconstruction engine for filling gaps in damaged manuscript text.
    
    Combines:
    - BERT's linguistic predictions (context-aware word suggestions)
    - OCR confidence scores (how legible the original was)
    
    Blending formula:
    final_score = (BERT_WEIGHT × BERT_probability) + (OCR_WEIGHT × OCR_confidence)
    """
    
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        """
        Initialize the reconstruction engine.
        
        Args:
            model_path: Path to fine-tuned BERT model directory
        
        Raises:
            FileNotFoundError: If model directory doesn't exist
            RuntimeError: If model fails to load
        """
        # Validate model exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"\n{'='*60}\n"
                f"ERROR: Model not found at '{model_path}'\n"
                f"{'='*60}\n"
                f"The BERT model must be trained first.\n\n"
                f"To fix this:\n"
                f"1. Run: python reconstruction/train_model.py\n"
                f"2. Wait for training to complete (~1-2 hours)\n"
                f"3. Model will be saved to: {model_path}\n"
                f"{'='*60}\n"
            )
        
        # Auto-detect device (GPU if available, else CPU)
        self.device = 0 if torch.cuda.is_available() else -1
        device_name = "GPU" if self.device == 0 else "CPU"
        
        print(f"Loading reconstruction model from: {model_path}")
        print(f"Using device: {device_name}")
        
        try:
            self.fill_mask_pipeline = pipeline(
                "fill-mask",
                model=model_path,
                device=self.device,
                top_k=TOP_K_PREDICTIONS
            )
            print("✓ Model loaded successfully\n")
            
        except Exception as e:
            raise RuntimeError(
                f"Failed to load model from '{model_path}': {str(e)}\n"
                f"The model files may be corrupted. Try retraining."
            )
    
    def reconstruct_with_blend(
        self, 
        masked_text: str, 
        ocr_confidence_map: dict
    ) -> dict:
        """
        Reconstruct text by filling [MASK] tokens with context-aware predictions.
        
        Args:
            masked_text: Input text with [MASK] tokens (e.g., "The [MASK] ruled wisely")
            ocr_confidence_map: Dictionary mapping mask positions to OCR confidence
                               Example: {0: 0.35, 1: 0.42} for first and second masks
        
        Returns:
            Dictionary with:
            - 'reconstructed_text': Best guess with masks filled
            - 'all_predictions': List of {position, word, score, alternatives} for each mask
        """
        if not masked_text or not masked_text.strip():
            return {
                'reconstructed_text': '',
                'all_predictions': []
            }
        
        if '[MASK]' not in masked_text:
            return {
                'reconstructed_text': masked_text,
                'all_predictions': []
            }
        
        print(f"Input: {masked_text}")
        print(f"OCR confidence map: {ocr_confidence_map}")
        
        try:
            # BERT predicts all masks at once
            predictions = self.fill_mask_pipeline(masked_text)
            
        except Exception as e:
            print(f"ERROR during BERT prediction: {e}")
            return {
                'reconstructed_text': masked_text.replace('[MASK]', '[ERROR]'),
                'all_predictions': []
            }
        
        # Handle single mask vs multiple masks
        if not isinstance(predictions[0], list):
            predictions = [predictions]
        
        final_text = masked_text
        all_results = []
        
        for mask_idx, mask_predictions in enumerate(predictions):
            # Get OCR confidence for this mask position
            ocr_conf = ocr_confidence_map.get(mask_idx, 0.0)
            
            # Blend BERT scores with OCR confidence
            blended_predictions = []
            
            for pred in mask_predictions:
                bert_score = pred['score']
                
                # Blended score = weighted combination
                blended_score = (BERT_WEIGHT * bert_score) + (OCR_WEIGHT * ocr_conf)
                
                blended_predictions.append({
                    'word': pred['token_str'].strip(),
                    'bert_score': bert_score,
                    'ocr_confidence': ocr_conf,
                    'final_score': blended_score
                })
            
            # Sort by final blended score
            blended_predictions.sort(key=lambda x: x['final_score'], reverse=True)
            
            # Use top prediction to fill the mask
            best_word = blended_predictions[0]['word']
            final_text = final_text.replace('[MASK]', best_word, 1)
            
            # Store all predictions for this mask
            all_results.append({
                'position': mask_idx,
                'chosen_word': best_word,
                'final_score': blended_predictions[0]['final_score'],
                'alternatives': blended_predictions[:TOP_K_PREDICTIONS]
            })
            
            print(f"\nMask {mask_idx}: Chose '{best_word}' (score: {blended_predictions[0]['final_score']:.3f})")
            print(f"  Alternatives: {[p['word'] for p in blended_predictions[:TOP_K_PREDICTIONS]]}")
        
        return {
            'reconstructed_text': final_text,
            'all_predictions': all_results
        }
    
    def reconstruct_for_api(
        self, 
        masked_text: str, 
        ocr_confidence_map: dict
    ) -> dict:
        """
        API-friendly wrapper for reconstruction.
        
        Args:
            masked_text: Input text with [MASK] tokens
            ocr_confidence_map: Dictionary mapping mask positions to OCR confidence
        
        Returns:
            Dictionary ready for JSON serialization
        """
        return self.reconstruct_with_blend(masked_text, ocr_confidence_map)


if __name__ == "__main__":
    print("="*60)
    print("MANUSCRIPT RECONSTRUCTION ENGINE - TEST")
    print("="*60 + "\n")
    
    try:
        # Initialize engine
        engine = ReconstructionEngine()
        
        # Test case: damaged historical text
        test_input = "The [MASK] of England [MASK] the treaty in 1215."
        test_ocr_map = {
            0: 0.35,  # First mask has low OCR confidence
            1: 0.72   # Second mask has decent OCR confidence
        }
        
        print("TEST INPUT:")
        print(f"  Text: {test_input}")
        print(f"  OCR Map: {test_ocr_map}\n")
        
        # Reconstruct
        result = engine.reconstruct_for_api(test_input, test_ocr_map)
        
        print("\n" + "="*60)
        print("FINAL RESULT:")
        print("="*60)
        print(f"Reconstructed: {result['reconstructed_text']}")
        
        print("\nDetailed predictions:")
        for pred in result['all_predictions']:
            print(f"\n  Position {pred['position']}:")
            print(f"    Chosen: '{pred['chosen_word']}' (score: {pred['final_score']:.3f})")
            print(f"    Top 3: {[alt['word'] for alt in pred['alternatives']]}")
        
    except FileNotFoundError as e:
        print(str(e))
    except Exception as e:
        print(f"Unexpected error: {e}")