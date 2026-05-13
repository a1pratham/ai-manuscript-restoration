import os
import torch
from datasets import load_dataset
from transformers import (
    BertTokenizer, 
    BertForMaskedLM, 
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling
)

# --- TRAINING CONFIGURATION ---

# Dataset
DATASET_PATH = "../datasets/synthetic_training_data.csv"
TRAIN_TEST_SPLIT = 0.1  # 10% for validation, 90% for training

# Model
BASE_MODEL_NAME = "bert-base-uncased"  # Pre-trained BERT model to fine-tune
OUTPUT_MODEL_PATH = "../models/manuscript-bert-final"

# Training Hyperparameters
BATCH_SIZE = 16            # Number of samples per training step (adjust based on GPU memory)
LEARNING_RATE = 5e-5       # How fast the model learns (too high = unstable, too low = slow)
NUM_EPOCHS = 3             # How many times to go through entire dataset
WEIGHT_DECAY = 0.01        # Regularization to prevent overfitting
WARMUP_STEPS = 500         # Gradual learning rate increase at start
LOGGING_STEPS = 100        # Log training metrics every N steps
SAVE_STEPS = 1000          # Save checkpoint every N steps
EVAL_STEPS = 500           # Evaluate on validation set every N steps

# Tokenization
MAX_SEQUENCE_LENGTH = 128  # Maximum tokens per input (longer = more memory)
MLM_PROBABILITY = 0.15     # Probability of masking tokens (BERT standard)

# Hardware
USE_GPU_IF_AVAILABLE = True


class TrainingError(Exception):
    """Custom exception for training failures."""
    pass


def validate_environment():
    """
    Check that all required files and directories exist before training.
    
    Raises:
        TrainingError: If dataset or directories are missing
    """
    print("Validating training environment...")
    
    # Check dataset exists
    if not os.path.exists(DATASET_PATH):
        raise TrainingError(
            f"\n{'='*60}\n"
            f"ERROR: Training dataset not found\n"
            f"{'='*60}\n"
            f"Expected location: {DATASET_PATH}\n\n"
            f"You need to generate the training dataset first:\n"
            f"1. cd dataset_generation\n"
            f"2. python build_dataset.py\n"
            f"3. Wait for 500K sentence pairs to be generated\n"
            f"4. Then run this training script\n"
            f"{'='*60}\n"
        )
    
    print(f"  ✓ Dataset found: {DATASET_PATH}")
    
    # Check output directory exists, create if not
    output_dir = os.path.dirname(OUTPUT_MODEL_PATH)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"  ✓ Created output directory: {output_dir}")
    else:
        print(f"  ✓ Output directory exists: {output_dir}")
    
    # Check GPU availability
    if torch.cuda.is_available() and USE_GPU_IF_AVAILABLE:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  ✓ GPU available: {gpu_name} ({gpu_memory:.1f}GB)")
        device = "cuda"
    else:
        print(f"  ⚠ No GPU detected - training will use CPU (much slower!)")
        print(f"    Expected training time: ~3-6 hours on CPU vs ~30-60 min on GPU")
        device = "cpu"
    
    print(f"  ✓ Device: {device.upper()}\n")
    
    return device


def load_and_prepare_dataset():
    """
    Load training dataset and prepare for BERT fine-tuning.
    
    Returns:
        Tuple of (train_dataset, eval_dataset, tokenizer)
    
    Raises:
        TrainingError: If dataset loading fails
    """
    print("="*60)
    print("LOADING DATASET")
    print("="*60 + "\n")
    
    try:
        # Load CSV dataset
        print(f"Loading dataset from: {DATASET_PATH}")
        dataset = load_dataset("csv", data_files=DATASET_PATH, split="train")
        print(f"  ✓ Loaded {len(dataset)} training pairs\n")
        
    except Exception as e:
        raise TrainingError(f"Failed to load dataset: {e}")
    
    # Split into train/validation
    print(f"Splitting dataset: {int((1-TRAIN_TEST_SPLIT)*100)}% train, {int(TRAIN_TEST_SPLIT*100)}% validation")
    dataset = dataset.train_test_split(test_size=TRAIN_TEST_SPLIT)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]
    
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Validation: {len(eval_dataset)} samples\n")
    
    # Load tokenizer
    print(f"Loading BERT tokenizer: {BASE_MODEL_NAME}")
    try:
        tokenizer = BertTokenizer.from_pretrained(BASE_MODEL_NAME)
        print(f"  ✓ Tokenizer loaded (vocab size: {tokenizer.vocab_size})\n")
    except Exception as e:
        raise TrainingError(f"Failed to load tokenizer: {e}")
    
    # Tokenize datasets
    print("Tokenizing datasets (this may take a few minutes)...")
    
    def tokenize_function(examples):
        """Tokenize the 'input' field (damaged text) for MLM training."""
        return tokenizer(
            examples["input"],
            padding="max_length",
            truncation=True,
            max_length=MAX_SEQUENCE_LENGTH
        )
    
    try:
        train_dataset = train_dataset.map(
            tokenize_function, 
            batched=True,
            remove_columns=train_dataset.column_names
        )
        eval_dataset = eval_dataset.map(
            tokenize_function, 
            batched=True,
            remove_columns=eval_dataset.column_names
        )
        print(f"  ✓ Tokenization complete\n")
        
    except Exception as e:
        raise TrainingError(f"Tokenization failed: {e}")
    
    return train_dataset, eval_dataset, tokenizer


def train_model(train_dataset, eval_dataset, tokenizer, device):
    """
    Fine-tune BERT model for manuscript reconstruction.
    
    Training Strategy:
    - Start with pre-trained BERT (already understands English)
    - Fine-tune on synthetic damaged text pairs
    - Learn to predict missing words from context
    - Save final model for inference
    
    Args:
        train_dataset: Tokenized training data
        eval_dataset: Tokenized validation data
        tokenizer: BERT tokenizer
        device: 'cuda' or 'cpu'
    
    Returns:
        Trained model
    
    Raises:
        TrainingError: If training fails
    """
    print("="*60)
    print("INITIALIZING MODEL")
    print("="*60 + "\n")
    
    try:
        print(f"Loading base model: {BASE_MODEL_NAME}")
        model = BertForMaskedLM.from_pretrained(BASE_MODEL_NAME)
        print(f"  ✓ Model loaded ({sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters)\n")
        
    except Exception as e:
        raise TrainingError(f"Failed to load model: {e}")
    
    # Data collator for masked language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=MLM_PROBABILITY
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_MODEL_PATH,
        overwrite_output_dir=True,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=WARMUP_STEPS,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        eval_steps=EVAL_STEPS,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,  # Keep only 2 best checkpoints to save disk space
        push_to_hub=False,
        report_to="none",  # Disable wandb/tensorboard if not configured
        no_cuda=(device == "cpu")
    )
    
    print("="*60)
    print("TRAINING CONFIGURATION")
    print("="*60)
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Max sequence length: {MAX_SEQUENCE_LENGTH}")
    print(f"  MLM probability: {MLM_PROBABILITY}")
    print(f"  Device: {device.upper()}")
    print(f"  Output: {OUTPUT_MODEL_PATH}")
    print("="*60 + "\n")
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    print("="*60)
    print("STARTING TRAINING")
    print("="*60)
    print("This will take approximately:")
    print(f"  - GPU: 30-60 minutes")
    print(f"  - CPU: 3-6 hours")
    print("\nProgress will be logged every {LOGGING_STEPS} steps.")
    print("You can monitor loss decreasing over time.\n")
    
    try:
        # Train!
        train_result = trainer.train()
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE!")
        print("="*60)
        print(f"  Final train loss: {train_result.training_loss:.4f}")
        print(f"  Total training time: {train_result.metrics['train_runtime']:.1f}s")
        print("="*60 + "\n")
        
    except Exception as e:
        raise TrainingError(f"Training failed: {e}")
    
    return model, trainer


def save_final_model(model, tokenizer):
    """
    Save the trained model and tokenizer for inference.
    
    Args:
        model: Trained BERT model
        tokenizer: BERT tokenizer
    
    Raises:
        TrainingError: If saving fails
    """
    print("="*60)
    print("SAVING MODEL")
    print("="*60 + "\n")
    
    try:
        print(f"Saving model to: {OUTPUT_MODEL_PATH}")
        model.save_pretrained(OUTPUT_MODEL_PATH)
        tokenizer.save_pretrained(OUTPUT_MODEL_PATH)
        
        # Verify files were created
        required_files = ["config.json", "pytorch_model.bin", "tokenizer_config.json"]
        for filename in required_files:
            filepath = os.path.join(OUTPUT_MODEL_PATH, filename)
            if not os.path.exists(filepath):
                raise TrainingError(f"Missing expected file: {filename}")
        
        print(f"  ✓ Model saved successfully")
        print(f"  ✓ Tokenizer saved successfully")
        
        # Calculate model size
        model_size = sum(
            os.path.getsize(os.path.join(OUTPUT_MODEL_PATH, f))
            for f in os.listdir(OUTPUT_MODEL_PATH)
            if os.path.isfile(os.path.join(OUTPUT_MODEL_PATH, f))
        ) / 1e6
        
        print(f"  ✓ Total model size: {model_size:.1f} MB")
        print("\n" + "="*60)
        print("MODEL READY FOR INFERENCE!")
        print("="*60)
        print(f"You can now use: ml_model/reconstruction_engine.py")
        print("="*60 + "\n")
        
    except TrainingError:
        raise
    except Exception as e:
        raise TrainingError(f"Failed to save model: {e}")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("MANUSCRIPT RECONSTRUCTION - MODEL TRAINING")
    print("="*60 + "\n")
    
    try:
        # Step 1: Validate environment
        device = validate_environment()
        
        # Step 2: Load dataset
        train_dataset, eval_dataset, tokenizer = load_and_prepare_dataset()
        
        # Step 3: Train model
        model, trainer = train_model(train_dataset, eval_dataset, tokenizer, device)
        
        # Step 4: Save final model
        save_final_model(model, tokenizer)
        
        print("🎉 SUCCESS! Training pipeline complete!")
        print("Next step: Test your model with test/gap_test.py\n")
        
    except TrainingError as e:
        print(f"\n❌ TRAINING FAILED\n")
        print(str(e))
        exit(1)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        print("Partial checkpoints may be saved in:", OUTPUT_MODEL_PATH)
        exit(1)
        
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR\n")
        print(f"{e}")
        import traceback
        traceback.print_exc()
        exit(1)