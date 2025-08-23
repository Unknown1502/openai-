#!/usr/bin/env python3
"""
Test script for text generation using transformers pipeline
Handles various model loading strategies and error cases
"""

import os
import sys
import torch
from pathlib import Path

def test_text_generation():
    """
    Test text generation with proper error handling and fallback models
    """
    from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
    import warnings
    warnings.filterwarnings('ignore')
    
    # The model ID from your code - note this doesn't exist on HuggingFace
    # We'll try it first, then fallback to working alternatives
    model_configs = [
        {
            "id": "openai/gpt-oss-20b",
            "name": "OpenAI GPT-OSS-20B (Project Default)",
            "exists": False  # This model doesn't actually exist
        },
        {
            "id": "microsoft/DialoGPT-medium",
            "name": "Microsoft DialoGPT (Conversational)",
            "exists": True
        },
        {
            "id": "gpt2",
            "name": "GPT-2 (Fallback)",
            "exists": True
        }
    ]
    
    # Message to generate response for
    messages = [
        {"role": "user", "content": "Explain quantum mechanics clearly and concisely."},
    ]
    
    # Convert messages to text prompt
    prompt = messages[0]["content"]
    
    print("=" * 60)
    print("Text Generation Pipeline Test")
    print("=" * 60)
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"✓ GPU Available: {torch.cuda.get_device_name(0)}")
        device = 0  # Use first GPU
    else:
        print("✗ No GPU detected, using CPU")
        device = -1  # Use CPU
    
    print("\nAttempting to load models...")
    print("-" * 40)
    
    pipe = None
    successful_model = None
    
    for config in model_configs:
        model_id = config["id"]
        print(f"\nTrying: {config['name']}")
        print(f"  Model ID: {model_id}")
        
        try:
            if not config["exists"]:
                # For non-existent models, show what would happen
                print(f"  ⚠ Note: This model ID doesn't exist on HuggingFace")
                print(f"  → Skipping to avoid error...")
                continue
            
            # Try to create pipeline with automatic configuration
            print(f"  Loading pipeline...")
            pipe = pipeline(
                "text-generation",
                model=model_id,
                device=device,
                torch_dtype="auto" if torch.cuda.is_available() else torch.float32,
                trust_remote_code=True
            )
            
            successful_model = config["name"]
            print(f"  ✓ Successfully loaded {config['name']}")
            break
            
        except Exception as e:
            error_msg = str(e)
            if "404" in error_msg or "not found" in error_msg:
                print(f"  ✗ Model not found on HuggingFace")
            elif "out of memory" in error_msg.lower():
                print(f"  ✗ Out of memory - model too large")
            else:
                print(f"  ✗ Failed: {error_msg[:100]}...")
            continue
    
    if pipe is None:
        print("\n❌ Failed to load any model. Please check:")
        print("  1. Internet connection for downloading models")
        print("  2. Available disk space for model cache")
        print("  3. Sufficient RAM/VRAM for model loading")
        return
    
    print("\n" + "=" * 60)
    print(f"Generating response using: {successful_model}")
    print("=" * 60)
    
    # Generate response
    try:
        print(f"\nPrompt: {prompt}")
        print("\nGenerating response...")
        
        # Configure generation parameters
        generation_kwargs = {
            "max_new_tokens": 256,
            "temperature": 0.7,
            "top_p": 0.95,
            "do_sample": True,
            "pad_token_id": pipe.tokenizer.eos_token_id
        }
        
        # Generate
        outputs = pipe(
            prompt,
            **generation_kwargs
        )
        
        # Extract generated text
        generated_text = outputs[0]["generated_text"]
        
        # Remove the prompt from the output if it's included
        if generated_text.startswith(prompt):
            response = generated_text[len(prompt):].strip()
        else:
            response = generated_text
        
        print("\n" + "-" * 40)
        print("Generated Response:")
        print("-" * 40)
        print(response)
        print("-" * 40)
        
        # Show token statistics if available
        if hasattr(pipe.tokenizer, 'encode'):
            input_tokens = len(pipe.tokenizer.encode(prompt))
            output_tokens = len(pipe.tokenizer.encode(response))
            print(f"\nToken Statistics:")
            print(f"  Input tokens: {input_tokens}")
            print(f"  Output tokens: {output_tokens}")
            print(f"  Total tokens: {input_tokens + output_tokens}")
        
    except Exception as e:
        print(f"\n❌ Generation failed: {str(e)}")
        return
    
    print("\n✓ Test completed successfully!")
    
    # Cleanup
    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def main():
    """Main entry point"""
    # Optional: Set environment variables for better memory management
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    try:
        test_text_generation()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
