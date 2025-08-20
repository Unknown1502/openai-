#!/usr/bin/env python3
"""
Complete script to load OpenAI GPT-OSS-20B model in Google Colab
This handles MXFP4 quantization and memory constraints
"""

import os
import sys
import torch
import gc
from pathlib import Path

def setup_environment():
    """Setup environment for model loading"""
    print("Setting up environment...")
    
    # Set memory allocation
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # Clear any existing cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("No GPU detected! Enable GPU in Runtime settings")
        return False
    
    return True

def install_dependencies():
    """Install required dependencies for MXFP4 quantization"""
    print("\nInstalling dependencies...")
    
    # Install triton for MXFP4 support
    os.system("pip install -q triton>=3.4.0")
    os.system("pip install -q kernels")
    
    # Update transformers for latest MXFP4 support
    os.system("pip install -q transformers>=4.45.0")
    os.system("pip install -q accelerate>=0.25.0")
    os.system("pip install -q bitsandbytes>=0.41.3")
    
    print("Dependencies installed")

def load_model_with_mxfp4():
    """Load model with MXFP4 quantization support"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print("\nLoading OpenAI GPT-OSS-20B model...")
    model_name = "openai/gpt-oss-20b"
    
    # First, load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    print("Tokenizer loaded")
    
    # Try different loading strategies
    print("\nAttempting to load model with MXFP4 support...")
    
    try:
        # Strategy 1: Load with auto device map and bf16
        print("Strategy 1: Auto device map with bfloat16...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            use_cache=True
        )
        print("Model loaded successfully with Strategy 1!")
        return model, tokenizer
        
    except Exception as e:
        print(f"Strategy 1 failed: {str(e)[:200]}")
        
    # Strategy 2: Load with CPU offloading
    try:
        print("\nStrategy 2: CPU offloading for large model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            offload_folder="offload",
            offload_state_dict=True,
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        print("Model loaded successfully with Strategy 2!")
        return model, tokenizer
        
    except Exception as e:
        print(f"Strategy 2 failed: {str(e)[:200]}")
    
    # Strategy 3: Force dequantization
    try:
        print("\nStrategy 3: Force dequantization to bf16...")
        os.environ['MXFP4_DEQUANTIZE'] = '1'
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        print("Model loaded successfully with Strategy 3!")
        return model, tokenizer
        
    except Exception as e:
        print(f"Strategy 3 failed: {str(e)[:200]}")
    
    return None, tokenizer

def test_model(model, tokenizer):
    """Test the loaded model"""
    if model is None:
        print("No model loaded to test")
        return
    
    print("\nTesting model...")
    
    test_prompt = "Hello, I am an AI assistant. How can I help you today?"
    
    try:
        inputs = tokenizer(test_prompt, return_tensors="pt")
        
        # Move to GPU if available
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=100,
                temperature=0.7,
                do_sample=True,
                top_p=0.95
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\nModel test successful!")
        print(f"Prompt: {test_prompt}")
        print(f"Response: {response}")
        
        return True
        
    except Exception as e:
        print(f"Model test failed: {str(e)[:200]}")
        return False

def main():
    """Main function to load and test the model"""
    print("=" * 60)
    print("OpenAI GPT-OSS-20B Model Loader for Google Colab")
    print("=" * 60)
    
    # Step 1: Setup environment
    if not setup_environment():
        print("\nEnvironment setup failed. Please enable GPU!")
        return
    
    # Step 2: Install dependencies
    install_dependencies()
    
    # Step 3: Load model
    model, tokenizer = load_model_with_mxfp4()
    
    # Step 4: Test model
    if model is not None:
        success = test_model(model, tokenizer)
        
        if success:
            print("\nModel loaded and tested successfully!")
            print("\nYou can now run:")
            print("  python run_competition.py")
            print("  python run_enhanced_competition.py --mode quick")
            
            # Save model reference for later use
            globals()['loaded_model'] = model
            globals()['loaded_tokenizer'] = tokenizer
        else:
            print("\nModel loaded but test failed")
    else:
        print("\nFailed to load model with any strategy")
        print("\nTroubleshooting:")
        print("1. Ensure GPU is enabled (Runtime → Change runtime type → GPU)")
        print("2. Try restarting runtime and running again")
        print("3. Check available GPU memory with: !nvidia-smi")

if __name__ == "__main__":
    main()
