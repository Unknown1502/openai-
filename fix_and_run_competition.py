#!/usr/bin/env python3
"""
Quick fix script for BitsAndBytesConfig compatibility issue and run competition
"""

import os
import sys
import json

def apply_fixes():
    """Apply all necessary fixes to run the competition"""
    
    print("🔧 Applying fixes for OpenAI GPT-OSS-20B competition...")
    
    # Fix 1: Disable quantization in config.json to avoid BitsAndBytesConfig issues
    config_path = 'config.json'
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Disable 8-bit and 4-bit loading
        config['hf_load_in_8bit'] = False
        config['hf_load_in_4bit'] = False
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print("✅ Disabled quantization in config.json")
    
    # Fix 2: Verify device mismatch fix is applied
    hf_local_path = 'src/backends/hf_local.py'
    if os.path.exists(hf_local_path):
        with open(hf_local_path, 'r') as f:
            content = f.read()
        
        if 'Move inputs to the same device as the model' in content:
            print("✅ Device mismatch fix already applied")
        else:
            print("⚠️  Device mismatch fix may need to be applied")
    
    # Fix 3: Set environment variables
    os.environ['TORCH_HOME'] = os.path.join(os.getcwd(), '.cache/torch')
    os.environ['CUDA_HOME'] = '/usr/local/cuda'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # Create cache directory
    os.makedirs('.cache/torch', exist_ok=True)
    
    print("✅ Environment variables set")
    print("\n🚀 Fixes applied! You can now run the competition.")
    
    return True

def run_competition(mode='quick'):
    """Run the competition after applying fixes"""
    
    # Apply fixes first
    if not apply_fixes():
        print("❌ Failed to apply fixes")
        return
    
    print(f"\n🏃 Running competition in {mode} mode...")
    
    # Import and run
    try:
        if mode == 'basic':
            os.system('python run_competition.py')
        else:
            os.system(f'python run_enhanced_competition.py --strategy standard --mode {mode}')
    except Exception as e:
        print(f"❌ Error running competition: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fix and run OpenAI GPT-OSS-20B competition")
    parser.add_argument(
        '--mode', 
        choices=['basic', 'quick', 'comprehensive', 'targeted'],
        default='quick',
        help='Competition mode to run'
    )
    parser.add_argument(
        '--fix-only',
        action='store_true',
        help='Only apply fixes without running competition'
    )
    
    args = parser.parse_args()
    
    if args.fix_only:
        apply_fixes()
    else:
        run_competition(args.mode)
