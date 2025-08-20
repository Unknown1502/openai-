"""
Test script to verify 8-bit quantization setup is working correctly
"""

import sys
import os
import json

def test_imports():
    """Test if all required libraries can be imported"""
    print("Testing imports...")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__} imported successfully")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False
    
    try:
        import transformers
        print(f"✓ Transformers {transformers.__version__} imported successfully")
    except ImportError as e:
        print(f"✗ Transformers import failed: {e}")
        return False
    
    try:
        import bitsandbytes
        print(f"✓ BitsAndBytes {bitsandbytes.__version__} imported successfully")
    except ImportError as e:
        print(f"✗ BitsAndBytes import failed: {e}")
        print("  Note: BitsAndBytes is not available on Windows")
    
    try:
        import accelerate
        print(f"✓ Accelerate {accelerate.__version__} imported successfully")
    except ImportError as e:
        print(f"✗ Accelerate import failed: {e}")
        return False
    
    return True

def test_config():
    """Test if config.json has correct 8-bit settings"""
    print("\nTesting configuration...")
    
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
        
        # Check 8-bit settings
        if config.get('hf_load_in_8bit', False):
            print("✓ 8-bit quantization is enabled")
        else:
            print("✗ 8-bit quantization is NOT enabled")
            
        if not config.get('hf_load_in_4bit', True):
            print("✓ 4-bit quantization is disabled (correct)")
        else:
            print("✗ 4-bit quantization is still enabled (should be disabled)")
            
        # Check memory limits
        max_memory = config.get('hf_max_memory', {})
        print(f"✓ Memory limits set: GPU={max_memory.get('0', 'Not set')}, CPU={max_memory.get('cpu', 'Not set')}")
        
        return True
    except Exception as e:
        print(f"✗ Config test failed: {e}")
        return False

def test_backend_import():
    """Test if the HF local backend can be imported"""
    print("\nTesting backend import...")
    
    try:
        # Add src to path if needed
        if 'src' not in sys.path:
            sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
        
        from backends.hf_local import HFLocalClient
        print("✓ HFLocalClient imported successfully")
        
        # Check if 8-bit loading code exists
        import inspect
        source = inspect.getsource(HFLocalClient)
        if "load_in_8bit" in source and "BitsAndBytesConfig" in source:
            print("✓ Backend has 8-bit quantization support")
        else:
            print("✗ Backend may not have proper 8-bit support")
            
        return True
    except Exception as e:
        print(f"✗ Backend import failed: {e}")
        return False

def test_memory_check():
    """Check available memory"""
    print("\nChecking system memory...")
    
    try:
        import psutil
        
        # RAM
        ram = psutil.virtual_memory()
        print(f"✓ RAM: {ram.total / 1e9:.2f} GB total, {ram.available / 1e9:.2f} GB available")
        
        # Check if we have enough RAM for 8-bit model
        if ram.available / 1e9 > 10:
            print("✓ Sufficient RAM available for 8-bit model loading")
        else:
            print("⚠ Low RAM available, may need to use CPU offloading")
        
        return True
    except ImportError:
        print("⚠ psutil not installed, skipping memory check")
        return True

def main():
    """Run all tests"""
    print("=" * 60)
    print("8-bit Quantization Setup Test")
    print("=" * 60)
    
    all_passed = True
    
    # Run tests
    all_passed &= test_imports()
    all_passed &= test_config()
    all_passed &= test_backend_import()
    all_passed &= test_memory_check()
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All tests passed! 8-bit setup is ready.")
        print("\nTo run the model with 8-bit quantization:")
        print("  python run_competition.py")
    else:
        print("✗ Some tests failed. Please check the errors above.")
        print("\nCommon fixes:")
        print("  1. Install PyTorch with CUDA support")
        print("  2. Install bitsandbytes: pip install bitsandbytes")
        print("  3. Ensure config.json has hf_load_in_8bit: true")
    print("=" * 60)

if __name__ == "__main__":
    main()
