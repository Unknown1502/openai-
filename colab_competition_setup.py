#!/usr/bin/env python3
"""
Google Colab Competition Setup Script
Handles model loading and environment setup for OpenAI GPT-OSS-20B
"""

import os
import sys
import subprocess
from pathlib import Path
import shutil

def setup_colab_environment():
    """Setup Google Colab environment for competition"""
    print("Setting up Google Colab environment...")
    
    # Determine project root
    if os.path.exists("/content/openai-"):
        project_root = Path("/content/openai-")
    elif os.path.exists("/content/kaggleproject"):
        project_root = Path("/content/kaggleproject")
    else:
        project_root = Path.cwd()
    
    print(f"Project root: {project_root}")
    
    # Add project to Python path
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    # Change to project directory
    os.chdir(project_root)
    
    return project_root

def install_dependencies():
    """Install required dependencies"""
    print("\nInstalling dependencies...")
    
    deps = [
        "torch",
        "transformers>=4.41.0",
        "accelerate>=0.25.0", 
        "bitsandbytes>=0.41.3",
        "aiohttp",
        "nest-asyncio",
        "tqdm",
        "numpy",
        "pandas",
        "sentencepiece",
        "protobuf"
    ]
    
    for dep in deps:
        print(f"Installing {dep}...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", dep], check=False)
    
    print("Dependencies installed successfully")

def fix_code_issues():
    """Fix known code issues"""
    print("\nApplying code fixes...")
    
    # Fix enhanced_attack_vectors.py if needed
    attack_vectors_path = Path("src/competition/enhanced_attack_vectors.py")
    if attack_vectors_path.exists():
        content = attack_vectors_path.read_text()
        if "ENHANCED_COMPETITION_ATTACKS = ENHANCED_COMPETITION_ATTACK_VECTORS" in content:
            content = content.replace(
                "ENHANCED_COMPETITION_ATTACKS = ENHANCED_COMPETITION_ATTACK_VECTORS",
                "# ENHANCED_COMPETITION_ATTACKS = ENHANCED_COMPETITION_ATTACK_VECTORS"
            )
            attack_vectors_path.write_text(content)
            print("Fixed enhanced_attack_vectors.py")
    
    print("Code fixes applied")

def setup_model_loading():
    """Setup for model loading"""
    print("\nPreparing for model loading...")
    
    # Create necessary directories
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("offload", exist_ok=True)
    
    # Set environment variables
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    # Check GPU availability
    try:
        import torch
        if torch.cuda.is_available():
            print(f"GPU available: {torch.cuda.get_device_name(0)}")
            print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            print("WARNING: No GPU detected! Model loading may fail.")
    except ImportError:
        print("PyTorch not installed yet")
    
    print("Model loading preparation complete")

def main():
    """Main setup function"""
    print("=" * 60)
    print("Google Colab Competition Setup")
    print("=" * 60)
    
    # Setup environment
    project_root = setup_colab_environment()
    
    # Install dependencies
    install_dependencies()
    
    # Fix code issues
    fix_code_issues()
    
    # Setup for model loading
    setup_model_loading()
    
    print("\n" + "=" * 60)
    print("Setup complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Run: python openai_gpt_model_loader.py")
    print("2. Or use the notebook: openai_gpt_model_setup_guide.ipynb")
    print("3. Then run: python run_competition.py")
    
    return project_root

if __name__ == "__main__":
    main()
