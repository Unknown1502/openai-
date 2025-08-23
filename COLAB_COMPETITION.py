#!/usr/bin/env python3
"""Colab Competition Winner - Fixes all dependency and CUDA issues for Google Colab."""

import os
import sys
import subprocess
import shutil
from pathlib import Path

class ColabCompetitionSetup:
    """Setup script specifically for Google Colab environment."""
    
    def __init__(self):
        self.is_colab = 'COLAB_GPU' in os.environ
        self.project_root = Path("/content/openai-")
        self.vendor_dir = self.project_root / "vendor"
        
    def detect_environment(self):
        """Detect and display environment information."""
        print(" Detecting Environment...")
        print(f"  Is Colab: {self.is_colab}")
        print(f"  Python: {sys.version}")
        
        # Check CUDA
        try:
            import torch
            print(f"  PyTorch: {torch.__version__}")
            print(f"  CUDA Available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"  CUDA Version: {torch.version.cuda}")
                print(f"  GPU: {torch.cuda.get_device_name(0)}")
                print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        except ImportError:
            print("  PyTorch: Not installed")
    
    def fix_vendor_issues(self):
        """Fix vendor directory issues specific to Colab."""
        print("\n Fixing Vendor Issues...")
        
        # Remove corrupted torch signal module
        signal_paths = [
            self.vendor_dir / "torch" / "signal",
            Path("/usr/local/lib/python3.10/dist-packages/torch/signal")
        ]
        
        for signal_path in signal_paths:
            if signal_path.exists():
                try:
                    # Check if it's corrupted
                    init_file = signal_path / "__init__.py"
                    if init_file.exists():
                        content = init_file.read_text()
                        if "fro" in content or len(content) < 10:
                            print(f"  Removing corrupted: {signal_path}")
                            shutil.rmtree(signal_path)
                except Exception as e:
                    print(f"  Error checking {signal_path}: {e}")
    
    def setup_environment(self):
        """Setup environment variables for Colab."""
        print("\n Setting up Environment...")
        
        # CUDA paths
        os.environ['CUDA_HOME'] = '/usr/local/cuda'
        os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda/lib64:' + os.environ.get('LD_LIBRARY_PATH', '')
        
        # PyTorch paths
        os.environ['TORCH_HOME'] = str(self.project_root / ".cache" / "torch")
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        
        # Update Python path for the new repository
        if str(self.project_root) not in sys.path:
            sys.path.insert(0, str(self.project_root))
        
        # Python paths
        if str(self.vendor_dir) not in sys.path:
            sys.path.insert(0, str(self.vendor_dir))
        
        print("  Environment variables set")
    
    def install_dependencies(self):
        """Install dependencies optimized for Colab."""
        print("\n Installing Dependencies...")
        
        # Check if we need to install
        requirements_file = self.project_root / "requirements_fixed_clean.txt"
        if not requirements_file.exists():
            print("  requirements_fixed_clean.txt not found!")
            return False
        
        # Install with specific flags for Colab
        cmd = [
            sys.executable, "-m", "pip", "install",
            "-r", str(requirements_file),
            "--no-deps",  # Avoid dependency conflicts
            "--force-reinstall",  # Force reinstall
            "--no-cache-dir"  # Don't use cache
        ]
        
        try:
            print("  Installing packages...")
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"   Installation warnings: {result.stderr}")
            else:
                print("   Dependencies installed")
            
            # Install bitsandbytes separately for 8-bit support
            print("  Installing bitsandbytes...")
            subprocess.run([
                sys.executable, "-m", "pip", "install",
                "bitsandbytes==0.41.3",
                "--no-deps"
            ], capture_output=True)
            print("   Bitsandbytes installed")
            
        except Exception as e:
            print(f"   Installation failed: {e}")
            return False
        
        return True
    
    def verify_setup(self):
        """Verify the setup is working correctly."""
        print("\n Verifying Setup...")
        
        # Test imports
        try:
            import torch
            import transformers
            import bitsandbytes
            print("   Core imports successful")
            
            # Test CUDA
            if torch.cuda.is_available():
                # Simple CUDA test
                x = torch.randn(100, 100).cuda()
                y = x @ x.T
                print(f"   CUDA computation successful")
                
                # Memory info
                print(f"  GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB used")
            else:
                print("   CUDA not available")
                
        except Exception as e:
            print(f"   Verification failed: {e}")
            return False
        
        return True
    
    def prepare_for_competition(self):
        """Prepare environment for running competition."""
        print("\n Preparing for Competition...")
        
        # Create necessary directories
        dirs_to_create = [
            self.project_root / "outputs",
            self.project_root / "enhanced_results",
            self.project_root / ".cache" / "torch"
        ]
        
        for dir_path in dirs_to_create:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Clear GPU memory
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                print("   GPU memory cleared")
        except:
            pass
        
        print("   Ready for competition")
    
    def run_full_setup(self):
        """Run the complete setup process."""
        print("=" * 60)
        print(" COLAB COMPETITION WINNER SETUP")
        print("=" * 60)
        
        # Step 1: Detect environment
        self.detect_environment()
        
        # Step 2: Fix vendor issues
        self.fix_vendor_issues()
        
        # Step 3: Setup environment
        self.setup_environment()
        
        # Step 4: Install dependencies
        if not self.install_dependencies():
            print("\n Setup failed at dependency installation")
            return False
        
        # Step 5: Verify setup
        if not self.verify_setup():
            print("\n Setup verification failed")
            return False
        
        # Step 6: Prepare for competition
        self.prepare_for_competition()
        
        print("\n" + "=" * 60)
        print(" SETUP COMPLETE - Ready to run competition!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Run standard competition: !python run_competition.py")
        print("2. Run enhanced competition: !python run_enhanced_competition.py --mode quick")
        print("3. For full scan: !python run_enhanced_competition.py --strategy aggressive")
        
        return True


def main():
    """Main entry point."""
    setup = ColabCompetitionSetup()
    
    if not setup.is_colab:
        print(" Warning: Not running in Google Colab environment")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    success = setup.run_full_setup()
    
    if success:
        # Optionally run a quick test
        print("\n Running quick test...")
        try:
            from src.competition.enhanced_attack_vectors import ENHANCED_COMPETITION_ATTACK_VECTORS
            print(f"  Loaded {len(ENHANCED_COMPETITION_ATTACK_VECTORS)} attack vectors")
        except Exception as e:
            print(f"   Test failed: {e}")


if __name__ == "__main__":
    main()
