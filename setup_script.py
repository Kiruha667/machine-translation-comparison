"""
Setup script to create project structure
Run this first: python setup.py
"""

import os


def create_directory_structure():
    """Create all necessary directories"""
    directories = [
        'data',
        'src',
        'src/models',
        'src/training',
        'src/utils',
        'scripts',
        'models',
        'outputs',
        'logs',
        'notebooks'
    ]
    
    print("="*80)
    print("CREATING PROJECT STRUCTURE")
    print("="*80)
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created: {directory}/")
    
    # Create __init__.py files
    init_files = [
        'src/__init__.py',
        'src/models/__init__.py',
        'src/training/__init__.py',
        'src/utils/__init__.py'
    ]
    
    for init_file in init_files:
        with open(init_file, 'w') as f:
            f.write('"""Package initialization"""\n')
        print(f"✓ Created: {init_file}")
    
    print("\n" + "="*80)
    print("PROJECT STRUCTURE CREATED SUCCESSFULLY")
    print("="*80)
    print("\nNext steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Download dataset: python data/download_data.py")
    print("3. Train models:")
    print("   - python scripts/train_seq2seq.py")
    print("   - python scripts/train_transformer.py")
    print("4. Compare models: python scripts/compare_models.py")


def create_gitignore():
    """Create .gitignore file"""
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# PyCharm
.idea/
*.iml

# VS Code
.vscode/

# Data
data/fra.txt
data/*.zip
data/*.tar.gz

# Models
models/*.pt
models/*.pth
checkpoints/

# Logs
logs/
*.log

# Outputs
outputs/
results/
*.png
*.pdf

# Jupyter
.ipynb_checkpoints/
*.ipynb_checkpoints

# OS
.DS_Store
Thumbs.db
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Desktop.ini
"""
    
    with open('.gitignore', 'w') as f:
        f.write(gitignore_content)
    
    print("✓ Created .gitignore")


def print_gpu_check():
    """Print GPU check instructions"""
    print("\n" + "="*80)
    print("GPU CHECK")
    print("="*80)
    print("\nTo verify GPU is detected:")
    print("  python -c \"import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')\"\n")
    print("Expected output:")
    print("  CUDA: True")
    print("  GPU: NVIDIA GeForce GTX 1660 SUPER")
    print("\nIf CUDA is False:")
    print("  1. Install NVIDIA drivers: https://www.nvidia.com/download/index.aspx")
    print("  2. Install CUDA Toolkit: https://developer.nvidia.com/cuda-downloads")
    print("  3. Reinstall PyTorch with CUDA: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")


if __name__ == '__main__':
    create_directory_structure()
    create_gitignore()
    print_gpu_check()
    
    print("\n" + "="*80)
    print("SETUP COMPLETE")
    print("="*80)
