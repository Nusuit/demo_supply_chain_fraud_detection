"""
Setup Script for Fraud Detection Streamlit Demo

This script helps set up the required files and preprocessors
for the Streamlit demo application.
"""

import os
import sys
import shutil
from pathlib import Path

# Add utils to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.preprocessing import create_mock_preprocessors

def print_header(text):
    """Print a formatted header"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)

def check_directory_structure():
    """Check if all required directories exist"""
    print_header("Checking Directory Structure")
    
    required_dirs = [
        'models',
        'data',
        'utils',
        'assets',
        'pages',
        '.streamlit'
    ]
    
    all_exist = True
    for directory in required_dirs:
        if os.path.exists(directory):
            print(f"✓ {directory}/ exists")
        else:
            print(f"✗ {directory}/ missing - creating...")
            os.makedirs(directory, exist_ok=True)
            all_exist = False
    
    return all_exist

def check_data_files():
    """Check if required data files exist"""
    print_header("Checking Data Files")
    
    required_files = [
        'data/test_results.json',
        'data/sna_results.json',
        'data/sample_transactions.csv'
    ]
    
    all_exist = True
    for file in required_files:
        if os.path.exists(file):
            print(f"✓ {file} exists")
        else:
            print(f"✗ {file} missing")
            all_exist = False
    
    return all_exist

def check_model_files():
    """Check if model files exist"""
    print_header("Checking Model Files")
    
    model_files = [
        'models/combined_model_seed42.keras',
        'models/combined_model_seed123.keras',
        'models/combined_model_seed456.keras'
    ]
    
    all_exist = True
    for file in model_files:
        if os.path.exists(file):
            print(f"✓ {file} exists")
        else:
            print(f"✗ {file} missing")
            all_exist = False
    
    if not all_exist:
        print("\n💡 To copy models from the original project:")
        print("   cp ../fraud_supplychain_year4/Fraud_SupplyChain/model/best_models/*.keras models/")
    
    return all_exist

def setup_preprocessors():
    """Setup scaler and PCA preprocessors"""
    print_header("Setting up Preprocessors")
    
    scaler_path = 'models/scaler.pkl'
    pca_path = 'models/pca.pkl'
    
    if os.path.exists(scaler_path) and os.path.exists(pca_path):
        print("✓ Preprocessors already exist")
        print(f"  - {scaler_path}")
        print(f"  - {pca_path}")
        return True
    else:
        print("Creating mock preprocessors for demo...")
        try:
            create_mock_preprocessors('models')
            print("✓ Preprocessors created successfully")
            return True
        except Exception as e:
            print(f"✗ Error creating preprocessors: {str(e)}")
            return False

def check_assets():
    """Check if asset files exist"""
    print_header("Checking Asset Files")
    
    asset_files = [
        'assets/confusion_matrix.png',
        'assets/roc_curve.png'
    ]
    
    all_exist = True
    for file in asset_files:
        if os.path.exists(file):
            print(f"✓ {file} exists")
        else:
            print(f"⚠ {file} missing (optional)")
            all_exist = False
    
    if not all_exist:
        print("\n💡 To create visualization assets, see: assets/README.md")
    
    return all_exist

def check_dependencies():
    """Check if required Python packages are installed"""
    print_header("Checking Dependencies")
    
    required_packages = {
        'streamlit': 'streamlit',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'tensorflow': 'tensorflow',
        'sklearn': 'scikit-learn',
        'joblib': 'joblib'
    }
    
    missing = []
    for module, package in required_packages.items():
        try:
            __import__(module)
            print(f"✓ {package} installed")
        except ImportError:
            print(f"✗ {package} not installed")
            missing.append(package)
    
    if missing:
        print(f"\n⚠ Missing packages: {', '.join(missing)}")
        print("\nTo install all dependencies:")
        print("   pip install -r requirements.txt")
        return False
    
    return True

def print_summary(checks):
    """Print setup summary"""
    print_header("Setup Summary")
    
    status_map = {
        'directories': 'Directory Structure',
        'data': 'Data Files',
        'models': 'Model Files',
        'preprocessors': 'Preprocessors',
        'assets': 'Asset Files',
        'dependencies': 'Dependencies'
    }
    
    print("\nStatus:")
    all_good = True
    for key, status in checks.items():
        icon = "✓" if status else "✗"
        name = status_map.get(key, key)
        print(f"  {icon} {name}")
        if not status:
            all_good = False
    
    print()
    if all_good:
        print("🎉 Everything is ready!")
        print("\nTo start the application:")
        print("   streamlit run Home.py")
    else:
        print("⚠ Some components are missing. Please check the messages above.")
        print("\nYou can still run the demo with mock data/models:")
        print("   streamlit run Home.py")
        print("\nFor full functionality, ensure all required files are present.")

def main():
    """Main setup function"""
    print_header("Fraud Detection Demo - Setup Script")
    print("This script will check and set up the required components.\n")
    
    # Change to script directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Run checks
    checks = {
        'directories': check_directory_structure(),
        'data': check_data_files(),
        'models': check_model_files(),
        'preprocessors': setup_preprocessors(),
        'assets': check_assets(),
        'dependencies': check_dependencies()
    }
    
    # Print summary
    print_summary(checks)
    
    print("\n" + "="*70)
    print("Setup complete!")
    print("="*70 + "\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Error during setup: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
