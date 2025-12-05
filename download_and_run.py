#!/usr/bin/env python3
"""
Complete Download and Run Script (Cross-Platform)
Downloads repository from GitHub and runs everything automatically
Works on Mac, Linux, and Windows
"""

import sys
import subprocess
import os
import shutil
from pathlib import Path

REPO_URL = "https://github.com/bvishnu08/diabetes-readmission-prediction-with-flagging-hisk-risk-patiences-.git"
REPO_NAME = "diabetes-readmission-prediction-with-flagging-hisk-risk-patiences-"

def run_command(cmd, shell=False, env=None, cwd=None):
    """Run a command and return success status"""
    try:
        if isinstance(cmd, str) and not shell:
            cmd = cmd.split()
        result = subprocess.run(cmd, shell=shell, check=True, capture_output=True, text=True, env=env, cwd=cwd)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stderr

def check_git():
    """Check if git is installed"""
    print("🔍 Checking Git installation...")
    success, output = run_command(["git", "--version"])
    if not success:
        print("❌ ERROR: Git is not installed!")
        print("   Please install Git first:")
        print("   - Mac: brew install git")
        print("   - Linux: sudo apt-get install git")
        print("   - Windows: https://git-scm.com/download/win")
        print(f"   Or download ZIP from: {REPO_URL}")
        return False
    print(f"✅ Found Git: {output.strip()}")
    return True

def download_repository():
    """Clone repository from GitHub"""
    repo_path = Path(REPO_NAME)
    
    if repo_path.exists() and (repo_path / ".git").exists():
        print(f"📁 Repository folder already exists: {REPO_NAME}")
        print("   Using existing folder...")
        return True
    
    print("📥 Downloading repository from GitHub...")
    print(f"   URL: {REPO_URL}")
    print()
    
    if not check_git():
        print()
        print("⚠️  Git not found. Alternative: Download ZIP manually")
        print(f"   1. Go to: {REPO_URL}")
        print("   2. Click 'Code' → 'Download ZIP'")
        print("   3. Extract the ZIP file")
        print(f"   4. Rename folder to: {REPO_NAME}")
        print("   5. Run this script again from that folder")
        return False
    
    print("⏳ Cloning repository... (This may take a minute)")
    success, output = run_command(["git", "clone", REPO_URL, REPO_NAME])
    
    if not success:
        print(f"❌ ERROR: Failed to clone repository!")
        print(f"   Error: {output}")
        print()
        print("⚠️  Alternative: Download ZIP manually")
        print(f"   1. Go to: {REPO_URL}")
        print("   2. Click 'Code' → 'Download ZIP'")
        print("   3. Extract the ZIP file")
        print(f"   4. Rename folder to: {REPO_NAME}")
        print("   5. Run this script again from that folder")
        return False
    
    # Verify the repository was cloned successfully
    if not repo_path.exists():
        print(f"❌ ERROR: Repository folder not found after cloning!")
        print(f"   Expected: {repo_path.absolute()}")
        return False
    
    print("✅ Repository downloaded successfully!")
    return True

def run_setup_script():
    """Run the setup script if it exists"""
    repo_path = Path(REPO_NAME)
    os.chdir(repo_path)
    
    # Check for platform-specific scripts
    if os.name == "nt":  # Windows
        script_name = "run_all.bat"
        if Path(script_name).exists():
            print(f"✅ Found {script_name} script")
            print("   Running automated setup...")
            print()
            success, output = run_command([script_name], shell=True)
            return success
    else:  # Mac/Linux
        script_name = "run_all.sh"
        if Path(script_name).exists():
            print(f"✅ Found {script_name} script")
            print("   Running automated setup...")
            print()
            # Make executable
            os.chmod(script_name, 0o755)
            success, output = run_command([f"./{script_name}"], shell=True)
            return success
    
    # Try Python script as fallback
    script_name = "run_all.py"
    if Path(script_name).exists():
        print(f"✅ Found {script_name} script")
        print("   Running automated setup...")
        print()
        venv_python = Path(".venv") / ("Scripts" if os.name == "nt" else "bin") / "python"
        if venv_python.exists():
            success, output = run_command([str(venv_python), script_name])
        else:
            success, output = run_command([sys.executable, script_name])
        return success
    
    return False

def manual_setup():
    """Manual setup if scripts don't exist"""
    print("⚠️  Automated scripts not found, running manual setup...")
    print()
    
    # Check Python
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ ERROR: Python 3.8+ required. Found Python {version.major}.{version.minor}")
        return False
    print(f"✅ Found Python {version.major}.{version.minor}.{version.micro}")
    print()
    
    # Create venv
    venv_path = Path(".venv")
    if not venv_path.exists():
        print("🔧 Creating virtual environment...")
        success, output = run_command([sys.executable, "-m", "venv", ".venv"])
        if not success:
            print(f"❌ Failed to create virtual environment: {output}")
            return False
        print("✅ Virtual environment created")
    else:
        print("✅ Virtual environment already exists")
    print()
    
    # Install requirements
    venv_python = venv_path / ("Scripts" if os.name == "nt" else "bin") / "python"
    print("📦 Installing required packages...")
    print("   (This may take a few minutes...)")
    success, _ = run_command([str(venv_python), "-m", "pip", "install", "--upgrade", "pip", "--quiet"])
    success, output = run_command([str(venv_python), "-m", "pip", "install", "-r", "requirements.txt", "--quiet"])
    if not success:
        print(f"❌ Failed to install packages: {output}")
        return False
    print("✅ All packages installed")
    print()
    
    # Check data
    data_path = Path("data/raw/diabetic_data.csv")
    if not data_path.exists():
        print(f"❌ ERROR: {data_path} not found!")
        return False
    print("✅ Data file found")
    print()
    
    # Run training
    print("🚀 Starting training...")
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    success, output = run_command([str(venv_python), "scripts/run_train.py"], env=env)
    if not success:
        print(f"❌ Training failed: {output}")
        return False
    print("✅ Training completed")
    print()
    
    # Run evaluation
    print("📊 Running evaluation...")
    success, output = run_command([str(venv_python), "scripts/run_eval.py"], env=env)
    if not success:
        print(f"❌ Evaluation failed: {output}")
        return False
    print("✅ Evaluation completed")
    print()
    
    return True

def main():
    """Main execution function"""
    print("==========================================")
    print("Diabetes Readmission Prediction")
    print("Complete Download & Run Script")
    print("==========================================")
    print()
    
    # Step 1: Download repository
    if not download_repository():
        sys.exit(1)
    print()
    
    # Step 2: Change to repository directory
    repo_path = Path(REPO_NAME)
    
    # Verify repository exists before changing directory
    if not repo_path.exists():
        print(f"❌ ERROR: Repository folder not found: {REPO_NAME}")
        print(f"   Expected location: {repo_path.absolute()}")
        print()
        print("💡 Solutions:")
        print(f"   1. Make sure you're in the correct directory")
        print(f"   2. If you downloaded ZIP, extract it and rename to: {REPO_NAME}")
        print(f"   3. Or run: git clone {REPO_URL} {REPO_NAME}")
        sys.exit(1)
    
    original_dir = Path.cwd()
    
    try:
        os.chdir(repo_path)
    except OSError as e:
        print(f"❌ ERROR: Cannot access repository folder!")
        print(f"   Path: {repo_path.absolute()}")
        print(f"   Error: {e}")
        print()
        print("💡 Solutions:")
        print(f"   1. Check if the folder exists: {repo_path.absolute()}")
        print("   2. Check folder permissions")
        print("   3. Try running as administrator (Windows)")
        sys.exit(1)
    
    try:
        print(f"📁 Working directory: {Path.cwd()}")
        print()
        
        # Step 3: Try to run setup script
        if not run_setup_script():
            # Step 4: Fall back to manual setup
            if not manual_setup():
                sys.exit(1)
        
        # Step 5: Summary
        print("==========================================")
        print("✅ COMPLETE! Everything is ready!")
        print("==========================================")
        print()
        print(f"📁 Project location: {Path.cwd()}")
        print()
        print("🎯 Next steps:")
        print("  1. View results in the terminal output above")
        print("  2. Launch dashboard:")
        if os.name == "nt":
            print(f"     cd {Path.cwd()}")
            print("     .venv\\Scripts\\activate")
        else:
            print(f"     cd {Path.cwd()}")
            print("     source .venv/bin/activate")
        print("     streamlit run dashboard.py")
        print()
        print("==========================================")
        
    finally:
        os.chdir(original_dir)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user")
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"\n\n❌ File not found error: {e}")
        print()
        print("💡 This usually means:")
        print("   1. The repository folder doesn't exist")
        print("   2. Git is not installed or not in PATH")
        print("   3. The script is trying to access a file that doesn't exist")
        print()
        print("💡 Solutions:")
        print(f"   1. Download ZIP from: {REPO_URL}")
        print("   2. Extract and rename folder to match repository name")
        print("   3. Run the script from inside that folder")
        print("   4. Or install Git and try again")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        print()
        print("💡 If this is a Windows error, try:")
        print("   1. Download ZIP from GitHub instead of using git clone")
        print("   2. Extract ZIP and run run_all.py directly")
        print(f"   3. Or manually clone: git clone {REPO_URL}")
        sys.exit(1)

