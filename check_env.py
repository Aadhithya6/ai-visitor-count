import sys
import importlib

def check_library(lib_name):
    try:
        importlib.import_module(lib_name)
        print(f"[OK] {lib_name} is installed and working.")
        return True
    except ImportError as e:
        print(f"[ERROR] {lib_name} is NOT installed. ({e})")
        return False

def main():
    print("==========================================")
    print("Python Environment Validation")
    print("==========================================")
    print(f"Python Version: {sys.version}")
    print(f"Virtual Environment: {sys.prefix != sys.base_prefix}")
    print("------------------------------------------")

    required_libraries = [
        "ultralytics",
        "insightface",
        "cv2",
        "numpy",
        "sqlalchemy",
        "deep_sort_realtime"
    ]

    success = True
    for lib in required_libraries:
        if not check_library(lib):
            success = False

    print("------------------------------------------")
    if success:
        print("Environment setup successful! All libraries are available.")
    else:
        print("Environment setup incomplete. Please run the setup script again.")
        sys.exit(1)

if __name__ == "__main__":
    main()
