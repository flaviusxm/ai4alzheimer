import shutil
import os

def create_zip():
    files = [
        "cnn.py", 
        "dataset.py",
        "data-analysis.py",
        "data-preprocessing.py",
        "evaluate.py",
        "global_constants.py", 
        "inference.py", 
        "requirements.txt",
        "train.py", 
        "utilities.py"
    ]
    dataset_dirs = [
        "mild-dementia", 
        "moderate-dementia", 
        "non-dementia", 
        "very-mild-dementia"
    ]
    zip_name = "alzheimer_project"
    temp_dir = "temp_colab_pack"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    
    print("Copying files...")
    for f in files:
        if os.path.exists(f):
            shutil.copy(f, temp_dir)
        else:
            print(f"Warning: {f} not found!")

    print("Copying dataset directories ")
    for d in dataset_dirs:
        if os.path.exists(d):
            shutil.copytree(d, os.path.join(temp_dir, d))
        else:
             print(f"Warning: {d} not found!")

    print("Zipping...")
    shutil.make_archive(zip_name, 'zip', temp_dir)
    shutil.rmtree(temp_dir)
    print(f"Done! Created {zip_name}.zip")

if __name__ == "__main__":
    create_zip()
