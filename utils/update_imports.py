#!/usr/bin/env python3
import os
import re

def update_imports(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Replace imports from neural_lib to ember_ml
    updated_content = re.sub(r'from\s+neural_lib', 'from ember_ml', content)
    updated_content = re.sub(r'import\s+neural_lib', 'import ember_ml', updated_content)
    
    # Replace imports from notebooks.neural_experiments to ember_ml
    updated_content = re.sub(r'from\s+notebooks\.neural_experiments', 'from ember_ml', updated_content)
    updated_content = re.sub(r'import\s+notebooks\.neural_experiments', 'import ember_ml', updated_content)
    
    # Replace imports from notebooks.binary_wave_neurons to ember_ml
    updated_content = re.sub(r'from\s+notebooks\.binary_wave_neurons', 'from ember_ml.wave', updated_content)
    updated_content = re.sub(r'import\s+notebooks\.binary_wave_neurons', 'import ember_ml.wave', updated_content)
    
    # Replace imports from notebooks.audio_processing to ember_ml.audio
    updated_content = re.sub(r'from\s+notebooks\.audio_processing', 'from ember_ml.audio', updated_content)
    updated_content = re.sub(r'import\s+notebooks\.audio_processing', 'import ember_ml.audio', updated_content)
    
    # Replace imports from notebooks.infinitemath to ember_ml.math.infinite
    updated_content = re.sub(r'from\s+notebooks\.infinitemath', 'from ember_ml.math.infinite', updated_content)
    updated_content = re.sub(r'import\s+notebooks\.infinitemath', 'import ember_ml.math.infinite', updated_content)
    
    if content != updated_content:
        with open(file_path, 'w') as f:
            f.write(updated_content)
        return True
    return False

def main():
    updated_files = 0
    for root, dirs, files in os.walk('ember_ml'):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                if update_imports(file_path):
                    updated_files += 1
                    print(f"Updated imports in {file_path}")
    
    print(f"\nUpdated imports in {updated_files} files")

if __name__ == "__main__":
    main()
