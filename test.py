import os
import torch
import torch.nn as nn
import numpy as np

# Import the OemgaSqueeze compiler
from oemgasqueeze import OemgaSqueeze

# ==========================================
# 1. Define a Basic Dummy Architecture
# ==========================================
# This model uses only the supported v1 operations: 
# Conv1d (groups=1), ReLU, MaxPool1d, Flatten, Linear
class Dummy1DCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv1d(in_channels=4, out_channels=8, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.flatten = nn.Flatten()
        # After two /2 poolings, a length of 64 becomes 16. 8 channels * 16 = 128 features.
        self.fc = nn.Linear(in_features=128, out_features=4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        x = self.flatten(x)
        x = self.fc(x)
        return x

def main():
    print("--- OemgaSqueeze End-to-End Demo ---")
    
    # ==========================================
    # 2. Generate Random Synthetic Data
    # ==========================================
    # Shape: (Batch, Channels, Length) -> e.g., 1 channel of 64 sensor readings
    input_shape = (1, 64) 
    num_classes = 4
    
    print("\nGenerating synthetic data...")
    # Single example input for graph tracing
    example_input = torch.randn(1, *input_shape)
    
    # Calibration data for INT8 quantization (e.g., 100 samples)
    calibration_data = torch.randn(100, *input_shape)
    
    # Test data for verifying the generated C code matches PyTorch
    X_test = torch.randn(50, *input_shape)
    y_test = torch.randint(0, num_classes, (50,))

    # Initialize the dummy model
    model = Dummy1DCNN()
    model.eval()

    # ==========================================
    # 3. Initialize the Compiler
    # ==========================================
    output_directory = "demo_build_zephyr/app"
    
    compiler = OemgaSqueeze(
        model=model,
        example_input=example_input,
        calibration_data=calibration_data,
        output_dir=output_directory,
    )

    # ==========================================
    # 4. Compile to Zephyr C Code
    # ==========================================
    print("\nStarting Compilation Pipeline...")
    artifacts = compiler.compile(backend="oemga_native_int8")
    
    print(f"\nCompilation Successful! Code generated at: {artifacts.output_root}")
    
    # ==========================================
    # 5. Verify C Model vs PyTorch Model
    # ==========================================
    print("\nStarting Verification...")
    report = compiler.verify(X_test, y_test)
    
    print("\n--- Verification Summary ---")
    print(f"PyTorch Macro F1: {report['pytorch']['macro_f1']:.4f}")
    print(f"C Model Macro F1: {report['c_model']['macro_f1']:.4f}")
    print(f"Prediction Agreement: {report['comparison']['prediction_agreement'] * 100:.2f}%")
    
    # Note: Because the model is completely untrained and the data is random noise, 
    # accuracy and F1 will be low (~25% for 4 classes). The important metric here 
    # is the "Prediction Agreement", which verifies the C port matches the PyTorch math.

if __name__ == "__main__":
    main()