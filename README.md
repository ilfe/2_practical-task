# 2_practical-task
# Tennis Ball Detection Model

This project demonstrates how to train a model to detect tennis balls in images using PyTorch and torchvision. The model is trained on a labeled dataset and fine-tuned on an unlabeled dataset.

## Requirements

- Python 3.x
- PyTorch
- torchvision
- OpenCV
- Matplotlib
- PIL (Pillow)

## Installation

1. Install the required libraries:
    ```bash
    pip install torch torchvision opencv-python matplotlib pillow
    ```

## Data Preparation

1. **Labeled Data**: Place your labeled images and corresponding label files in the `train_with` directory.
    - Example structure:
      ```
      data1/
      ├── train_with/
      │   ├── train/
      │   │   ├── images/
      │   │   │   ├── image1.jpg
      │   │   │   ├── image2.jpg
      │   │   │   └── ...
      │   │   ├── labels/
      │   │   │   ├── image1.txt
      │   │   │   ├── image2.txt
      │   │   │   └── ...
      │   ├── valid/
      │   │   ├── images/
      │   │   │   ├── image1.jpg
      │   │   │   ├── image2.jpg
      │   │   │   └── ...
      │   │   ├── labels/
      │   │   │   ├── image1.txt
      │   │   │   ├── image2.txt
      │   │   │   └── ...
      ```

2. **Unlabeled Data**: Place your unlabeled images in the `res` directory.
    - Example structure:
      ```
      res/
      ├── image1.jpg
      ├── image2.jpg
      └── ...
      ```

## Training the Model

1. **Data Transformations**: The images are resized to 224x224 pixels, converted to tensors, and normalized.
    ```python
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    ```

2. **Custom Datasets**: Two custom datasets are created for labeled and unlabeled data.
    ```python
    class LabeledDataset(Dataset):
        # Implementation here

    class UnlabeledDataset(Dataset):
        # Implementation here
    ```

3. **Data Loaders**: Data loaders are created for training, validation, and unlabeled data.
    ```python
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=batch_size, shuffle=False)
    ```

4. **Model Definition**: A ResNet-18 model is used for training.
    ```python
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 2)
    ```

5. **Loss Function and Optimizer**: Cross-entropy loss and Adam optimizer are used.
    ```python
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    ```

6. **Training and Validation Functions**: Functions for training and validating the model are defined.
    ```python
    def train_epoch(model, loader, criterion, optimizer):
        # Implementation here

    def validate_epoch(model, loader, criterion):
        # Implementation here
    ```

7. **Training Loop**: The model is trained for a specified number of epochs.
    ```python
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion)
        # Logging and visualization here
    ```

8. **Model Saving**: The trained model is saved to a specified path.
    ```python
    torch.save(model.state_dict(), model_save_path)
    ```

## Visualization

1. **Training and Validation Loss/Accuracy**: The training and validation loss/accuracy are plotted using Matplotlib.
    ```python
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.title('Accuracy')
    plt.legend()

    plt.show()
    ```

## Loading the Model

1. **Load Model**: A function to load the saved model is provided.
    ```python
    def load_model(model_save_path):
        if os.path.exists(model_save_path):
            model.load_state_dict(torch.load(model_save_path))
            model.eval()
            return model
        else:
            print(f"Model file not found at: {model_save_path}")
            return None
    ```

## Conclusion

This project demonstrates how to train a model to detect tennis balls in images using PyTorch and torchvision. The model is trained on a labeled dataset and fine-tuned on an unlabeled dataset. The training process is visualized using Matplotlib, and the trained model is saved for future use.
