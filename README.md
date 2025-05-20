# YoloCameraApp

A real-time object detection desktop application built using **WPF (.NET)** and **YOLOv4** via **Emgu.CV** (OpenCV wrapper for .NET).

## ✨ Features

- Real-time object detection using YOLOv4 and webcam input.
- Display of bounding boxes and labels for detected objects.
- Scaled overlays rendered on a WPF `Canvas` for accurate visualization.
- Live summary of detected object categories.
- GPU acceleration via CUDA (if available).

## 📂 Project Structure

```
YoloCameraApp/
│
├── Helpers/
│   └── YoloProcessor.cs       # Loads model and runs predictions
│
├── Models/
│   ├── YoloPrediction.cs      # Represents a single detection result
│   └── OverlayItem.cs         # Represents a bounding box overlay in the UI
│
├── ViewModels/
│   ├── MainViewModel.cs       # Handles camera and prediction logic
│   └── ViewModelBase.cs       # Implements INotifyPropertyChanged
│
├── MainWindow.xaml            # WPF UI with camera and overlay canvas
└── Models/
    ├── yolov4.cfg             # YOLOv4 config
    ├── yolov4.weights         # YOLOv4 weights
    └── coco.names             # Class labels
```

## ⚙ Requirements

- .NET 6 or later
- Emgu.CV (OpenCV wrapper)
- CUDA-compatible GPU (recommended for better performance)
- YOLOv4 model files: `yolov4.cfg`, `yolov4.weights`, `coco.names` in `Models/` folder

## 🛠 Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/YoloCameraApp.git
cd YoloCameraApp
```

2. Restore NuGet packages:

```bash
dotnet restore
```

3. Make sure model files (`.cfg`, `.weights`, `coco.names`) exist in the `Models` folder.

4. Run the project:

```bash
dotnet run
```

## 📷 How It Works

- Captures frames from the default webcam.
- Uses Emgu.CV to process the frame and detect objects via YOLOv4.
- Detected objects are rendered as colored bounding boxes with confidence scores.
- A summary of object counts is displayed dynamically.

## ✅ Output Example

- Person (98%), Dog (85%), Car (92%)
- Bounding boxes with labels on live camera feed.

## 📜 License

This project is licensed under the MIT License.
