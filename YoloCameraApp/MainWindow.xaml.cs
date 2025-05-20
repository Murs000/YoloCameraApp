using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Dnn;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media.Imaging;
using System.Windows.Threading;

namespace YoloCameraApp
{
    public partial class MainWindow : Window
    {
        private VideoCapture _capture;
        private DispatcherTimer _timer;
        private Net _yoloNet;
        private List<string> _classLabels;
        private string[] _outputLayerNames;

        public MainWindow()
        {
            InitializeComponent();
            LoadYoloModel();
            StartCamera();
        }

        private void LoadYoloModel()
        {
            string baseDir = AppDomain.CurrentDomain.BaseDirectory;
            string modelPath = Path.Combine(baseDir, "Models");

            string configPath = Path.Combine(modelPath, "yolov4.cfg");
            string weightsPath = Path.Combine(modelPath, "yolov4.weights");
            string namesPath = Path.Combine(modelPath, "coco.names");

            _classLabels = File.ReadAllLines(namesPath).ToList();
            _yoloNet = DnnInvoke.ReadNetFromDarknet(configPath, weightsPath);

            //_yoloNet.SetPreferableBackend(Emgu.CV.Dnn.Backend.OpenCV);
            //_yoloNet.SetPreferableTarget(Target.Cpu);
            _yoloNet.SetPreferableBackend(Emgu.CV.Dnn.Backend.Cuda);
            _yoloNet.SetPreferableTarget(Target.Cuda);

            _outputLayerNames = _yoloNet.UnconnectedOutLayersNames;
        }

        private void StartCamera()
        {
            _capture = new VideoCapture(0, VideoCapture.API.DShow); // Or API.Any

            // Get camera's native resolution
            int width = (int)_capture.Get(CapProp.FrameWidth);
            int height = (int)_capture.Get(CapProp.FrameHeight);

            _capture.ImageGrabbed += ProcessFrame;
            _capture.Start();
        }

        private void ProcessFrame(object sender, EventArgs e)
        {
            Mat frame = new Mat();
            _capture.Retrieve(frame);

            if (frame.IsEmpty)
                return;

            Application.Current.Dispatcher.Invoke(() =>
            {
                // Only set this once
                if (CameraImage.Source == null)
                {
                    CameraImage.Width = frame.Width;
                    CameraImage.Height = frame.Height;

                    OverlayCanvas.Width = frame.Width;
                    OverlayCanvas.Height = frame.Height;
                }

                CameraImage.Source = ConvertMatToBitmapImage(frame);
            });

            using (Mat blob = DnnInvoke.BlobFromImage(frame, 1 / 255.0, new System.Drawing.Size(416, 416), new MCvScalar(), true, false))
            {
                _yoloNet.SetInput(blob);
                using (VectorOfMat output = new VectorOfMat())
                {
                    _yoloNet.Forward(output, _outputLayerNames);
                    PostProcess(frame, output);
                }
            }
        }

        private void PostProcess(Mat frame, VectorOfMat outputs)
        {
            int frameWidth = frame.Width;
            int frameHeight = frame.Height;

            float xScale = frameWidth / 416.0f;
            float yScale = frameHeight / 416.0f;

            List<Rectangle> boxes = new List<Rectangle>();
            List<int> classIds = new List<int>();
            List<float> confidences = new List<float>();

            for (int i = 0; i < outputs.Size; i++)
            {
                float[,] data = (float[,])outputs[i].GetData();

                for (int j = 0; j < data.GetLength(0); j++)
                {
                    float confidence = data[j, 4];
                    if (confidence > 0.5)
                    {
                        float[] scores = new float[_classLabels.Count];
                        for (int k = 0; k < scores.Length; k++)
                            scores[k] = data[j, 5 + k];

                        int classId = Array.IndexOf(scores, scores.Max());
                        float score = scores[classId];

                        if (score > 0.5)
                        {
                            // Detected in 416x416, need to rescale
                            float centerX = data[j, 0] * 416;
                            float centerY = data[j, 1] * 416;
                            float width = data[j, 2] * 416;
                            float height = data[j, 3] * 416;

                            // Scale to actual frame size
                            int scaledX = (int)((centerX - width / 2) * xScale);
                            int scaledY = (int)((centerY - height / 2) * yScale);
                            int scaledWidth = (int)(width * xScale);
                            int scaledHeight = (int)(height * yScale);

                            boxes.Add(new Rectangle(scaledX, scaledY, scaledWidth, scaledHeight));
                            classIds.Add(classId);
                            confidences.Add(score);
                        }
                    }
                }
            }

            int[] indices = DnnInvoke.NMSBoxes(boxes.ToArray(), confidences.ToArray(), 0.5f, 0.4f);

            Application.Current.Dispatcher.Invoke(() =>
            {
                OverlayCanvas.Children.Clear();

                for (int i = 0; i < indices.Length; i++)
                {
                    int idx = indices[i];
                    Rectangle box = boxes[idx];
                    string label = _classLabels[classIds[idx]];

                    System.Windows.Shapes.Rectangle rect = new System.Windows.Shapes.Rectangle
                    {
                        Width = box.Width,
                        Height = box.Height,
                        Stroke = System.Windows.Media.Brushes.Green,
                        StrokeThickness = 2
                    };
                    Canvas.SetLeft(rect, box.X);
                    Canvas.SetTop(rect, box.Y);
                    OverlayCanvas.Children.Add(rect);

                    TextBlock text = new TextBlock
                    {
                        Text = label,
                        Foreground = System.Windows.Media.Brushes.White,
                        Background = System.Windows.Media.Brushes.Black,
                        FontWeight = FontWeights.Bold
                    };
                    Canvas.SetLeft(text, box.X);
                    Canvas.SetTop(text, box.Y - 20);
                    OverlayCanvas.Children.Add(text);
                }
            });
        }

        private BitmapImage ConvertMatToBitmapImage(Mat mat)
        {
            using (MemoryStream ms = new MemoryStream())
            {
                mat.ToImage<Bgr, byte>().ToBitmap().Save(ms, System.Drawing.Imaging.ImageFormat.Bmp);
                ms.Seek(0, SeekOrigin.Begin);
                BitmapImage image = new BitmapImage();
                image.BeginInit();
                image.StreamSource = ms;
                image.CacheOption = BitmapCacheOption.OnLoad;
                image.EndInit();
                return image;
            }
        }

        protected override void OnClosed(EventArgs e)
        {
            _capture?.Dispose();
            base.OnClosed(e);
        }
    }
}
