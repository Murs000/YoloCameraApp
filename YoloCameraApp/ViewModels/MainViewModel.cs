using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Dnn;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using System;
using System.Collections.ObjectModel;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Media.Imaging;
using System.Windows.Threading;
using YoloCameraApp.Helpers;
using YoloCameraApp.Models;

namespace YoloCameraApp.ViewModels
{
    public class MainViewModel : ViewModelBase
    {
        private readonly YoloProcessor _processor;
        private VideoCapture _capture;
        private DispatcherTimer _timer;
        private BitmapImage _cameraImage;
        private string _summary;

        public double DisplayedImageWidth { get; set; }
        public double DisplayedImageHeight { get; set; }

        public ObservableCollection<YoloPrediction> Predictions { get; set; } = new();
        public ObservableCollection<OverlayItem> OverlayItems { get; set; } = new();

        public BitmapImage CameraImage
        {
            get => _cameraImage;
            set => SetProperty(ref _cameraImage, value);
        }

        public string Summary
        {
            get => _summary;
            set => SetProperty(ref _summary, value);
        }

        public MainViewModel()
        {
            _processor = new YoloProcessor();
            StartCamera();
        }

        private void StartCamera()
        {
            _capture = new VideoCapture(0, VideoCapture.API.DShow);

            DisplayedImageWidth = (int)_capture.Get(CapProp.FrameWidth);
            DisplayedImageHeight = (int)_capture.Get(CapProp.FrameHeight);

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
                CameraImage = _processor.ConvertMatToBitmapImage(frame);
            });

            var results = _processor.Predict(frame);

            Application.Current.Dispatcher.Invoke(() =>
            {
                Predictions.Clear();
                OverlayItems.Clear();

                // Calculate scale factors from original frame size to displayed image size
                double scaleX = 1.0, scaleY = 1.0;

                if (DisplayedImageWidth > 0 && DisplayedImageHeight > 0)
                {
                    scaleX = DisplayedImageWidth / frame.Width;
                    scaleY = DisplayedImageHeight / frame.Height;
                }

                foreach (var item in results)
                {
                    Predictions.Add(item);

                    // Scale the bounding box coords before adding to OverlayItems
                    OverlayItems.Add(new OverlayItem
                    {
                        Left = item.Box.X * scaleX,
                        Top = item.Box.Y * scaleY,
                        Width = item.Box.Width * scaleX,
                        Height = item.Box.Height * scaleY,
                        Label = $"{item.Label} ({item.Confidence:P0})"
                    });
                }

                Summary = Predictions.Count > 0
                    ? string.Join(", ", Predictions.GroupBy(p => p.Label).Select(g => $"{g.Key}: {g.Count()}"))
                    : "None";
            });

        }

        public void Dispose()
        {
            _capture?.Dispose();
        }
    }
}
