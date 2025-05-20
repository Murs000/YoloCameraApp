using Emgu.CV;
using System.Drawing;
using System.IO;
using System.Text;
using System.Text.RegularExpressions;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using System.Windows.Threading;

namespace YoloCameraApp
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        private VideoCapture _capture;
        private DispatcherTimer _timer;

        public MainWindow()
        {
            InitializeComponent();
            StartCamera();
        }

        private void StartCamera()
        {
            _capture = new VideoCapture(0); // 0 = default camera
            _capture.ImageGrabbed += ProcessFrame;
            _capture.Start();
        }

        private void ProcessFrame(object sender, EventArgs e)
        {
            Mat frame = new Mat();
            _capture.Retrieve(frame);

            Application.Current.Dispatcher.Invoke(() =>
            {
                CameraImage.Source = ConvertMatToBitmapImage(frame);
            });
        }

        private BitmapImage ConvertMatToBitmapImage(Mat mat)
        {
            Bitmap bitmap = mat.ToBitmap();
            using (MemoryStream ms = new MemoryStream())
            {
                bitmap.Save(ms, System.Drawing.Imaging.ImageFormat.Bmp);
                ms.Seek(0, SeekOrigin.Begin);
                BitmapImage img = new BitmapImage();
                img.BeginInit();
                img.StreamSource = ms;
                img.CacheOption = BitmapCacheOption.OnLoad;
                img.EndInit();
                return img;
            }
        }

        protected override void OnClosed(EventArgs e)
        {
            _capture?.Dispose();
            base.OnClosed(e);
        }
    }
}