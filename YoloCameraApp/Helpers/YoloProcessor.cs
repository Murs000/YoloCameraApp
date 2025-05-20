using Emgu.CV;
using Emgu.CV.Dnn;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Windows.Media.Imaging;
using YoloCameraApp.Models;

namespace YoloCameraApp.Helpers
{
    public class YoloProcessor
    {
        private readonly Net _net;
        private readonly List<string> _classLabels;
        private readonly string[] _outputLayerNames;

        public YoloProcessor()
        {
            string baseDir = AppDomain.CurrentDomain.BaseDirectory;
            string modelPath = Path.Combine(baseDir, "Models");

            string configPath = Path.Combine(modelPath, "yolov4.cfg");
            string weightsPath = Path.Combine(modelPath, "yolov4.weights");
            string namesPath = Path.Combine(modelPath, "coco.names");

            _classLabels = File.ReadAllLines(namesPath).ToList();
            _net = DnnInvoke.ReadNetFromDarknet(configPath, weightsPath);

            _net.SetPreferableBackend(Emgu.CV.Dnn.Backend.Cuda);
            _net.SetPreferableTarget(Target.Cuda);

            _outputLayerNames = _net.UnconnectedOutLayersNames;
        }

        public List<YoloPrediction> Predict(Mat frame)
        {
            List<YoloPrediction> results = new();

            float xScale = frame.Width / 416.0f;
            float yScale = frame.Height / 416.0f;

            using (var blob = DnnInvoke.BlobFromImage(frame, 1 / 255.0, new Size(416, 416), new MCvScalar(), true, false))
            {
                _net.SetInput(blob);
                using (VectorOfMat output = new VectorOfMat())
                {
                    _net.Forward(output, _outputLayerNames);

                    List<Rectangle> boxes = new();
                    List<int> classIds = new();
                    List<float> confidences = new();

                    for (int i = 0; i < output.Size; i++)
                    {
                        float[,] data = (float[,])output[i].GetData();

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
                                    float centerX = data[j, 0] * 416;
                                    float centerY = data[j, 1] * 416;
                                    float width = data[j, 2] * 416;
                                    float height = data[j, 3] * 416;

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

                    foreach (int idx in indices)
                    {
                        results.Add(new YoloPrediction
                        {
                            Label = _classLabels[classIds[idx]],
                            Box = boxes[idx],
                            Confidence = confidences[idx]
                        });
                    }
                }
            }

            return results;
        }

        public BitmapImage ConvertMatToBitmapImage(Mat mat)
        {
            using MemoryStream ms = new();
            mat.ToImage<Bgr, byte>().ToBitmap().Save(ms, System.Drawing.Imaging.ImageFormat.Bmp);
            ms.Seek(0, SeekOrigin.Begin);
            BitmapImage image = new();
            image.BeginInit();
            image.StreamSource = ms;
            image.CacheOption = BitmapCacheOption.OnLoad;
            image.EndInit();
            return image;
        }
    }
}
