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

        public List<YoloPrediction> Predict(Mat originalFrame)
        {
            List<YoloPrediction> results = new();

            int originalWidth = originalFrame.Width;
            int originalHeight = originalFrame.Height;

            // 1. Keep original frame (no resize)
            Mat resized = originalFrame.Clone();

            // 2. Create blob from original image size
            using var blob = DnnInvoke.BlobFromImage(resized, 1 / 255.0, new Size(originalWidth, originalHeight), new MCvScalar(), true, false);
            _net.SetInput(blob);

            // 3. Run forward pass
            using VectorOfMat output = new();
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
                            float centerX = data[j, 0] * originalWidth;
                            float centerY = data[j, 1] * originalHeight;
                            float width = data[j, 2] * originalWidth;
                            float height = data[j, 3] * originalHeight;

                            int x = (int)(centerX - width / 2);
                            int y = (int)(centerY - height / 2);
                            int w = (int)(width);
                            int h = (int)(height);

                            boxes.Add(new Rectangle(x, y, w, h));
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
