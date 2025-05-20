using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Media;

namespace YoloCameraApp.Models
{
    public class OverlayItem
    {
        public double Left { get; set; }
        public double Top { get; set; }
        public double Width { get; set; }
        public double Height { get; set; }
        public string Label { get; set; }
        public Brush Stroke { get; set; } = Brushes.Red;
        public double StrokeThickness { get; set; } = 2;
    }
}
