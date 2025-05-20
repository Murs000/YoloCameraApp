using System.ComponentModel;
using System.Runtime.CompilerServices;

namespace YoloCameraApp.ViewModels
{
    public class ViewModelBase : INotifyPropertyChanged
    {
        public event PropertyChangedEventHandler PropertyChanged;
        protected void SetProperty<T>(ref T field, T value, [CallerMemberName] string propName = "")
        {
            if (!Equals(field, value))
            {
                field = value;
                PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propName));
            }
        }
    }
}
 