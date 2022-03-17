using System.Drawing;
using System.Windows.Forms;

namespace redberry
{
    public class menu_strip_renderer : ToolStripProfessionalRenderer
    {
        public menu_strip_renderer() : base(new menu_strip_colors()) { }
    }

    public class menu_strip_colors : ProfessionalColorTable
    {
        public override Color MenuItemSelected
        {
            get { return Color.White; }
        }
        public override Color MenuItemSelectedGradientBegin
        {
            get { return Color.White; }
        }
        public override Color MenuItemSelectedGradientEnd
        {
            get { return Color.White; }
        }
        public override Color MenuItemBorder
        {
            get { return Color.LightGray; }
        }
    }
}
