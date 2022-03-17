using System.Windows.Forms;
using System.Drawing;

namespace redberry
{
    public partial class Redberry : Form
    {
        #region variables and controls

        CheckBox upper_index_button = new CheckBox();
        CheckBox lower_index_button = new CheckBox();

        #endregion

        private void load_index_buttons()
        {
            load_checkbox_as_button(upper_index_button);
            load_checkbox_as_button(lower_index_button);
            upper_index_button.Size = new Size(2 * font_size, 2 * font_size);
            lower_index_button.Size = new Size(2 * font_size, 2 * font_size);
            upper_index_button.Location = new Point(min_blank, 8 * menu_strip.Height / 5 - 1);
            lower_index_button.Location = new Point(min_blank, upper_index_button.Location.Y + upper_index_button.Height + min_blank);
            upper_index_button.Visible = true;
            lower_index_button.Visible = true;
            upper_index_button.Text = "^";
            lower_index_button.Text = "_";
            upper_index_button.CheckedChanged += upper_index_button_CheckedChanged;
            lower_index_button.CheckedChanged += lower_index_button_CheckedChanged;
        }
    }
}
