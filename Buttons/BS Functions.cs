using System.Windows.Forms;
using System.Drawing;

namespace redberry
{
    public partial class Redberry : Form
    {
        private void load_checkbox_as_button(CheckBox checkbox)
        {
            checkbox.BackColor = Color.White;
            checkbox.Appearance = Appearance.Button;
            checkbox.FlatStyle = FlatStyle.Flat;
            checkbox.FlatAppearance.BorderColor = Color.Gray;
            checkbox.FlatAppearance.BorderSize = border_width;
            checkbox.CheckState = CheckState.Unchecked;
            checkbox.TextAlign = ContentAlignment.MiddleCenter;
            checkbox.Visible = false;

            checkbox.CheckedChanged += button_checkbox_checked;

            Controls.Add(checkbox);
        }

        private void load_greek_button(Button button)
        {
            button.Font = new Font("Times New Roman", font_size, FontStyle.Italic);
            button.Size = new Size(3 * font_size, 3 * font_size);
            button.Visible = false;
            button.BackColor = Color.White;
            button.FlatStyle = FlatStyle.Flat;
            button.FlatAppearance.BorderColor = Color.Gray;
            button.FlatAppearance.BorderSize = border_width;

            button.Click += greek_letter_click;
            
            Controls.Add(button);
        }
    }
}