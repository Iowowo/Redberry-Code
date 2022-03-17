using System;
using System.Windows.Forms;
using System.Drawing;

namespace redberry
{
    public partial class Redberry : Form
    {
        private void button_checkbox_checked(object sender, EventArgs e)
        {
            if ((sender as CheckBox).Checked == true) (sender as CheckBox).BackColor = Color.LightGray;
            else (sender as CheckBox).BackColor = Color.White;
        }

        private void upper_index_button_CheckedChanged(object sender, EventArgs e)
        {
            if (opened_tabs_control.TabCount != 0)
            {
                NumberedRTB NRTB = get_active_NRTB();

                if (upper_index_button.Checked == true)
                {
                    lower_index_button.Checked = false;
                    NRTB.RichTextBox.SelectionCharOffset = 3 * font_size / 4;
                    NRTB.RichTextBox.SelectionFont = new Font("Times New Roman", 4 * font_size / 5, FontStyle.Italic);
                }
                else
                {
                    NRTB.RichTextBox.SelectionCharOffset = 0;
                    NRTB.RichTextBox.SelectionFont = new Font("Times New Roman", font_size, FontStyle.Italic);
                }

                NRTB.Focus();
            }
            else upper_index_button.Checked = false;
        }

        private void lower_index_button_CheckedChanged(object sender, EventArgs e)
        {
            if (opened_tabs_control.TabCount != 0)
            {
                NumberedRTB NRTB = get_active_NRTB();

                if (lower_index_button.Checked == true)
                {
                    upper_index_button.Checked = false;
                    NRTB.RichTextBox.SelectionCharOffset = -font_size / 5;
                    NRTB.RichTextBox.SelectionFont = new Font("Times New Roman", 4 * font_size / 5, FontStyle.Italic);
                }
                else
                {
                    NRTB.RichTextBox.SelectionCharOffset = 0;
                    NRTB.RichTextBox.SelectionFont = new Font("Times New Roman", font_size, FontStyle.Italic);
                }

                NRTB.Focus();
            }
            else lower_index_button.Checked = false;
        }
    }
}