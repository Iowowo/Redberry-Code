using System;
using System.Windows.Forms;

namespace redberry
{
    public partial class Redberry : Form
    {
        private void greek_letter_click(object sender, EventArgs e)
        {
            if (opened_tabs_control.TabCount > 0)
            {
                NumberedRTB NRTB = get_active_NRTB();

                NRTB.RichTextBox.SelectedText = (sender as Button).Text;
            }
        }

        private void greek_alphabet_caps_changed(object sender, EventArgs e)
        {
            if (greek_alphabet_caps.Text.Equals(greek_alphabet_caps_name_lowercase[language]))
            {
                greek_alphabet_caps_checked = true;
                greek_alphabet_caps.Text = greek_alphabet_caps_name[language];
                hide_greek_alphabet();
                visualize_greek_alphabet();
            }
            else
            {
                greek_alphabet_caps_checked = false;
                greek_alphabet_caps.Text = greek_alphabet_caps_name_lowercase[language];
                hide_GREEK_alphabet();
                visualize_greek_alphabet();
            }
        }
    }
}