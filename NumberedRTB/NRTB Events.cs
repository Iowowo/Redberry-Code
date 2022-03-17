using System.Windows.Forms;

namespace redberry
{
    public partial class Redberry : Form
    {
        private void NRTB_press_enter(object sender, KeyEventArgs e)
        {
            if (e.KeyCode == Keys.Enter)
            {
                e.SuppressKeyPress = true;

                upper_index_button.Checked = false;
                lower_index_button.Checked = false;

                ushort left_bracket_counter = 0;
                ushort right_bracket_counter = 0;

                for (int i = 0; i < (sender as RichTextBox).SelectionStart; i++)
                {
                    if ((sender as RichTextBox).Text[i].Equals('{')) left_bracket_counter++;
                    else if ((sender as RichTextBox).Text[i].Equals('}')) right_bracket_counter++;
                }

                (sender as RichTextBox).SelectedText += "\n";
                for (int i = 0; i < left_bracket_counter - right_bracket_counter; i++) (sender as RichTextBox).SelectedText += "\t";
            }
        }

        private void NRTB_input_right_bracket(object sender, KeyEventArgs e)
        {
            if (((sender as RichTextBox).Text.Length > 1) && (e.Modifiers == Keys.Shift && e.KeyCode == Keys.Oem6) && ((sender as RichTextBox).Text[(sender as RichTextBox).SelectionStart - 1].Equals('\t')))
            {
                e.SuppressKeyPress = true;
                (sender as RichTextBox).SelectionStart -= 1;
                (sender as RichTextBox).SelectionLength = 1;
                (sender as RichTextBox).SelectedText = "}";
            }
        }
    }
}