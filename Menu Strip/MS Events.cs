using System;
using System.Drawing;
using System.Windows.Forms;
using System.IO;

namespace redberry
{
    public partial class Redberry : Form
    {
        private void new_tab_button_click(object sender, EventArgs e)
        {
            TabPage new_tab = new TabPage(new_tab_name[language]);
            NumberedRTB new_NRTB = new NumberedRTB();

            new_NRTB.RichTextBox.Font = new Font("Times New Roman", font_size, FontStyle.Italic);
            new_NRTB.RichTextBox.AcceptsTab = true;
            new_NRTB.RichTextBox.WordWrap = false;
            new_NRTB.RichTextBox.DetectUrls = false;
            new_NRTB.RichTextBox.KeyDown += NRTB_press_enter;
            new_NRTB.RichTextBox.KeyDown += NRTB_input_right_bracket;

            if (syntax_highlight_button.Checked) new_NRTB.RichTextBox.TextChanged += new_NRTB.RichTextBox.syntax_highlight;

            new_tab.Controls.Add(new_NRTB);

            new_tab.Tag = new tabTag();
            new_tab.Name = new_tab.Text;
            opened_tabs_control.TabPages.Add(new_tab);
            opened_tabs_control.SelectedTab = new_tab;
            opened_tabs_control.Visible = true;
            resize_NRTB(new_NRTB);
            new_NRTB.Focus();
        }

        private void open_greek_alphabet_button_click(object sender, EventArgs e)
        {
            if (open_greek_alphabet_button.Checked == true) { visualize_greek_alphabet(); reposition_opened_tabs_control(); }
            else { greek_alphabet_caps.Visible = false; greek_alphabet_background.Visible = false; hide_greek_alphabet(); hide_GREEK_alphabet(); reposition_opened_tabs_control(); }
        }

        void open_button_click(object sender, EventArgs e, bool isResult)
        {
            OpenFileDialog new_file = new OpenFileDialog();
            string fileContent = "";

            if (isResult) new_file.Filter = "txt files (*.txt)|*.txt|groovy files (*.groovy)|*.groovy|All files (*.*)|*.*";
            else new_file.Filter = "groovy files (*.groovy)|*.groovy|txt files (*.txt)|*.txt|All files (*.*)|*.*";

            if (new_file.ShowDialog() == DialogResult.OK)
            {
                string filePath = new_file.SafeFileName;

                var fileStream = new_file.OpenFile();

                using (StreamReader reader = new StreamReader(fileStream)) fileContent = reader.ReadToEnd();

                fileContent = fileContent.Replace("    ", "\t");

                if (isResult) fileContent = change_slashed_to_greek(fileContent);
                else fileContent = change_double_slashed_to_greek(fileContent);

                new_tab_button_click(sender, e);

                ((tabTag)opened_tabs_control.SelectedTab.Tag).path = new_file.FileName;

                NumberedRTB new_NRTB = get_active_NRTB();

                new_NRTB.Visible = false;
                new_NRTB.RichTextBox.Text = fileContent;
                print_indices(new_NRTB.RichTextBox);
                new_NRTB.Visible = true;

                opened_tabs_control.SelectedTab.Text = filePath;
                opened_tabs_control.SelectedTab.Name = filePath;

                ((tabTag)opened_tabs_control.SelectedTab.Tag).isResult = isResult;
                ((tabTag)opened_tabs_control.SelectedTab.Tag).changed = false;
            }
        }

        private void change_language(object sender, EventArgs e, ushort chosen_language)
        {
            language = chosen_language;

            if ((sender as ToolStripMenuItem).Checked == false)
            {
                if (language == 0)
                {
                    russian_language_button.Checked = false;
                }
                else if (language == 1)
                {
                    english_language_button.Checked = false;
                }

                (sender as ToolStripMenuItem).Checked = true;

                file_strip.Text = file_strip_name[language];
                view_strip.Text = view_strip_name[language];
                new_tab_button.Text = new_tab_button_name[language];
                open_file_strip.Text = open_file_strip_name[language];
                open_code_button.Text = open_code_button_name[language];
                open_result_button.Text = open_result_button_name[language];
                open_greek_alphabet_button.Text = open_greek_alphabet_button_name[language];
                syntax_highlight_button.Text = syntax_highlight_button_name[language];

                if (greek_alphabet_caps_checked == true) greek_alphabet_caps.Text = greek_alphabet_caps_name[language];
                else greek_alphabet_caps.Text = greek_alphabet_caps_name_lowercase[language];

                for (int i = 0; i < 5; i++)  code_context_menu.Items[i].Text = code_context_menu_names[i, language];
                for (int i = 0; i < 4; i++) result_context_menu.Items[i].Text = result_context_menu_names[i, language];
            }
        }
    }
}