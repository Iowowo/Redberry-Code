using System.Drawing;
using System.Windows.Forms;

namespace redberry
{
    public partial class Redberry : Form
    {
        #region variables and controls

        MenuStrip menu_strip = new MenuStrip();
        ToolStripMenuItem file_strip = new ToolStripMenuItem(file_strip_name[language]);
        ToolStripMenuItem view_strip = new ToolStripMenuItem(view_strip_name[language]);
        ToolStripMenuItem language_strip = new ToolStripMenuItem("Language");
        ToolStripMenuItem new_tab_button = new ToolStripMenuItem(new_tab_button_name[language]);
        ToolStripMenuItem open_file_strip = new ToolStripMenuItem(open_file_strip_name[language]);
        ToolStripMenuItem open_code_button = new ToolStripMenuItem(open_code_button_name[language]);
        ToolStripMenuItem open_result_button = new ToolStripMenuItem(open_result_button_name[language]);
        ToolStripMenuItem open_greek_alphabet_button = new ToolStripMenuItem(open_greek_alphabet_button_name[language]) { Checked = true, CheckOnClick = true };
        ToolStripMenuItem syntax_highlight_button = new ToolStripMenuItem(syntax_highlight_button_name[language]) { Checked = true, CheckOnClick = true };
        ToolStripMenuItem english_language_button = new ToolStripMenuItem("English") { Checked = true, CheckOnClick = false };
        ToolStripMenuItem russian_language_button = new ToolStripMenuItem("Russian") { Checked = false, CheckOnClick = false };

        Panel menu_strip_line = new Panel();

        #endregion

        private void load_menu_strip()
        {
            menu_strip.Renderer = new menu_strip_renderer();

            file_strip.DropDownItems.Add(new_tab_button);
            file_strip.DropDownItems.Add(open_file_strip);
            open_file_strip.DropDownItems.Add(open_code_button);
            open_file_strip.DropDownItems.Add(open_result_button);

            view_strip.DropDownItems.Add(open_greek_alphabet_button);
            view_strip.DropDownItems.Add(syntax_highlight_button);

            language_strip.DropDownItems.Add(english_language_button);
            language_strip.DropDownItems.Add(russian_language_button);

            menu_strip.Items.Add(file_strip);
            menu_strip.Items.Add(view_strip);
            menu_strip.Items.Add(language_strip);
            menu_strip.Location = new Point(0, 0);
            menu_strip.BackColor = Color.FromArgb(242, 242, 242);

            new_tab_button.Click += new_tab_button_click;
            open_code_button.Click += (sender, e) => open_button_click(sender, e, false);
            open_result_button.Click += (sender, e) => open_button_click(sender, e, true);
            open_greek_alphabet_button.Click += open_greek_alphabet_button_click;
            syntax_highlight_button.Click += syntax_highlight_toggle;
            english_language_button.Click += (sender, e) => change_language(sender, e, 0);
            russian_language_button.Click += (sender, e) => change_language(sender, e, 1);

            Controls.Add(menu_strip);

            menu_strip_line.Size = new Size(Width, menu_strip.Height / 5);
            menu_strip_line.Location = new Point(0, 6 * menu_strip.Height / 5 - 1);
            menu_strip_line.BackColor = Color.FromArgb(214, 214, 214);
            menu_strip_line.Visible = true;

            Controls.Add(menu_strip_line);
        }
    }
}