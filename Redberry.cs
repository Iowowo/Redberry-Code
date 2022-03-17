using System;
using System.ComponentModel;
using System.Drawing;
using System.Windows.Forms;

namespace redberry
{
    public partial class Redberry : Form
    {
        #region variables and controls

        ushort font_size = 20;
        ushort border_width = 2;
        ushort min_blank = 2;
        bool Cancel = false;
        static ushort language = 0;

        #endregion

        public Redberry()
        {
            InitializeComponent();
            BackColor = Color.FromArgb(242, 242, 242);
            SizeChanged += form_size_changed;
            FormClosing += form_close;
            load_menu_strip();
            load_index_buttons();
            load_opened_tabs_control();
            load_code_context_menu();
            load_result_context_menu();
            load_greek_alphabet();

            reposition_greek_alphabet();
            if (open_greek_alphabet_button.Checked == true) visualize_greek_alphabet();
            reposition_opened_tabs_control();

            MinimumSize = new Size(960, 540);
        }

        private void form_size_changed(object sender, EventArgs e)
        {
            reposition_greek_alphabet();
            reposition_opened_tabs_control();
            menu_strip_line.Size = new Size(ClientSize.Width, menu_strip.Height / 5);
        }

        private void form_close(object sender, CancelEventArgs e)
        {
            foreach (TabPage tab in opened_tabs_control.TabPages)
            {
                clicked_tab = tab;
                opened_tabs_control.SelectedTab = tab;
                close_tab_click(sender, e);
                if (Cancel) e.Cancel = true;
            }
        }
    }
}