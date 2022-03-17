using System.Windows.Forms;
using System.Drawing;

namespace redberry
{
    public partial class Redberry : Form
    {
        #region variables and controls

        TabControl opened_tabs_control = new TabControl();

        #endregion

        private void load_opened_tabs_control()
        {
            opened_tabs_control.Visible = false;
            opened_tabs_control.Location = new Point(upper_index_button.Location.X + upper_index_button.Width + min_blank + 1, upper_index_button.Location.Y);
            opened_tabs_control.ItemSize = new Size(opened_tabs_control.ItemSize.Width, upper_index_button.Height - min_blank);
            opened_tabs_control.Selecting += opened_tabs_control_selecting;
            opened_tabs_control.MouseClick += open_tab_context_menu;

            Controls.Add(opened_tabs_control);
        }

        private void reposition_opened_tabs_control()
        {
            if (open_greek_alphabet_button.Checked == true) opened_tabs_control.Size = new Size(greek_alphabet_background.Location.X - opened_tabs_control.Location.X, ClientSize.Height - opened_tabs_control.Location.Y);
            else opened_tabs_control.Size = new Size(ClientSize.Width - opened_tabs_control.Location.X, ClientSize.Height - opened_tabs_control.Location.Y);

            resize_all_NRTBs();
        }

        public NumberedRTB get_active_NRTB()
        {
            foreach (Control control in opened_tabs_control.SelectedTab.Controls)
                if (control.GetType() == typeof(NumberedRTB))
                {
                    NumberedRTB new_NRTB = (NumberedRTB)control;
                    return new_NRTB;
                }
            return null;
        }

        private void resize_NRTB(NumberedRTB NRTB)
        {
            NRTB.Size = new Size(opened_tabs_control.SelectedTab.Size.Width, opened_tabs_control.SelectedTab.Size.Height);
        }

        private void resize_all_NRTBs()
        {
            foreach (TabPage tab in opened_tabs_control.TabPages)
                foreach (Control control in tab.Controls)
                    if (control.GetType() == typeof(NumberedRTB))
                        control.Size = new Size(opened_tabs_control.SelectedTab.Size.Width, opened_tabs_control.SelectedTab.Size.Height);
        }
    }
}