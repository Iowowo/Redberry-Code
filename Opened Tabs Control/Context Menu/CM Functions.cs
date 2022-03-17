using System.Windows.Forms;

namespace redberry
{
    public partial class Redberry : Form
    {
        #region variables and controls

        ContextMenuStrip code_context_menu = new ContextMenuStrip();
        ContextMenuStrip result_context_menu = new ContextMenuStrip();
        TabPage clicked_tab;

        #endregion

        private void load_code_context_menu()
        {
            code_context_menu.Items.Add(code_context_menu_names[0, language], null, run_code_click);
            code_context_menu.Items.Add(code_context_menu_names[1, language], null, (sender, e) => save_file_click(sender, e, true));
            code_context_menu.Items.Add(code_context_menu_names[2, language], null, (sender, e) => save_file_click(sender, e, false));
            code_context_menu.Items.Add(code_context_menu_names[3, language], null, close_tab_click);
            code_context_menu.Items.Add(code_context_menu_names[4, language], null, change_isResult);
        }

        private void load_result_context_menu()
        {
            result_context_menu.Items.Add(result_context_menu_names[0, language], null, (sender, e) => save_file_click(sender, e, true));
            result_context_menu.Items.Add(result_context_menu_names[1, language], null, (sender, e) => save_file_click(sender, e, false));
            result_context_menu.Items.Add(result_context_menu_names[2, language], null, close_tab_click);
            result_context_menu.Items.Add(result_context_menu_names[3, language], null, change_isResult);
        }
    }
}