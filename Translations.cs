using System.Windows.Forms;

namespace redberry
{
    public partial class Redberry : Form
    {
        static string[] file_strip_name = { "File", "Файл" };
        static string[] view_strip_name = { "View", "Вид" };
        static string[] new_tab_button_name = { "Create file", "Создать файл" };
        static string[] new_tab_name = { "New tab", "Новая вкладка" };
        static string[] open_file_strip_name = { "Open file", "Открыть файл" };
        static string[] open_code_button_name = { "Code", "Программа" };
        static string[] open_result_button_name = { "Result", "Результат" };
        static string[] open_greek_alphabet_button_name = { "Greek alphabet", "Греческий алфавит" };
        static string[] syntax_highlight_button_name = { "Syntax highlight", "Посветка синтаксиса" };

        static string[] greek_alphabet_caps_name_lowercase = { "Lowercase letters", "Строчные буквы" };
        static string[] greek_alphabet_caps_name = { "Capital letters", "Заглавные буквы" };

        static string[,] code_context_menu_names = { { "Run", "Запустить" }, { "Save", "Сохранить" }, { "Save as...", "Сохранить как..." }, { "Close", "Закрыть" }, { "Code", "Программа" } };
        static string[,] result_context_menu_names = { { "Save", "Сохранить" }, { "Save as...", "Сохранить как..." }, { "Close", "Закрыть" }, { "Result", "Результат" } };
        static string[] result_of_name = { "Result of", "Результат" };
        static string[] close_tab_messagebox_name = { "Unsaved changes", "Несохранённые изменения" };
        static string[] close_tab_message = { "File wasn't saved. Save changes?", "Файл не был сохранен. Сохранить изменения?" };
        static string[] change_isResult_message = { "Do you want to change tab status?", "Вы уверены, что хотите поменять статус вкладки?" };
    }
}