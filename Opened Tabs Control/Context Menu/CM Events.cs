using System;
using System.Windows.Forms;
using System.Diagnostics;
using System.IO;

namespace redberry
{
    public partial class Redberry : Form
    {
        private void run_code_click(object sender, EventArgs e)
        {
            save_file_click(null, null, true);

            if (((tabTag)opened_tabs_control.SelectedTab.Tag).path != "")
            {
                string filePath = ((tabTag)opened_tabs_control.SelectedTab.Tag).path;
                string fileName = Path.GetFileNameWithoutExtension(filePath);
                var proc = new Process();
                var startInfo = new ProcessStartInfo("cmd.exe", $"/C groovy \"{filePath}\"");
                startInfo.CreateNoWindow = true;
                startInfo.RedirectStandardOutput = true;
                proc.StartInfo = startInfo;

                proc.Start();
                string output = proc.StandardOutput.ReadToEnd();
                proc.WaitForExit();

                output = output.Replace("    ", "\t");
                new_tab_button_click(null, null);

                NumberedRTB new_NRTB = get_active_NRTB();
                new_NRTB.RichTextBox.Text = change_slashed_to_greek(output);

                opened_tabs_control.SelectedTab.Name = $"{result_of_name[language]} {fileName}";
                opened_tabs_control.SelectedTab.Text = opened_tabs_control.SelectedTab.Name;

                ((tabTag)opened_tabs_control.SelectedTab.Tag).changed = false;
                ((tabTag)opened_tabs_control.SelectedTab.Tag).isResult = true;
            }
        }

        private void close_tab_click(object sender, EventArgs e)
        {
            Cancel = false;
            if (((tabTag)clicked_tab.Tag).changed == true)
            {
                DialogResult result = MessageBox.Show(close_tab_message[language], close_tab_messagebox_name[language], MessageBoxButtons.YesNoCancel);
                if (result == DialogResult.Yes)
                {
                    save_file_click(null, null, true);
                    opened_tabs_control.TabPages.Remove(clicked_tab);
                    if (opened_tabs_control.TabCount == 0) opened_tabs_control.Visible = false;
                }
                else if (result == DialogResult.No)
                {
                    opened_tabs_control.TabPages.Remove(clicked_tab);
                    if (opened_tabs_control.TabCount == 0) opened_tabs_control.Visible = false;
                }
                else Cancel = true;
            }
            else
            {
                opened_tabs_control.TabPages.Remove(clicked_tab);
                if (opened_tabs_control.TabCount == 0) opened_tabs_control.Visible = false;
            }
        }

        private void save_file_click(object sender, EventArgs e, bool flag)
        {
            NumberedRTB new_NRTB = get_active_NRTB();

            if (flag && (((tabTag)clicked_tab.Tag).path != null))
            {
                using (StreamWriter writer = new StreamWriter(((tabTag)clicked_tab.Tag).path))
                {
                    writer.Write(change_greek_to_double_slashed(create_code(new_NRTB.RichTextBox)));

                    clicked_tab.Text = clicked_tab.Name;
                }
            }

            else
            {
                SaveFileDialog saveFile = new SaveFileDialog();
                saveFile.Filter = "groovy files (*.groovy)|*.groovy|txt files (*.txt)|*.txt|All files (*.*)|*.*";

                if (saveFile.ShowDialog() == DialogResult.OK)
                {
                    Stream fileStream;
                    if ((fileStream = saveFile.OpenFile()) != null)
                    {
                        string fileName = saveFile.FileName;
                        using (StreamWriter writer = new StreamWriter(fileStream))
                        {
                            writer.Write(change_greek_to_double_slashed(create_code(new_NRTB.RichTextBox)));
                        }
                        clicked_tab.Text = new FileInfo(saveFile.FileName).Name;
                        clicked_tab.Name = clicked_tab.Text;
                        fileStream.Close();

                        ((tabTag)clicked_tab.Tag).changed = true;
                        ((tabTag)clicked_tab.Tag).path = fileName;
                    }
                }
            }
        }

        private void change_isResult(object sender, EventArgs e)
        {
            DialogResult result = MessageBox.Show(change_isResult_message[language], clicked_tab.Text, MessageBoxButtons.YesNo);
            if (result == DialogResult.Yes)
            {
                ((tabTag)opened_tabs_control.SelectedTab.Tag).isResult ^= true;
            }
        }
    }
}