using System;
using System.Windows.Forms;
using System.Text.RegularExpressions;
using System.Drawing;

namespace redberry
{
    public partial class Redberry : Form
    {
        public void syntax_highlight_toggle(object sender, EventArgs e)
        {
            CustomRichTextBox RTB = new CustomRichTextBox();

            foreach (TabPage tab in opened_tabs_control.TabPages)
            {
                foreach (Control control in tab.Controls)
                    if (control.GetType() == typeof(NumberedRTB)) RTB = ((NumberedRTB)control).RichTextBox;


                if (syntax_highlight_button.Checked)
                {
                    RTB.syntax_highlight(null, null);
                    RTB.highlightOn();
                }

                else
                {
                    RTB.highlightOff();
                    opened_tabs_control.Focus();
                    int index = RTB.SelectionStart;
                    RTB.SelectAll();
                    RTB.SelectionColor = Color.Black;
                    RTB.SelectionStart = index;
                    RTB.SelectionLength = 0;
                    RTB.Focus();
                }
            }
        }

        private void print_indices(CustomRichTextBox RTB)
        {
            if (syntax_highlight_button.Checked)
                RTB.highlightOff();

            opened_tabs_control.Focus();
            int index = RTB.SelectionStart;
            RTB.SelectAll();
            RTB.SelectionColor = Color.Black;
            RTB.SelectionStart = index;
            RTB.SelectionLength = 0;
            RTB.Focus();

            MatchCollection tensors = Regex.Matches(RTB.Rtf, "'(?:[^\"'\\\\]|\\\\.)*?'\\.t");
            string tensorReplaced;

            foreach (Match tensor in tensors)
            {
                tensorReplaced = tensor.Value.Replace("_\\{", "\\dn6\\fs32 \\v0").Replace("^\\{", "\\up22\\fs32 \\v0").Replace("\\}", "\\up0\\fs41 \\v0");
                RTB.Rtf = RTB.Rtf.Replace(tensor.Value, tensorReplaced);
            }
            if (syntax_highlight_button.Checked)
            {
                RTB.highlightOn();
                RTB.syntax_highlight(null, null);
            }
        }

        private string create_code(RichTextBox RTB)
        {
            string created_code = "";

            int lastCarretPos = RTB.SelectionStart;
            clicked_tab.Focus();
            for (int n = 0; n < RTB.Text.Length; n++)
            {
                RTB.SelectionStart = n;
                RTB.SelectionLength = 1;

                if (RTB.SelectionCharOffset == 0) created_code += RTB.Text[n].ToString();

                else if (RTB.SelectionCharOffset > 0)
                {
                    created_code += "^{";
                    created_code += RTB.Text[n].ToString();
                    created_code += "}";
                }

                else if (RTB.SelectionCharOffset < 0)
                {
                    created_code += "_{";
                    created_code += RTB.Text[n].ToString();
                    created_code += "}";
                }
            }
            RTB.Parent.Focus();
            RTB.SelectionStart = lastCarretPos;
            RTB.SelectionLength = 0;

            return created_code;
        }

        private string change_double_slashed_to_greek(string text)
        {
            string new_text = text.Replace(@"\\alpha", "α").Replace(@"\\beta", "β").Replace(@"\\gamma", "γ").Replace(@"\\delta", "δ").Replace(@"\\epsilon", "ε").Replace(@"\\zeta", "ζ").Replace(@"\\eta", "η").Replace(@"\\theta", "θ").Replace(@"\\iota", "ι").Replace(@"\\kappa", "κ").Replace(@"\\lambda", "λ").Replace(@"\\mu", "μ").Replace(@"\\nu", "ν").Replace(@"\\xi", "ξ").Replace(@"\\omicron", "ο").Replace(@"\\pi", "π").Replace(@"\\rho", "ρ").Replace(@"\\sigma", "σ").Replace(@"\\tau", "τ").Replace(@"\\upsilon", "υ").Replace(@"\\phi", "φ").Replace(@"\\chi", "χ").Replace(@"\\psi", "ψ").Replace(@"\\omega", "ω").Replace(@"\\ALPHA", "Α").Replace(@"\\BETA", "Β").Replace(@"\\GAMMA", "Γ").Replace(@"\\DELTA", "Δ").Replace(@"\\EPSILON", "Ε").Replace(@"\\ZETA", "Ζ").Replace(@"\\ETA", "Η").Replace(@"\\THETA", "Θ").Replace(@"\\IOTA", "Ι").Replace(@"\\KAPPA", "Κ").Replace(@"\\LAMBDA", "Λ").Replace(@"\\MU", "Μ").Replace(@"\\NU", "Ν").Replace(@"\\XI", "Ξ").Replace(@"\\OMICRON", "Ο").Replace(@"\\PI", "Π").Replace(@"\\RHO", "Ρ").Replace(@"\\SIGMA", "Σ").Replace(@"\\TAU", "Τ").Replace(@"\\UPSILON", "Υ").Replace(@"\\PHI", "Φ").Replace(@"\\CHI", "Χ").Replace(@"\\PSI", "Ψ").Replace(@"\\OMEGA", "Ω");
            return new_text;
        }

        private string change_greek_to_double_slashed(string text)
        {
            string new_text = text.Replace("α", @"\\alpha").Replace("β", @"\\beta").Replace("γ", @"\\gamma").Replace("δ", @"\\delta").Replace("ε", @"\\epsilon").Replace("ζ", @"\\zeta").Replace("η", @"\\eta").Replace("θ", @"\\theta").Replace("ι", @"\\iota").Replace("κ", @"\\kappa").Replace("λ", @"\\lambda").Replace("μ", @"\\mu").Replace("ν", @"\\nu").Replace("ξ", @"\\xi").Replace("ο", @"\\omicron").Replace("π", @"\\pi").Replace("ρ", @"\\rho").Replace("σ", @"\\sigma").Replace("τ", @"\\tau").Replace("υ", @"\\upsilon").Replace("φ", @"\\phi").Replace("χ", @"\\chi").Replace("ψ", @"\\psi").Replace("ω", @"\\omega").Replace("Α", @"\\ALPHA").Replace("Β", @"\\BETA").Replace("Γ", @"\\GAMMA").Replace("Δ", @"\\DELTA").Replace("Ε", @"\\EPSILON").Replace("Ζ", @"\\ZETA").Replace("Η", @"\\ETA").Replace("Θ", @"\\THETA").Replace("Ι", @"\\IOTA").Replace("Κ", @"\\KAPPA").Replace("Λ", @"\\LAMBDA").Replace("Μ", @"\\MU").Replace("Ν", @"\\NU").Replace("Ξ", @"\\XI").Replace("Ο", @"\\OMICRON").Replace("Π", @"\\PI").Replace("Ρ", @"\\RHO").Replace("Σ", @"\\SIGMA").Replace("Τ", @"\\TAU").Replace("Υ", @"\\UPSILON").Replace("Φ", @"\\PHI").Replace("Χ", @"\\CHI").Replace("Ψ", @"\\PSI").Replace("Ω", @"\\OMEGA");
            return new_text;
        }

        private string change_slashed_to_greek(string text)
        {
            string new_text = text.Replace(@"\alpha", "α").Replace(@"\beta", "β").Replace(@"\gamma", "γ").Replace(@"\delta", "δ").Replace(@"\epsilon", "ε").Replace(@"\zeta", "ζ").Replace(@"\eta", "η").Replace(@"\theta", "θ").Replace(@"\iota", "ι").Replace(@"\kappa", "κ").Replace(@"\lambda", "λ").Replace(@"\mu", "μ").Replace(@"\nu", "ν").Replace(@"\xi", "ξ").Replace(@"\omicron", "ο").Replace(@"\pi", "π").Replace(@"\rho", "ρ").Replace(@"\sigma", "σ").Replace(@"\tau", "τ").Replace(@"\upsilon", "υ").Replace(@"\phi", "φ").Replace(@"\chi", "χ").Replace(@"\psi", "ψ").Replace(@"\omega", "ω").Replace(@"\ALPHA", "Α").Replace(@"\BETA", "Β").Replace(@"\GAMMA", "Γ").Replace(@"\DELTA", "Δ").Replace(@"\EPSILON", "Ε").Replace(@"\ZETA", "Ζ").Replace(@"\ETA", "Η").Replace(@"\THETA", "Θ").Replace(@"\IOTA", "Ι").Replace(@"\KAPPA", "Κ").Replace(@"\LAMBDA", "Λ").Replace(@"\MU", "Μ").Replace(@"\NU", "Ν").Replace(@"\XI", "Ξ").Replace(@"\OMICRON", "Ο").Replace(@"\PI", "Π").Replace(@"\RHO", "Ρ").Replace(@"\SIGMA", "Σ").Replace(@"\TAU", "Τ").Replace(@"\UPSILON", "Υ").Replace(@"\PHI", "Φ").Replace(@"\CHI", "Χ").Replace(@"\PSI", "Ψ").Replace(@"\OMEGA", "Ω");
            return new_text;
        }
    }
}