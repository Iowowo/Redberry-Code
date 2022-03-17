using System.Drawing;
using System.Windows.Forms;

namespace redberry
{
    public partial class Redberry : Form
    {
        #region variables and controls

        bool greek_alphabet_caps_checked = false;

        Button greek_alphabet_caps = new Button();
        Panel greek_alphabet_background = new Panel();

        Button alpha_button = new Button() { Text = "α", TextAlign = ContentAlignment.MiddleCenter };
        Button beta_button = new Button() { Text = "β", TextAlign = ContentAlignment.MiddleCenter };
        Button gamma_button = new Button() { Text = "γ", TextAlign = ContentAlignment.MiddleCenter };
        Button delta_button = new Button() { Text = "δ", TextAlign = ContentAlignment.MiddleCenter };
        Button epsilon_button = new Button() { Text = "ε", TextAlign = ContentAlignment.MiddleCenter };
        Button zeta_button = new Button() { Text = "ζ", TextAlign = ContentAlignment.MiddleCenter };
        Button eta_button = new Button() { Text = "η", TextAlign = ContentAlignment.MiddleCenter };
        Button theta_button = new Button() { Text = "θ", TextAlign = ContentAlignment.MiddleCenter };
        Button iota_button = new Button() { Text = "ι", TextAlign = ContentAlignment.MiddleCenter };
        Button kappa_button = new Button() { Text = "κ", TextAlign = ContentAlignment.MiddleCenter };
        Button lambda_button = new Button() { Text = "λ", TextAlign = ContentAlignment.MiddleCenter };
        Button mu_button = new Button() { Text = "μ", TextAlign = ContentAlignment.MiddleCenter };
        Button nu_button = new Button() { Text = "ν", TextAlign = ContentAlignment.MiddleCenter };
        Button xi_button = new Button() { Text = "ξ", TextAlign = ContentAlignment.MiddleCenter };
        Button omicron_button = new Button() { Text = "ο", TextAlign = ContentAlignment.MiddleCenter };
        Button pi_button = new Button() { Text = "π", TextAlign = ContentAlignment.MiddleCenter };
        Button rho_button = new Button() { Text = "ρ", TextAlign = ContentAlignment.MiddleCenter };
        Button sigma_button = new Button() { Text = "σ", TextAlign = ContentAlignment.MiddleCenter };
        Button tau_button = new Button() { Text = "τ", TextAlign = ContentAlignment.MiddleCenter };
        Button upsilon_button = new Button() { Text = "υ", TextAlign = ContentAlignment.MiddleCenter };
        Button phi_button = new Button() { Text = "φ", TextAlign = ContentAlignment.MiddleCenter };
        Button chi_button = new Button() { Text = "χ", TextAlign = ContentAlignment.MiddleCenter };
        Button psi_button = new Button() { Text = "ψ", TextAlign = ContentAlignment.MiddleCenter };
        Button omega_button = new Button() { Text = "ω", TextAlign = ContentAlignment.MiddleCenter };

        Button ALPHA_button = new Button() { Text = "Α", TextAlign = ContentAlignment.MiddleCenter };
        Button BETA_button = new Button() { Text = "Β", TextAlign = ContentAlignment.MiddleCenter };
        Button GAMMA_button = new Button() { Text = "Γ", TextAlign = ContentAlignment.MiddleCenter };
        Button DELTA_button = new Button() { Text = "Δ", TextAlign = ContentAlignment.MiddleCenter };
        Button EPSILON_button = new Button() { Text = "Ε", TextAlign = ContentAlignment.MiddleCenter };
        Button ZETA_button = new Button() { Text = "Ζ", TextAlign = ContentAlignment.MiddleCenter };
        Button ETA_button = new Button() { Text = "Η", TextAlign = ContentAlignment.MiddleCenter };
        Button THETA_button = new Button() { Text = "Θ", TextAlign = ContentAlignment.MiddleCenter };
        Button IOTA_button = new Button() { Text = "Ι", TextAlign = ContentAlignment.MiddleCenter };
        Button KAPPA_button = new Button() { Text = "Κ", TextAlign = ContentAlignment.MiddleCenter };
        Button LAMBDA_button = new Button() { Text = "Λ", TextAlign = ContentAlignment.MiddleCenter };
        Button MU_button = new Button() { Text = "Μ", TextAlign = ContentAlignment.MiddleCenter };
        Button NU_button = new Button() { Text = "Ν", TextAlign = ContentAlignment.MiddleCenter };
        Button XI_button = new Button() { Text = "Ξ", TextAlign = ContentAlignment.MiddleCenter };
        Button OMICRON_button = new Button() { Text = "Ο", TextAlign = ContentAlignment.MiddleCenter };
        Button PI_button = new Button() { Text = "Π", TextAlign = ContentAlignment.MiddleCenter };
        Button RHO_button = new Button() { Text = "Ρ", TextAlign = ContentAlignment.MiddleCenter };
        Button SIGMA_button = new Button() { Text = "Σ", TextAlign = ContentAlignment.MiddleCenter };
        Button TAU_button = new Button() { Text = "Τ", TextAlign = ContentAlignment.MiddleCenter };
        Button UPSILON_button = new Button() { Text = "Υ", TextAlign = ContentAlignment.MiddleCenter };
        Button PHI_button = new Button() { Text = "Φ", TextAlign = ContentAlignment.MiddleCenter };
        Button CHI_button = new Button() { Text = "Χ", TextAlign = ContentAlignment.MiddleCenter };
        Button PSI_button = new Button() { Text = "Ψ", TextAlign = ContentAlignment.MiddleCenter };
        Button OMEGA_button = new Button() { Text = "Ω", TextAlign = ContentAlignment.MiddleCenter };

        #endregion

        private void load_greek_alphabet()
        {
            #region greek alphabet

            load_greek_button(alpha_button);
            load_greek_button(beta_button);
            load_greek_button(gamma_button);
            load_greek_button(delta_button);
            load_greek_button(epsilon_button);
            load_greek_button(zeta_button);
            load_greek_button(eta_button);
            load_greek_button(theta_button);
            load_greek_button(iota_button);
            load_greek_button(kappa_button);
            load_greek_button(lambda_button);
            load_greek_button(mu_button);
            load_greek_button(nu_button);
            load_greek_button(xi_button);
            load_greek_button(omicron_button);
            load_greek_button(pi_button);
            load_greek_button(rho_button);
            load_greek_button(sigma_button);
            load_greek_button(tau_button);
            load_greek_button(upsilon_button);
            load_greek_button(phi_button);
            load_greek_button(chi_button);
            load_greek_button(psi_button);
            load_greek_button(omega_button);
            load_greek_button(ALPHA_button);
            load_greek_button(BETA_button);
            load_greek_button(GAMMA_button);
            load_greek_button(DELTA_button);
            load_greek_button(EPSILON_button);
            load_greek_button(ZETA_button);
            load_greek_button(ETA_button);
            load_greek_button(THETA_button);
            load_greek_button(IOTA_button);
            load_greek_button(KAPPA_button);
            load_greek_button(LAMBDA_button);
            load_greek_button(MU_button);
            load_greek_button(NU_button);
            load_greek_button(XI_button);
            load_greek_button(OMICRON_button);
            load_greek_button(PI_button);
            load_greek_button(RHO_button);
            load_greek_button(SIGMA_button);
            load_greek_button(TAU_button);
            load_greek_button(UPSILON_button);
            load_greek_button(PHI_button);
            load_greek_button(CHI_button);
            load_greek_button(PSI_button);
            load_greek_button(OMEGA_button);

            #endregion

            greek_alphabet_caps.Size = new Size(4 * alpha_button.Size.Width + 3 * min_blank, 2 * font_size);
            greek_alphabet_caps.Visible = false;
            greek_alphabet_caps.TextAlign = ContentAlignment.MiddleCenter;
            greek_alphabet_caps.BackColor = Color.White;
            greek_alphabet_caps.FlatStyle = FlatStyle.Flat;
            greek_alphabet_caps.FlatAppearance.BorderColor = Color.Gray;
            greek_alphabet_caps.FlatAppearance.BorderSize = border_width;
            greek_alphabet_caps.Text = greek_alphabet_caps_name_lowercase[language];
            greek_alphabet_caps.Click += greek_alphabet_caps_changed;
            Controls.Add(greek_alphabet_caps);

            greek_alphabet_background.Size = new Size(greek_alphabet_caps.Size.Width + 2 * min_blank, greek_alphabet_caps.Size.Height + 6 * alpha_button.Size.Height + 8 * min_blank);
            greek_alphabet_background.Visible = false;
            greek_alphabet_background.BackColor = Color.White;
            Controls.Add(greek_alphabet_background);
        }

        private void reposition_greek_alphabet()
        {
            greek_alphabet_background.Location = new Point(ClientSize.Width - greek_alphabet_background.Width, upper_index_button.Location.Y - min_blank);
            greek_alphabet_caps.Location = new Point(greek_alphabet_background.Location.X + min_blank, upper_index_button.Location.Y);

            #region greek alphabet

            alpha_button.Location = new Point(greek_alphabet_background.Location.X + min_blank, greek_alphabet_caps.Location.Y + greek_alphabet_caps.Height + min_blank);
            ALPHA_button.Location = new Point(alpha_button.Location.X, alpha_button.Location.Y);
            beta_button.Location = new Point(alpha_button.Location.X + alpha_button.Width + min_blank, alpha_button.Location.Y);
            BETA_button.Location = new Point(beta_button.Location.X, alpha_button.Location.Y);
            gamma_button.Location = new Point(beta_button.Location.X + beta_button.Width + min_blank, alpha_button.Location.Y);
            GAMMA_button.Location = new Point(gamma_button.Location.X, alpha_button.Location.Y);
            delta_button.Location = new Point(gamma_button.Location.X + gamma_button.Width + min_blank, alpha_button.Location.Y);
            DELTA_button.Location = new Point(delta_button.Location.X, alpha_button.Location.Y);
            epsilon_button.Location = new Point(alpha_button.Location.X, alpha_button.Location.Y + alpha_button.Height + min_blank);
            EPSILON_button.Location = new Point(alpha_button.Location.X, epsilon_button.Location.Y);
            zeta_button.Location = new Point(beta_button.Location.X, epsilon_button.Location.Y);
            ZETA_button.Location = new Point(beta_button.Location.X, epsilon_button.Location.Y);
            eta_button.Location = new Point(gamma_button.Location.X, epsilon_button.Location.Y);
            ETA_button.Location = new Point(gamma_button.Location.X, epsilon_button.Location.Y);
            theta_button.Location = new Point(delta_button.Location.X, epsilon_button.Location.Y);
            THETA_button.Location = new Point(delta_button.Location.X, epsilon_button.Location.Y);
            iota_button.Location = new Point(alpha_button.Location.X, epsilon_button.Location.Y + epsilon_button.Height + min_blank);
            IOTA_button.Location = new Point(alpha_button.Location.X, iota_button.Location.Y);
            kappa_button.Location = new Point(beta_button.Location.X, iota_button.Location.Y);
            KAPPA_button.Location = new Point(beta_button.Location.X, iota_button.Location.Y);
            lambda_button.Location = new Point(gamma_button.Location.X, iota_button.Location.Y);
            LAMBDA_button.Location = new Point(gamma_button.Location.X, iota_button.Location.Y);
            mu_button.Location = new Point(delta_button.Location.X, iota_button.Location.Y);
            MU_button.Location = new Point(delta_button.Location.X, iota_button.Location.Y);
            nu_button.Location = new Point(alpha_button.Location.X, iota_button.Location.Y + iota_button.Height + min_blank);
            NU_button.Location = new Point(alpha_button.Location.X, nu_button.Location.Y);
            xi_button.Location = new Point(beta_button.Location.X, nu_button.Location.Y);
            XI_button.Location = new Point(beta_button.Location.X, nu_button.Location.Y);
            omicron_button.Location = new Point(gamma_button.Location.X, nu_button.Location.Y);
            OMICRON_button.Location = new Point(gamma_button.Location.X, nu_button.Location.Y);
            pi_button.Location = new Point(delta_button.Location.X, nu_button.Location.Y);
            PI_button.Location = new Point(delta_button.Location.X, nu_button.Location.Y);
            rho_button.Location = new Point(alpha_button.Location.X, nu_button.Location.Y + nu_button.Height + min_blank);
            RHO_button.Location = new Point(alpha_button.Location.X, rho_button.Location.Y);
            sigma_button.Location = new Point(beta_button.Location.X, rho_button.Location.Y);
            SIGMA_button.Location = new Point(beta_button.Location.X, rho_button.Location.Y);
            tau_button.Location = new Point(gamma_button.Location.X, rho_button.Location.Y);
            TAU_button.Location = new Point(gamma_button.Location.X, rho_button.Location.Y);
            upsilon_button.Location = new Point(delta_button.Location.X, rho_button.Location.Y);
            UPSILON_button.Location = new Point(delta_button.Location.X, rho_button.Location.Y);
            phi_button.Location = new Point(alpha_button.Location.X, rho_button.Location.Y + rho_button.Height + min_blank);
            PHI_button.Location = new Point(alpha_button.Location.X, phi_button.Location.Y);
            chi_button.Location = new Point(beta_button.Location.X, phi_button.Location.Y);
            CHI_button.Location = new Point(beta_button.Location.X, phi_button.Location.Y);
            psi_button.Location = new Point(gamma_button.Location.X, phi_button.Location.Y);
            PSI_button.Location = new Point(gamma_button.Location.X, phi_button.Location.Y);
            omega_button.Location = new Point(delta_button.Location.X, phi_button.Location.Y);
            OMEGA_button.Location = new Point(delta_button.Location.X, phi_button.Location.Y);

            #endregion
        }

        private void visualize_greek_alphabet()
        {
            greek_alphabet_caps.Visible = true;
            greek_alphabet_background.Visible = true;
            greek_alphabet_background.SendToBack();

            #region greek alphabet

            if (greek_alphabet_caps.Text.Equals(greek_alphabet_caps_name[language]))
            {
                ALPHA_button.Visible = true;
                BETA_button.Visible = true;
                GAMMA_button.Visible = true;
                DELTA_button.Visible = true;
                EPSILON_button.Visible = true;
                ZETA_button.Visible = true;
                ETA_button.Visible = true;
                THETA_button.Visible = true;
                IOTA_button.Visible = true;
                KAPPA_button.Visible = true;
                LAMBDA_button.Visible = true;
                MU_button.Visible = true;
                NU_button.Visible = true;
                XI_button.Visible = true;
                OMICRON_button.Visible = true;
                PI_button.Visible = true;
                RHO_button.Visible = true;
                SIGMA_button.Visible = true;
                TAU_button.Visible = true;
                UPSILON_button.Visible = true;
                PHI_button.Visible = true;
                CHI_button.Visible = true;
                PSI_button.Visible = true;
                OMEGA_button.Visible = true;
            }
            else
            {
                alpha_button.Visible = true;
                beta_button.Visible = true;
                gamma_button.Visible = true;
                delta_button.Visible = true;
                epsilon_button.Visible = true;
                zeta_button.Visible = true;
                eta_button.Visible = true;
                theta_button.Visible = true;
                iota_button.Visible = true;
                kappa_button.Visible = true;
                lambda_button.Visible = true;
                mu_button.Visible = true;
                nu_button.Visible = true;
                xi_button.Visible = true;
                omicron_button.Visible = true;
                pi_button.Visible = true;
                rho_button.Visible = true;
                sigma_button.Visible = true;
                tau_button.Visible = true;
                upsilon_button.Visible = true;
                phi_button.Visible = true;
                chi_button.Visible = true;
                psi_button.Visible = true;
                omega_button.Visible = true;
            }

            #endregion
        }

        private void hide_greek_alphabet()
        {
            alpha_button.Visible = false;
            beta_button.Visible = false;
            gamma_button.Visible = false;
            delta_button.Visible = false;
            epsilon_button.Visible = false;
            zeta_button.Visible = false;
            eta_button.Visible = false;
            theta_button.Visible = false;
            iota_button.Visible = false;
            kappa_button.Visible = false;
            lambda_button.Visible = false;
            mu_button.Visible = false;
            nu_button.Visible = false;
            xi_button.Visible = false;
            omicron_button.Visible = false;
            pi_button.Visible = false;
            rho_button.Visible = false;
            sigma_button.Visible = false;
            tau_button.Visible = false;
            upsilon_button.Visible = false;
            phi_button.Visible = false;
            chi_button.Visible = false;
            psi_button.Visible = false;
            omega_button.Visible = false;
        }

        private void hide_GREEK_alphabet()
        {
            ALPHA_button.Visible = false;
            BETA_button.Visible = false;
            GAMMA_button.Visible = false;
            DELTA_button.Visible = false;
            EPSILON_button.Visible = false;
            ZETA_button.Visible = false;
            ETA_button.Visible = false;
            THETA_button.Visible = false;
            IOTA_button.Visible = false;
            KAPPA_button.Visible = false;
            LAMBDA_button.Visible = false;
            MU_button.Visible = false;
            NU_button.Visible = false;
            XI_button.Visible = false;
            OMICRON_button.Visible = false;
            PI_button.Visible = false;
            RHO_button.Visible = false;
            SIGMA_button.Visible = false;
            TAU_button.Visible = false;
            UPSILON_button.Visible = false;
            PHI_button.Visible = false;
            CHI_button.Visible = false;
            PSI_button.Visible = false;
            OMEGA_button.Visible = false;
        }
    }
}