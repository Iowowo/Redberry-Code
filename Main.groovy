import cc.redberry.core.context.CC
import cc.redberry.groovy.Redberry
import static cc.redberry.core.context.OutputFormat.*
import static cc.redberry.groovy.RedberryPhysics.*
import static cc.redberry.groovy.RedberryStatic.*
import static cc.redberry.core.tensor.Tensors.*

use(Redberry){
    CC.current().setMetricName('\\eta')
    addSymmetry 'h_mn[x^o]',  [1, 0].p
    def dim = 'd^a_a = 4'.t
    def haa = 'h^a_a[x^o] = h[x^o]'.t
    def check = 'a = 1'.t & 'b = 0'.t & 'c = 0'.t & 'd = 0'.t & 'e = 0'.t & 'f = 0'.t & 'l = 0'.t & 'm = 0'.t & 'n = 0'.t & 'o = 0'.t & 'p = 0'.t & 'q = 0'.t & 'r = 0'.t & 'A = 0'.t & 'B = 0'.t & 'C = 0'.t & 'D = 0'.t & 'E = 0'.t & 'F = 0'.t & 'G = 0'.t & 'H = 0'.t & 'J = 0'.t & 'K = 0'.t & 'L = 0'.t & 'M = 0'.t
    def checkLagr = '\\alpha = 1'.t & '\\Lambda = 0'.t & '\\beta = 0'.t & '\\gamma = 0'.t & '\\delta = 0'.t
    def fix = 'a**2 = a**(2.0)'.t & 'b**2 = b**(2.0)'.t
    def collect = Collect['a', 'b', 'c', 'd', 'e', 'f', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', Factor]
    def collectnf = Collect['a', 'b', 'c', 'd', 'e', 'f', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M']

    def g__mn = ['\\eta_mn'.t, 'a * h_mn[x^o]   +   b * \\eta_mn * h[x^o]'.t,
                 'c * h_mn[x^o] * h[x^o]   +   d * h_ma[x^o] * h^a_n[x^o]   +   e * \\eta_mn * (h[x^o])**2   +   f * \\eta_mn * h^ab[x^o] * h_ab[x^o]'.t,
                 'l * h_mn[x^o] * (h[x^o])**2   +   m * h_mn[x^o] * h^ab[x^o] * h_ab[x^o]   +   n * h_ma[x^o] * h^a_n[x^o] * h[x^o]   +   o * h_ma[x^o] * h^ab[x^o] * h_bn[x^o]   +   p * \\eta_mn * (h[x^o])**3   +   q * \\eta_mn * h^ab[x^o] * h_ab[x^o] * h[x^o]   +   r * \\eta_nm * h_ab[x^o] * h^bc[x^o] * h_c^a[x^o]'.t,
                 'A * h_mn[x^o] * (h[x^o])**3   +   B * h_mn[x^o] * h_ab[x^o] * h^ab[x^o] * h[x^o]   +   C * h_mn[x^o] * h_ab[x^o] * h^bc[x^o] * h_c^a[x^o]   +   D * h_ma[x^o] * h^a_n[x^o] * (h[x^o])**2   +   E * h_ma[x^o] * h^a_n[x^o] * h_bc[x^o] * h^bc[x^o]   +   F * h_ma[x^o] * h^ab[x^o] * h_bn[x^o] * h[x^o]   +   G * h_ma[x^o] * h^ab[x^o] * h_bc[x^o] * h^c_n[x^o]   +   H * \\eta_mn * (h[x^o])**4   +   J * \\eta_mn * h_ab[x^o] * h^ab[x^o] * (h[x^o])**2   +   K * \\eta_mn * h_ab[x^o] * h^ab[x^o] * h_cd[x^o] * h^cd[x^o]   +   L * \\eta_mn * h_ab[x^o] * h^bc[x^o] * h_cd[x^o] * h^da[x^o]   +   M * \\eta_mn * h_ab[x^o] * h^bc[x^o] * h_c^a[x^o] * h[x^o]'.t]

    def g_mn = ['\\eta^mn'.t, '0'.t, '0'.t, '0'.t, '0'.t]
    def __mn$__ab = '{_m -> _a, _n -> _b}'.mapping
    def _n$_a = '{^n -> ^a}'.mapping
    for (int i = 1; i < 5; i++){
        for (int j = 0; j < i; j++){
            g_mn[i] = sum(g_mn[i], -(__mn$__ab >> g__mn[i-j]) * (_n$_a >> g_mn[j]) * '\\eta^nb'.t)
        }
        g_mn[i] = dim >> (haa >> (ExpandAndEliminate >> g_mn[i]))
    }
    /*
    for (int i = 0; i < 5; i++){
        new File("results/g_mn/g_mn[${i}].txt").withWriter('utf-8') {
            writer -> writer.writeLine(g_mn[i].toString())
        }
    }
    */
    def Gamma_l_mn = ['0'.t, '0'.t, '0'.t, '0'.t, '0'.t]
    def _mn$_lk = '{^m -> ^l, ^n -> ^k}'.mapping
    def __mn$__nk = '{_m -> _n, _n -> _k}'.mapping
    def __n$__k = '{_n -> _k}'.mapping

    for (int i = 0; i < 5; i++) {
        for (int j = 0; j <= i; j++) {
            temp1 = Differentiate['x^m'] >> (__mn$__nk >> g__mn[i-j])
            temp2 = Differentiate['x^n'] >> (__n$__k >> g__mn[i-j])
            temp3 = Differentiate['x^k'] >> g__mn[i-j]
            temp = sum(temp1, temp2, -temp3)
            Gamma_l_mn[i] = sum(Gamma_l_mn[i],   0.5   *   (_mn$_lk >> g_mn[j])   *   temp)
        }
        Gamma_l_mn[i] = collect >> (dim >> (haa >> (ExpandAndEliminate >> Gamma_l_mn[i])))
    }
    /*
    for (int i = 0; i < 5; i++){
        new File("results/Gamma/Gamma[${i}].txt").withWriter('utf-8') {
            writer -> writer.writeLine(Gamma_l_mn[i].toString())
        }
    }
    */
    def R_l__mnk = ['0'.t, '0'.t, '0'.t, '0'.t, '0'.t]
    def _l__n$_a__k = '{^l -> ^a, _n -> _k}'.mapping
    def __m$__a = '{_m -> _a}'.mapping
    def _l$_a = '{^l -> ^a}'.mapping
    def __mn$__ak = '{_m -> _a, _n -> _k}'.mapping

    for (int i = 0; i < 5; i++){
        temp1 = Differentiate['x^n'] >> (__n$__k >> Gamma_l_mn[i])
        temp2 = Differentiate['x^k'] >> (-Gamma_l_mn[i])
        temp3 = '0'.t
        temp4 = '0'.t
        for (int j = 0; j <= i; j++){
            temp3 = sum(temp3, (_l__n$_a__k >> Gamma_l_mn[j]) * (__m$__a >> Gamma_l_mn[i-j]))
            temp4 = sum(temp4, -(_l$_a >> Gamma_l_mn[j]) * (__mn$__ak >> Gamma_l_mn[i-j]))
        }
        R_l__mnk[i] = sum(temp1, temp2, temp3, temp4)
        R_l__mnk[i] = CollectNonScalars >> (dim >> (haa >> (ExpandAndEliminate >> R_l__mnk[i])))
    }
    /*
    for (int i = 0; i < 5; i++){
        new File("results/R_l__mnk/R_l__mnk[${i}].txt").withWriter('utf-8') {
            writer -> writer.writeLine(R_l__mnk[i].toString())
        }
    }
    */
    def R__mn = ['0'.t, '0'.t, '0'.t, '0'.t, '0'.t]
    def __k$__n = '{_k -> _n}'.mapping

    for (int i = 0; i < 5; i++){
        R__mn[i] = collectnf >> (__k$__n >> (dim >> (haa >> (ExpandAndEliminate >> 'd_a^n'.t * 'd_l^a'.t * R_l__mnk[i]))))
    }
    /*
    for (int i = 0; i < 5; i++){
        new File("results/R__mn/R__mn[${i}].txt").withWriter('utf-8') {
            writer -> writer.writeLine(R__mn[i].toString())
        }
    }
    */
    def R = ['0'.t, '0'.t, '0'.t, '0'.t, '0'.t]

    for (int i = 0; i < 5; i++){
        for (int j = 0; j <= i; j++){
            R[i] = sum(R[i], g_mn[j] * R__mn[i-j])
        }
        R[i] = collectnf >> (dim >> (haa >> (ExpandAndEliminate >> R[i])))
    }
    /*
    for (int i = 0; i < 5; i++){
        new File("results/R/R[${i}].txt").withWriter('utf-8') {
            writer -> writer.writeLine(R[i].toString())
        }
    }
    */
    def R_m__n = ['0'.t, '0'.t, '0'.t, '0'.t, '0'.t]
    def R_mn = ['0'.t, '0'.t, '0'.t, '0'.t, '0'.t]
    def RR = ['0'.t, '0'.t, '0'.t, '0'.t, '0'.t]
    def _m$_a = '{^m -> ^a}'.mapping
    def __n$__a = '{_n -> _a}'.mapping

    for (int i = 0; i < 5; i++){
        for (int j = 0; j <= i; j++){
            R_m__n[i] = sum(R_m__n[i], (_n$_a >> g_mn[j]) * (__m$__a >> R__mn[i-j]))
        }
        R_m__n[i] = dim >> (haa >> (ExpandAndEliminate >> R_m__n[i]))
    }
    for (int i = 0; i < 5; i++){
        for (int j = 0; j <= i; j++){
            R_mn[i] = sum(R_mn[i], (_m$_a >> g_mn[j]) * (__n$__a >> R_m__n[i-j]))
        }
        R_mn[i] = dim >> (haa >> (ExpandAndEliminate >> R_mn[i]))
    }
    for (int i = 0; i < 5; i++){
        for (int j = 0; j <= i; j++){
            RR[i] = sum(RR[i], R__mn[j] * R_mn[i-j])
        }
        RR[i] = collectnf >> (dim >> (haa >> (ExpandAndEliminate >> RR[i])))
    }
    /*
    for (int i = 0; i < 5; i++){
        new File("results/RR/RR[${i}].txt").withWriter('utf-8') {
            writer -> writer.writeLine(RR[i].toString())
        }
    }
    */
    def R2 = ['0'.t, '0'.t, '0'.t, '0'.t, '0'.t]

    for (int i = 0; i < 5; i++){
        for (int j = 0; j <= i; j++){
            R2[i] = sum(R2[i], R[j] * R[i-j])
        }
        R2[i] = collectnf >> (dim >> (haa >> (ExpandAndEliminate >> R2[i])))
    }
    /*
    for (int i = 0; i < 5; i++){
        new File("results/R2/R2[${i}].txt").withWriter('utf-8') {
            writer -> writer.writeLine(R2[i].toString())
        }
    }
    */
    def nnR = ['0'.t, '0'.t, '0'.t, '0'.t, '0'.t]

    for (int i = 0; i < 5; i++){
        temp1 = Differentiate['x^a'] >> (Differentiate['x_a'] >> R[i])
        temp2 = '0'.t
        for (int j = 0; j <= i; j++){
            temp2 = sum(temp2, ('d_l^m'.t * 'd^n_b'.t * Gamma_l_mn[j]) * (Differentiate['x_b'] >> R[i-j]))
        }
        new File("results/GammaR/GammaR[${i}].txt").withWriter('utf-8') {
            writer -> writer.writeLine((collectnf >> (dim >> (haa >> (ExpandAndEliminate >> temp2)))).toString())
        }
        nnR[i] = sum(temp1, temp2)
        nnR[i] = collectnf >> (dim >> (haa >> (ExpandAndEliminate >> nnR[i])))
    }
    /*
    for (int i = 0; i < 5; i++){
        new File("results/nnR/nnR[${i}].txt").withWriter('utf-8') {
            writer -> writer.writeLine(nnR[i].toString())
        }
    }
    */
    def g_m_n = ['0'.t, '0'.t, '0'.t, '0'.t, '0'.t]
    def __mn$_m_n = '{_m -> ^m}'.mapping
    for (int i = 0; i < 5; i++) {
        g_m_n[i] = __mn$_m_n >> g__mn[i]
    }

    def _m_n$_m_a = '{_n -> _a}'.mapping
    def _m_n$_a_n = '{^m -> ^a}'.mapping
    def _m_n$_a_b = '{^m -> ^a, _n -> _b}'.mapping
    def _m_n$_b_n = '{^m -> ^b}'.mapping
    def _m_n$_b_c = '{^m -> ^b, _n -> _c}'.mapping
    def _m_n$_c_n = '{^m -> ^c}'.mapping

    def H_m_n = ['0'.t, '0'.t, '0'.t, '0'.t, '0'.t]
    H_m_n[0] = '0'.t
    H_m_n[1] = dim >> (haa >> (ExpandAndEliminate >> g_m_n[1] / 2))
    H_m_n[2] = dim >> (haa >> (ExpandAndEliminate >> sum(g_m_n[2], -(_m_n$_m_a >> g_m_n[1]) * (_m_n$_a_n >> g_m_n[1]) / 2) / 2))
    H_m_n[3] = dim >> (haa >> (ExpandAndEliminate >> sum(g_m_n[3], -(_m_n$_m_a >> g_m_n[1]) * (_m_n$_a_n >> g_m_n[2]) / 2, -(_m_n$_m_a >> g_m_n[2]) * (_m_n$_a_n >> g_m_n[1]) / 2, (_m_n$_m_a >> g_m_n[1]) * (_m_n$_a_b >> g_m_n[1]) * (_m_n$_b_n >> g_m_n[1]) / 3) / 2))
    H_m_n[4] = dim >> (haa >> (ExpandAndEliminate >> sum(g_m_n[4], -(_m_n$_m_a >> g_m_n[1]) * (_m_n$_a_n >> g_m_n[3]) / 2, (_m_n$_m_a >> g_m_n[2]) * (_m_n$_a_n >> g_m_n[2]) / 2, (_m_n$_m_a >> g_m_n[3]) * (_m_n$_a_n >> g_m_n[1]) / 2, (_m_n$_m_a >> g_m_n[1]) * (_m_n$_a_b >> g_m_n[1]) * (_m_n$_b_n >> g_m_n[2]) / 3, (_m_n$_m_a >> g_m_n[1]) * (_m_n$_a_b >> g_m_n[2]) * (_m_n$_b_n >> g_m_n[1]) / 3, (_m_n$_m_a >> g_m_n[2]) * (_m_n$_a_b >> g_m_n[1]) * (_m_n$_b_n >> g_m_n[1]) / 3, -(_m_n$_m_a >> g_m_n[1]) * (_m_n$_a_b >> g_m_n[1]) * (_m_n$_b_c >> g_m_n[1]) * (_m_n$_c_n >> g_m_n[1]) / 4) / 2))

    def det = ['1'.t, '0'.t, '0'.t, '0'.t, '0'.t]
    det[1] = ExpandAndEliminate >> (dim >> (haa >> (ExpandAndEliminate >> ('d^n_m'.t * H_m_n[1]))))
    det[2] = ExpandAndEliminate >> (dim >> (haa >> (ExpandAndEliminate >> sum('d^n_m'.t * H_m_n[2], (ExpandAndEliminate >> ('d^n_m'.t * H_m_n[1]))**2 / 2))))
    det[3] = ExpandAndEliminate >> (dim >> (haa >> (ExpandAndEliminate >> sum('d^n_m'.t * H_m_n[3], (ExpandAndEliminate >> ('d^n_m'.t * H_m_n[1])) * (ExpandAndEliminate >> ('d^n_m'.t * H_m_n[2])), (ExpandAndEliminate >> ('d^n_m'.t * H_m_n[1]))**3 / 6))))
    det[4] = ExpandAndEliminate >> (dim >> (haa >> (ExpandAndEliminate >> sum('d^n_m'.t * H_m_n[4], (ExpandAndEliminate >> ('d^n_m'.t * H_m_n[2]))**2 / 2, ('d^n_m'.t * H_m_n[1])**4 / 24))))
    /*
    for (int i = 0; i < 5; i++){
        new File("results/det/det[${i}].txt").withWriter('utf-8') {
            writer -> writer.writeLine(det[i].toString())
        }
    }
    */
    def L = ['0'.t, '0'.t, '0'.t, '0'.t, '0'.t]
    for (int i = 0; i < 5; i++) {
        def temp1 = '0'.t
        for (int j = 0; j <= i; j++) {
            temp1 = sum(temp1, '\\alpha'.t * det[j] * R[i-j])
        }
        temp1 = ExpandAndEliminate >> (dim >> (haa >> (ExpandAndEliminate >> temp1)))

        def temp2 = '\\Lambda'.t * det[i]

        def temp3 = '0'.t
        for (int j = 0; j <= i; j++) {
            temp3 = sum(temp3, '\\beta'.t * det[j] * R2[i-j])
        }
        temp3 = ExpandAndEliminate >> (dim >> (haa >> (ExpandAndEliminate >> temp3)))

        def temp4 = '0'.t
        for (int j = 0; j <= i; j++) {
            temp4 = sum(temp4, '\\gamma'.t * det[j] * RR[i-j])
        }
        temp4 = ExpandAndEliminate >> (dim >> (haa >> (ExpandAndEliminate >> temp4)))

        def temp5 = '0'.t
        for (int j = 0; j <= i; j++) {
            temp5 = sum(temp5, '\\delta'.t * det[j] * nnR[i-j])
        }
        temp5 = ExpandAndEliminate >> (dim >> (haa >> (ExpandAndEliminate >> temp5)))

        L[i] = CollectNonScalars >> (CollectScalars >> (ExpandAndEliminate >> (dim >> (haa >> (ExpandAndEliminate >> sum(temp1, temp2, temp3, temp4, temp5))))))
    }
    /*
    for (int i = 0; i < 5; i++){
        new File("results/L/L[${i}].txt").withWriter('utf-8') {
            writer -> writer.writeLine(L[i].toString())
        }
    }
    */
    new File("results/L[2]_parts/Lambda.txt").withWriter('utf-8') {
        writer -> writer.writeLine((Differentiate['\\Lambda'] >> L[2]).toString())
    }
    new File("results/L[2]_parts/alpha.txt").withWriter('utf-8') {
        writer -> writer.writeLine((fix >> (Differentiate['\\alpha'] >> L[2])).toString())
    }
    new File("results/L[2]_parts/beta.txt").withWriter('utf-8') {
        writer -> writer.writeLine((fix >> (Differentiate['\\beta'] >> L[2])).toString())
    }
    new File("results/L[2]_parts/gamma.txt").withWriter('utf-8') {
        writer -> writer.writeLine((fix >> (Differentiate['\\gamma'] >> L[2])).toString())
    }
    new File("results/L[2]_parts/delta.txt").withWriter('utf-8') {
        writer -> writer.writeLine((fix >> (Differentiate['\\delta'] >> L[2])).toString())
    }

    // Выделение полных дивергенций

    def div = 'h~(1)_l[x^a] * h~(1)^l[x^a] = -h[x^a] * h~(2)^l_l[x^a]'.t &
            'h~(1)^mn_m[x^a] * h~(1)^a_na[x^a] = -h^mn[x^a] * h~(2)^a_n_a_m[x^a]'.t &
            'h~(1)^mn_l[x^a] * h~(1)_mn^l[x^a] = -h^mn[x^a] * h~(2)_mn^l_l[x^a]'.t &
            'h~(1)_a[x^a] * h~(1)^ab_b[x^a] = -h_mn[x^a] * h~(2)^mn[x^a]'.t &
            'h~(1)^mn_a[x^a] * h~(1)^a_nm[x^a] = -h^mn[x^a] * h~(2)^a_n_a_m[x^a]'.t &
            'h~(1)^mn_l[x^a] * h~(3)_mns^ls[x^a] = -h^mn[x^a] * h~(4)_mnsl^sl[x^a]'.t &
            'h~(1)^mnl[x^a] * h~(3)_mnl[x^a] = -h[x^a] * h~(4)^mnl_mnl[x^a]'.t &
            'h~(1)^mn_m[x^a] * h~(3)_nl^l[x^a] = -h[x^a] * h~(4)^mnl_mnl[x^a]'.t &
            'h~(1)^mn_l[x^a] * h~(3)^a_mna^l[x^a] = -h^mn[x^a] * h~(4)^a_mnal^l[x^a]'.t &
            'h~(1)^mn_m[x^a] * h~(3)^a_nal^l[x^a] = -h^mn[x^a] * h~(4)^a_mnal^l[x^a]'.t &
            'h~(1)^mn_a[x^a] * h~(3)^a_nml^l[x^a] = -h^mn[x^a] * h~(4)^a_mnal^l[x^a]'.t &
            'h~(1)_l[x^a] * h~(3)^abl_ab[x^a] = -h[x^a] * h~(4)^mnl_mnl[x^a]'.t &
            'h~(1)_l[x^a] * h~(3)^ls_s[x^a] = -h[x^a] * h~(4)^ls_ls[x^a]'.t &
            'h~(1)_a[x^a] * h~(3)^abl_bl[x^a] = -h[x^a] * h~(4)^mnl_mnl[x^a]'.t &
            'h~(2)^mn_ls[x^a] * h~(2)_mn^ls[x^a] = h^mn[x^a] * h~(4)_mnls^ls[x^a]'.t &
            'h~(2)^mn_la[x^a] * h~(2)_n^al_m[x^a] = h^mn[x^a] * h~(4)_n^al_mal[x^a]'.t &
            'h~(2)^mn_mn[x^a] * h~(2)^ab_ab[x^a] = h^mn[x^a] * h~(4)^ab_mnab[x^a]'.t & '(h~(2)^mn_mn[x^a])**2 = h^mn[x^a] * h~(4)^ab_mnab[x^a]'.t & '(h~(2)^mn_mn[x^a])**(2.0) = h^mn[x^a] * h~(4)^ab_mnab[x^a]'.t &
            'h~(2)^mn_an[x^a] * h~(2)^ab_mb[x^a] = h^mn[x^a] * h~(4)^ab_mnab[x^a]'.t &
            'h~(2)^l_l[x^a] * h~(2)^s_s[x^a] = h[x^a] * h~(4)^ls_ls[x^a]'.t & '(h~(2)^l_l[x^a])**2 = h[x^a] * h~(4)^ls_ls[x^a]'.t & '(h~(2)^l_l[x^a])**(2.0) = h[x^a] * h~(4)^ls_ls[x^a]'.t &
            'h~(2)^mnl_l[x^a] * h~(2)_mn[x^a] = h[x^a] * h~(4)^mnl_mnl[x^a]'.t &
            'h~(2)^mn_ml[x^a] * h~(2)^a_na^l[x^a] = h^mn[x^a] * h~(4)^a_mnal^l[x^a]'.t &
            'h~(2)_ls[x^a] * h~(2)^ls[x^a] = h[x^a] * h~(4)^ls_ls[x^a]'.t &
            'h~(2)^mnl_l[x^a] * h~(2)_mns^s[x^a] = h^mn[x^a] * h~(4)_mnls^ls[x^a]'.t &
            'h~(2)^mn_ml[x^a] * h~(2)_n^l[x^a] = h[x^a] * h~(4)^mnl_mnl[x^a]'.t &
            'h~(2)^mn_ma[x^a] * h~(2)^a_nl^l[x^a] = h^mn[x^a] * h~(4)^a_mnal^l[x^a]'.t &
            'h~(2)^mn_mn[x^a] * h~(2)_l^l[x^a] = h[x^a] * h~(4)^mnl_mnl[x^a]'.t &
            'h^mn[x^a] * h~(4)_mnl^l[x^a] = h[x^a] * h~(4)^mn_mnl^l[x^a]'.t

    def L2 = Collect['\\Lambda', '\\alpha', '\\beta', '\\gamma', '\\delta'] >> (Expand >> (fix >> (div >> L[2])))

    // Подсчёт операторов D в координатном представлении

    def repl_D4 = 'h[x^a] * h[x^a] = \\eta^mn * \\eta^ab * H_mn * H_ab'.t & '(h[x^a])**2 = \\eta^mn * \\eta^ab * H_mn * H_ab'.t & '(h[x^a])**(2.0) = \\eta^mn * \\eta^ab * H_mn * H_ab'.t &
            'h^mn[x^a] * h_mn[x^a] = \\eta^ma * \\eta^nb * H_mn * H_ab'.t

    def D4 = '\\Lambda'.t * (Collect['H_mn'.t, 'H_ab'.t] >> (repl_D4 >> (ExpandAndEliminate >> (fix >> (Differentiate['\\Lambda'] >> L2)))))
    new File("results/D_coor/D4.txt").withWriter('utf-8') {
        writer -> writer.writeLine(D4.toString())
    }

    def repl_D2 = 'h_mn[x^a] * h~(2)^mn[x^a] = d^m_l * d^n_s * \\eta^ab * H_mn * H_ab * D^ls'.t &
            'h_mn[x^a] * h~(2)^mn_l^l[x^a] = \\eta^ma * \\eta^nb * \\eta_ls * H_mn * H_ab * D^ls'.t &
            'h_mn[x^a] * h~(2)^an^m_a[x^a] = d^m_l * \\eta^na * d^b_s * H_mn * H_ab * D^ls'.t

    def badRepl_D2 = 'h~(2)^ab_ab[x^a] = \\eta^mn * d^a_l * d^b_s * H_mn * H_ab * D^ls'.t &
            'h~(2)^l_l[x^a] = \\eta^mn * \\eta^ab * \\eta_ls * H_mn * H_ab * D^ls'.t &
            'h[x^a] = 1'.t

    def D2 = '\\alpha'.t * (Collect['H_mn'.t, 'H_ab'.t, 'D^ls'.t] >> (badRepl_D2 >>(repl_D2 >> (ExpandAndEliminate >> (fix >> (Differentiate['\\alpha'] >> L2))))))
    new File("results/D_coor/D2.txt").withWriter('utf-8') {
        writer -> writer.writeLine(D2.toString())
    }

    def repl_D0 = 'h^mn[x^a] * h~(4)_mnl^l[x^a] = d^m_l * d^n_s * \\eta_gd * \\eta^ab * H_mn * H_ab * D^lsgd'.t &
            'h^mn[x^a] * h~(4)_mnls^ls[x^a] = \\eta^ma * \\eta^nb * \\eta_ls * \\eta_gd * H_mn * H_ab * D^lsgd'.t &
            'h^mn[x^a] * h~(4)^a_n_alm^l[x^a] = d^m_l * \\eta^na * d^b_s * \\eta_gd * H_mn * H_ab * D^lsgd'.t &
            'h^{mn}[x^{a}] * h~(4)^{ab}_{mnab}[x^{a}] = d^m_l * d^n_s * d^a_g * d^b_d * H_mn * H_ab * D^lsgd'.t

    def badRepl_D0 = 'h~(4)_ls^ls[x^a] = \\eta^mn * \\eta^ab * \\eta_ls * \\eta_gd * H_mn * H_ab * D^lsgd'.t &
            'h~(4)_abl^abl[x^a] = \\eta^mn * \\eta_ls * d^a_g * d^b_d * H_mn * H_ab * D^lsgd'.t &
            'h[x^a] = 1'.t

    def D0_b = '\\beta'.t * (badRepl_D0 >> (repl_D0 >> (ExpandAndEliminate >> (fix >> (Differentiate['\\beta'] >> L2)))))
    def D0_g = '\\gamma'.t * (badRepl_D0 >> (repl_D0 >> (ExpandAndEliminate >> (fix >> (Differentiate['\\gamma'] >> L2)))))
    def D0_d = '\\delta'.t * (badRepl_D0 >> (repl_D0 >> (ExpandAndEliminate >> (fix >> (Differentiate['\\delta'] >> L2)))))
    def D0 = Collect['H_mn'.t, 'H_ab'.t, 'D^lsgd'.t] >> sum(D0_b, D0_g, D0_d)
    new File("results/D_coor/D0.txt").withWriter('utf-8') {
        writer -> writer.writeLine(D0.toString())
    }

    // Переход в импульсное представление

    def momentum_D4 = '\\eta^mn * \\eta^ab * H_mn * H_ab = 4 * \\eta^mn * \\eta^ab * H_mn * H_ab'.t &
            '\\eta^ma * \\eta^nb * H_mn * H_ab = 2 * (\\eta^ma * \\eta^nb + \\eta^mb * \\eta^na) * H_mn * H_ab'.t
    def D4_m = Collect['H_mn'.t, 'H_ab'.t, '\\Lambda'.t] >> (momentum_D4 >> (Expand >> D4))
    new File("results/D_momentum/D4.txt").withWriter('utf-8') {
        writer -> writer.writeLine(D4_m.toString())
    }

    def momentum_D2 = '\\eta^mn * \\eta^ab * \\eta_ls * H_mn * H_ab * D^ls = -4 * k**(2.0) * \\eta^mn * \\eta^ab * H_mn * H_ab'.t &
            '\\eta^ma * \\eta^nb * \\eta_ls * H_mn * H_ab * D^ls = -2 * k**(2.0) * (\\eta^ma * \\eta^nb + \\eta^mb * \\eta^na) * H_mn * H_ab'.t &
            '\\eta^mn * d^a_l * d^b_s * H_mn * H_ab * D^ls = -2 * (\\eta^mn * k^a * k^b + \\eta^ab * k^m * k^n) * H_mn * H_ab'.t &
            'd^m_l * \\eta^na * d^b_s * H_mn * H_ab * D^ls = -(\\eta^ma * k^n * k^b + \\eta^na * k^m * k^b + \\eta^mb * k^n * k^a + \\eta^nb * k^m * k^a) * H_mn * H_ab'.t
    def D2_m = Collect['H_mn'.t, 'H_ab'.t, '\\alpha'.t] >> (momentum_D2 >> (Expand >> D2))
    new File("results/D_momentum/D2.txt").withWriter('utf-8') {
        writer -> writer.writeLine(D2_m.toString())
    }

    def momentum_D0 = '\\eta^mn * \\eta^ab * \\eta_ls * \\eta_gd * H_mn * H_ab * D^lsgd = 4 * k**(4.0) * \\eta^mn * \\eta^ab * H_mn * H_ab'.t &
            '\\eta^ma * \\eta^nb * \\eta_ls * \\eta_gd * H_mn * H_ab * D^lsgd = 2 * k**(4.0) * (\\eta^ma * \\eta^nb + \\eta^mb * \\eta^na) * H_mn * H_ab'.t &
            '\\eta_{ls}*d^{a}_{g}*\\eta^{mn}*d^{b}_{d} * H_mn * H_ab * D^lsgd = 2 * k**(2.0) * (\\eta^mn * k^a * k^b + \\eta^ab * k^m * k^n) * H_mn * H_ab'.t &
            'd^m_l * \\eta^na * d^b_s * \\eta_gd * H_mn * H_ab * D^lsgd = k**(2.0) * (\\eta^ma * k^n * k^b + \\eta^na * k^m * k^b + \\eta^mb * k^n * k^a + \\eta^nb * k^m * k^a) * H_mn * H_ab'.t &
            'd^m_l * d^n_s * d^a_g * d^b_d * H_mn * H_ab * D^lsgd = 4 * k^m * k^n * k^a * k^b * H_mn * H_ab'.t
    def D0_m = Collect['H_mn'.t, 'H_ab'.t] >> (momentum_D0 >> (Expand >> D0))
    new File("results/D_momentum/D0.txt").withWriter('utf-8') {
        writer -> writer.writeLine(D0_m.toString())
    }

    // Разложение по проекционным операторам

    def projection_D4 = '\\eta^mn * \\eta^ab * H_mn * H_ab = 4 * (3 * R + S + T + U)'.t &
            '\\eta^ma * \\eta^nb * H_mn * H_ab = 4 * (P + Q + R + S)'.t
    D4 = Collect['P'.t, 'Q'.t, 'R'.t, 'S'.t, 'T'.t, 'U'.t] >> (projection_D4 >> (Expand >> D4))

    def projection_D2 = '\\eta^mn * \\eta^ab * \\eta_ls * H_mn * H_ab * D^ls = -4 * k**(2.0) * (3 * R + S + T + U)'.t &
            '\\eta^ma * \\eta^nb * \\eta_ls * H_mn * H_ab * D^ls = -4 * k**(2.0) * (P + Q + R + S)'.t &
            '\\eta^mn * d^a_l * d^b_s * H_mn * H_ab * D^ls = -2 * k**(2.0) * (2 * S + T + U)'.t &
            'd^m_l * \\eta^na * d^b_s * H_mn * H_ab * D^ls = -k**(2.0) * (2 * P + 4 * S)'.t
    D2 = Collect['P'.t, 'Q'.t, 'R'.t, 'S'.t, 'T'.t, 'U'.t] >> (projection_D2 >> (Expand >> D2))

    def projection_D0 = '\\eta^mn * \\eta^ab * \\eta_ls * \\eta_gd * H_mn * H_ab * D^lsgd = 4 * k**(4.0) * (3 * R + S + T + U)'.t &
            '\\eta^ma * \\eta^nb * \\eta_ls * \\eta_gd * H_mn * H_ab * D^lsgd = 4 * k**(4.0) * (P + Q + R + S)'.t &
            '\\eta^mn * \\eta_ls * d^a_g * d^b_d * H_mn * H_ab * D^lsgd = 2 * k**(4.0) * (2 * S + T + U)'.t &
            'd^m_l * \\eta^na * d^b_s * \\eta_gd * H_mn * H_ab * D^lsgd = k**(4.0) * (2 * P + 4 * S)'.t &
            'd^m_l * d^n_s * d^a_g * d^b_d * H_mn * H_ab * D^lsgd = 4 * k**(4.0) * S'.t
    D0 = Collect['P'.t, 'Q'.t, 'R'.t, 'S'.t, 'T'.t, 'U'.t] >> (projection_D0 >> (Expand >> D0))

    def D = sum(D0, D2, D4)
    def projectors = ['P'.t, 'Q'.t, 'R'.t, 'S'.t, 'T'.t, 'U'.t]
    def L_parts = ['\\Lambda'.t, '\\alpha'.t, '\\beta'.t, '\\gamma'.t, '\\delta'.t]
    def x = []
    for (int i = 0; i < 6; i++){
        x[i] = Differentiate[projectors[i]] >> D
        println (check >> x[i])
    }
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 5; j++) {
            new File("results/x/x[${i + 1}]/x[${i + 1}][${L_parts[j].toString()}].txt").withWriter('utf-8') {
                writer -> writer.writeLine((Differentiate[L_parts[j]] >> x[i]).toString())
            }
        }
    }

    def Delta = Collect['\\Lambda'.t, '\\alpha'.t, '\\beta'.t, '\\gamma'.t, '\\delta'.t] >> sum(x[2] * x[3], -x[4] * x[5] * 3)
    def Delta_LL = fix >> (Differentiate['\\Lambda'] >> (Differentiate['\\Lambda'] >> ((0.5) * Delta)))
    def Delta_aa = fix >> (Differentiate['\\alpha'] >> (Differentiate['\\alpha'] >> ((0.5) * Delta)))
    def Delta_bb = fix >> (Differentiate['\\beta'] >> (Differentiate['\\beta'] >> ((0.5) * Delta)))
    def Delta_gg = fix >> (Differentiate['\\gamma'] >> (Differentiate['\\gamma'] >> ((0.5) * Delta)))
    def Delta_La = fix >> (Differentiate['\\Lambda'] >> (Differentiate['\\alpha'] >> Delta))
    def Delta_Lb = fix >> (Differentiate['\\Lambda'] >> (Differentiate['\\beta'] >> Delta))
    def Delta_Lg = fix >> (Differentiate['\\Lambda'] >> (Differentiate['\\gamma'] >> Delta))
    def Delta_ab = fix >> (Differentiate['\\alpha'] >> (Differentiate['\\beta'] >> Delta))
    def Delta_ag = fix >> (Differentiate['\\alpha'] >> (Differentiate['\\beta'] >> Delta))
    def Delta_bg = fix >> (Differentiate['\\alpha'] >> (Differentiate['\\beta'] >> Delta))
    Delta = sum('\\Lambda**(2.0)'.t * Delta_LL, '\\alpha**(2.0)'.t * Delta_aa, '\\beta**(2.0)'.t * Delta_bb, '\\gamma**(2.0)'.t * Delta_gg,
            '\\Lambda * \\alpha'.t * Delta_La, '\\Lambda * \\beta'.t * Delta_Lb, '\\Lambda * \\gamma'.t * Delta_Lg,
            '\\alpha * \\beta'.t * Delta_ab, '\\alpha * \\gamma'.t * Delta_ag, '\\beta * \\gamma'.t * Delta_bg)
    new File("results/x/Delta/Delta.txt").withWriter('utf-8') {
        writer -> writer.writeLine(Delta.toString())
    }
    def Delta_ = [Delta_LL, Delta_aa, Delta_bb, Delta_gg, Delta_La, Delta_Lb, Delta_Lg, Delta_ab, Delta_ag, Delta_bg]
    for (int i = 0; i < 10; i++) {
        new File("results/x/Delta/Delta_[${i}].txt").withWriter('utf-8') {
            writer -> writer.writeLine(Delta_[i].toString())
        }
    }

}