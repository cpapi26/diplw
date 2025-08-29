class AnalyticalRailPath3D:
    """
    Path που προχωρά ευθεία κατά z, με κατακόρυφη καμπύλη (ανηφόρα-κορυφή-κατηφόρα)
    τύπου raised-cosine. Προαιρετικά μπορεί να ενεργοποιηθεί και οριζόντια χάραξη,
    αλλά από προεπιλογή είναι απενεργοποιημένη (x=0).
    Άξονες: x=αριστερά, y=πάνω, z=μπροστά.
    """
    def __init__(self, 
                 # οριζόντια γεωμετρία (προαιρετική)
                 R=600.0, Ls=72.0, arc_angle_deg=60.0,
                 gauge=1.435, E=0.12, Ud=0.075,
                 use_horizontal_curve=False,
                 # κατακόρυφη γεωμετρία (ζητούμενη)
                 H=20.0,            # μέγιστο ύψος "bump"
                 Lvert=300.0,       # συνολικό μήκος της ανηφόρας+κατηφόρας
                 s0=0.0, Lend=10000.0):           # σημείο έναρξης της κάθετης μετάβασης

        # --- οριζόντια παράμετροι (αν τυχόν τις θες αργότερα) ---
        self.R = R
        self.Ls = Ls
        self.gauge = gauge
        self.E = E
        self.Ud = Ud
        self.arc_angle = np.deg2rad(arc_angle_deg)
        self.s_arc = R * self.arc_angle
        self.A = np.sqrt(max(R*Ls, 1e-12))  # clothoid parameter (ασφάλεια)

        self.use_horizontal_curve = use_horizontal_curve
        self.h_total = 2*Ls + self.s_arc if use_horizontal_curve else 0.0

        # --- κατακόρυφη παράμετρος (ζητούμενη) ---
        self.H = H
        self.Lvert = max(Lvert, 1e-12)
        self.s0 = s0
        self.Lend = Lend
    
        self.v_total = self.s0 + self.Lvert + self.Lend
        self.s_total = max(self.h_total, self.v_total)

    # ---------- προαιρετική οριζόντια χάραξη (x offset) ----------
    def _clothoid_xy(self, s, sign=1.0):
        sigma = s / self.A
        S, C = fresnel(sigma / np.sqrt(np.pi))
        x = self.A * C
        z_like = self.A * S * sign  # ΔΕΝ το χρησιμοποιούμε σαν z!
        return x, z_like

    def _arc_xy(self, s, phi_offset=0.0):
        phi = s / self.R
        x = self.R * np.sin(phi + phi_offset)
        z_like = self.R * (1 - np.cos(phi + phi_offset))  # ΔΕΝ το χρησιμοποιούμε σαν z!
        return x, z_like

    def _x_horizontal(self, s):
        """Πλευρική μετατόπιση x(s) μόνο αν έχει ενεργοποιηθεί η οριζόντια χάραξη."""
        if not self.use_horizontal_curve:
            return 0.0
        if s < self.Ls:
            x, _ = self._clothoid_xy(s, sign=1.0)
            return x
        elif s < self.Ls + self.s_arc:
            s_arc = s - self.Ls
            x0, _ = self._clothoid_xy(self.Ls, sign=1.0)
            x1, _ = self._arc_xy(s_arc)
            return x0 + x1
        elif s < self.h_total:
            s_spiral = s - (self.Ls + self.s_arc)
            x0, _ = self._clothoid_xy(self.Ls, sign=1.0)
            x_arc, _ = self._arc_xy(self.s_arc)
            x1, _ = self._clothoid_xy(s_spiral, sign=-1.0)
            return x0 + x_arc + x1
        else:
            # πέρα από την οριζόντια χάραξη, κράτα σταθερό
            return self._x_horizontal(self.h_total) if self.h_total > 0 else 0.0

    # ---------- κατακόρυφη χάραξη (ζητούμενη) ----------
    def vertical_y(self, s):
        """Raised-cosine bump + τελικό ευθύγραμμο τμήμα."""
        if s < self.s0:
            return 0.0
        if s <= self.s0 + self.Lvert:
            u = (s - self.s0) / self.Lvert
            return 0.5 * self.H * (1.0 - np.cos(2.0*np.pi * u))
        if s <= self.v_total:  # τελικό ευθύγραμμο τμήμα
            return 0.0
        return 0.0

    def vertical_dy(self, s):
        """Παράγωγος dy/ds του bump (0 στην αρχή, στο μέσο και στο τέλος)."""
        if s < self.s0 or s > self.s0 + self.Lvert:
            return 0.0
        u = (s - self.s0) / self.Lvert
        return 0.5 * self.H * (2.0*np.pi / self.Lvert) * np.sin(2.0*np.pi * u)

    # ---------- βασικές συναρτήσεις ----------
    def position(self, s):
        """
        Επιστρέφει (x, y, z) με z=s (ευθεία κατά μήκος),
        y από την κατακόρυφη καμπύλη και x πλευρικό (συνήθως 0).
        """
        s = np.array(s, ndmin=1, dtype=float)
        x = np.zeros_like(s)
        y = np.zeros_like(s)
        z = np.zeros_like(s)

        for i, ss in enumerate(s):
            x[i] = self._x_horizontal(ss)      # 0 αν use_horizontal_curve=False
            y[i] = self.vertical_y(ss)         # raised-cosine bump
            z[i] = ss                          # ΠΑΝΤΑ προχωράμε κατά z

        return x, y, z

    def tangent(self, s):
        """
        Επιστρέφει εφαπτομένη (dx/ds, dy/ds, dz/ds) κανονικοποιημένη.
        Με z=s ⇒ dz/ds=1. dx/ds=0 (εκτός αν ενεργοποιήσεις οριζόντια χάραξη).
        """
        s = np.array(s, ndmin=1, dtype=float)
        dx = np.zeros_like(s)
        dy = np.zeros_like(s)
        dz = np.ones_like(s)   # z = s ⇒ dz/ds = 1

        if self.use_horizontal_curve:
            # Απλό (σταθερό) προσέγγιση: διαφοράς για dx/ds ώστε να μην
            # μπλέξουμε με καμπυλο-μήκη (αφού z=s). Παίρνουμε μικρό βήμα.
            eps = 1e-3
            for i, ss in enumerate(s):
                x_f = self._x_horizontal(ss + eps)
                x_b = self._x_horizontal(max(ss - eps, 0.0))
                dx[i] = (x_f - x_b) / (eps + max(ss, eps) - max(ss - eps, 0.0))

        for i, ss in enumerate(s):
            dy[i] = self.vertical_dy(ss)
            # Κανονικοποίηση
            n = np.sqrt(dx[i]**2 + dy[i]**2 + dz[i]**2)
            if n > 1e-12:
                dx[i] /= n
                dy[i] /= n
                dz[i] /= n

        if np.isscalar(s):
            return dx[0], dy[0], dz[0]
        return dx, dy, dz

    def curvature(self, s):
        """
        Προαιρετικά: καμπυλότητα μόνο από το κατακόρυφο bump με z=s.
        Για απλότητα εδώ δίνουμε κ κάθετα (δεν περιλαμβάνει οριζόντια κ).
        """
        s = np.array(s, ndmin=1, dtype=float)
        kappa = np.zeros_like(s)

        # αριθμητική 2ης παραγώγου στο y για το bump
        eps = 1e-3
        for i, ss in enumerate(s):
            y_f = self.vertical_y(ss + eps)
            y_0 = self.vertical_y(ss)
            y_b = self.vertical_y(max(ss - eps, 0.0))
            d2y = (y_f - 2*y_0 + y_b) / (eps**2)
            # προσεγγιστική καμπυλότητα σε επίπεδο zy
            kappa[i] = abs(d2y) / ((1 + (self.vertical_dy(ss))**2)**1.5 + 1e-12)
        return kappa

    def superelevation(self, s):
        """Παραμένει ως έχει/προαιρετικό — δεν επηρεάζει το ζητούμενο z/y."""
        s = np.array(s, ndmin=1, dtype=float)
        return np.zeros_like(s) + self.E