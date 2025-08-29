class WheelRailSystemAnalytical:
    def __init__(self, wheel_center, wheel_quat, V, omega, wheel_type="right"):
        self.wheel_center = np.array(wheel_center)
        self.V = np.array(V)
        self.omega = np.array(omega)
        self.wheel_quat = np.array(wheel_quat)
        self.wheel_type = wheel_type
        self.rot = R.from_quat(wheel_quat)

        # Προφίλ τροχού / ράγας
        self.WHEEL_POINTS_X = -np.array([39.959, 33.543, 30.832, 25.252,
                                        15.553, -68.447, -71.645, -75.008, -75.947]) + 7.943
        self.WHEEL_POINTS_R = np.abs(np.array([-475.088, -469.879, -462.389,
                                               -453.968, -451.327, -447.130, -446.470, -444.859, -442.370]))

        self.RAIL_POINTS_X = np.array([-33.632, -34.622, -35, -35, -31.523, -23, 0,
                                       23, 31.523, 35, 35, 34.622, 33.632])
        self.RAIL_POINTS_Y = np.array([109.244, 109.972, 111.142, 126.2, 135.002, 139.114,
                                       140, 139.114, 135.002, 126.2, 111.142, 109.972, 109.244]) - 140

        # δημιουργία αναλυτικού path με τις νέες παραμέτρους
        self.create_extended_analytical_path()

    def create_extended_analytical_path(self):
        # --- παραδείγματα τιμών ---
        H = 200.0       # μέγιστο ύψος του bump
        Lvert = 100000.0  # συνολικό μήκος ανηφόρας+κατηφόρας
        s0 = 10000.0       # από που ξεκινά η μετάβαση
    
        self.analytical_path = AnalyticalRailPath3D(
            use_horizontal_curve=False,  # ευθεία κατά z, x=0
            H=H, Lvert=Lvert, s0=s0,
            # τα υπόλοιπα μπορούν να μείνουν ως έχουν
            R=600.0, Ls=100.0, arc_angle_deg=20.0,
            gauge=1.435, E=0.12, Ud=0.075
        )
    
        self.path_func = {
            'length': self.analytical_path.s_total,
            'gauge': 1535
        }
        self.x_shift = self.path_func['gauge']/2.0 if self.wheel_type == "left" else -self.path_func['gauge']/2.0
        self.y_shift = 140.0

    def get_path_point(self, s):
        """Επιστρέφει σημείο στη διαδρομή για δεδομένο s"""
        x, y, z = self.analytical_path.position(s)
        return np.array([x[0] + self.x_shift, y[0] + self.y_shift, z[0]])

    def get_frenet_frame(self, s):
        """Υπολογίζει το Frenet-Serret frame"""
        point = self.get_path_point(s)
        
        # Χρήση αναλυτικής εφαπτομένης
        tangent = np.array(self.analytical_path.tangent(s))
        
        # Υπολογισμός καμπυλότητας
        h = 1.0
        s_plus = min(s + h, self.path_func['length'])
        s_minus = max(s - h, 0)
        
        point_plus = self.get_path_point(s_plus)
        point_minus = self.get_path_point(s_minus)
        
        second_derivative = (point_plus - 2*point + point_minus) / (h**2)
        
        if np.linalg.norm(second_derivative) > 1e-10:
            normal = second_derivative / np.linalg.norm(second_derivative)
        else:
            normal = np.array([0, 1, 0])
        
        # Ορθογωνιοποίηση ως προς την εφαπτομένη
        normal = normal - np.dot(normal, tangent) * tangent
        if np.linalg.norm(normal) > 1e-10:
            normal /= np.linalg.norm(normal)
        else:
            normal = np.array([0, 1, 0])
            
        tangent = tangent.flatten()[:3]
        normal = normal.flatten()[:3]
        binormal = np.cross(tangent, normal)
        binormal /= np.linalg.norm(binormal)
        
        # Επαναϋπολογισμός normal για διασφάλιση ορθογωνιότητας
        normal = np.cross(binormal, tangent)
        normal /= np.linalg.norm(normal)
        
        return {
            'origin': point,
            'tangent': tangent,
            'normal': normal,
            'binormal': binormal
        }

    def get_first_plane_analytical(self):
        """Αναλυτικός ορισμός επιπέδου επαφής με διασφάλιση θετικού y-axis"""
        wheel_z = self.wheel_center[2]
        s_guess = wheel_z
        
        # Frenet frame
        frame = self.get_frenet_frame(s_guess)
        
        # Άξονας περιστροφής τροχού (τοπικός x)
        x_axis = self.rot.apply([1, 0, 0])
        x_axis = x_axis / np.linalg.norm(x_axis)
        
        # Βεβαιώνουμε ότι το normal είναι κάθετο στον x-άξονα
        normal = frame['normal']
        projection = np.dot(normal, x_axis) * x_axis
        normal_corrected = normal - projection
        
        # Έλεγχος για μηδενικό διάνυσμα
        if np.linalg.norm(normal_corrected) < 1e-12:
            normal_corrected = np.array([0, 1, 0])
        
        normal_corrected = normal_corrected / np.linalg.norm(normal_corrected)
        
        # ΔΙΟΡΘΩΣΗ: Βεβαιώνουμε ότι ο y-axis έχει θετική συνιστώσα στον καθολικό Y
        if normal_corrected[1] < 0:
            normal_corrected = -normal_corrected
        
        # Y-axis = corrected normal, Z-axis = x × y
        y_axis = normal_corrected
        z_axis = np.cross(x_axis, y_axis)
        z_axis /= np.linalg.norm(z_axis)
        
        # Επιπλέον έλεγχος: αν ο z-axis δείχνει προς τα κάτω, αντιστρέφουμε
        if z_axis[2] < 0:
            z_axis = -z_axis
            y_axis = np.cross(z_axis, x_axis)  # Επαναϋπολογισμός y-axis
        
        contact_plane = {
            'point': self.wheel_center,
            'normal': z_axis,
            'x_axis': x_axis,
            'y_axis': y_axis,
            'z_axis': z_axis,
            'frenet_frame': frame,
            's_position': s_guess
        }
        
        return contact_plane

    def get_rail_profile_at_s(self, s):
        """Rail profile με πλήρη έλεγχο προσανατολισμού"""
        frame = self.get_frenet_frame(s)
        
        # Διόρθωση προσανατολισμού των αξόνων
        tangent = frame['tangent']
        normal = frame['normal']
        binormal = frame['binormal']
        
        # Βεβαιώνουμε ότι το normal δείχνει προς τα πάνω (θετικό Y)
        if normal[1] < 0:
            normal = -normal
            binormal = np.cross(tangent, normal)  # Επαναϋπολογισμός binormal
        
        # Βεβαιώνουμε ότι το binormal έχει τη σωστή φορά για τον τύπο τροχού
        if self.wheel_type == "left":
            # Για αριστερό τροχό, το binormal πρέπει να δείχνει προς τα αριστερά (αρνητικό X)
            if binormal[0] > 0:
                binormal = -binormal
                normal = np.cross(binormal, tangent)  # Επαναϋπολογισμός normal
        else:
            # Για δεξί τροχό, το binormal πρέπει να δείχνει προς τα δεξιά (θετικό X)
            if binormal[0] > 0:
                binormal = -binormal
                normal = np.cross(binormal, tangent)
        
        # Μετασχηματισμός rail profile points
        rail_points_3d = []
        for i in range(len(self.RAIL_POINTS_X)):
            point_3d = (frame['origin'] + 
                       self.RAIL_POINTS_X[i] * binormal + 
                       self.RAIL_POINTS_Y[i] * normal)
            rail_points_3d.append(point_3d)
        
        # Εκτύπωση για έλεγχο
        print(f"Rail profile at s={s}:")
        print(f"First point: {rail_points_3d[0]}")
        print(f"Last point: {rail_points_3d[-1]}")
        
        return np.array(rail_points_3d), {'tangent': tangent, 'normal': normal, 'binormal': binormal, 'origin': frame['origin']}
    def set_twist_table(self, twist_points, degrees=False):
        """
        Ορίζει τον πίνακα twist points.
        twist_points: iterable από (s, angle) όπου angle σε radians (default) ή degrees αν degrees=True.
        Παράδειγμα: system.set_twist_table([(0.0, 0.0), (500.0, 0.1)], degrees=False)
        """
        arr = np.array(sorted(twist_points, key=lambda x: x[0]), dtype=float)
        if arr.size == 0:
            self.twist_s = np.array([0.0])
            self.twist_theta = np.array([0.0])
        else:
            self.twist_s = arr[:, 0]
            self.twist_theta = arr[:, 1]
            if degrees:
                self.twist_theta = np.deg2rad(self.twist_theta)
    
    def get_twist_angle(self, s):
        """
        Επιστρέφει τη γωνία twist θ(s) σε radians. Γίνεται γραμμική παρεμβολή
        μεταξύ των σημάτων που ορίστηκαν με set_twist_table.
        Αν s εκτός του εύρους → χρησιμοποιείται const extrapolation (edge values).
        """
        if not hasattr(self, 'twist_s'):
            # default: no twist
            return 0.0
        # np.interp δίνει γραμμική παρεμβολή + constant extrapolation στα άκρα
        return float(np.interp(s, self.twist_s, self.twist_theta, left=self.twist_theta[0], right=self.twist_theta[-1]))
    
    def get_twisted_frame(self, s):
        """
        Επιστρέφει frame με twist: axes (origin, tangent, normal_twisted, binormal_twisted).
        Χρησιμοποιεί το Frenet-like frame που έχει ήδη η κλάση (get_frenet_frame)
        και εφαρμόζει περιστροφή γύρω από τον tangent κατά θ(s).
        """
        frame = self.get_frenet_frame(s)
        origin = frame['origin']
        tangent = np.array(frame['tangent'], dtype=float)
        # εξασφαλίζουμε ομαλοποιημένο tangent
        tnorm = np.linalg.norm(tangent)
        if tnorm < 1e-12:
            tangent = np.array([0., 1., 0.])
            tnorm = 1.0
        tangent = tangent / tnorm
    
        normal = np.array(frame['normal'], dtype=float)
        binormal = np.array(frame['binormal'], dtype=float)
    
        # Η γωνία twist
        theta = self.get_twist_angle(s)
    
        # Περιστροφή γύρω από τον tangent κατά θ (right-hand rule)
        # Χρήση scipy Rotation: rotvec = axis * angle
        rot = R.from_rotvec(tangent * theta)
        normal_tw = rot.apply(normal)
        binormal_tw = rot.apply(binormal)
    
        # Επανα-ορθονομήση για ασφάλεια
        normal_tw = normal_tw - np.dot(normal_tw, tangent) * tangent
        if np.linalg.norm(normal_tw) < 1e-12:
            # fallback
            normal_tw = np.array([0., 1., 0.])
            normal_tw = normal_tw - np.dot(normal_tw, tangent) * tangent
        normal_tw /= np.linalg.norm(normal_tw)
        binormal_tw = np.cross(tangent, normal_tw)
        binormal_tw /= np.linalg.norm(binormal_tw)
    
        return {
            'origin': origin,
            'tangent': tangent,
            'normal': normal_tw,
            'binormal': binormal_tw
        }
    
    def get_rail_profile_at_s_with_twist(self, s):
        """
        Επιστρέφει rail profile 3D σημεία στο σημείο s λαμβάνοντας υπόψη twist(s).
        Επιστρέφει (rail_points_3d, frame_dict) όπως η παλιά get_rail_profile_at_s.
        Χρησιμοποιεί το υπάρχον self.RAIL_POINTS_X, self.RAIL_POINTS_Y.
        """
        frame = self.get_twisted_frame(s)
    
        tangent = frame['tangent']
        normal = frame['normal']
        binormal = frame['binormal']
        origin = frame['origin']
    
        rail_points_3d = []
        for i in range(len(self.RAIL_POINTS_X)):
            # Σημείο profile στο τοπικό (binormal, normal) basis
            point_3d = origin + self.RAIL_POINTS_X[i] * binormal + self.RAIL_POINTS_Y[i] * normal
            rail_points_3d.append(point_3d)
    
        rail_points_3d = np.array(rail_points_3d)
    
        # Για debugging: (μπορείς να σχολιάσεις/αφαιρέσεις)
        # print(f"Rail profile at s={s} with twist={self.get_twist_angle(s):.6f} rad -> first/last: {rail_points_3d[0]}, {rail_points_3d[-1]}")
    
        return rail_points_3d, {'tangent': tangent, 'normal': normal, 'binormal': binormal, 'origin': origin}

class RailTwistFunction:
    def __init__(self, s_points, theta_points, degrees=False):
        self.s_points = np.array(s_points)
        self.theta_points = np.array(theta_points)
        if degrees:
            self.theta_points = np.deg2rad(self.theta_points)
    def twist(self, s):
        return float(np.interp(s, self.s_points, self.theta_points, left=self.theta_points[0], right=self.theta_points[-1]))