def export_rail_paths_to_csv(system, filename_prefix="rail_path", n_points=1000):
    """
    Εξαγωγή 3 CSV αρχείων για: αριστερό path, δεξί path, και κεντρικό path
    
    Parameters:
    -----------
    system : WheelRailSystemAnalytical
        Το σύστημα τροχού-ράγας
    filename_prefix : str
        Πρόθεμα για τα ονόματα των CSV αρχείων
    n_points : int
        Αριθμός σημείων διακριτοποίησης
    """
    # Δημιουργία σημείων κατά μήκος του path
    s_vals = np.linspace(0, system.path_func['length'], n_points)
    
    # Λίστες για τα δεδομένα
    left_data, right_data, center_data = [], [], []
    
    gauge = system.path_func['gauge']
    
    for s in s_vals:
        # Βασικό σημείο (κέντρο)
        x_center, y_center, z_center = system.analytical_path.position(s)
        
        # Frenet frame για τον προσανατολισμό
        frame = system.get_frenet_frame(s)
        
        # Υπολογισμός θέσεων για αριστερό, δεξιό και κεντρικό path
        # Αριστερό path: x = -gauge/2
        left_point = (frame['origin'] + 
                     (-gauge/2) * frame['binormal'] + 
                     (605.57508546) * frame['normal'])
        
        # Δεξιό path: x = +gauge/2
        right_point = (frame['origin'] + 
                      (gauge/2) * frame['binormal'] + 
                      (605.57508546) * frame['normal'])
        
        # Κεντρικό path: x = 0
        center_point = (frame['origin'] + 
                       (605.57508546) * frame['normal'])
        
        # Προσθήκη στα δεδομένα
        left_data.append({
            's': s,
            'x': left_point[0],
            'y': left_point[1],
            'z': left_point[2],
            'curvature': system.analytical_path.curvature(s)[0],
            'superelevation': system.analytical_path.superelevation(s)[0]
        })
        
        right_data.append({
            's': s,
            'x': right_point[0],
            'y': right_point[1],
            'z': right_point[2],
            'curvature': system.analytical_path.curvature(s)[0],
            'superelevation': system.analytical_path.superelevation(s)[0]
        })
        
        center_data.append({
            's': s,
            'x': center_point[0],
            'y': center_point[1],
            'z': center_point[2],
            'curvature': system.analytical_path.curvature(s)[0],
            'superelevation': system.analytical_path.superelevation(s)[0]
        })
    
    # Δημιουργία DataFrames και εξαγωγή σε CSV
    df_left = pd.DataFrame(left_data)
    df_right = pd.DataFrame(right_data)
    df_center = pd.DataFrame(center_data)
    
    # Εξαγωγή σε CSV
    df_left.to_csv(f"{filename_prefix}_left.csv", index=False)
    df_right.to_csv(f"{filename_prefix}_right.csv", index=False)
    df_center.to_csv(f"{filename_prefix}_center.csv", index=False)
    
    print(f"Εξήχθησαν 3 CSV αρχεία:")
    print(f"- {filename_prefix}_left.csv ({len(df_left)} σημεία)")
    print(f"- {filename_prefix}_right.csv ({len(df_right)} σημεία)")
    print(f"- {filename_prefix}_center.csv ({len(df_center)} σημεία)")
    
    return df_left, df_right, df_center

def save_path_to_csv(results, filename="rail_path_system.csv"):
    """Αποθήκευση του rail path σε CSV με μετατόπιση"""
    system = results["system"]
    s_vals = np.linspace(0, system.path_func['length'], 100)
    path_points = np.array([system.get_path_point(s) for s in s_vals])
    
    # Μετατόπιση των συντεταγμένων
    path_points[:, 0] -= 1531.374 / 2  # Μετατόπιση X
    path_points[:, 1] += 605.57508546 - 140  # Μετατόπιση Y
    # Το Z παραμένει αμετάβλητο
    
    # Δημιουργία DataFrame
    df = pd.DataFrame(path_points, columns=['X', 'Y', 'Z'])
    
    # Αποθήκευση σε CSV
    df.to_csv(filename, index=False)
    print(f"Rail path saved to {filename}")
    
    return df