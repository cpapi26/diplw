if __name__ == "__main__":
    # Παράμετροι εισόδου
    wheel_center = [-746, 800+0, 0+10000]
    V = [0, 0, 1000]
    omega = [0, 10, 0]
    euler_angles = [0/57.3, 0.0/57.3, 0]
    
    # Κλήση βελτιωμένης GFOSUB
    results = GFOSUB_enhanced_analytical(
        wheel_center, V, omega, euler_angles,
        wheel_type="", return_plot_data=True
    )
    
    # Οπτικοποίηση
    plot_3d_results_with_forces(results, 0.1)
    
    save_path_to_csv(results)