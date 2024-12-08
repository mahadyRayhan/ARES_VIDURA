import numpy as np
import pandas as pd

# Define constants
R = 8.314  # Gas constant, J/(mol K)
A = 6000 #10000  # Base growth rate potential in micrometers per second # checnge this one to check the effect of growth rate
E_a = 50000  # Activation energy, J/mol
gamma_0 = 0.01  # Base decay rate, per second
k = 0.0025  # Decay rate temperature factor, per °C  
T_ref = 700 + 273.15  # Reference temperature in Kelvin (1000 °C in Kelvin)
d_opt = 1.2 #2 #1.2  # Optimal Fe catalyst thickness in nm  # checnge this one to check the effect of growth rate
sigma_d = 1 #500 #0.2  # Standard deviation for Gaussian thickness effect in nm  # checnge this one to check the effect of growth rate

# Function to calculate growth rate
def growth_rate(T_p_C, t, d):
    """
    Calculate the growth rate based on temperature, time, and catalyst thickness.

    Parameters:
    T_p_C : float : Precursor temperature in Celsius
    t : int or float : Time in seconds
    d : float : Fe catalyst thickness in nm

    Returns:
    float : Growth rate in micrometers per second
    """
    # Convert Tp from Celsius to Kelvin
    T_p_K = T_p_C + 273.15
    
    # Catalyst decay factor, gamma, which depends on Tp
    gamma = gamma_0 * (1 + k * (T_p_C - (T_ref - 273.15)))  # Temperature adjustment in Celsius
    
    # Thickness-dependent factor, f(d)
    f_d = np.exp(-((d - d_opt) ** 2) / (2 * sigma_d ** 2))
    
    # Calculate the growth rate using the model
    growth_rate_value = A * f_d * np.exp(-E_a / (R * T_p_K)) * (1 - np.exp(-gamma * t))
    # Generate a random noise percentage between 1% and 10%
    noise_percentage = np.random.uniform(0.01, 0.10)  # Random value between 0.01 and 0.10
    # Calculate the noise
    sigma = noise_percentage * growth_rate_value  # Noise relative to the growth rate
    noise = np.random.normal(0, sigma)  # Mean 0, standard deviation sigma
    # Add the noise to the growth rate
    growth_rate_value_with_noise = growth_rate_value + noise
    return growth_rate_value_with_noise

# Function to simulate and save data for each combination of temperature and thickness to Excel
def simulate_and_save(temperature_range, thickness_range, max_time=1200):
    """
    Simulate CNT growth rate over time for a range of temperatures and thicknesses
    and save the results to an Excel file.

    Parameters:
    temperature_range : list : List of temperatures in Celsius
    thickness_range : list : List of catalyst thicknesses in nm
    max_time : int : Maximum simulation time in seconds (default: 1200)
    """
    
    # Create a list to store the data
    data = []
    experiment_number = 1  # Experiment counter for each combination of temperature and thickness

    # Loop over each temperature and thickness combination
    for T_p_C in temperature_range:
        for thickness in thickness_range:
            # Loop over each second from t=0 to t=max_time
            for t in range(max_time + 1):
                # Calculate growth rate for each time point
                growth_rate_value = growth_rate(T_p_C, t, thickness)
                
                # Append the data as a dictionary
                data.append({
                    'Experiment Number': experiment_number,
                    'Temperature (Tp, °C)': T_p_C,
                    'Time (t, s)': t,
                    'Catalyst Thickness (d, nm)': thickness,
                    'CNT-G (micrometers/s)': growth_rate_value
                })
            
            # Increment experiment number after each temperature-thickness combination
            experiment_number += 1

    # Convert data list to a DataFrame
    df = pd.DataFrame(data)
    
    # Save the DataFrame to an Excel file
    output_filename = "CNT_Growth_Rate_Simulation_with_noise.xlsx"
    df.to_excel(output_filename, index=False)
    print(f"Data successfully saved to {output_filename}")

# Example usage
# Input ranges
temperature_range = [600, 625, 650,675, 700, 725, 750, 775, 800, 825, 850, 875, 900, 925, 950, 975, 1000]  # List of temperatures in Celsius
thickness_range = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]  # List of catalyst thicknesses in nm
# temperature_range = [625]  # List of temperatures in Celsius
# thickness_range = [2] 
simulate_and_save(temperature_range, thickness_range)
