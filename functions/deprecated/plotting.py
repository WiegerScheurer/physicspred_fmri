import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.polynomial.polynomial import polyfit, polyval
import matplotlib.colors as mcolors
from scipy.optimize import curve_fit


def logistic_function(x, L, x0, k, b):
    """
    Standard logistic function with parameters:
    L: the curve's maximum value (typically 1 for probability)
    x0: the x-value of the sigmoid's midpoint
    k: the steepness of the curve
    b: the minimum value (typically 0 for probability)
    """
    return L / (1 + np.exp(-k * (x - x0))) + b
        
# Define the Gaussian function
def gaussian(x, a, b, c):
    return a * np.exp(-((x - b)**2) / (2 * c**2))

def weibull_cdf(x, lamb, beta):
    return 1 - np.exp(-(x / lamb) ** beta)

def create_psychometric_plots(data, title=None, prob_ylim=None, rt_ylim=None, 
                              prob_poly_fit=None, rt_poly_fit=None, rt_gaussian_fit=None, split_hypotheses=False,
                              save_fig:bool=False, cmap:str="gist_earth", line_alpha:float=.4, fit_alpha:float=.9,
                              prob_type:str="o-", rt_type:str="o-", fit_type:str="--",
                              rt_correct_only:bool=False, fit_width:float=8.5, line_width:float=3.5,
                              string_x_labels:bool=False):
    """
    Create two psychometric plots with customizable features.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing the columns 'ball_color_change', 'response', and 'rt'
    title : str, optional
        Custom title for the overall figure
    prob_ylim : tuple, optional
        Custom y-axis range for probability plot (min, max)
    rt_ylim : tuple, optional
        Custom y-axis range for reaction time plot (min, max)
    prob_poly_fit : int or None, optional
        Use sigmoid fit for probability data (if not None)
    rt_poly_fit : int or None, optional
        Degree of polynomial to fit to reaction time data (None for no fit)
    split_hypotheses : bool, optional
        Whether to split the data into four hypotheses using filter_condition
    """
    # Ensure polynomial degrees are integers
    if rt_poly_fit is not None:
        rt_poly_fit = int(rt_poly_fit)
    
    color_spectrum = plt.get_cmap(cmap)

    hypotheses = ['CC', 'CI', 'IC', 'II']

    # Normalize the indices to map them to the colormap
    norm = mcolors.Normalize(vmin=0, vmax=len(hypotheses) - 1)
        
    # Figure setup
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    plt.tight_layout(pad=4)
    
    if not split_hypotheses:
        # Original implementation for a single dataset
        # Extract unique values of ball_color_change and ensure they are in order
        color_changes = sorted(data['ball_color_change'].unique())
        
        # Convert color changes to numeric for fitting
        x_numeric = np.array([float(x) for x in color_changes])
        
        # Calculate probability of "brighter" response for each ball_color_change value
        prob_brighter = []
        for change in color_changes:
            subset = data[data['ball_color_change'] == change]
            prob = (subset['response'] == 'brighter').mean()
            prob_brighter.append(prob)
        
        # Calculate mean reaction time for each ball_color_change value
        mean_rt = []
        rt_error = []  # for standard error
        for change in color_changes:
            subset = data[data['ball_color_change'] == change]
            mean_rt.append(subset['rt'].mean())
            rt_error.append(subset['rt'].std() / np.sqrt(len(subset)))
        
        # Plot 1: Probability of "brighter" response
        ax1.plot(color_changes, prob_brighter, 'o-', color='blue', markersize=8, label='Data')
        
        # Add sigmoid fit if requested
        if prob_poly_fit is not None and len(color_changes) > 3:  # Need at least 4 points for a reliable sigmoid
            try:
                # Initial parameter guesses: L=1, x0=midpoint, k=1, b=0
                p0 = [1, np.median(x_numeric), 1, 0]
                
                # Bounds to constrain parameters to reasonable values
                # L between 0.9 and 1.1, x0 within range, k positive, b between -0.1 and 0.1
                bounds = ([0.9, min(x_numeric), 0.001, -0.1], 
                          [1.1, max(x_numeric), 10, 0.1])
                
                # Fit the sigmoid function to the data
                params, _ = curve_fit(logistic_function, x_numeric, prob_brighter, 
                                     p0=p0, bounds=bounds, maxfev=10000)
                
                # Create a more fine-grained x for smooth curve
                x_fit = np.linspace(min(x_numeric), max(x_numeric), 100)
                y_fit = logistic_function(x_fit, *params)
                
                # Extract parameters for labeling
                L, x0, k, b = params
                
                # Plot fitted curve
                ax1.plot(x_fit, y_fit, 'r--', linewidth=2, 
                         label=f'Sigmoid fit\nMidpoint: {x0:.2f}, Slope: {k:.2f}')
            except Exception as e:
                print(f"Error fitting sigmoid: {e}")
        
        # Plot 2: Reaction time
        ax2.errorbar(color_changes, mean_rt, yerr=rt_error, fmt='o-', color='green', 
                    markersize=8, capsize=5, label='Data')
        
        # Add polynomial fit if requested for RT data
        if rt_poly_fit is not None and len(color_changes) > rt_poly_fit:
            try:
                # Fit polynomial of specified degree
                coefs = polyfit(x_numeric, mean_rt, rt_poly_fit)
                
                # Create a more fine-grained x for smooth curve
                x_fit = np.linspace(min(x_numeric), max(x_numeric), 100)
                y_fit = polyval(x_fit, coefs)
                
                # Plot fitted curve
                ax2.plot(x_fit, y_fit, 'r--', linewidth=2, 
                         label=f'{rt_poly_fit}-degree polynomial fit')
            except Exception as e:
                print(f"Error fitting RT polynomial: {e}")
        
        # Add labels for number of trials per condition
        for i, change in enumerate(color_changes):
            count = len(data[data['ball_color_change'] == change])
            ax1.annotate(f'n={count}', xy=(change, prob_brighter[i]), 
                        xytext=(0, 10), textcoords='offset points', 
                        ha='center', va='bottom', fontsize=8)
    
    else:
        # Implementation for split hypotheses
                    
        # Get data for each hypothesis
        cc = filter_condition(data, True, True)
        ci = filter_condition(data, True, False)
        ic = filter_condition(data, False, True)
        ii = filter_condition(data, False, False)
        
        # Define colors and labels for each hypothesis
        hypotheses = [
            (cc, 'SC_EC', color_spectrum(norm(0))),
            (ci, 'SC_EI', color_spectrum(norm(1))),
            (ic, 'SI_EC', color_spectrum(norm(2))),
            (ii, 'SI_EI', color_spectrum(norm(3)))
        ]
        
        # Find the common set of color changes across all hypotheses
        all_changes = set()
        for hyp_data, _, _ in hypotheses:
            all_changes.update(hyp_data['ball_color_change'].unique())
        color_changes = sorted(all_changes)
        
        # Plot for each hypothesis
        for hyp_data, label, color in hypotheses:
            hyp_color_changes = sorted(hyp_data['ball_color_change'].unique())
            
            # Skip if no data for this hypothesis
            if len(hyp_color_changes) == 0:
                continue
                
            # Convert to numeric values for fitting
            hyp_x_numeric = np.array([float(x) for x in hyp_color_changes])
            
            # Calculate probability of "brighter" response
            prob_brighter = []
            for change in hyp_color_changes:
                subset = hyp_data[hyp_data['ball_color_change'] == change]
                if len(subset) > 0:
                    prob = (subset['response'] == 'brighter').mean()
                    prob_brighter.append(prob)
                else:
                    prob_brighter.append(np.nan)
            
            # Calculate mean reaction time and error
            mean_rt = []
            rt_error = []
            for change in hyp_color_changes:
                subset = hyp_data[hyp_data['ball_color_change'] == change]
                if len(subset) > 0:
                    mean_rt.append(subset['rt'].mean())
                    rt_error.append(subset['rt'].std() / np.sqrt(len(subset)))
                else:
                    mean_rt.append(np.nan)
                    rt_error.append(np.nan)
            
            # Convert lists to numpy arrays for easier manipulation
            prob_brighter_array = np.array(prob_brighter)
            mean_rt_array = np.array(mean_rt)
            
            # Plot probability data
            ax1.plot(hyp_color_changes, prob_brighter, prob_type, color=color, 
                    markersize=6, label=f'Trialtype {label}', alpha=line_alpha, linewidth=line_width)
            
            # Add sigmoid fit if requested
            if prob_poly_fit is not None:
                # Check if we have enough valid data points for fitting
                valid_indices = ~np.isnan(prob_brighter_array)
                num_valid = np.sum(valid_indices)
                
                if num_valid > 3:  # Need at least 4 points for reliable sigmoid
                    try:
                        # Extract valid x and y values
                        valid_x = hyp_x_numeric[valid_indices]
                        valid_y = prob_brighter_array[valid_indices]
                        
                        # Initial parameter guesses
                        p0 = [1, np.median(valid_x), 1, 0]
                        
                        # Bounds to constrain parameters
                        bounds = ([0.9, min(valid_x), 0.001, -0.1], 
                                  [1.1, max(valid_x), 10, 0.1])
                        
                        # Fit sigmoid function
                        params, _ = curve_fit(logistic_function, valid_x, valid_y, 
                                             p0=p0, bounds=bounds, maxfev=10000)
                        
                        # Create a more fine-grained x for smooth curve within the VALID range
                        x_fit = np.linspace(min(valid_x), max(valid_x), 100)
                        y_fit = logistic_function(x_fit, *params)
                        
                        # Extract parameters for labeling
                        L, x0, k, b = params
                        
                        # Plot fitted curve
                        ax1.plot(x_fit, y_fit, fit_type, color=color, linewidth=fit_width, 
                                alpha=fit_alpha, label=f'Slope: {k:.2f}\nMidpoint: {x0:.2f}')
                    except Exception as e:
                        print(f"Error fitting sigmoid for {label}: {e}")
                        print(f"Debug - valid_x shape: {valid_x.shape}, valid_y shape: {valid_y.shape}")
            
            # # Plot reaction time data
            # ax2.errorbar(hyp_color_changes, mean_rt, yerr=rt_error, fmt=rt_type, 
            #             color=color, markersize=6, capsize=4, 
            #             label=f'Condition {label}', alpha=line_alpha, linewidth=line_width)
            
            ax2.errorbar(hyp_color_changes, mean_rt, yerr=rt_error, fmt=rt_type, 
                        color=color, markersize=8, capsize=5, capthick=2, elinewidth=1.5,
                        markerfacecolor='white', markeredgewidth=2,
                        label=f'Trialtype {label}', alpha=line_alpha, linewidth=line_width)

            
            if rt_gaussian_fit:  # Replace rt_poly_fit with a boolean flag for Gaussian fit
                # Check if we have enough valid data points for fitting
                valid_indices = ~np.isnan(mean_rt_array)
                num_valid = np.sum(valid_indices)
                
                if num_valid > 3:  # Gaussian fit requires at least 3 points
                    try:
                        # Extract valid x and y values
                        valid_x = hyp_x_numeric[valid_indices]
                        valid_y = mean_rt_array[valid_indices]
                        
                        # Initial guess for parameters (a, b, c)
                        p0 = [np.max(valid_y), np.mean(valid_x), np.std(valid_x)]
                        
                        # Fit Gaussian function
                        popt, _ = curve_fit(gaussian, valid_x, valid_y, p0=p0)
                        
                        # Create a more fine-grained x for smooth curve within the VALID range
                        x_fit = np.linspace(min(valid_x), max(valid_x), 100)
                        y_fit = gaussian(x_fit, *popt)
                        
                        
                        # Plot fitted curve
                        ax2.plot(x_fit, y_fit, fit_type, color=color, linewidth=fit_width,
                                alpha=fit_alpha, label=f'Gaussian fit')
                        
                    except Exception as e:
                        print(f"Error in gaussian fitting: {e}")
                        
            if rt_poly_fit:  # Replace rt_poly_fit with a boolean flag for polynomial fit
                # Check if we have enough valid data points for fitting
                valid_indices = ~np.isnan(mean_rt_array)
                num_valid = np.sum(valid_indices)
                
                if num_valid > 3:  # Polynomial fit requires at least 3 points
                    try:
                        # Extract valid x and y values
                        valid_x = hyp_x_numeric[valid_indices]
                        valid_y = mean_rt_array[valid_indices]
                        
                        # Degree of the polynomial
                        degree = rt_poly_fit  # You can change this to the desired polynomial degree
                        
                        # Fit polynomial function
                        p = np.polyfit(valid_x, valid_y, degree)
                        poly = np.poly1d(p)
                        
                        # Create a more fine-grained x for smooth curve within the VALID range
                        x_fit = np.linspace(min(valid_x), max(valid_x), 100)
                        y_fit = poly(x_fit)
                        
                        # Plot fitted curve
                        ax2.plot(x_fit, y_fit, fit_type, color=color, linewidth=fit_width,
                                alpha=fit_alpha, label=f'{degree}deg polynom fit')
                    except Exception as e:
                        print(f"Error in polynomial fitting: {e}")

    # Common settings for both plots
    ax1.set_xlabel('Ball Luminance Change')
    ax1.set_ylabel('Probability of "brighter" Response')
    ax1.set_title('Psychometric Function: Probability of "brighter" Response')
    
    # Set custom y-axis limits if provided
    if prob_ylim:
        ax1.set_ylim(prob_ylim)
    else:
        ax1.set_ylim(0, 1)  # Default range for probability
    
    # ax1.grid(True, alpha=0.2)
    ax1.yaxis.grid(True, alpha=.25)
    ax1.legend(loc='best')
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    # ax1.set_xticks([-0.3, -.15, 0, .15, .3])
    # ax1.set_xticks([-0.16, -.08, 0, .08, .16])
    ax1.set_xticks([-2, -1, 0, 1, 2])
    ax1.set_xticklabels(['darkest', 'darker', 'same', 'brighter', 'brightest'])    
    
    # Add horizontal line at 0.5 probability for reference
    ax1.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
    ax1.set_xlim(-2, 2)
    ax2.set_xlabel('Ball Luminance Change')
    ax2.set_ylabel('Reaction Time (s)')
    ax2.set_title('Reaction Time by Ball Lumin Change')
    
    # Set custom y-axis limits if provided
    if rt_ylim:
        ax2.set_ylim(rt_ylim)
    
    # ax2.grid(True, alpha=0.1)
    ax2.yaxis.grid(True, alpha=.25)
    ax2.legend(loc='best')
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.set_xlim(-2, 2)
    # ax2.set_xticks([-0.3, -.15, 0, .15, .3])
    # ax2.set_xticks([-0.16, -.08, 0, .08, .16])
    ax2.set_xticks([-2, -1, 0, 1, 2])
    # ax2.set_xticklabels(["brightest", "brighter", "same", "darker", "darkest"])
    ax2.set_xticklabels(['darkest', 'darker', 'same', 'brighter', 'brightest'])

    # Set custom title if provided
    if title:
        plt.suptitle(title, fontsize=16)
    else:
        plt.suptitle('Psychometric Analysis of Ball Hue Task', fontsize=16)
    
    if save_fig:
        plt.savefig('psychometric_plots.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig