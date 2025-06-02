import matplotlib.pyplot as plt
import numpy as np

'''
{'min_k_DP': [5310.0, 7364.0, 2538.0, 2834.0, 4330.0, 8030.0, 12752.0, 12176.0, 13446.0, 15750.0],
 'order_DP': [5912.0, 8574.0, 2554.0, 3578.0, 5470.0, 9288.0, 12752.0, 12176.0, 13446.0, 15750.0],
 'merge_batches': [5310.0, 7364.0, 2538.0, 3132.0, 4330.0, 8030.0, 12752.0, 12176.0, 13446.0, 15750.0],
 'SetCover1': [5888.0, 9244.0, 3412.0, 3132.0, 4490.0, 8086.0, 12752.0, 12176.0, 13446.0, 15750.0],
 'ApproximateGreedy': [5888.0, 9244.0, 3412.0, 3198.0, 4490.0, 8086.0, 12752.0, 12176.0, 13446.0, 15750.0]}

'''

def plot_latency():
    latencies_by_method = \
{'min_k_DP': [5310.0, 7364.0, 2538.0, 2834.0, 4330.0, 8030.0, 12752.0, 12176.0, 13446.0, 15750.0],
 'order_DP': [5912.0, 8574.0, 2554.0, 3578.0, 5470.0, 9288.0, 12752.0, 12176.0, 13446.0, 15750.0],
 'merge_batches': [5310.0, 7364.0, 2538.0, 3132.0, 4330.0, 8030.0, 12752.0, 12176.0, 13446.0, 15750.0],
 'SetCover1': [5888.0, 9244.0, 3412.0, 3132.0, 4490.0, 8086.0, 12752.0, 12176.0, 13446.0, 15750.0],
 'ApproximateGreedy': [5888.0, 9244.0, 3412.0, 3198.0, 4490.0, 8086.0, 12752.0, 12176.0, 13446.0, 15750.0]}
    circuit_group = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    # Number of bars (methods)
    num_methods = len(latencies_by_method)
    
    # Width of each bar
    bar_width = 0.2  # Adjusted for better spacing
    
    # Create an array of positions for each set of bars
    index = np.arange(len(circuit_group))
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    
    # Plot bars for each method
    for i, (method, latencies) in enumerate(latencies_by_method.items()):
        plt.bar(index + i * bar_width, latencies, bar_width, label=method)
    
    # Adding labels and title
    plt.xlabel('Circuit Group with 5 circuits')
    plt.ylabel('Total Execution Time')
    plt.title('Memory Distribution with high Variance')
    
    # Set the x-axis with the circuit group numbers in the middle of the grouped bars
    plt.xticks(index + bar_width * (num_methods - 1) / 2, circuit_group)
    
    # Add a legend
    plt.legend()
    
    # Show the plot
    plt.tight_layout()
    plt.savefig('LATENCY3.png')
    plt.show()


plot_latency() 
