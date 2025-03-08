import os
import matplotlib.pyplot as plt
import numpy as np

def get_plot(path, train_data, val_data, title, train_label, val_label, x_label, y_label):
	"""
	Create and save a plot of training and validation metrics.
	
	Args:
		path (str): Path to save the plot
		train_data (list): Training metrics data
		val_data (list or None): Validation metrics data, can be None
		title (str): Plot title
		train_label (str): Label for training data
		val_label (str): Label for validation data
		x_label (str): Label for x-axis
		y_label (str): Label for y-axis
	"""
	try:
		plt.figure(figsize=(10, 6))
		
		# Plot training data
		plt.plot(train_data, 'b-', label=train_label)
		
		# Plot validation data if available
		if val_data is not None:
			plt.plot(val_data, 'r-', label=val_label)
		
		plt.title(title)
		plt.xlabel(x_label)
		plt.ylabel(y_label)
		plt.legend()
		plt.grid(True)
		
		# Save the plot
		plt.savefig(path + title + '.png')
		plt.close()
	except Exception as e:
		print(f"Error creating plot {title}: {e}")
		# Try a simpler plot as fallback
		try:
			plt.figure()
			plt.plot(train_data)
			if val_data is not None:
				plt.plot(val_data)
			plt.title(title)
			plt.savefig(path + title + '.png')
			plt.close()
		except Exception as e2:
			print(f"Failed to create even a simple plot: {e2}")
