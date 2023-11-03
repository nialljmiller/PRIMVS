!!!!!README WRITTEN BY CHATGPT!!!!!!

# PRIMVS (PeRiodic Infrared Multiclass Variable Stars)

PRIMVS is a powerful tools package designed for astronomers who have access to the UHHPC cluster. This package provides essential tools to interface with astronomical data, particularly for the study of variable stars in the infrared spectrum.

## Features

### PRIMVS_autoencoder.py
- An autoencoder for the catalog, enabling feature-based cross-matching.

### PRIMVS_gif.py

`PRIMVS_gif.py` is a Python script designed for users of the PRIMVS (PeRiodic Infrared Multiclass Variable Stars) package. It specializes in creating animated GIFs that display the light curve of variable stars, with a particular emphasis on precise image stacking and alignment.

### Key Feature: Image Stacking and Alignment

One of the central features of `PRIMVS_gif.py` is the meticulous stacking and alignment of astronomical images for creating smooth and informative animated GIFs. Here's how this process is achieved:

- **Cross-Correlation for Alignment:** The script employs cross-correlation to determine the optimal translation (shift) required to align each frame in the light curve. This technique ensures that the position of the variable star remains consistent throughout the animation.

- **Reprojection for Consistency:** Prior to stacking, each image is reprojected to a common reference frame. Reprojection involves transforming image coordinates to match a predefined reference frame, such as the first image or a master image. This step is vital for ensuring precise alignment, even in the presence of small positional variations.

- **Normalization for Clarity:** To enhance visibility of variable star features, the script normalizes image data by subtracting the median intensity. This normalization guarantees that the brightest features in each frame are consistently presented, enabling astronomers to identify variations and patterns in the light curve more effectively.

- **Frame Selection and Sorting:** The script organizes frames in the desired sequence based on phase or time, allowing for a coherent representation of the variable star's behavior. This sorting assists viewers in tracking the star's brightness changes over time.

- **GIF Generation:** Once the images are properly stacked, aligned, and sorted, the script compiles them into an animated GIF. This GIF serves as a dynamic visual representation of the variable star's light curve, making it a valuable tool for astronomical analysis.




### PRIMVS_gif_batch.py
- Generate multiple GIFs within a given feature space, making it efficient for large-scale analyses.

### PRIMVS_histograms.py
- Create histograms of the catalog data, aiding in data visualization and analysis.

### PRIMVS_isolationforest.py
- Utilize an isolation forest for anomaly detection in the catalog, which can be crucial for identifying unique astronomical objects.

### PRIMVS_plotter.py
- Plot light curves and other relevant data from PRIMVS, providing visual insights into the astronomical data.

## Getting Started

To get started with PRIMVS, follow these steps:

1. Clone the repository to your local machine.
2. Install the necessary dependencies (provide a list of dependencies in your repository's documentation).
3. Use the PRIMVS tools according to your research needs. Refer to the specific tool's documentation for usage details.

For more information on how to use these tools, consult the package's documentation and examples provided in this repository.

## Contributors

- List the contributors or maintainers of the package, if applicable.

## License

Specify the license under which the PRIMVS package is distributed (e.g., MIT License).

For detailed documentation and usage instructions, please visit the [Documentation](link-to-documentation) section.

Feel free to create issues and pull requests if you have suggestions or improvements for the package.

Happy researching with PRIMVS!
