# AFM Dashboard

**AFM Dashboard** is an interactive Jupyter Notebook designed to streamline the analysis of Atomic Force Microscopy (AFM) data. With a variety of powerful features for exploring, comparing, and visualizing data, AFM Dashboard provides a customizable platform for researchers to easily compare multiple data files and include customized analysis functionalities.

## Key Features

1. **File Browser**: 
   - Navigate and compare multiple channel data from a single measurement. Easily find good measurement data.
    ![file_browser](https://github.com/user-attachments/assets/c6d51abf-03d4-46f6-a969-67c4ea636198)

2. **Spectroscopy Data Plotting**:
   - Plot spectroscopy data from multiple channels or compare multiple data files.
   - Generate correlation plots to analyze the relationships between different data channels.
    ![spectroscopy](https://github.com/user-attachments/assets/bbdeecc8-592d-4630-9211-42a2561bb06b)

3. **Extracted Spectroscopy Parameters**:
   - Automatically extract key parameters from spectroscopy curves, including: Adhesion, Snap-in distance, Stiffness, Amplitude growth rate, Amplitude slope
   - Additional parameter extraction functions can be easily included.

4. **Image Segmentation and Analysis**:
   - Segment AFM images into distinct regions and analyze channel data for each distinct segment.
   - Perform statistical analysis on individual segments and generate correlation plots for each segment.
     ![segmentation](https://github.com/user-attachments/assets/a740aa12-4d5d-4763-88e8-58a400cb2da6)

5. **Force-Volume Data Viewer**:
   - Visualize force-volume data across any chosen XYZ plane for each measurement channel.
    ![force_volume](https://github.com/user-attachments/assets/19442448-7f7c-466c-9995-6fd75bd9d3ca)

6. **3D Plotting**:
   - Create 3D plots with the ability to overlay two channels — using one as a surface and the other as a color map — for enhanced data visualization.
     ![3d_plot](https://github.com/user-attachments/assets/5422c9cd-20bc-4cfb-8834-d857bdbe78bd)

7. **Cantilever Dynamics Simulation**:
   - Simulate cantilever dynamics for a given tip-sample interaction function, useful to predict the behaviour of the cantilever against a particular sample.
   - Currently supports Tapping mode and Constant-phase Amplitude Modulation mode


## Installation

To use AFM Dashboard, you'll need Python (>=3.10.12) installed in your system.

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/AFM-Dashboard.git
   cd AFM-Dashboard
   ```
   
2. Install the necessary dependencies (preferably inside a virtual environment) by running:
   
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Jupyter Lab within the virtual environment:

```bash

    jupyter lab
```

2. Open the AFM_Dashboard.ipynb file and follow the instructions in the notebook!

## Contributions

Contributions are welcome! Feel free to open issues for bug reports, feature requests, or submit pull requests with improvements.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
