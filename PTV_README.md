# Rocker or Pump? Transcriptomic Response of Endothelial Cells Exposed to Peristaltic Pump-Based Unidirectional Flow vs. Rocker-Induced Bidirectional Flow


## Table of Contents
- [Project Overview](#project-overview)
- [Setup](#setup)
- [Usage](#usage)


## Project Overview

This code focuses on analysing flow velocity on a microfluidic chip using both experimental and theoretical approaches.

- **`PTV.py`**: This script implements a pipeline to extract the flow velocity of fluorescent beads from videos using TrackPy and OpenCV. Users can rotate the video, define the region of interest (ROI), set the binary threshold and select the relevant particle size interactively. The user-defined settings are saved.  TrackPy is used to detect particles in each frame and link them across frames into trajectories. The velocity is calculated based on the particle displacement and a calibrated pixel-to-millimetre conversion. The flow rate in µL/min assumes a 2D flow representation and is computed from the calculated velocity and cross-sectional area. The flow rate data is smoothed using a median filter and exported as an Excel spreadsheet. 

- **`theoretical_calculation.py`**: This script estimates the theoretical flow rate in a tilted microfluidic channel using analytical and ODE-based models. The exponential decay model is based on the hydrostatic pressure drop due to gravity-driven flow in a rectangular channel, assuming Poiseuille flow. A coupled reservoir ODE model takes the changing heights of the inlet and outlet reservoirs over time into account. Flow is based on the law of mass conservation. Theoretical calculations are compared with the experimental flow rate data from the PTV.py script.


### Key Features

- **Interactive Calibration & Masking**: Select frame regions and thresholds manually.
- **Automatic Tracking**: Track particles with user-defined diameter using TrackPy.
- **Flow Rate Estimation**: Compute µL/min rates using trajectory data and cross-sectional area.
- **Smoothing & Visualisation**: Median filtering and Matplotlib plots for smoothed velocity profiles.
- **Theoretical Modelling**:
  - **Exponential Decay Model** based on pressure head.
  - **Coupled Reservoir ODE System** for simulating inlet/outlet height dynamics.
- **Data Export**: Save flow rates and plots to Excel and PDF.

## Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/nvahdani/ooc_flow.git
cd ooc_flow

### Step 2: Set Up a Virtual Environment
Create and activate a conda environment (Python ≥ 3.7):

```bash
conda create --name ptv_env python=3.7
conda activate ptv_env

### Step 3: Install Requirements

```bash
pip install -r requirements.txt

### Step 4: Verify Installation

```bash
python --version
pip list


## Usage
To run the PTV.py and the theoretical_calculations.py scripts, ensure the following folder structure. 

/
├── Data/
│   └── Video/
│       ├── Video1/
│       ├── Video2/
│       ├── Video3/
│       └── Video4/
├── Video/
│
├── settings/
├── PTV.py
├── theoretical_calculation.py
├── README.md
 