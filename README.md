# Low-Cost SDR-Based Wireless Data Analysis System

This project presents a low-cost, software-defined wireless signal processing system developed for IoT and Smart Energy applications. The system is based on a Software-Defined Radio (SDR) device, GNU Radio, and Python-based digital signal processing techniques.

## Project Overview
The goal of this project is to analyze the impact of noise and Signal-to-Noise Ratio (SNR) on wireless signal quality under realistic conditions. FM broadcast signals are used as representative wireless signals, and the system evaluates signal quality using both basic filtering and spectral noise reduction techniques.

## System Components
- **SDR Device:** NooElec NESDR SMArt
- **Software Tools:** GNU Radio, Python
- **Key Techniques:** Filtering, demodulation, spectral analysis, noise reduction, SNR evaluation

## Folder Structure
- `grc/` – GNU Radio Companion flowgraphs
- `src/` – Python scripts for signal analysis and evaluation
- `data/` – Audio datasets generated during the project
- `results/` – Analysis outputs and evaluation figures
- `report/` – Final project report

## Evaluation
The system performance is evaluated using Signal-to-Noise Ratio (SNR) as the primary metric. Different signal processing methods are compared to assess their effectiveness under noisy wireless conditions.

## Academic Context
This project was developed as part of the **Business Project in Computer Science (M608)** module and demonstrates the integration of software-based system design, digital signal processing, and data-driven evaluation within a real-world business-oriented wireless scenario.
