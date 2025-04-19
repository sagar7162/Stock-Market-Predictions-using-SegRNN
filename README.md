# Stock Price Forecasting with SegRNN

This project implements a time series forecasting system for stock price prediction using a Segment-based Recurrent Neural Network (SegRNN) architecture. The system includes data preprocessing, model training, evaluation, and a web-based dashboard for interactive visualization of forecasts.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
  - [Web Dashboard](#web-dashboard)
  - [Command Line Interface](#command-line-interface)
- [Data Format](#data-format)
- [Training Process](#training-process)
- [Prediction](#prediction)
- [Performance Analysis](#performance-analysis)
- [Customization](#customization)
- [Troubleshooting](#troubleshooting)

## Overview

Stock Price Forecasting with SegRNN is designed to predict future stock prices based on historical data using advanced neural network techniques. The project leverages a novel segment-based approach to time series forecasting that divides input sequences into smaller segments for more effective temporal processing.

**Key Features:**

- Interactive web dashboard for stock selection and prediction visualization
- Support for multiple stocks (currently 30+ stocks from the NIFTY50 index)
- Flexible training parameters and configuration
- Real-time prediction and visualization
- Comparative analysis with historical data

## Project Structure

The codebase is organized as follows:

```
.
├── checkpoints/         # Saved model checkpoints
├── data_provider/       # Data loading and preprocessing
├── dataset/             # Stock data CSV files
├── exp/                 # Experiment setup and execution
├── layers/              # Custom neural network layers
├── logs/                # Training and execution logs
├── models/              # Model implementations (including SegRNN)
├── results/             # Prediction results and visualizations
├── scripts/             # Automation scripts for training and evaluation
├── test_results/        # Test evaluation results
├── utils/               # Utility functions for metrics, tools, etc.
├── webapp/              # Flask web application
├── run_longExp.py       # Main experiment execution script
└── README.md            # This documentation
```

## Model Architecture

### SegRNN (Segment-based Recurrent Neural Network)

SegRNN is designed specifically for time series forecasting with the following key components:

1. **Segment-based Processing**:

   - Divides input time series into fixed-length segments
   - Processes segments rather than individual time steps
   - Reduces sequence length for more efficient training

2. **Network Architecture**:

   - Value Embedding Layer: Transforms segments to embedding space
   - RNN Backbone: Processes embedded sequences (supports RNN, GRU, or LSTM)
   - Decoding Mechanism: Two methods available:
     - RMF (Recurrent Multi-step Forecasting): Autoregressive decoding
     - PMF (Parallel Multi-step Forecasting): Single-step decoding with position encoding

3. **Hyperparameters**:
   - `seq_len`: Input sequence length (historical data window)
   - `pred_len`: Prediction length (forecast horizon)
   - `seg_len`: Segment length for dividing sequences
   - `d_model`: Model dimension for embeddings
   - `rnn_type`: Type of RNN cell (rnn, gru, lstm)
   - `dec_way`: Decoding method (rmf, pmf)
   - `channel_id`: Enable/disable channel position encoding

### Advantages Over Traditional Models

- **Efficiency**: Processing segments reduces computational complexity
- **Long-Term Dependencies**: Better capture of long-range patterns
- **Scalability**: Works well with different sequence lengths and prediction horizons
- **Flexibility**: Adaptable to various time series characteristics

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/sagar7162/Stock-Market-Predictions-using-SegRNN.git
   cd DC
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

   Key dependencies include:

   - PyTorch
   - Flask
   - pandas
   - numpy
   - plotly
   - matplotlib

3. **Prepare your environment**:
   - Set the environment variable to resolve MKL threading issues:
     ```bash
     export MKL_THREADING_LAYER=GNU
     ```

## Usage

### Web Dashboard

1. **Start the web server**:

   ```bash
   python webapp/main.py
   ```

2. **Access the dashboard** at `http://127.0.0.1:5000/`

3. **Using the dashboard**:
   - Select a stock from the dropdown menu
   - Click "Train Model" to train the SegRNN model for the selected stock
   - After training completes, select a target date from the prediction range
   - Click "Show Prediction" to see the forecast

### Command Line Interface

For batch processing or automated workflows, use the provided shell scripts:

1. **Train a model for a specific stock**:

   ```bash
   sh scripts/SegRNN/stock_predict.sh --stock STOCKNAME --is_training 1
   ```

2. **Generate predictions using a trained model**:

   ```bash
   sh scripts/SegRNN/stock_predict.sh --stock STOCKNAME --is_training 0
   ```

3. **Custom parameter configuration**:

   ```bash
   sh scripts/SegRNN/stock_predict.sh --stock STOCKNAME --pred_len 192 --seq_len 720 --seg_len 48
   ```

4. **Direct use of the Python script**:
   ```bash
   python run_longExp.py --is_training 1 --model SegRNN --data custom --data_path STOCKNAME.csv --seq_len 720 --pred_len 96 --model_id STOCKNAME_720_96
   ```

## Data Format

The system requires stock data in CSV format with the following columns:

- Date: Date of trading
- Open: Opening price
- High: Highest price during the day
- Low: Lowest price during the day
- Close: Closing price (target for prediction)
- Volume: Trading volume

CSV files should be named as `STOCKNAME.csv` and placed in the `dataset/` directory.

## Training Process

The training process involves the following steps:

1. **Data Preprocessing**:

   - Loading data from CSV files
   - Normalization using feature-wise scaling
   - Splitting into training, validation, and test sets
   - Creating sliding windows for sequence input

2. **Model Configuration**:

   - Setting up hyperparameters based on command-line arguments
   - Initializing model architecture
   - Setting up optimizer and loss function

3. **Training Loop**:

   - Epoch-based training with early stopping
   - Validation after each epoch
   - Learning rate adjustment based on validation performance
   - Checkpoint saving for best models

4. **Evaluation**:
   - Testing on held-out test set
   - Calculating metrics (MAE, MSE, RMSE)
   - Generating visualizations of predictions

## Prediction

The prediction process generates forecasts for future time points beyond the available data:

1. **Loading Trained Model**:

   - Retrieving the best checkpoint
   - Initializing model with saved weights

2. **Generating Predictions**:

   - Using the latest available data window as input
   - Running forward pass through the model
   - De-normalizing outputs to get actual price values

3. **Saving Results**:
   - Storing predictions in CSV format
   - Generating visualization plots
   - Making results available through the web dashboard

## Performance Analysis

The model's performance is evaluated using several metrics:

- **MAE (Mean Absolute Error)**: Average absolute difference between predictions and actual values
- **MSE (Mean Squared Error)**: Average squared difference, penalizing large errors more heavily
- **RMSE (Root Mean Squared Error)**: Square root of MSE, interpretable in the original data scale

Results from various experiments show that SegRNN generally outperforms traditional methods and other deep learning approaches, particularly for longer prediction horizons.

## Customization

The system is highly customizable:

1. **Model Parameters**: Adjust in the script or through command-line arguments:

   - Sequence length (`--seq_len`)
   - Prediction length (`--pred_len`)
   - Segment length (`--seg_len`)
   - Model dimension (`--d_model`)
   - RNN type (`--rnn_type`)
   - Decoding way (`--dec_way`)

2. **Training Parameters**:

   - Batch size (`--batch_size`)
   - Learning rate (`--learning_rate`)
   - Training epochs (`--train_epochs`)
   - Patience for early stopping (`--patience`)

3. **Feature Selection**:
   - Choose which columns to use as features (`--features`)
   - Select the target variable (`--target`)

## Troubleshooting

Common issues and solutions:

1. **CUDA Out of Memory**:

   - Reduce batch size
   - Decrease sequence length
   - Lower model dimension

2. **Poor Prediction Quality**:

   - Try different hyperparameter combinations
   - Ensure sufficient training data
   - Check for data quality issues
   - Experiment with different RNN types (rnn, gru, lstm)

3. **Training Too Slow**:

   - Reduce sequence length or segment length
   - Use GPU acceleration if available
   - Decrease model complexity (d_model, dropout)

4. **Web Dashboard Issues**:
   - Check logs for error messages
   - Ensure model is properly trained before prediction
   - Verify file permissions for results directory

For any other issues, please check the logs in the `logs/` directory for detailed error messages.

---

**Acknowledgments:**

- This project builds upon research in time series forecasting and deep learning
- Utilizes PyTorch for model implementation and training
- Flask for web dashboard development
