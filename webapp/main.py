import os
import sys
import numpy as np
import pandas as pd
import json
import plotly
import plotly.express as px
import plotly.graph_objects as go
import torch
import subprocess
from flask import Flask, render_template, request, jsonify
from datetime import datetime, timedelta

# Add parent directory to path so we can import from project
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from exp.exp_main import Exp_Main
from utils.tools import dotdict

# Set environment variable to resolve MKL threading issue
os.environ['MKL_THREADING_LAYER'] = 'GNU'

app = Flask(__name__)

# Global variable to track if model is trained
MODEL_TRAINED = {}

# Get list of available stocks
def get_available_stocks():
    dataset_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dataset')
    csv_files = [f.split('.')[0] for f in os.listdir(dataset_dir) 
                if f.endswith('.csv') and not f.startswith('NIFTY') and not f == 'stock_metadata.csv']
    return sorted(csv_files)

# Get prediction results for a stock
def get_prediction_results(stock):
    """
    Get prediction results from the future_predictions.csv file in the results directory
    for the given stock.
    """
    # Get the absolute path to the DC directory
    dc_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Look for the results directory for this stock
    results_dir = os.path.join(dc_dir, 'results')
    
    # Find the matching directory for this stock
    if os.path.exists(results_dir):
        matching_dirs = [d for d in os.listdir(results_dir) if d.startswith(f"{stock}_")]
        
        if matching_dirs:
            # Use the first matching directory
            stock_dir = os.path.join(results_dir, matching_dirs[0])
            predictions_file = os.path.join(stock_dir, 'future_predictions.csv')
            
            if os.path.exists(predictions_file):
                try:
                    return pd.read_csv(predictions_file)
                except Exception as e:
                    print(f"Error reading predictions file: {str(e)}")
                    return None
    
    return None

# Get date range from prediction results
def get_prediction_date_range(stock):
    predictions_df = get_prediction_results(stock)
    if predictions_df is not None and 'Date' in predictions_df.columns:
        predictions_df['Date'] = pd.to_datetime(predictions_df['Date'])
        return {
            'start': predictions_df['Date'].min().strftime('%Y-%m-%d'),
            'end': predictions_df['Date'].max().strftime('%Y-%m-%d')
        }
    elif predictions_df is not None and 'date' in predictions_df.columns:
        predictions_df['date'] = pd.to_datetime(predictions_df['date'])
        return {
            'start': predictions_df['date'].min().strftime('%Y-%m-%d'),
            'end': predictions_df['date'].max().strftime('%Y-%m-%d')
        }
    return None

# Read stock data
def get_stock_data(stock):
    try:
        file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                               'dataset', f"{stock}.csv")
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error reading stock data: {e}")
        return None

# Configure model arguments
def get_model_args(stock):
    args = dotdict()
    args.model = 'SegRNN'
    args.data = stock
    args.root_path = './dataset/'
    args.data_path = f"{stock}.csv"
    args.features = 'MS'  # Multivariate time series
    args.target = 'Close'
    args.freq = 'd'
    args.seq_len = 720
    args.pred_len = 96
    args.label_len = 48

    # Dynamically determine input/output feature counts
    stock_csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                               'dataset', f"{stock}.csv")
    try:
        df = pd.read_csv(stock_csv_path)
        num_features = len(df.columns) - (1 if 'date' in df.columns else 0)
        args.enc_in = 5
        args.dec_in = 5
        args.c_out = 1  # Usually predicting 'Close' price
    except Exception as e:
        args.enc_in = 5
        args.dec_in = 5
        args.c_out = 1

    args.d_model = 512
    args.dropout = 0.5
    args.use_amp = False
    args.output_attention = False
    args.test_flop = False
    args.embed = 'timeF'
    args.factor = 5
    args.patience = 3
    args.des = 'test'
    args.lradj = 'type1'
    args.use_gpu = True if torch.cuda.is_available() else False
    args.gpu = 0
    args.use_multi_gpu = False
    args.itr = 1
    args.train_epochs = 10
    args.batch_size = 32
    args.learning_rate = 0.001
    args.loss = 'mae'
    args.seg_len = 48
    args.win_len = 48
    args.rnn_type = 'rnn'
    args.dec_way = 'pmf'
    args.pct_start = 0.3
    
    return args

@app.route('/')
def index():
    stocks = get_available_stocks()
    return render_template('index.html', stocks=stocks)

@app.route('/stock_data', methods=['GET'])
def stock_data():
    stock = request.args.get('stock', 'AXISBANK')
    df = get_stock_data(stock)
    
    if df is None:
        return jsonify({'error': 'Stock data not found'})
        
    # Convert date to proper format and ensure it's sorted
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
    
    # Get the last year of data for display
    if len(df) > 365:
        df = df.iloc[-365:]
    
    # Create historical price chart
    date_col = 'date' if 'date' in df.columns else df.index
    price_col = 'Close' if 'Close' in df.columns else df.columns[1]
    
    fig = px.line(df, x=date_col, y=price_col, title=f"{stock} Historical Price")
    graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    # Use the prediction date range instead of the full dataset
    date_range = get_prediction_date_range(stock)
    
    return jsonify({
        'graph': graph_json,
        'date_range': date_range if date_range else {
            'start': df['date'].iloc[0].strftime('%Y-%m-%d') if 'date' in df.columns else 'N/A',
            'end': df['date'].iloc[-1].strftime('%Y-%m-%d') if 'date' in df.columns else 'N/A'
        }
    })

@app.route('/train_model', methods=['POST'])
def train_model():
    data = request.json
    stock = data.get('stock', 'AXISBANK')
    
    # Check if model is already trained for this stock
    if stock in MODEL_TRAINED and MODEL_TRAINED[stock]:
        return jsonify({
            'status': 'success',
            'message': f'Model for {stock} already trained. Ready for predictions.'
        })
    
    try:
        # Get the absolute path to the DC directory
        dc_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Use the stock_predict.sh script for training
        script_path = os.path.join(dc_dir, 'scripts', 'SegRNN', 'stock_predict.sh')
        
        # Run the script with training mode from the DC root directory
        command = f"cd {dc_dir} && sh scripts/SegRNN/stock_predict.sh --stock {stock} --is_training 1"
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Error in training: {result.stderr}")
            return jsonify({'status': 'error', 'message': f"Training failed: {result.stderr}"})
        
        # Mark as trained
        MODEL_TRAINED[stock] = True
        
        return jsonify({
            'status': 'success',
            'message': f'Model for {stock} trained successfully. Ready for predictions.'
        })
    except Exception as e:
        print(f"Exception during training: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/make_prediction', methods=['POST'])
def make_prediction():
    data = request.json
    stock = data.get('stock', 'AXISBANK')
    target_date = data.get('date')
    
    # Get the prediction results directly from the CSV file
    predictions_df = get_prediction_results(stock)
    
    if predictions_df is None:
        # If predictions don't exist yet, try to generate them
        try:
            # Get the absolute path to the DC directory
            dc_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            
            # Run the script with prediction mode from the DC root directory
            command = f"cd {dc_dir} && sh scripts/SegRNN/stock_predict.sh --stock {stock} --is_training 0"
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"Error in prediction: {result.stderr}")
                return jsonify({'status': 'error', 'message': f"Prediction failed: {result.stderr}"})
            
            # Try to get the prediction results again after generating
            predictions_df = get_prediction_results(stock)
            
            if predictions_df is None:
                return jsonify({'status': 'error', 'message': 'Failed to find prediction results after generation'})
        except Exception as e:
            print(f"Exception during prediction: {str(e)}")
            return jsonify({'status': 'error', 'message': str(e)})
    
    # Process the predictions data
    if 'Date' not in predictions_df.columns and 'date' in predictions_df.columns:
        predictions_df.rename(columns={'date': 'Date'}, inplace=True)
    
    # Convert Date column to datetime and sort
    predictions_df['Date'] = pd.to_datetime(predictions_df['Date'])
    predictions_df = predictions_df.sort_values('Date')
    
    # Get the available date range
    min_date = predictions_df['Date'].min().strftime('%Y-%m-%d')
    max_date = predictions_df['Date'].max().strftime('%Y-%m-%d')
    
    # Filter predictions for the target date if provided
    if target_date:
        target_date = datetime.strptime(target_date, '%Y-%m-%d')
        
        # Find the exact date match or closest date
        if target_date in predictions_df['Date'].values:
            prediction_value = predictions_df[predictions_df['Date'] == target_date]['Predicted'].iloc[0]
        else:
            closest_date_idx = (predictions_df['Date'] - target_date).abs().idxmin()
            closest_date = predictions_df.loc[closest_date_idx, 'Date']
            prediction_value = predictions_df.loc[closest_date_idx, 'Predicted']
            target_date = closest_date  # Use the closest date
    else:
        # If no date specified, use the first date in predictions
        target_date = predictions_df['Date'].iloc[0]
        prediction_value = predictions_df['Predicted'].iloc[0]
    
    # Create prediction chart for entire date range
    fig = px.line(predictions_df, x='Date', y='Predicted',
                  title=f"{stock} Price Predictions")
    fig.update_traces(line=dict(color='red'))
    
    # Highlight the selected point if there's a target date
    fig.add_trace(go.Scatter(
        x=[target_date],
        y=[prediction_value],
        mode='markers',
        marker=dict(color='blue', size=10),
        name='Selected Date'
    ))
    
    graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    return jsonify({
        'status': 'success',
        'graph': graph_json,
        'date_range': {
            'start': min_date,
            'end': max_date
        },
        'prediction_value': float(prediction_value),
        'selected_date': target_date.strftime('%Y-%m-%d')
    })

if __name__ == '__main__':
    app.run(debug=True)