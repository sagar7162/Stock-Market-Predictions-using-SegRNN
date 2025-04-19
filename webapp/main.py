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
    Get prediction results from the future_predictions.csv file for a specific stock
    """
    try:
        # Path to the results directory
        results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
        
        # Find the right folder for this stock (with the format STOCK_720_96_SegRNN_...)
        stock_folders = [folder for folder in os.listdir(results_dir) 
                        if folder.startswith(f"{stock}_720_96_SegRNN")]
        
        if stock_folders:
            # Use the first matching folder
            print("inside folder")
            stock_folder = stock_folders[0]
            predictions_path = os.path.join(results_dir, stock_folder, 'future_predictions.csv')
        else:
            print(f"No matching folders found for {stock} in results directory")
            return None
        
        # Read the CSV file
        df = pd.read_csv(predictions_path)
        return df
    except Exception as e:
        print(f"Error reading predictions file: {str(e)}")
        return None

# Get date range from prediction results
def get_prediction_date_range(stock):
    predictions_df = get_prediction_results(stock)
    if predictions_df is not None:
        date_col = 'Date' if 'Date' in predictions_df.columns else 'date'
        if date_col in predictions_df.columns:
            predictions_df[date_col] = pd.to_datetime(predictions_df[date_col])
            dates_list = predictions_df[date_col].dt.strftime('%Y-%m-%d').tolist()
            return {
                'start': predictions_df[date_col].min().strftime('%Y-%m-%d'),
                'end': predictions_df[date_col].max().strftime('%Y-%m-%d'),
                'dates': dates_list
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
    
    # Use the prediction date range instead of generating a historical graph
    date_range = get_prediction_date_range(stock)
    
    return jsonify({
        'date_range': date_range if date_range else {
            'start': df['date'].iloc[0].strftime('%Y-%m-%d') if 'date' in df.columns else 'N/A',
            'end': df['date'].iloc[-1].strftime('%Y-%m-%d') if 'date' in df.columns else 'N/A',
            'dates': []
        }
    })

@app.route('/train_model', methods=['POST'])
def train_model():
    data = request.json
    stock = data.get('stock', 'AXISBANK')
    
    print(f"Training model for stock: {stock}")
    
    # Check if model is already trained for this stock
    if stock in MODEL_TRAINED and MODEL_TRAINED[stock]:
        print(f"Model for {stock} already trained. Returning success.")
        return jsonify({
            'status': 'success',
            'message': f'Model for {stock} already trained. Ready for predictions.'
        })
    
    try:
        # Get the absolute path to the DC directory
        dc_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        print(f"DC directory: {dc_dir}")
        
        # Check if script exists
        script_path = os.path.join(dc_dir, 'scripts', 'SegRNN', 'stock_predict.sh')
        print(f"Script path: {script_path}")
        print(f"Script exists: {os.path.exists(script_path)}")
        
        # Run the script with training mode from the DC root directory
        command = f"cd {dc_dir} && sh scripts/SegRNN/stock_predict.sh --stock {stock} --is_training 1"
        print(f"Running command: {command}")
        
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        
        print(f"Command return code: {result.returncode}")
        print(f"Command stdout: {result.stdout}")
        print(f"Command stderr: {result.stderr}")
        
        if result.returncode != 0:
            print(f"Error in training: {result.stderr}")
            return jsonify({'status': 'error', 'message': f"Training failed: {result.stderr}"})
        
        # Mark as trained
        MODEL_TRAINED[stock] = True
        print(f"Model for {stock} marked as trained successfully")
        
        return jsonify({
            'status': 'success',
            'message': f'Model for {stock} trained successfully. Ready for predictions.'
        })
    except Exception as e:
        print(f"Exception during training: {str(e)}")
        import traceback
        traceback.print_exc()
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
    
    # Process the predictions data - handle column name differences
    if 'Date' not in predictions_df.columns and 'date' in predictions_df.columns:
        predictions_df.rename(columns={'date': 'Date'}, inplace=True)
        
    # Handle prediction column name differences
    if 'Predicted' not in predictions_df.columns:
        # Try to find the prediction column using common naming patterns
        pred_columns = [col for col in predictions_df.columns if 'predict' in col.lower() or 'forecast' in col.lower()]
        if pred_columns:
            predictions_df.rename(columns={pred_columns[0]: 'Predicted'}, inplace=True)
        elif 'Close' in predictions_df.columns:  # Assume Close is the prediction if no clear prediction column
            predictions_df.rename(columns={'Close': 'Predicted'}, inplace=True)
        else:
            # If no suitable column found, use the second column (assuming first is Date)
            col_name = predictions_df.columns[1]
            predictions_df.rename(columns={col_name: 'Predicted'}, inplace=True)
    
    # Convert Date column to datetime and sort
    predictions_df['Date'] = pd.to_datetime(predictions_df['Date'])
    predictions_df = predictions_df.sort_values('Date')
    
    # Ensure the Predicted column is numeric
    predictions_df['Predicted'] = pd.to_numeric(predictions_df['Predicted'], errors='coerce')
    
    # Get the available date range
    min_date = predictions_df['Date'].min().strftime('%Y-%m-%d')
    max_date = predictions_df['Date'].max().strftime('%Y-%m-%d')
    
    # Filter predictions for the target date if provided
    if target_date:
        target_date = datetime.strptime(target_date, '%Y-%m-%d')
        
        # Find the exact date match or closest date
        if any(predictions_df['Date'] == target_date):
            prediction_row = predictions_df[predictions_df['Date'] == target_date].iloc[0]
            prediction_value = prediction_row['Predicted']
            actual_value = prediction_row['Actual'] if 'Actual' in predictions_df.columns else None
        else:
            closest_date_idx = (predictions_df['Date'] - target_date).abs().idxmin()
            closest_date = predictions_df.loc[closest_date_idx, 'Date']
            prediction_row = predictions_df.loc[closest_date_idx]
            prediction_value = prediction_row['Predicted']
            actual_value = prediction_row['Actual'] if 'Actual' in predictions_df.columns else None
            target_date = closest_date  # Use the closest date
    else:
        # If no date specified, use the first date in predictions
        prediction_row = predictions_df.iloc[0]
        target_date = prediction_row['Date']
        prediction_value = prediction_row['Predicted']
        actual_value = prediction_row['Actual'] if 'Actual' in predictions_df.columns else None
    
    # Calculate price change and percentage
    price_change = 0
    price_change_pct = 0
    
    # Try to get the previous day's value for comparison
    target_date_idx = predictions_df[predictions_df['Date'] == target_date].index[0]
    if target_date_idx > 0:
        prev_value = predictions_df.iloc[target_date_idx - 1]['Predicted']
        price_change = round(prediction_value - prev_value, 2)
        price_change_pct = round((price_change / prev_value) * 100, 2)
    
    # Create prediction chart using data from future_predictions.csv
    fig = go.Figure()
    
    # Add predicted values line
    fig.add_trace(go.Scatter(
        x=predictions_df['Date'],
        y=predictions_df['Predicted'],
        mode='lines',
        name='Predicted Price',
        line=dict(color='red', width=2)
    ))
    
    # Simple layout
    fig.update_layout(
        title=f"{stock} Price Predictions with SegRNN",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_white"
    )
    
    graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    # Create prediction data for cards (without confidence level)
    prediction_data = {
        'date': target_date.strftime('%Y-%m-%d'),
        'price': f"${prediction_value:.2f}",
        'change': f"${price_change:.2f}",
        'change_percent': f"{price_change_pct:.2f}%"
    }
    
    return jsonify({
        'status': 'success',
        'graph': graph_json,
        'date_range': {
            'start': min_date,
            'end': max_date
        },
        'prediction_value': float(prediction_value),
        'selected_date': target_date.strftime('%Y-%m-%d'),
        'prediction': prediction_data
    })

if __name__ == '__main__':
    app.run(debug=True)