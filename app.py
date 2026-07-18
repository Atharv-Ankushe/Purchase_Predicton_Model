from flask import Flask, request, jsonify, render_template_string
import pickle
import numpy as np

app = Flask(__name__)

# Load the pickle model
try:
    with open('vaive_model.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    model = None
    print("Warning: vaive_model.pkl not found. Please place it in the same directory.")

# HTML Template with built-in CSS (Slate Blue & Teal Palette)
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Model Prediction Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #f4f6f9;
            --card-bg: #ffffff;
            --primary: #4f46e5;
            --primary-hover: #4338ca;
            --text-main: #1f2937;
            --text-muted: #6b7280;
            --border: #e5e7eb;
            --success: #10b981;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Inter', sans-serif;
        }

        body {
            background-color: var(--bg-color);
            color: var(--text-main);
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            padding: 2rem;
        }

        .container {
            background: var(--card-bg);
            max-width: 550px;
            width: 100%;
            padding: 2.5rem;
            border-radius: 16px;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05), 0 8px 10px -6px rgba(0, 0, 0, 0.05);
        }

        header {
            text-align: center;
            margin-bottom: 2rem;
        }

        header h1 {
            font-size: 1.75rem;
            font-weight: 600;
            color: var(--text-main);
            margin-bottom: 0.5rem;
        }

        header p {
            color: var(--text-muted);
            font-size: 0.95rem;
        }

        .form-group {
            margin-bottom: 1.5rem;
        }

        .form-group label {
            display: block;
            font-size: 0.9rem;
            font-weight: 500;
            margin-bottom: 0.5rem;
            color: var(--text-main);
        }

        .form-group input {
            width: 100%;
            padding: 0.75rem 1rem;
            border: 1px solid var(--border);
            border-radius: 8px;
            font-size: 0.95rem;
            transition: all 0.2s ease;
            background-color: #f9fafb;
        }

        .form-group input:focus {
            outline: none;
            border-color: var(--primary);
            background-color: #fff;
            box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.1);
        }

        .btn {
            width: 100%;
            padding: 0.85rem;
            background-color: var(--primary);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 1rem;
            font-weight: 500;
            cursor: pointer;
            transition: background-color 0.2s ease;
            margin-top: 1rem;
        }

        .btn:hover {
            background-color: var(--primary-hover);
        }

        .result-box {
            margin-top: 2rem;
            padding: 1.25rem;
            background-color: #f0fdf4;
            border: 1px solid #bbf7d0;
            border-radius: 8px;
            text-align: center;
        }

        .result-box h3 {
            color: #166534;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.25rem;
        }

        .result-value {
            font-size: 1.5rem;
            font-weight: 600;
            color: var(--success);
        }
        
        .error-box {
            margin-top: 2rem;
            padding: 1rem;
            background-color: #fef2f2;
            border: 1px solid #fee2e2;
            border-radius: 8px;
            color: #991b1b;
            text-align: center;
            font-size: 0.95rem;
        }
    </style>
</head>
<body>

<div class="container">
    <header>
        <h1>Model Predictor</h1>
        <p>Enter the required metrics below to generate a prediction.</p>
    </header>

    <form method="POST" action="/">
        <!-- Update these input fields based on your model's expected features -->
        <div class="form-group">
            <label for="feature1">Feature 1 Metric</label>
            <input type="number" step="any" name="feature1" id="feature1" required placeholder="e.g. 12.5">
        </div>
        
        <div class="form-group">
            <label for="feature2">Feature 2 Metric</label>
            <input type="number" step="any" name="feature2" id="feature2" required placeholder="e.g. 0.85">
        </div>

        <div class="form-group">
            <label for="feature3">Feature 3 Metric</label>
            <input type="number" step="any" name="feature3" id="feature3" required placeholder="e.g. 104">
        </div>

        <button type="submit" class="btn">Generate Prediction</button>
    </form>

    {% if prediction_text %}
    <div class="result-box">
        <h3>Prediction Result</h3>
        <div class="result-value">{{ prediction_text }}</div>
    </div>
    {% endif %}

    {% if error_text %}
    <div class="error-box">
        {{ error_text }}
    </div>
    {% endif %}
</div>

</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction_text = None
    error_text = None
    
    if request.method == 'POST':
        if model is None:
            error_text = "Model file 'vaive_model.pkl' is missing or not loaded properly."
            return render_template_string(HTML_TEMPLATE, error_text=error_text)
            
        try:
            # Extract data from form (ensure keys match the HTML input 'name' attributes)
            f1 = float(request.form['feature1'])
            f2 = float(request.form['feature2'])
            f3 = float(request.form['feature3'])
            
            # Format inputs into an array matching your model's expected input shape
            input_data = np.array([[f1, f2, f3]])
            
            # Run prediction
            prediction = model.predict(input_data)
            prediction_text = str(prediction[0])
            
        except Exception as e:
            error_text = f"Error processing input: {str(e)}"

    return render_template_string(HTML_TEMPLATE, prediction_text=prediction_text, error_text=error_text)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
