from flask import Flask, request, jsonify, render_template_string
import pickle
import numpy as np

app = Flask(__name__)

# Load the pickle model safely
try:
    with open('vaive_model.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    model = None
    print("Warning: vaive_model.pkl not found. Please place it in the same directory.")

# Advanced Dashboard Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Analytics & Prediction Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-gradient: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
            --panel-bg: rgba(255, 255, 255, 0.03);
            --panel-border: rgba(255, 255, 255, 0.08);
            --accent-glow: rgba(99, 102, 241, 0.15);
            
            --brand-primary: #6366f1;
            --brand-gradient: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --input-bg: rgba(15, 23, 42, 0.6);
            
            --success: #10b981;
            --success-bg: rgba(16, 185, 129, 0.1);
            --error: #ef4444;
            --error-bg: rgba(239, 68, 68, 0.1);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Plus Jakarta Sans', sans-serif;
            -webkit-font-smoothing: antialiased;
        }

        body {
            background: var(--bg-gradient);
            color: var(--text-primary);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 1.5rem;
        }

        .dashboard-card {
            background: var(--panel-bg);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid var(--panel-border);
            max-width: 640px;
            width: 100%;
            border-radius: 24px;
            padding: 3rem;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
        }

        .brand-header {
            text-align: center;
            margin-bottom: 2.5rem;
        }

        .brand-badge {
            display: inline-block;
            padding: 0.35rem 1rem;
            background: rgba(99, 102, 241, 0.1);
            border: 1px solid rgba(99, 102, 241, 0.2);
            color: #818cf8;
            border-radius: 100px;
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            margin-bottom: 0.75rem;
        }

        .brand-header h1 {
            font-size: 2rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            background: linear-gradient(to right, #fff, #94a3b8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }

        .brand-header p {
            color: var(--text-secondary);
            font-size: 0.95rem;
        }

        .input-grid {
            display: grid;
            grid-template-columns: 1fr;
            gap: 1.25rem;
            margin-bottom: 2rem;
        }

        @media (min-width: 480px) {
            .input-grid {
                grid-template-columns: repeat(2, 1fr);
            }
            .full-width-field {
                grid-column: span 2;
            }
        }

        .input-wrapper {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }

        .input-wrapper label {
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--text-secondary);
            letter-spacing: 0.02em;
        }

        .input-field {
            background: var(--input-bg);
            border: 1px solid var(--panel-border);
            border-radius: 12px;
            padding: 0.85rem 1.1rem;
            color: var(--text-primary);
            font-size: 0.95rem;
            transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        }

        .input-field:focus {
            outline: none;
            border-color: var(--brand-primary);
            box-shadow: 0 0 0 4px var(--accent-glow);
            background: rgba(15, 23, 42, 0.8);
        }

        .input-field::placeholder {
            color: #475569;
        }

        .action-btn {
            width: 100%;
            padding: 1rem;
            background: var(--brand-gradient);
            color: white;
            border: none;
            border-radius: 12px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.15s ease, opacity 0.2s ease;
            box-shadow: 0 10px 20px -5px rgba(99, 102, 241, 0.4);
        }

        .action-btn:hover {
            opacity: 0.95;
        }

        .action-btn:active {
            transform: scale(0.98);
        }

        .output-panel {
            margin-top: 2.5rem;
            border-radius: 16px;
            animation: slideUp 0.4s ease-out;
        }

        .output-success {
            background: var(--success-bg);
            border: 1px solid rgba(16, 185, 129, 0.2);
            padding: 1.5rem;
            text-align: center;
        }

        .output-success span {
            display: block;
            font-size: 0.8rem;
            font-weight: 700;
            text-transform: uppercase;
            color: #34d399;
            letter-spacing: 0.05em;
            margin-bottom: 0.5rem;
        }

        .output-value {
            font-size: 2.25rem;
            font-weight: 700;
            color: #fff;
            text-shadow: 0 0 20px rgba(16, 185, 129, 0.3);
        }

        .output-error {
            background: var(--error-bg);
            border: 1px solid rgba(239, 68, 68, 0.2);
            padding: 1.25rem;
            color: #fca5a5;
            font-size: 0.9rem;
            border-radius: 12px;
            text-align: center;
        }

        @keyframes slideUp {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
</head>
<body>

<div class="dashboard-card">
    <div class="brand-header">
        <div class="brand-badge">ML Inference Engine</div>
        <h1>Predictive Intelligence</h1>
        <p>Supply telemetry metrics to run operational analysis.</p>
    </div>

    <form method="POST" action="/">
        <div class="input-grid">
            <div class="input-wrapper">
                <label for="feature1">Operational Metric A</label>
                <input type="number" step="any" name="feature1" id="feature1" class="input-field" required placeholder="0.00">
            </div>
            
            <div class="input-wrapper">
                <label for="feature2">Operational Metric B</label>
                <input type="number" step="any" name="feature2" id="feature2" class="input-field" required placeholder="0.00">
            </div>

            <div class="input-wrapper full-width-field">
                <label for="feature3">System Performance Constant</label>
                <input type="number" step="any" name="feature3" id="feature3" class="input-field" required placeholder="0.00">
            </div>
        </div>

        <button type="submit" class="action-btn">Execute Model Evaluation</button>
    </form>

    {% if prediction_text %}
    <div class="output-panel output-success">
        <span>Inference Output</span>
        <div class="output-value">{{ prediction_text }}</div>
    </div>
    {% endif %}

    {% if error_text %}
    <div class="output-panel output-error">
        <strong>System Error:</strong> {{ error_text }}
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
            error_text = "The binary model 'vaive_model.pkl' could not be initialized on the server backend."
            return render_template_string(HTML_TEMPLATE, error_text=error_text)
            
        try:
            # Parse form fields safely
            f1 = float(request.form['feature1'])
            f2 = float(request.form['feature2'])
            f3 = float(request.form['feature3'])
            
            # Structuring shape matching the model properties
            input_data = np.array([[f1, f2, f3]])
            
            prediction = model.predict(input_data)
            prediction_text = str(prediction[0])
            
        except Exception as e:
            error_text = f"Invalid payload structure or computational error: {str(e)}"

    return render_template_string(HTML_TEMPLATE, prediction_text=prediction_text, error_text=error_text)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
