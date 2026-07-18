import os
import pickle
import numpy as np
from flask import Flask, request, render_template_string

app = Flask(__name__)

# --------------------------------------------------------------------------
# Model Initialization
# --------------------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'vaive_model.pkl')
model = None

try:
    with open(MODEL_PATH, 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    print(f"CRITICAL: Model file not found at {MODEL_PATH}")
except Exception as e:
    print(f"CRITICAL: Failed to load model file: {str(e)}")

# --------------------------------------------------------------------------
# UI Template Layout (Modern Glassmorphism Design)
# --------------------------------------------------------------------------
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Predictive Analytics Platform</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-gradient: linear-gradient(135deg, #0b0f19 0%, #111827 100%);
            --card-bg: rgba(255, 255, 255, 0.02);
            --card-border: rgba(255, 255, 255, 0.07);
            --input-bg: rgba(17, 24, 39, 0.8);
            --input-border: rgba(255, 255, 255, 0.1);
            
            --primary-glow: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
            --text-main: #f9fafb;
            --text-muted: #9ca3af;
            
            --success-border: rgba(16, 185, 129, 0.2);
            --success-bg: rgba(16, 185, 129, 0.08);
            --error-border: rgba(239, 68, 68, 0.2);
            --error-bg: rgba(239, 68, 68, 0.08);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Inter', sans-serif;
            -webkit-font-smoothing: antialiased;
        }

        body {
            background: var(--bg-gradient);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 2rem 1rem;
        }

        .app-container {
            max-width: 580px;
            width: 100%;
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 24px;
            padding: 3rem;
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
        }

        .header {
            text-align: center;
            margin-bottom: 2.5rem;
        }

        .badge {
            display: inline-block;
            padding: 0.35rem 0.85rem;
            background: rgba(79, 70, 229, 0.1);
            border: 1px solid rgba(79, 70, 229, 0.2);
            color: #a5b4fc;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.75rem;
        }

        .header h1 {
            font-size: 1.85rem;
            font-weight: 700;
            letter-spacing: -0.025em;
            margin-bottom: 0.5rem;
            background: linear-gradient(to bottom right, #ffffff, #d1d5db);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .header p {
            color: var(--text-muted);
            font-size: 0.95rem;
        }

        .form-grid {
            display: grid;
            grid-template-columns: 1fr;
            gap: 1.25rem;
            margin-bottom: 2rem;
        }

        @media(min-width: 480px) {
            .form-grid {
                grid-template-columns: repeat(2, 1fr);
            }
            .grid-span-2 {
                grid-column: span 2;
            }
        }

        .field-group {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }

        .field-group label {
            font-size: 0.85rem;
            font-weight: 600;
            color: #d1d5db;
        }

        .form-control {
            width: 100%;
            background: var(--input-bg);
            border: 1px solid var(--input-border);
            padding: 0.85rem 1rem;
            border-radius: 12px;
            color: var(--text-main);
            font-size: 0.95rem;
            transition: all 0.2s ease;
        }

        .form-control:focus {
            outline: none;
            border-color: #6366f1;
            box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.15);
        }

        select.form-control {
            appearance: none;
            background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='%239ca3af' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><polyline points='6 9 12 15 18 9'></polyline></svg>");
            background-repeat: no-repeat;
            background-position: right 1rem center;
            background-size: 1.1rem;
            padding-right: 2.5rem;
            cursor: pointer;
        }

        select.form-control option {
            background-color: #111827;
            color: var(--text-main);
        }

        .submit-btn {
            width: 100%;
            padding: 1rem;
            background: var(--primary-glow);
            color: #ffffff;
            border: none;
            border-radius: 12px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: opacity 0.2s ease, transform 0.1s ease;
            box-shadow: 0 4px 20px rgba(79, 70, 229, 0.3);
        }

        .submit-btn:hover {
            opacity: 0.95;
        }

        .submit-btn:active {
            transform: scale(0.99);
        }

        .alert-box {
            margin-top: 2rem;
            border-radius: 14px;
            padding: 1.5rem;
            text-align: center;
            animation: fadeIn 0.3s ease-in-out;
        }

        .alert-success {
            background: var(--success-bg);
            border: 1px solid var(--success-border);
        }

        .alert-success h3 {
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #34d399;
            margin-bottom: 0.5rem;
        }

        .alert-success .val {
            font-size: 2rem;
            font-weight: 700;
            color: #ffffff;
            text-shadow: 0 0 20px rgba(16, 185, 129, 0.2);
        }

        .alert-danger {
            background: var(--error-bg);
            border: 1px solid var(--error-border);
            color: #fca5a5;
            font-size: 0.9rem;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(8px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
</head>
<body>

<div class="app-container">
    <div class="header">
        <span class="badge">GaussianNB Inference Platform</span>
        <h1>Predictive Intelligence</h1>
        <p>Analyze user demographics to process targeting classifications.</p>
    </div>

    <form method="POST" action="/">
        <div class="form-grid">
            <div class="field-group">
                <label for="gender">Gender</label>
                <select name="gender" id="gender" class="form-control" required>
                    <option value="" disabled selected>Select option</option>
                    <option value="1">Male</option>
                    <option value="0">Female</option>
                </select>
            </div>

            <div class="field-group">
                <label for="age">Age (Years)</label>
                <input type="number" min="1" max="120" name="age" id="age" class="form-control" placeholder="e.g., 35" required>
            </div>

            <div class="field-group grid-span-2">
                <label for="salary">Estimated Annual Salary ($)</label>
                <input type="number" min="0" step="any" name="salary" id="salary" class="form-control" placeholder="e.g., 75000" required>
            </div>
        </div>

        <button type="submit" class="submit-btn">Run Engine Execution</button>
    </form>

    {% if prediction_text %}
    <div class="alert-box alert-success">
        <h3>Target Classification</h3>
        <div class="val">{{ prediction_text }}</div>
    </div>
    {% endif %}

    {% if error_text %}
    <div class="alert-box alert-danger">
        <strong>Initialization Fault:</strong> {{ error_text }}
    </div>
    {% endif %}
</div>

</body>
</html>
"""

# --------------------------------------------------------------------------
# Controller & Processing Pipelines
# --------------------------------------------------------------------------
@app.route('/', methods=['GET', 'POST'])
def dashboard():
    prediction_text = None
    error_text = None
    
    if request.method == 'POST':
        if model is None:
            error_text = "The binary GaussianNB pipeline file 'vaive_model.pkl' could not be instantiated on the server ecosystem."
            return render_template_string(HTML_TEMPLATE, error_text=error_text)
            
        try:
            # Safely fetch parameter matrices matching ['Gender', 'Age', 'EstimatedSalary']
            gender = float(request.form['gender'])
            age = float(request.form['age'])
            salary = float(request.form['salary'])
            
            # Format explicitly for Scikit-Learn validation structural shapes
            features_array = np.array([[gender, age, salary]])
            
            # Execute Model evaluation
            raw_prediction = model.predict(features_array)[0]
            
            # Label translation if evaluation yields a boolean structure
            if str(raw_prediction) == "1":
                prediction_text = "Positive Output (1)"
            elif str(raw_prediction) == "0":
                prediction_text = "Negative Output (0)"
            else:
                prediction_text = f"Result Class: {raw_prediction}"
                
        except Exception as e:
            error_text = f"Payload parse failure or evaluation anomaly: {str(e)}"

    return render_template_string(HTML_TEMPLATE, prediction_text=prediction_text, error_text=error_text)

if __name__ == '__main__':
    # Render binds dynamically using environment configurations
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
