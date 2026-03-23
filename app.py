from flask import Flask, request, jsonify, render_template_string
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

app = Flask(__name__)

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train all 3 models
models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5)
}

accuracies = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    accuracies[name] = round(accuracy_score(y_test, preds) * 100, 2)

print("Model Accuracies:", accuracies)

HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Iris Flower Classifier</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', sans-serif;
            min-height: 100vh;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }
        .container { width: 100%; max-width: 560px; }
        .header { text-align: center; margin-bottom: 24px; color: white; }
        .header h1 { font-size: 30px; font-weight: 700; }
        .header p { color: rgba(255,255,255,0.6); margin-top: 6px; font-size: 14px; }
        .card {
            background: rgba(255,255,255,0.05);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 24px;
            padding: 32px;
        }

        /* Model selector */
        .model-selector {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 10px;
            margin-bottom: 24px;
        }
        .model-btn {
            padding: 10px 6px;
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.15);
            border-radius: 12px;
            color: rgba(255,255,255,0.6);
            font-size: 12px;
            font-weight: 600;
            cursor: pointer;
            text-align: center;
            transition: all 0.3s;
        }
        .model-btn.active {
            background: linear-gradient(135deg, #e94560, #0f3460);
            border-color: transparent;
            color: white;
        }
        .model-btn .acc {
            display: block;
            font-size: 16px;
            font-weight: 700;
            color: #4caf50;
            margin-top: 4px;
        }
        .model-btn.active .acc { color: white; }

        /* Inputs */
        .input-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 14px;
            margin-bottom: 20px;
        }
        .input-group label {
            display: block;
            color: rgba(255,255,255,0.7);
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 6px;
        }
        .input-group input {
            width: 100%;
            padding: 12px 14px;
            background: rgba(255,255,255,0.08);
            border: 1px solid rgba(255,255,255,0.15);
            border-radius: 10px;
            color: white;
            font-size: 15px;
            outline: none;
            transition: all 0.3s;
        }
        .input-group input:focus {
            border-color: #e94560;
            background: rgba(255,255,255,0.12);
        }
        .input-group input::placeholder { color: rgba(255,255,255,0.3); }

        button.predict-btn {
            width: 100%;
            padding: 15px;
            background: linear-gradient(135deg, #e94560, #0f3460);
            color: white;
            border: none;
            border-radius: 12px;
            font-size: 15px;
            font-weight: 700;
            cursor: pointer;
            letter-spacing: 1px;
            transition: transform 0.2s, opacity 0.2s;
        }
        button.predict-btn:hover { transform: translateY(-2px); opacity: 0.9; }

        /* Result */
        .result-card {
            margin-top: 20px;
            padding: 24px;
            border-radius: 16px;
            display: none;
            animation: slideUp 0.4s ease;
        }
        @keyframes slideUp {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .result-card.setosa { background: linear-gradient(135deg, #134e5e, #71b280); }
        .result-card.versicolor { background: linear-gradient(135deg, #1a1a2e, #4776e6); }
        .result-card.virginica { background: linear-gradient(135deg, #4a0072, #9c27b0); }

        .result-top {
            display: flex;
            align-items: center;
            gap: 14px;
            margin-bottom: 16px;
        }
        .flower-emoji { font-size: 48px; }
        .result-info h2 { color: white; font-size: 22px; font-weight: 700; }
        .result-info p { color: rgba(255,255,255,0.7); font-size: 13px; margin-top: 3px; }
        .badge {
            display: inline-block;
            background: rgba(255,255,255,0.2);
            color: white;
            padding: 3px 10px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
            margin-top: 6px;
        }
        .confidence-section h4 {
            color: rgba(255,255,255,0.8);
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 10px;
        }
        .prob-bar { margin-bottom: 8px; }
        .prob-label {
            display: flex;
            justify-content: space-between;
            color: rgba(255,255,255,0.8);
            font-size: 12px;
            margin-bottom: 3px;
        }
        .prob-track {
            height: 7px;
            background: rgba(255,255,255,0.1);
            border-radius: 4px;
            overflow: hidden;
        }
        .prob-fill {
            height: 100%;
            border-radius: 4px;
            background: rgba(255,255,255,0.7);
            transition: width 0.8s ease;
        }

        .model-used {
            margin-top: 14px;
            text-align: center;
            color: rgba(255,255,255,0.5);
            font-size: 12px;
        }
    </style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>🌸 Iris Classifier</h1>
        <p>Compare 3 ML models — choose your predictor</p>
    </div>

    <div class="card">
        <!-- Model Selector -->
        <div class="model-selector">
            <div class="model-btn active" onclick="selectModel('Decision Tree', this)">
                Decision Tree
                <span class="acc" id="acc-dt"></span>
            </div>
            <div class="model-btn" onclick="selectModel('Random Forest', this)">
                Random Forest
                <span class="acc" id="acc-rf"></span>
            </div>
            <div class="model-btn" onclick="selectModel('KNN', this)">
                KNN
                <span class="acc" id="acc-knn"></span>
            </div>
        </div>

        <!-- Inputs -->
        <div class="input-grid">
            <div class="input-group">
                <label>Sepal Length (cm)</label>
                <input type="number" id="sl" placeholder="e.g. 5.1" step="0.1">
            </div>
            <div class="input-group">
                <label>Sepal Width (cm)</label>
                <input type="number" id="sw" placeholder="e.g. 3.5" step="0.1">
            </div>
            <div class="input-group">
                <label>Petal Length (cm)</label>
                <input type="number" id="pl" placeholder="e.g. 1.4" step="0.1">
            </div>
            <div class="input-group">
                <label>Petal Width (cm)</label>
                <input type="number" id="pw" placeholder="e.g. 0.2" step="0.1">
            </div>
        </div>

        <button class="predict-btn" onclick="predict()">⚡ Predict Species</button>

        <div class="result-card" id="result">
            <div class="result-top">
                <div class="flower-emoji" id="emoji"></div>
                <div class="result-info">
                    <h2 id="species-name"></h2>
                    <p id="description"></p>
                    <span class="badge" id="confidence-badge"></span>
                </div>
            </div>
            <div class="confidence-section">
                <h4>Prediction Confidence</h4>
                <div class="prob-bar">
                    <div class="prob-label">
                        <span>🌺 Setosa</span>
                        <span id="p0">0%</span>
                    </div>
                    <div class="prob-track">
                        <div class="prob-fill" id="b0" style="width:0%"></div>
                    </div>
                </div>
                <div class="prob-bar">
                    <div class="prob-label">
                        <span>🌼 Versicolor</span>
                        <span id="p1">0%</span>
                    </div>
                    <div class="prob-track">
                        <div class="prob-fill" id="b1" style="width:0%"></div>
                    </div>
                </div>
                <div class="prob-bar">
                    <div class="prob-label">
                        <span>🌷 Virginica</span>
                        <span id="p2">0%</span>
                    </div>
                    <div class="prob-track">
                        <div class="prob-fill" id="b2" style="width:0%"></div>
                    </div>
                </div>
            </div>
            <div class="model-used" id="model-used"></div>
        </div>
    </div>
</div>

<script>
let selectedModel = 'Decision Tree';
let accuracyData = {};

window.onload = async function() {
    const res = await fetch('/accuracies');
    accuracyData = await res.json();
    document.getElementById('acc-dt').textContent = accuracyData['Decision Tree'] + '%';
    document.getElementById('acc-rf').textContent = accuracyData['Random Forest'] + '%';
    document.getElementById('acc-knn').textContent = accuracyData['KNN'] + '%';
}

function selectModel(name, el) {
    selectedModel = name;
    document.querySelectorAll('.model-btn').forEach(b => b.classList.remove('active'));
    el.classList.add('active');
    document.getElementById('result').style.display = 'none';
}

async function predict() {
    const sl = document.getElementById('sl').value;
    const sw = document.getElementById('sw').value;
    const pl = document.getElementById('pl').value;
    const pw = document.getElementById('pw').value;

    if (!sl || !sw || !pl || !pw) {
        alert('Please fill in all 4 measurements!');
        return;
    }

    const response = await fetch('/predict', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({sl, sw, pl, pw, model: selectedModel})
    });

    const data = await response.json();

    const emojis = {setosa: '🌺', versicolor: '🌼', virginica: '🌷'};
    const descriptions = {
        setosa: 'Small, delicate flower found in arctic regions',
        versicolor: 'Medium sized flower from eastern North America',
        virginica: 'Large, elegant flower from eastern North America'
    };

    const result = document.getElementById('result');
    result.className = 'result-card ' + data.species;
    result.style.display = 'block';

    document.getElementById('emoji').textContent = emojis[data.species];
    document.getElementById('species-name').textContent =
        'Iris ' + data.species.charAt(0).toUpperCase() + data.species.slice(1);
    document.getElementById('description').textContent = descriptions[data.species];
    document.getElementById('confidence-badge').textContent =
        '✓ ' + data.confidence + '% Confident';
    document.getElementById('model-used').textContent =
        'Predicted using: ' + selectedModel + ' (Accuracy: ' + accuracyData[selectedModel] + '%)';

    const probs = data.probabilities;
    for (let i = 0; i < 3; i++) {
        const pct = Math.round(probs[i] * 100);
        document.getElementById('p' + i).textContent = pct + '%';
        document.getElementById('b' + i).style.width = pct + '%';
    }
}
</script>
</body>
</html>
'''

@app.route('/')
def home():
    return render_template_string(HTML)

@app.route('/accuracies')
def get_accuracies():
    return jsonify(accuracies)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = [[
        float(data['sl']),
        float(data['sw']),
        float(data['pl']),
        float(data['pw'])
    ]]
    model_name = data.get('model', 'Random Forest')
    model = models[model_name]
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    confidence = round(float(np.max(probabilities)) * 100, 1)
    species = iris.target_names[prediction]
    return jsonify({
        'species': species,
        'confidence': confidence,
        'probabilities': probabilities.tolist()
    })

if __name__ == '__main__':
    app.run(debug=True)
