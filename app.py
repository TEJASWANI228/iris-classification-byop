from flask import Flask, request, jsonify, render_template_string
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

app = Flask(__name__)

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>🌸 Iris Flower Classifier</title>
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700;800&display=swap" rel="stylesheet">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            font-family: 'Poppins', sans-serif;
            min-height: 100vh;
            background: linear-gradient(135deg, #ffd6e7 0%, #e8d5f5 35%, #d5f5e8 100%);
            overflow-x: hidden;
        }

        .blob {
            position: fixed;
            border-radius: 50%;
            filter: blur(80px);
            opacity: 0.4;
            animation: float 8s ease-in-out infinite;
            z-index: 0;
        }
        .blob1 { width: 400px; height: 400px; background: #ff85c1; top: -100px; left: -100px; animation-delay: 0s; }
        .blob2 { width: 350px; height: 350px; background: #c084fc; top: 50%; right: -100px; animation-delay: 2s; }
        .blob3 { width: 300px; height: 300px; background: #86efac; bottom: -100px; left: 30%; animation-delay: 4s; }
        .blob4 { width: 250px; height: 250px; background: #fda4af; top: 30%; left: 20%; animation-delay: 6s; }

        @keyframes float {
            0%, 100% { transform: translateY(0px) scale(1); }
            50% { transform: translateY(-30px) scale(1.05); }
        }

        .main {
            position: relative;
            z-index: 1;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 40px 20px;
        }

        .header {
            text-align: center;
            margin-bottom: 36px;
        }
        .header .tag {
            display: inline-block;
            background: linear-gradient(135deg, #f472b6, #a855f7);
            color: white;
            padding: 6px 18px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
            letter-spacing: 2px;
            text-transform: uppercase;
            margin-bottom: 16px;
        }
        .header h1 {
            font-size: 44px;
            font-weight: 800;
            color: #3b0764;
            line-height: 1.1;
        }
        .header h1 span {
            background: linear-gradient(135deg, #f472b6, #a855f7, #34d399);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .header p {
            color: #7c3aed;
            margin-top: 10px;
            font-size: 14px;
            opacity: 0.7;
        }

        .card {
            background: rgba(255,255,255,0.65);
            border: 1px solid rgba(255,255,255,0.9);
            border-radius: 28px;
            padding: 40px;
            width: 100%;
            max-width: 580px;
            backdrop-filter: blur(20px);
            box-shadow: 0 20px 60px rgba(168,85,247,0.15);
        }

        .model-label {
            color: #7c3aed;
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 2px;
            margin-bottom: 12px;
        }
        .model-selector {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 10px;
            margin-bottom: 28px;
        }
        .model-btn {
            padding: 14px 8px;
            background: rgba(255,255,255,0.5);
            border: 2px solid rgba(168,85,247,0.2);
            border-radius: 14px;
            color: #7c3aed;
            font-size: 12px;
            font-weight: 600;
            cursor: pointer;
            text-align: center;
            transition: all 0.3s;
            font-family: 'Poppins', sans-serif;
        }
        .model-btn:hover {
            border-color: #a855f7;
            background: rgba(168,85,247,0.1);
        }
        .model-btn.dt.active { background: linear-gradient(135deg, #f472b6, #ec4899); border-color: transparent; color: white; }
        .model-btn.rf.active { background: linear-gradient(135deg, #a855f7, #7c3aed); border-color: transparent; color: white; }
        .model-btn.knn.active { background: linear-gradient(135deg, #34d399, #059669); border-color: transparent; color: white; }
        .model-btn .acc {
            display: block;
            font-size: 20px;
            font-weight: 800;
            color: #059669;
            margin-top: 4px;
        }
        .model-btn.active .acc { color: rgba(255,255,255,0.95); }

        .divider {
            height: 1px;
            background: rgba(168,85,247,0.15);
            margin: 24px 0;
        }

        .input-label {
            color: #7c3aed;
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 2px;
            margin-bottom: 12px;
        }
        .input-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
            margin-bottom: 24px;
        }
        .input-group label {
            display: block;
            color: #6d28d9;
            font-size: 11px;
            font-weight: 600;
            margin-bottom: 6px;
        }
        .input-group input {
            width: 100%;
            padding: 13px 16px;
            background: rgba(255,255,255,0.7);
            border: 2px solid rgba(168,85,247,0.2);
            border-radius: 12px;
            color: #3b0764;
            font-size: 15px;
            font-family: 'Poppins', sans-serif;
            outline: none;
            transition: all 0.3s;
        }
        .input-group input:focus {
            border-color: #a855f7;
            background: white;
        }
        .input-group input::placeholder { color: rgba(124,58,237,0.3); }

        .predict-btn {
            width: 100%;
            padding: 16px;
            background: linear-gradient(135deg, #f472b6, #a855f7, #34d399);
            background-size: 200% 200%;
            animation: gradientShift 3s ease infinite;
            color: white;
            border: none;
            border-radius: 14px;
            font-size: 16px;
            font-weight: 700;
            cursor: pointer;
            font-family: 'Poppins', sans-serif;
            letter-spacing: 1px;
            transition: transform 0.2s;
            box-shadow: 0 8px 24px rgba(168,85,247,0.35);
        }
        @keyframes gradientShift {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        .predict-btn:hover { transform: translateY(-2px); }

        .result-card {
            margin-top: 24px;
            border-radius: 20px;
            overflow: hidden;
            display: none;
            animation: popIn 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            box-shadow: 0 12px 40px rgba(168,85,247,0.2);
        }
        @keyframes popIn {
            from { opacity: 0; transform: scale(0.8); }
            to { opacity: 1; transform: scale(1); }
        }

        .result-header {
            padding: 28px;
            display: flex;
            align-items: center;
            gap: 20px;
        }
        .result-card.setosa .result-header { background: linear-gradient(135deg, #f472b6, #ec4899); }
        .result-card.versicolor .result-header { background: linear-gradient(135deg, #a855f7, #7c3aed); }
        .result-card.virginica .result-header { background: linear-gradient(135deg, #34d399, #059669); }

        .flower-big { font-size: 60px; }
        .result-text h2 { color: white; font-size: 26px; font-weight: 800; }
        .result-text p { color: rgba(255,255,255,0.85); font-size: 13px; margin-top: 4px; }
        .confidence-pill {
            display: inline-block;
            background: rgba(255,255,255,0.3);
            color: white;
            padding: 4px 14px;
            border-radius: 20px;
            font-size: 13px;
            font-weight: 700;
            margin-top: 8px;
        }

        .result-body {
            background: rgba(255,255,255,0.85);
            padding: 24px 28px;
        }
        .conf-title {
            color: #7c3aed;
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 2px;
            margin-bottom: 16px;
        }
        .prob-bar { margin-bottom: 12px; }
        .prob-label {
            display: flex;
            justify-content: space-between;
            color: #3b0764;
            font-size: 13px;
            font-weight: 600;
            margin-bottom: 6px;
        }
        .prob-track {
            height: 8px;
            background: rgba(168,85,247,0.1);
            border-radius: 4px;
            overflow: hidden;
        }
        .prob-fill {
            height: 100%;
            border-radius: 4px;
            transition: width 1s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .fill-setosa { background: linear-gradient(90deg, #f472b6, #ec4899); }
        .fill-versicolor { background: linear-gradient(90deg, #a855f7, #7c3aed); }
        .fill-virginica { background: linear-gradient(90deg, #34d399, #059669); }

        .model-tag {
            margin-top: 16px;
            text-align: center;
            color: #7c3aed;
            font-size: 11px;
            opacity: 0.6;
        }
    </style>
</head>
<body>
    <div class="blob blob1"></div>
    <div class="blob blob2"></div>
    <div class="blob blob3"></div>
    <div class="blob blob4"></div>

    <div class="main">
        <div class="header">
            <div class="tag">🌿 ML Powered</div>
            <h1>Iris Flower<br><span>Species Classifier</span></h1>
            <p>Enter measurements · Choose your model · Get instant prediction</p>
        </div>

        <div class="card">
            <div class="model-label">Choose ML Model</div>
            <div class="model-selector">
                <button class="model-btn dt active" onclick="selectModel('Decision Tree', this)">
                    Decision Tree
                    <span class="acc" id="acc-dt"></span>
                </button>
                <button class="model-btn rf" onclick="selectModel('Random Forest', this)">
                    Random Forest
                    <span class="acc" id="acc-rf"></span>
                </button>
                <button class="model-btn knn" onclick="selectModel('KNN', this)">
                    KNN
                    <span class="acc" id="acc-knn"></span>
                </button>
            </div>

            <div class="divider"></div>

            <div class="input-label">Flower Measurements</div>
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

            <button class="predict-btn" onclick="predict()">
                ⚡ Predict Species Now
            </button>

            <div class="result-card" id="result">
                <div class="result-header">
                    <div class="flower-big" id="emoji"></div>
                    <div class="result-text">
                        <h2 id="species-name"></h2>
                        <p id="description"></p>
                        <span class="confidence-pill" id="conf-pill"></span>
                    </div>
                </div>
                <div class="result-body">
                    <div class="conf-title">Prediction Confidence Breakdown</div>
                    <div class="prob-bar">
                        <div class="prob-label">
                            <span>🌺 Iris Setosa</span>
                            <span id="p0">0%</span>
                        </div>
                        <div class="prob-track">
                            <div class="prob-fill fill-setosa" id="b0" style="width:0%"></div>
                        </div>
                    </div>
                    <div class="prob-bar">
                        <div class="prob-label">
                            <span>🌼 Iris Versicolor</span>
                            <span id="p1">0%</span>
                        </div>
                        <div class="prob-track">
                            <div class="prob-fill fill-versicolor" id="b1" style="width:0%"></div>
                        </div>
                    </div>
                    <div class="prob-bar">
                        <div class="prob-label">
                            <span>🌷 Iris Virginica</span>
                            <span id="p2">0%</span>
                        </div>
                        <div class="prob-track">
                            <div class="prob-fill fill-virginica" id="b2" style="width:0%"></div>
                        </div>
                    </div>
                    <div class="model-tag" id="model-tag"></div>
                </div>
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

    // Validate realistic ranges
    const ranges = {
        'Sepal Length': {val: parseFloat(sl), min: 4.0, max: 8.0},
        'Sepal Width': {val: parseFloat(sw), min: 2.0, max: 4.5},
        'Petal Length': {val: parseFloat(pl), min: 1.0, max: 7.0},
        'Petal Width': {val: parseFloat(pw), min: 0.1, max: 2.5}
    };

    for (const [name, r] of Object.entries(ranges)) {
        if (r.val < r.min || r.val > r.max) {
            alert(`⚠️ ${name} must be between ${r.min} and ${r.max} cm.\nYou entered: ${r.val} cm\n\nPlease enter realistic flower measurements!`);
            return;
        }
    }

    const response = await fetch('/predict', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({sl, sw, pl, pw, model: selectedModel})
    });

    const data = await response.json();

    const emojis = {setosa: '🌺', versicolor: '🌼', virginica: '🌷'};
    const descriptions = {
        setosa: 'Small, delicate flower · Arctic & subarctic regions',
        versicolor: 'Medium sized flower · Eastern North America',
        virginica: 'Large, elegant flower · Eastern North America'
    };

    const result = document.getElementById('result');
    result.className = 'result-card ' + data.species;
    result.style.display = 'block';

    document.getElementById('emoji').textContent = emojis[data.species];
    document.getElementById('species-name').textContent =
        'Iris ' + data.species.charAt(0).toUpperCase() + data.species.slice(1);
    document.getElementById('description').textContent = descriptions[data.species];
    document.getElementById('conf-pill').textContent = '✓ ' + data.confidence + '% Confident';
    document.getElementById('model-tag').textContent =
        'Predicted using ' + selectedModel + ' · Accuracy: ' + accuracyData[selectedModel] + '%';

    const probs = data.probabilities;
    for (let i = 0; i < 3; i++) {
        const pct = Math.round(probs[i] * 100);
        document.getElementById('p' + i).textContent = pct + '%';
        setTimeout(() => {
            document.getElementById('b' + i).style.width = pct + '%';
        }, 100);
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
