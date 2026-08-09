import React, { useEffect, useState } from 'react';

function StatTile({ label, value }) {
  return (
    <div style={{ border: '1px solid #ddd', borderRadius: 8, padding: 16, flex: 1 }}>
      <div style={{ fontSize: 13, color: '#666' }}>{label}</div>
      <div style={{ fontSize: 24, fontWeight: 600 }}>{value}</div>
    </div>
  );
}

export default function App() {
  const [info, setInfo] = useState(null);
  const [history, setHistory] = useState([]);
  const [features, setFeatures] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [busy, setBusy] = useState(false);

  const loadInfo = () => fetch('/api/model/info').then((r) => r.json()).then(setInfo);
  const loadHistory = () => fetch('/api/model/history').then((r) => r.json()).then(setHistory);

  useEffect(() => {
    loadInfo();
    loadHistory();
  }, []);

  const trySample = async () => {
    setBusy(true);
    setPrediction(null);
    try {
      const sample = await fetch('/api/model/sample').then((r) => r.json());
      setFeatures(sample);
      const result = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ features: sample.features }),
      }).then((r) => r.json());
      setPrediction(result);
    } finally {
      setBusy(false);
    }
  };

  const retrain = async () => {
    setBusy(true);
    try {
      await fetch('/api/retrain', { method: 'POST' });
      await loadInfo();
      await loadHistory();
    } finally {
      setBusy(false);
    }
  };

  return (
    <div style={{ maxWidth: 720, margin: '40px auto', fontFamily: 'system-ui', padding: '0 16px' }}>
      <h1>ML Predictive Analytics Platform</h1>
      <p style={{ color: '#666' }}>
        Real scikit-learn models trained on the diabetes progression dataset, with lightweight
        AutoML selection (linear regression, ridge, random forest, gradient boosting — best R² wins)
        and a background job that retrains on an interval.
      </p>

      {info && (
        <div style={{ display: 'flex', gap: 12, marginBottom: 24 }}>
          <StatTile label="Active model" value={info.model_name} />
          <StatTile label="R²" value={info.r2.toFixed(4)} />
          <StatTile label="MAE" value={info.mae.toFixed(2)} />
          <StatTile label="RMSE" value={info.rmse.toFixed(2)} />
        </div>
      )}

      <div style={{ display: 'flex', gap: 8, marginBottom: 24 }}>
        <button onClick={trySample} disabled={busy}>Predict from a real sample</button>
        <button onClick={retrain} disabled={busy}>Retrain now</button>
      </div>

      {features && prediction && (
        <div style={{ border: '1px solid #ddd', borderRadius: 8, padding: 16, marginBottom: 24 }}>
          <strong>Prediction:</strong> {prediction.prediction.toFixed(2)} (model: {prediction.model_name})
          <div style={{ fontSize: 12, color: '#888', marginTop: 8 }}>
            Input features: {features.feature_names.map((n, i) => `${n}=${features.features[i].toFixed(3)}`).join(', ')}
          </div>
        </div>
      )}

      <h2>Training history</h2>
      <table style={{ width: '100%', borderCollapse: 'collapse' }}>
        <thead>
          <tr style={{ textAlign: 'left', borderBottom: '1px solid #ccc' }}>
            <th>Model</th><th>R²</th><th>MAE</th><th>RMSE</th><th>Trained at</th>
          </tr>
        </thead>
        <tbody>
          {history.map((run, i) => (
            <tr key={i} style={{ borderBottom: '1px solid #eee' }}>
              <td>{run.model_name}</td>
              <td>{run.r2.toFixed(4)}</td>
              <td>{run.mae.toFixed(2)}</td>
              <td>{run.rmse.toFixed(2)}</td>
              <td>{new Date(run.trained_at).toLocaleString()}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
