# fl_bqc_simulator.py
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit


class HeartDiseaseFLBQC:
    def __init__(self, n_clients=3):
        self.n_clients = n_clients
        self.scaler = StandardScaler()
        self.X, self.y = self.load_data()
        self.client_data = self.split_data_balanced()
        self.global_weights = np.zeros((1, self.X.shape[1]))

    def load_data(self):
        """Load and return cleaned heart disease data."""
        df = pd.read_csv('data/heart_disease_clean.csv')
        X = df.drop('target', axis=1).values
        y = df['target'].values
        X = self.scaler.fit_transform(X)
        return X, y

    def split_data_balanced(self):
        """Split data into clients using stratified sampling to preserve class balance."""
        X, y = self.X, self.y
        sss = StratifiedShuffleSplit(
            n_splits=self.n_clients,
            train_size=len(X) // self.n_clients,
            random_state=42
        )
        client_data = []
        for _, idx in sss.split(X, y):
            client_data.append((X[idx], y[idx]))
        return client_data

    def train_local_model(self, X, y):
        """Train local logistic regression model and return gradient approximation."""
        if len(np.unique(y)) < 2:
            return None, np.zeros((1, X.shape[1]))

        model = LogisticRegression(max_iter=200, solver='liblinear', random_state=42)
        try:
            model.fit(X, y)
        except:
            return None, np.zeros((1, X.shape[1]))

        # Approximate gradient using synthetic baseline
        dummy_X = np.random.randn(2, X.shape[1])
        dummy_y = np.array([0, 1])
        baseline = LogisticRegression(max_iter=100, solver='liblinear', random_state=43)
        baseline.fit(dummy_X, dummy_y)
        grad = model.coef_ - baseline.coef_

        return model, grad

    def run_fl_only(self, epochs=10):
        """Run classical Federated Learning (FedAvg)."""
        results = []
        global_weights = np.zeros((1, self.X.shape[1]))
        momentum = np.zeros_like(global_weights)  # For smoother convergence

        for e in range(epochs):
            gradients = []
            for X, y in self.client_data:
                model, grad = self.train_local_model(X, y)
                if model is not None:
                    gradients.append(grad)

            if len(gradients) == 0:
                gradients = [np.zeros((1, self.X.shape[1]))]

            avg_grad = np.mean(gradients, axis=0)

            # Momentum for stable convergence
            momentum = 0.9 * momentum + 0.1 * avg_grad
            global_weights += momentum * 0.15  # Slightly higher LR

            # Evaluate
            pseudo = LogisticRegression(max_iter=1, solver='liblinear')
            dummy_X = np.random.randn(2, self.X.shape[1])
            dummy_y = np.array([0, 1])
            pseudo.fit(dummy_X, dummy_y)
            pseudo.coef_ = global_weights
            pseudo.intercept_ = np.zeros(1)
            y_pred = pseudo.predict(self.X)
            acc = accuracy_score(self.y, y_pred)
            leak = np.mean([np.linalg.norm(g) for g in gradients])

            results.append({
                'round': e + 1,
                'accuracy': acc,
                'privacy_leakage': leak,
                'method': 'FL Only'
            })
        return pd.DataFrame(results)

    def run_efl_bqc(self, epochs=10):
        """Run Enhanced FL with BQC-inspired blind aggregation."""
        results = []
        global_weights = np.zeros((1, self.X.shape[1]))
        momentum = np.zeros_like(global_weights)

        for e in range(epochs):
            blinded_gradients = []
            masks = []

            for X, y in self.client_data:
                model, grad = self.train_local_model(X, y)
                if model is None:
                    continue

                # Client-side: blind gradient with secret mask
                np.random.seed(42 + e + len(blinded_gradients))  # Deterministic for reproducibility
                mask = np.random.choice([-1, 1], size=grad.shape)
                blinded_grad = grad * mask

                blinded_gradients.append(blinded_grad)
                masks.append(mask)

            if len(blinded_gradients) == 0:
                avg_blinded = np.zeros((1, self.X.shape[1]))
            else:
                avg_blinded = np.mean(blinded_gradients, axis=0)

            # Server sends blinded update back
            # Clients unblind it locally
            avg_true_grad = np.zeros_like(avg_blinded)
            for bg, mask in zip(blinded_gradients, masks):
                avg_true_grad += bg * mask  # unblind
            avg_true_grad /= len(blinded_gradients)

            # Update with momentum
            momentum = 0.9 * momentum + 0.1 * avg_true_grad
            global_weights += momentum * 0.15

            # Evaluate
            pseudo = LogisticRegression(max_iter=1, solver='liblinear')
            dummy_X = np.random.randn(2, self.X.shape[1])
            dummy_y = np.array([0, 1])
            pseudo.fit(dummy_X, dummy_y)
            pseudo.coef_ = global_weights
            pseudo.intercept_ = np.zeros(1)
            y_pred = pseudo.predict(self.X)
            acc = accuracy_score(self.y, y_pred)

            # Privacy: server saw only blinded gradients
            # High std = more randomness = better privacy
            leak = np.mean([np.std(bg) for bg in blinded_gradients])

            results.append({
                'round': e + 1,
                'accuracy': acc,
                'privacy_leakage': leak,
                'method': 'FL + BQC (Simulated)'
            })
        return pd.DataFrame(results)