import numpy as np


class MLPClassifier:
    """MLP classifier implemented with NumPy only."""

    def __init__(
        self,
        num_classes: int,
        input_dim: int = 32 * 32 * 3,
        hidden_dims=(1024, 512),
        dropout: float = 0.3,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        seed: int = 42,
    ):
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.hidden_dims = tuple(hidden_dims)
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.rng = np.random.default_rng(seed)
        self.training = True
        self.t = 0

        dims = (input_dim,) + self.hidden_dims + (num_classes,)
        self.weights = []
        self.biases = []
        for fan_in, fan_out in zip(dims[:-1], dims[1:]):
            scale = np.sqrt(2.0 / fan_in)
            self.weights.append(self.rng.normal(0.0, scale, size=(fan_in, fan_out)).astype(np.float32))
            self.biases.append(np.zeros(fan_out, dtype=np.float32))

        self.m_w = [np.zeros_like(w) for w in self.weights]
        self.v_w = [np.zeros_like(w) for w in self.weights]
        self.m_b = [np.zeros_like(b) for b in self.biases]
        self.v_b = [np.zeros_like(b) for b in self.biases]

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def _prepare_input(self, x):
        x = np.asarray(x, dtype=np.float32)
        return x.reshape(x.shape[0], -1)

    def _forward(self, x, training=False):
        activations = [self._prepare_input(x)]
        pre_activations = []
        dropout_masks = []

        a = activations[0]
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            z = a @ w + b
            pre_activations.append(z)
            a = np.maximum(z, 0.0)

            if training and self.dropout > 0.0:
                keep_prob = 1.0 - self.dropout
                mask = (self.rng.random(a.shape) < keep_prob).astype(np.float32) / keep_prob
                a = a * mask
            else:
                mask = None

            dropout_masks.append(mask)
            activations.append(a)

        logits = a @ self.weights[-1] + self.biases[-1]
        pre_activations.append(logits)
        activations.append(logits)
        return logits, activations, pre_activations, dropout_masks

    def predict_logits(self, x):
        logits, _, _, _ = self._forward(x, training=False)
        return logits

    def predict(self, x):
        return np.argmax(self.predict_logits(x), axis=1)

    def _loss_and_grad_logits(self, logits, labels, include_regularization=True):
        labels = np.asarray(labels, dtype=np.int64)
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_scores = np.exp(shifted)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        batch_size = labels.shape[0]
        loss = -np.log(probs[np.arange(batch_size), labels] + 1e-12).mean()
        if include_regularization and self.weight_decay > 0.0:
            loss += 0.5 * self.weight_decay * sum(np.sum(w * w) for w in self.weights)

        grad_logits = probs
        grad_logits[np.arange(batch_size), labels] -= 1.0
        grad_logits /= batch_size
        return loss, grad_logits

    def train_batch(self, x, labels):
        self.train()
        logits, activations, pre_activations, dropout_masks = self._forward(x, training=True)
        loss, grad = self._loss_and_grad_logits(logits, labels, include_regularization=True)

        grad_w = [None] * len(self.weights)
        grad_b = [None] * len(self.biases)

        for layer_idx in reversed(range(len(self.weights))):
            grad_w[layer_idx] = activations[layer_idx].T @ grad
            if self.weight_decay > 0.0:
                grad_w[layer_idx] += self.weight_decay * self.weights[layer_idx]
            grad_b[layer_idx] = np.sum(grad, axis=0)

            if layer_idx > 0:
                grad = grad @ self.weights[layer_idx].T
                mask = dropout_masks[layer_idx - 1]
                if mask is not None:
                    grad = grad * mask
                grad = grad * (pre_activations[layer_idx - 1] > 0.0)

        self._adam_step(grad_w, grad_b)
        return loss, np.argmax(logits, axis=1)

    def _adam_step(self, grad_w, grad_b, beta1=0.9, beta2=0.999, eps=1e-8):
        self.t += 1
        for i in range(len(self.weights)):
            self.m_w[i] = beta1 * self.m_w[i] + (1.0 - beta1) * grad_w[i]
            self.v_w[i] = beta2 * self.v_w[i] + (1.0 - beta2) * (grad_w[i] * grad_w[i])
            self.m_b[i] = beta1 * self.m_b[i] + (1.0 - beta1) * grad_b[i]
            self.v_b[i] = beta2 * self.v_b[i] + (1.0 - beta2) * (grad_b[i] * grad_b[i])

            m_w_hat = self.m_w[i] / (1.0 - beta1 ** self.t)
            v_w_hat = self.v_w[i] / (1.0 - beta2 ** self.t)
            m_b_hat = self.m_b[i] / (1.0 - beta1 ** self.t)
            v_b_hat = self.v_b[i] / (1.0 - beta2 ** self.t)

            self.weights[i] -= self.lr * m_w_hat / (np.sqrt(v_w_hat) + eps)
            self.biases[i] -= self.lr * m_b_hat / (np.sqrt(v_b_hat) + eps)

    def loss_and_predictions(self, x, labels):
        self.eval()
        logits = self.predict_logits(x)
        loss, _ = self._loss_and_grad_logits(logits, labels, include_regularization=False)
        return loss, np.argmax(logits, axis=1)

    def save(self, path):
        payload = {
            "num_classes": np.array(self.num_classes),
            "input_dim": np.array(self.input_dim),
            "dropout": np.array(self.dropout),
            "lr": np.array(self.lr),
            "weight_decay": np.array(self.weight_decay),
            "hidden_dims": np.array(self.hidden_dims),
            "t": np.array(self.t),
        }
        for i, (w, b, mw, vw, mb, vb) in enumerate(
            zip(self.weights, self.biases, self.m_w, self.v_w, self.m_b, self.v_b)
        ):
            payload[f"w_{i}"] = w
            payload[f"b_{i}"] = b
            payload[f"mw_{i}"] = mw
            payload[f"vw_{i}"] = vw
            payload[f"mb_{i}"] = mb
            payload[f"vb_{i}"] = vb
        np.savez(path, **payload)

    def load(self, path):
        data = np.load(path, allow_pickle=False)
        self.t = int(data["t"])
        for i in range(len(self.weights)):
            self.weights[i] = data[f"w_{i}"].astype(np.float32)
            self.biases[i] = data[f"b_{i}"].astype(np.float32)
            self.m_w[i] = data[f"mw_{i}"].astype(np.float32)
            self.v_w[i] = data[f"vw_{i}"].astype(np.float32)
            self.m_b[i] = data[f"mb_{i}"].astype(np.float32)
            self.v_b[i] = data[f"vb_{i}"].astype(np.float32)
