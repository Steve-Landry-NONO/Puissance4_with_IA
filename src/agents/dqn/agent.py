# DQN logic(epsilon, target net...)

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.agents.dqn.model import build_q_network
from src.agents.dqn.replay_buffer import ReplayBuffer


class DQNAgent:
    def __init__(
        self,
        gamma: float = 0.99,
        lr: float = 1e-3,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 50_000,
        target_update_every: int = 1000,
        buffer_capacity: int = 50_000,
    ) -> None:
        self.gamma = float(gamma)

        self.q = build_q_network()
        self.target = build_q_network()
        self.target.set_weights(self.q.get_weights())

        self.opt = keras.optimizers.Adam(learning_rate=lr)
        self.buffer = ReplayBuffer(capacity=buffer_capacity)

        self.eps_start = float(epsilon_start)
        self.eps_end = float(epsilon_end)
        self.eps_decay_steps = int(epsilon_decay_steps)
        self.target_update_every = int(target_update_every)

        self.step_count = 0

    def act_greedy(self, state: np.ndarray, mask: np.ndarray) -> int:
        q = self.q(np.expand_dims(state, axis=0), training=False).numpy()[0]
        q_masked = np.where(mask, q, -1e9)
        return int(np.argmax(q_masked))

    def epsilon(self) -> float:
        t = min(1.0, self.step_count / self.eps_decay_steps)
        return self.eps_start + t * (self.eps_end - self.eps_start)

    def act_greedy(self, state, mask):
        q = self.model.predict(state[None, ...], verbose=0)[0]
        q = np.where(mask, q, -1e9)
        return int(np.argmax(q))

    def act(self, state: np.ndarray, action_mask: np.ndarray) -> int:
        """
        state: (6,7,2) float32
        action_mask: (7,) bool
        """
        eps = self.epsilon()
        self.step_count += 1

        valid = np.where(action_mask)[0]
        if len(valid) == 0:
            return 0

        if np.random.rand() < eps:
            return int(np.random.choice(valid))

        q_values = self.q(np.expand_dims(state, axis=0), training=False).numpy()[0]  # (7,)

        # Masking : on met -inf sur les actions invalides
        masked_q = np.where(action_mask, q_values, -1e9)
        return int(np.argmax(masked_q))

    def remember(self, s, a, r, s2, done, mask2) -> None:
        self.buffer.add(s, a, r, s2, done, mask2)

    def train_step(self, batch_size: int = 64) -> float:
        if len(self.buffer) < batch_size:
            return 0.0

        batch = self.buffer.sample(batch_size)

        s = tf.convert_to_tensor(batch.s, dtype=tf.float32)
        a = tf.convert_to_tensor(batch.a, dtype=tf.int32)
        r = tf.convert_to_tensor(batch.r, dtype=tf.float32)
        s2 = tf.convert_to_tensor(batch.s2, dtype=tf.float32)
        done = tf.convert_to_tensor(batch.done, dtype=tf.float32)
        mask2 = tf.convert_to_tensor(batch.mask2, dtype=tf.bool)

        loss = self._train_step_tf(s, a, r, s2, done, mask2)

        # target network update (hors tf.function = plus simple)
        if self.step_count % self.target_update_every == 0:
            self.target.set_weights(self.q.get_weights())

        return float(loss.numpy())

    @tf.function
    def _train_step_tf(self, s, a, r, s2, done, mask2):
        q2 = self.target(s2, training=False)  # (B,7)
        q2_masked = tf.where(mask2, q2, tf.fill(tf.shape(q2), tf.constant(-1e9, tf.float32)))
        max_q2 = tf.reduce_max(q2_masked, axis=1)  # (B,)
        y = r + (1.0 - done) * self.gamma * max_q2  # (B,)

        with tf.GradientTape() as tape:
            q_pred_all = self.q(s, training=True)  # (B,7)
            idx = tf.stack([tf.range(tf.shape(a)[0]), a], axis=1)
            q_pred = tf.gather_nd(q_pred_all, idx)  # (B,)
            loss = tf.reduce_mean(tf.square(y - q_pred))

        grads = tape.gradient(loss, self.q.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.q.trainable_variables))
        return loss

    def save(self, path: str) -> None:
        self.q.save(path)

    def load(self, path: str):
        self.model = tf.keras.models.load_model(path)
        if hasattr(self, "target_model"):
            self.target_model = tf.keras.models.clone_model(self.model)
            self.target_model.set_weights(self.model.get_weights())


