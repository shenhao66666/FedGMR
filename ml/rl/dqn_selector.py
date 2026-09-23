import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class QNet(nn.Module):
    def __init__(self, state_dim: int, num_actions: int, hidden: int = 64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, num_actions)
        )
    def forward(self, x):
        return self.fc(x)

class ReplayBuffer:
    def __init__(self, capacity: int = 10000):
        self.buf = deque(maxlen=capacity)
    def push(self, *transition):
        self.buf.append(tuple(transition))
    def sample(self, batch_size: int):
        return random.sample(self.buf, batch_size)
    def __len__(self):
        return len(self.buf)

class DQNExpertSelector:
    def __init__(self, state_dim: int, num_experts: int,
                 lr: float = 1e-3, gamma: float = 0.5,
                 eps_start: float = 1.0, eps_end: float = 0.05, eps_decay: int = 200,
                 buffer_size: int = 1000, batch_size: int = 16,
                 target_update: int = 50,
                 num_groups: int = 1,
                 experts_per_group: int = None):
        self.num_experts = num_experts
        self.num_groups = int(num_groups) if num_groups is not None else 1
        self.experts_per_group = int(experts_per_group) if experts_per_group is not None else int(num_experts)
        self.multi_group = self.num_groups > 1
        self.num_actions = self.num_groups * self.experts_per_group if self.multi_group else self.num_experts
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = QNet(state_dim, self.num_actions).to(self.device)
        self.target = QNet(state_dim, self.num_actions).to(self.device)
        self.target.load_state_dict(self.net.state_dict())
        self.opt = optim.Adam(self.net.parameters(), lr=lr)
        self.gamma = gamma
        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.step_cnt = 0
        self.buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.target_update = target_update

    def advance_epsilon(self):
        self.step_cnt += 1
        self.eps = max(self.eps_end,
                       self.eps - (1.0 - self.eps_end) / self.eps_decay)

    def _greedy_action(self, state: np.ndarray, group_candidates=None):
        s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        q = self.net(s)
        if self.multi_group:
            q = q.view(1, self.num_groups, self.experts_per_group)
            q_np = q.squeeze(0).detach().cpu().numpy()  # [G, E]
            chosen = []
            for g in range(self.num_groups):
                candidates = None
                if isinstance(group_candidates, (list, tuple)) and g < len(group_candidates):
                    raw = group_candidates[g]
                    if isinstance(raw, np.ndarray):
                        raw = raw.astype(np.int64).tolist()
                    if isinstance(raw, (list, tuple)):
                        candidates = [int(e) for e in raw if 0 <= int(e) < self.experts_per_group]
                if candidates:
                    best_e = max(candidates, key=lambda e: float(q_np[g, e]))
                else:
                    best_e = int(np.argmax(q_np[g]))
                chosen.append(best_e)
            return np.asarray(chosen, dtype=np.int64)
        return int(q.argmax().item())

    def select_greedy(self, state: np.ndarray, group_candidates=None):
        return self._greedy_action(state, group_candidates=group_candidates)

    def select(self, state: np.ndarray, group_candidates=None):
        self.advance_epsilon()
        if random.random() < self.eps:
            if self.multi_group:
                sampled = []
                for g in range(self.num_groups):
                    candidates = None
                    if isinstance(group_candidates, (list, tuple)) and g < len(group_candidates):
                        raw = group_candidates[g]
                        if isinstance(raw, np.ndarray):
                            raw = raw.astype(np.int64).tolist()
                        if isinstance(raw, (list, tuple)):
                            candidates = [int(e) for e in raw if 0 <= int(e) < self.experts_per_group]
                    if candidates:
                        sampled.append(int(random.choice(candidates)))
                    else:
                        sampled.append(int(random.randrange(self.experts_per_group)))
                return np.asarray(sampled, dtype=np.int64)
            return random.randrange(self.num_experts)
        return self._greedy_action(state, group_candidates=group_candidates)

    def store_transition(self, s, a, r, s_next, done=True):
        self.buffer.push(s, a, r, s_next, done)

    def update(self):
        if len(self.buffer) < self.batch_size:
            return
        batch = self.buffer.sample(self.batch_size)
        s, a, r, s2, d = zip(*batch)
        s = torch.tensor(np.stack(s), dtype=torch.float32, device=self.device)
        r = torch.tensor(r, dtype=torch.float32, device=self.device).unsqueeze(1)
        s2 = torch.tensor(np.stack(s2), dtype=torch.float32, device=self.device)
        d = torch.tensor(d, dtype=torch.float32, device=self.device).unsqueeze(1)

        if self.multi_group:
            a = torch.tensor(np.stack(a), dtype=torch.long, device=self.device).unsqueeze(2)  # [B, G, 1]
            q = self.net(s).view(-1, self.num_groups, self.experts_per_group)
            q_vals = q.gather(2, a).squeeze(2).sum(dim=1, keepdim=True)
            with torch.no_grad():
                q_next = self.target(s2).view(-1, self.num_groups, self.experts_per_group).max(2)[0].sum(dim=1, keepdim=True)
                q_target = r + self.gamma * q_next * (1 - d)
        else:
            a = torch.tensor(a, dtype=torch.long, device=self.device).unsqueeze(1)
            q_vals = self.net(s).gather(1, a)
            with torch.no_grad():
                q_next = self.target(s2).max(1, keepdim=True)[0]
                q_target = r + self.gamma * q_next * (1 - d)

        loss = nn.functional.mse_loss(q_vals, q_target)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
        self.opt.step()

        if self.step_cnt % self.target_update == 0:
            self.target.load_state_dict(self.net.state_dict())