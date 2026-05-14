import random
import numpy as np

from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

from model import DQN


class Agent:

    def __init__(

        self,

        state_size,

        action_size,

        learning_rate=0.001,

        gamma=0.99,

        epsilon=1.0,

        epsilon_decay=0.995
    ):

        # -----------------------------------
        # Basic Parameters
        # -----------------------------------

        self.state_size = state_size

        self.action_size = action_size

        # -----------------------------------
        # Replay Memory
        # -----------------------------------

        self.memory = deque(
            maxlen=10000
        )

        # -----------------------------------
        # Hyperparameters
        # -----------------------------------

        self.gamma = gamma

        self.epsilon = epsilon

        self.epsilon_min = 0.01

        self.epsilon_decay = epsilon_decay

        self.lr = learning_rate

        self.batch_size = 64

        # -----------------------------------
        # Device
        # -----------------------------------

        self.device = torch.device(
            "cpu"
        )

        # -----------------------------------
        # Main DQN Model
        # -----------------------------------

        self.model = DQN(

            state_size,

            action_size

        ).to(self.device)

        # -----------------------------------
        # Target DQN Model
        # -----------------------------------

        self.target_model = DQN(

            state_size,

            action_size

        ).to(self.device)

        # -----------------------------------
        # Optimizer
        # -----------------------------------

        self.optimizer = optim.Adam(

            self.model.parameters(),

            lr=self.lr
        )

        # -----------------------------------
        # Initialize Target Network
        # -----------------------------------

        self.update_target()

    # -----------------------------------
    # Update Target Network
    # -----------------------------------

    def update_target(self):

        self.target_model.load_state_dict(

            self.model.state_dict()
        )

    # -----------------------------------
    # Store Experience
    # -----------------------------------

    def remember(

        self,

        state,

        action,

        reward,

        next_state,

        done
    ):

        self.memory.append(

            (
                state,

                action,

                reward,

                next_state,

                done
            )
        )

    # -----------------------------------
    # Select Action
    # -----------------------------------

    def act(self, state):

        # Exploration

        if np.random.rand() <= self.epsilon:

            return random.randrange(

                self.action_size
            )

        # Convert State

        state = torch.FloatTensor(

            state

        ).unsqueeze(0)

        # Predict Q Values

        q_values = self.model(state)

        # Exploitation

        return torch.argmax(

            q_values

        ).item()

    # -----------------------------------
    # Train Using Replay Buffer
    # -----------------------------------

    def replay(self):

        # Minimum Samples Check

        if len(self.memory) < self.batch_size:

            return

        # Random Mini Batch

        batch = random.sample(

            self.memory,

            self.batch_size
        )

        # -----------------------------------
        # Train On Batch
        # -----------------------------------

        for (

            state,

            action,

            reward,

            next_state,

            done

        ) in batch:

            # Convert States

            state = torch.FloatTensor(

                state
            )

            next_state = torch.FloatTensor(

                next_state
            )

            # -----------------------------------
            # Compute Target
            # -----------------------------------

            target = reward

            if not done:

                target += (

                    self.gamma

                    * torch.max(

                        self.target_model(
                            next_state
                        )

                    ).item()
                )

            # -----------------------------------
            # Current Prediction
            # -----------------------------------

            target_f = self.model(state)

            target_f[action] = target

            # -----------------------------------
            # Compute Loss
            # -----------------------------------

            loss = nn.MSELoss()(

                self.model(state),

                target_f.detach()
            )

            # -----------------------------------
            # Backpropagation
            # -----------------------------------

            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()

        # -----------------------------------
        # Epsilon Decay
        # -----------------------------------

        if self.epsilon > self.epsilon_min:

            self.epsilon *= self.epsilon_decay