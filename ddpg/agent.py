import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from dotenv import load_dotenv
from pathlib import Path
from ddpg.memory import ReplayBuffer
from ddpg.networks import ActorNetwork, CriticNetwork
from ddpg.random_process import OrnsteinUhlenbeckProcess

load_dotenv()
home = os.getenv("PYTHONPATH")
if not home:
    raise EnvironmentError("PYTHONPATH not set in environment variables.")

class Agent():

    def __init__(
            self,
            alpha,
            beta,
            input_dims,
            tau,
            env,
            gamma=0.99,
            n_actions=2,
            max_size=1000000,
            layer1_size=400,
            layer2_size=300,
            batch_size=64,
            chkpt_dir=str(Path(home) / "McHorcrux" / "checkpoints" / "ddpg")
    ):
        self.input_dims = input_dims  # for nets
        input_shape = (input_dims, )  # tuple for memory

        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_shape, n_actions)
        self.batch_size = batch_size

        self.actor = ActorNetwork(alpha,
                                  input_dims,
                                  layer1_size,
                                  layer2_size,
                                  chkpt_dir,
                                  n_actions=n_actions,
                                  name="Actor")
        self.target_actor = ActorNetwork(alpha,
                                         input_dims,
                                         layer1_size,
                                         layer2_size,
                                         chkpt_dir,
                                         n_actions=n_actions,
                                         name="TargetActor")

        self.critic = CriticNetwork(beta,
                                    input_dims,
                                    layer1_size,
                                    layer2_size,
                                    chkpt_dir,
                                    n_actions=n_actions,
                                    name="Critic")
        self.target_critic = CriticNetwork(beta,
                                           input_dims,
                                           layer1_size,
                                           layer2_size,
                                           chkpt_dir,
                                           n_actions=n_actions,
                                           name="TargetCritic")

        self.noise = OrnsteinUhlenbeckProcess(theta=0.15,
                                              mu=np.zeros(2),
                                              sigma=0.2,
                                              dt=1e-2,
                                              x0=None,
                                              size=2,
                                              sigma_min=0.05,
                                              n_steps_annealing=10000)

        self.update_network_parameters(tau=1)

        self.low = np.array([env.v_min, env.r_min], dtype=np.float32)
        self.high = np.array([env.v_max, env.r_max], dtype=np.float32)

    def choose_action(self, observation):
        self.actor.eval()
        observation = T.tensor(observation,
                               dtype=T.float).to(self.actor.device)
        mu = self.actor(observation).to(self.actor.device)
        mu_prime = T.clamp(
            mu +
            T.tensor(self.noise.sample(), dtype=T.float).to(self.actor.device),
            -1.0, 1.0)
        # print(mu_prime)
        self.actor.train()
        return mu_prime.cpu().detach().numpy()

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def learn(self):
        if self.memory.mem_count < self.batch_size:
            return

        state, new_state, action, reward, done = self.memory.sample_buffer(
            self.batch_size)

        state = T.tensor(state, dtype=T.float).to(self.critic.device)
        action = T.tensor(action, dtype=T.float).to(self.critic.device)
        reward = T.tensor(reward,
                          dtype=T.float).to(self.critic.device).view(-1, 1)
        new_state = T.tensor(new_state, dtype=T.float).to(self.critic.device)
        done = T.tensor(done, dtype=T.float).to(self.critic.device).view(-1, 1)

        self.target_actor.eval()
        self.target_critic.eval()
        self.critic.train()
        self.actor.train()

        with T.no_grad():
            target_actions = self.target_actor(new_state)
            critic_value_ = self.target_critic(new_state, target_actions)
        critic_value = self.critic(state, action)

        target = []

        target = reward + self.gamma * critic_value_ * (1 - done)
        target = T.tensor(target).to(self.critic.device)
        target = target.view(self.batch_size, 1)

        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(target, critic_value)
        critic_loss.backward()
        self.critic.optimizer.step()
        self.critic.eval()

        self.actor.optimizer.zero_grad()
        mu = self.actor(state)
        actor_loss = -self.critic(state, mu)
        actor_loss = T.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau == None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        actor_state_dict = dict(actor_params)
        critic_state_dict = dict(critic_params)
        target_actor_dict = dict(target_actor_params)
        target_critic_dict = dict(target_critic_params)

        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + \
                (1 - tau)*target_critic_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)

        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                (1 - tau)*target_actor_dict[name].clone()

        self.target_actor.load_state_dict(actor_state_dict)

    def save_models(self):
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()
