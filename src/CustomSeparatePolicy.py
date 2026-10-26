from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from torch import nn
import torch as th

class ExportableActor(nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.actor_net = policy.mlp_extractor.actor_net
        self.action_net = policy.action_net
        self.actor_dim = policy.actor_dim

    def forward(self, obs):
        obs_actor = obs[..., :self.actor_dim]
        latent_pi = self.actor_net(obs_actor)
        action = self.action_net(latent_pi)
        return action

class SeparateMlpExtractor(nn.Module):
    def __init__(self, obs_dim, actor_dim, critic_dim, net_arch_actor, net_arch_critic, activation_fn):
        super().__init__()
        self.actor_dim = actor_dim
        self.critic_dim = critic_dim

        # Sieć aktora
        actor_layers = []
        last_dim = actor_dim
        for size in net_arch_actor:
            actor_layers.append(nn.Linear(last_dim, size))
            actor_layers.append(activation_fn())
            last_dim = size
        self.actor_net = nn.Sequential(*actor_layers)

        # Sieć krytyka
        critic_layers = []
        last_dim = critic_dim
        for size in net_arch_critic:
            critic_layers.append(nn.Linear(last_dim, size))
            critic_layers.append(activation_fn())
            last_dim = size
        self.critic_net = nn.Sequential(*critic_layers)

        self.latent_dim_pi = net_arch_actor[-1]
        self.latent_dim_vf = net_arch_critic[-1]

    def forward(self, obs):
        #obs = [obs_actor, obs_critic]`
        obs_actor = obs[..., :self.actor_dim]
        obs_critic = obs[..., self.actor_dim:self.actor_dim + self.critic_dim]

        latent_pi = self.actor_net(obs_actor)
        latent_vf = self.critic_net(obs_critic)
        return latent_pi, latent_vf
    
    def forward_actor(self, obs):
        obs_actor = obs[..., :self.actor_dim]
        return self.actor_net(obs_actor)

    def forward_critic(self, obs):
        obs_critic = obs[..., self.actor_dim:self.actor_dim + self.critic_dim]
        return self.critic_net(obs_critic)

class CustomSeparatePolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule,
                 net_arch_actor=[64, 64], net_arch_critic=[64, 64],
                 activation_fn=nn.Tanh, actor_dim=10, critic_dim=6, **kwargs):

        self.actor_dim = actor_dim
        self.critic_dim = critic_dim
        self.observation_space = observation_space
        self.action_space = action_space
        self.lr_schedule = lr_schedule
        self.net_arch_actor = net_arch_actor
        self.net_arch_critic = net_arch_critic
        self.activation_fn = activation_fn

        try:
            assert observation_space.shape[0] == actor_dim + critic_dim
        except AssertionError:
            raise ValueError(f"Invalid observation space shape: {observation_space.shape[0]}, "
                             f"expected: {actor_dim + critic_dim}")

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=[],
            activation_fn=activation_fn,
            **kwargs,
        )

        # Tworzymy własny extractor, który podzieli obserwację

        # Główne głowy:
        latent_dim_pi = net_arch_actor[-1]
        latent_dim_vf = net_arch_critic[-1]

        self.action_net = nn.Linear(latent_dim_pi, self.action_space.shape[0])
        self.value_net = nn.Linear(latent_dim_vf, 1)

        self._build(lr_schedule)

    def _build_mlp_extractor(self):
        self.mlp_extractor = SeparateMlpExtractor(
            obs_dim=self.observation_space.shape[0],
            actor_dim=self.actor_dim,
            critic_dim=self.critic_dim,
            net_arch_actor=self.net_arch_actor,
            net_arch_critic=self.net_arch_critic,
            activation_fn=self.activation_fn
        )



class SeparateLSTMExtractor(nn.Module):
    def __init__(self, obs_dim, actor_dim, critic_dim, lstm_hidden_actor, lstm_hidden_critic, activation_fn):
        super().__init__()
        self.actor_dim = actor_dim
        self.critic_dim = critic_dim
        self.lstm_hidden_actor = lstm_hidden_actor
        self.lstm_hidden_critic = lstm_hidden_critic

        # LSTM dla aktora
        self.actor_lstm = nn.LSTM(input_size=actor_dim, hidden_size=lstm_hidden_actor, batch_first=True)
        self.actor_activation = activation_fn()
        self.actor_net = nn.Linear(lstm_hidden_actor, lstm_hidden_actor//2)

        # LSTM dla krytyka
        self.critic_lstm = nn.LSTM(input_size=critic_dim, hidden_size=lstm_hidden_critic, batch_first=True)
        self.critic_activation = activation_fn()
        self.critic_net = nn.Linear(lstm_hidden_critic, lstm_hidden_critic//2)

        self.latent_dim_pi = lstm_hidden_actor//2
        self.latent_dim_vf = lstm_hidden_critic//2

    def forward(self, obs):
        obs_actor = obs[..., :self.actor_dim]
        obs_critic = obs[..., self.actor_dim:self.actor_dim + self.critic_dim]

        _, (h_actor, _) = self.actor_lstm(obs_actor)
        _, (h_critic, _) = self.critic_lstm(obs_critic)

        latent_pi = self.actor_activation(h_actor[-1])  # [batch, hidden]
        latent_vf = self.critic_activation(h_critic[-1])  # [batch, hidden]

        latent_pi = self.actor_net(latent_pi)
        latent_vf = self.critic_net(latent_vf)

        return latent_pi, latent_vf

    def forward_actor(self, obs):
        obs_actor = obs[..., :self.actor_dim]
        _, (h_actor, _) = self.actor_lstm(obs_actor)
        return self.actor_activation(self.actor_net(h_actor[-1])) 

    def forward_critic(self, obs):
        obs_critic = obs[..., self.actor_dim:self.actor_dim + self.critic_dim]
        _, (h_critic, _) = self.critic_lstm(obs_critic)
        return self.critic_activation(self.critic_net(h_critic[-1]))


class CustomSeparateLSTMPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule,
                 lstm_hidden_actor=64, lstm_hidden_critic=64,
                 activation_fn=nn.ReLU, actor_dim=10, critic_dim=6, **kwargs):

        self.actor_dim = actor_dim
        self.critic_dim = critic_dim
        self.observation_space = observation_space
        self.action_space = action_space
        self.lr_schedule = lr_schedule
        self.lstm_hidden_actor = lstm_hidden_actor
        self.lstm_hidden_critic = lstm_hidden_critic
        self.activation_fn = activation_fn

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=[],
            activation_fn=activation_fn,
            **kwargs,
        )
        print(self.action_space.shape[0])
        self.action_net = nn.Linear(lstm_hidden_actor//2, self.action_space.shape[0])
        self.value_net = nn.Linear(lstm_hidden_critic//2, 1)

        self._build(lr_schedule)

    def _build_mlp_extractor(self):
        self.mlp_extractor = SeparateLSTMExtractor(
            obs_dim=self.observation_space.shape[1],  # seq_len expected in shape
            actor_dim=self.actor_dim,
            critic_dim=self.critic_dim,
            lstm_hidden_actor=self.lstm_hidden_actor,
            lstm_hidden_critic=self.lstm_hidden_critic,
            activation_fn=self.activation_fn
        )

class NoopExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=observation_space.shape[-1])
        
    def forward(self, observations):
        return observations