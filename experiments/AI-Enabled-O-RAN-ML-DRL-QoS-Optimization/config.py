from dataclasses import dataclass
@dataclass
class Config:
    n_antennas:int=64
    n_users:int=8
    bandwidth_hz:float=20e6
    noise_power_w:float=4e-13
    p_max_w:float=10.0
    pa_efficiency:float=.40
    p_static_w:float=20.0
    p_circuit_w:float=80.0
    min_rate_base_bps:float=1e6
    min_rate_max_bps:float=5e6
    traffic_history:int=8
    seed:int=17
    ppo_episodes:int=300
    steps_per_episode:int=48
    gamma:float=.98
    gae_lambda:float=.95
    clip_epsilon:float=.2
    ppo_epochs:int=6
    minibatch_size:int=128
    hidden_dim:int=128
    learning_rate:float=3e-4
    entropy_coef:float=.005
    value_coef:float=.5
    alpha_energy:float=.55
    beta_throughput:float=.45
    qos_penalty:float=2.5
    smoothness_penalty:float=.02
    @property
    def fixed_power_w(self): return self.p_static_w+self.p_circuit_w
