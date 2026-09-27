"""Six-field attack interface; cached noise and unchanged SecureLink physical step."""
import math
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from securelink.environment import SecureLinkEnv

KEYS = ("slot","remaining_slots","willie_remaining_j","willie_peak_w","alice_x_est_m","alice_y_est_m")
DEFENDERS = (11001,11002,11003)
PURPOSES = {"precheck":"p1_w1_interface_check_v1","training":"p1_w1_train_v1",
            "development":"p1_w1_development_v1","feasibility":"p1_w1_feasibility_v1"}
RANGES = {"precheck":range(39001,39006),"training":range(33001,34001),
          "development":range(34001,34031),"feasibility":range(35001,35101)}


def encode(observation):
    if set(observation) != set(KEYS):
        raise ValueError("Exactly the six registered fields required")
    values = np.array([observation[key] for key in KEYS],dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite allowed observation")
    return (values/np.array([100,100,6,1,1000,1000])).astype(np.float32)


def power_request(action, peak):
    values = np.asarray(action)
    if values.shape != (1,) or not np.isfinite(values).all() or not math.isfinite(peak) or peak <= 0:
        raise ValueError("One finite scalar action and positive peak required")
    clipped = float(np.clip(values[0],-1,1))
    power = .001*math.expm1((clipped+1)/2*math.log1p(peak/.001))
    return power,clipped


def inverse_power(power, peak=1.):
    if not 0 <= power <= peak:
        raise ValueError("Requested power outside peak")
    return np.array([2*math.log1p(power/.001)/math.log1p(peak/.001)-1],dtype=np.float64)


class PendingPower:
    """Callback owns only the precommitted power; no environment/model/log object."""
    def __init__(self):
        self.value = None
        self.calls = 0

    def __call__(self, observation):
        if set(observation) != set(KEYS) or self.value is None:
            raise RuntimeError("Missing committed attack decision or invalid callback observation")
        self.calls += 1
        return self.value


class CachedAttackEnv(SecureLinkEnv):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.cached = None
        self.noise_draw_calls = 0
        self.attack_policy = PendingPower()

    def reset(self,**kwargs):
        self.cached = None
        self.noise_draw_calls = 0
        self.attack_policy.calls = 0
        self.attack_policy.value = None
        return super().reset(**kwargs)

    def prepare_attack(self):
        if self.episode_summary is not None:
            raise RuntimeError("No attack observation after termination")
        if self.cached is None:
            self.cached = SecureLinkEnv._attack_observation(self)
            self.noise_draw_calls += 1
        return dict(self.cached)

    def _attack_observation(self):
        if self.cached is None:
            raise RuntimeError("Attack decision must precede physical callback")
        return dict(self.cached)


class W1Env(gym.Env):
    """PPO sees only this Env's six-vector and privileged negative-rate reward."""
    def __init__(self, config, defenders, stage, ledger, run_seed=0, defender_seed=11001):
        if stage not in PURPOSES or defender_seed not in DEFENDERS:
            raise ValueError("Unknown stage or frozen defender")
        self.config,self.defenders,self.stage,self.ledger = config,defenders,stage,ledger
        self.run_seed,self.defender_seed = run_seed,defender_seed
        if stage == "training" and run_seed not in (21001,21002,21003):
            raise ValueError("Unregistered attack run")
        self.episode_index = -1
        self.base = None
        self.defender_observation = None
        self.attack_observation = None
        self.last_transition = None
        self.completed = []
        self.decision_events = []
        self.observation_space = spaces.Box(-np.inf,np.inf,shape=(6,),dtype=np.float32)
        self.action_space = spaces.Box(-1.,1.,shape=(1,),dtype=np.float32)

    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed)
        if self.stage == "training":
            if self.ledger.calls >= self.ledger.cap:
                # SB3's automatic terminal reset must not start an extra budget-end episode.
                return np.zeros(6,dtype=np.float32),{}
            self.episode_index += 1
            seed = 33001+self.episode_index
            index = (self.episode_index+(self.run_seed-21001))%3
            self.defender_seed = DEFENDERS[index]
        if seed not in RANGES[self.stage]:
            raise ValueError("Scenario outside this stage; no final-test option")
        self.ledger.begin_episode(stage=self.stage,scenario=seed,attack_run=self.run_seed,defender=self.defender_seed)
        if self.base is not None:
            self.base.close()
        self.base = CachedAttackEnv(self.config,method="B10",attack_rule="W1",
            stream_context={"purpose":PURPOSES[self.stage],"run_seed":self.run_seed if self.stage=="training" else 0,"worker_id":0})
        self.defender_observation,_ = self.base.reset(seed=seed)
        self.attack_observation = self.base.prepare_attack()
        return encode(self.attack_observation),{}

    def step(self, action):
        if self.base is None or self.base.episode_summary is not None:
            raise RuntimeError("Step after termination or before reset")
        request,clipped = power_request(action,float(self.attack_observation["willie_peak_w"]))
        self.base.attack_policy.value = request
        # Attack is irrevocably formed before access to the defender's current prediction.
        self.decision_events = ["attack_committed"]
        defender_action = self.defenders[self.defender_seed].predict(self.defender_observation,deterministic=True)[0]
        self.decision_events.append("defender_predicted")
        before = len(self.base.step_records)
        observation_snapshot = dict(self.attack_observation)
        self.ledger.reserve(stage=self.stage,scenario=self.base.seed_value,slot=self.base.slot,defender=self.defender_seed)
        self.defender_observation,reward,terminated,truncated,_ = self.base.step(defender_action)
        self.decision_events.append("physical_step")
        appended = self.base.step_records[before:]
        real = [row for row in appended if not row["padded_failure"]]
        self.ledger.commit(len(real),sum(row["padded_failure"] for row in appended))
        for row in appended:
            row.update(attack_observation=observation_snapshot if not row["padded_failure"] else None,
                       attack_input=encode(observation_snapshot).tolist() if not row["padded_failure"] else None,
                       attack_clipped_action=clipped if not row["padded_failure"] else None,
                       willie_requested_w=request if not row["padded_failure"] else None,
                       attack_run_seed=self.run_seed,defender_seed=self.defender_seed,
                       attack_reward=-float(row["secrecy_rate_bpshz"]),
                       reward_source="simulator_privileged_reward")
        self.last_transition = {"rows":appended,"real":len(real),"padded":len(appended)-len(real),
            "reward":-float(reward),"terminated":bool(terminated),"context":observation_snapshot,
            "defender_seed":self.defender_seed,"scenario":self.base.seed_value}
        if self.base.executed_hard_violations:
            raise RuntimeError("Executed hard constraint violation")
        if self.base.willie_remaining_j < -1e-8 or request > 1+1e-8:
            raise RuntimeError("Attack budget/peak check failed")
        self.base.cached = None
        if terminated or truncated:
            self.completed.append({"episode":dict(self.base.episode_summary),
                                   "steps":list(self.base.step_records),
                                   "defender_seed":self.defender_seed,"attack_run_seed":self.run_seed})
            next_observation = np.zeros(6,dtype=np.float32)
        else:
            self.attack_observation = self.base.prepare_attack()
            next_observation = encode(self.attack_observation)
        # No hidden base info, summary, log references or defender state reaches the policy interface.
        return next_observation,-float(reward),terminated,truncated,{}

    def partial(self):
        if self.base is None or self.base.episode_summary is not None:
            return None
        return {"partial":True,"scenario":self.base.seed_value,"defender_seed":self.defender_seed,
                "attack_run_seed":self.run_seed,"steps":[dict(row) for row in self.base.step_records],
                "no_extra_step_to_complete":True}

    def close(self):
        if self.base is not None:
            self.base.close()
