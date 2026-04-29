import gym
from gym.wrappers.time_limit import TimeLimit

def _patched_step(self, action):
    step_out = self.env.step(action)
    if len(step_out) == 5:
        observation, reward, terminated, truncated, info = step_out
        done = terminated or truncated
    else:
        observation, reward, done, info = step_out
        terminated, truncated = bool(done), False

    self._elapsed_steps += 1
    if self._elapsed_steps >= self._max_episode_steps:
        truncated = True
        done = True
        info = dict(info)
        info["TimeLimit.truncated"] = not terminated

    return observation, reward, done, info

def _patched_reset(self, **kwargs):
    self._elapsed_steps = 0
    reset_out = self.env.reset(**kwargs)
    if isinstance(reset_out, tuple) and len(reset_out) == 2:
        obs, _info = reset_out
        return obs
    return reset_out

TimeLimit.step = _patched_step
TimeLimit.reset = _patched_reset
