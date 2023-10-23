import numpy as np

from rlkit.envs.mujoco.ant_multitask_base import MultitaskAntEnv
from rlkit.envs import register_env

# Copy task structure from https://github.com/jonasrothfuss/ProMP/blob/master/meta_policy_search/envs/mujoco_envs/ant_rand_goal.py
@register_env('ant-goal')
class AntGoalEnv(MultitaskAntEnv):
    def __init__(self, task={}, n_tasks=2, max_episode_steps=200, randomize_tasks=False, **kwargs):
        self._max_episode_steps = max_episode_steps 
        self.randomize_tasks = randomize_tasks
        super(AntGoalEnv, self).__init__(task, n_tasks, **kwargs)

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        xposafter = np.array(self.get_body_com("torso"))

        goal_reward = -np.sum(np.abs(xposafter[:2] - self._goal)) # make it happy, not suicidal

        ctrl_cost = .1 * np.square(action).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 0.0
        reward = goal_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        done = False
        ob = self._get_obs()
        self._step += 1
        if self._step >= self._max_episode_steps:
            done = True
        return ob, reward, done, dict(
            goal_forward=goal_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward,
        )

    def reset(self):
        self._step = 0
        return super().reset()

    def sample_tasks(self, num_tasks):
        if self.randomize_tasks:
            np.random.seed(1334)
            a = np.random.random(num_tasks) * 2 * np.pi
            # r = 3 * np.random.random(num_tasks) ** 0.5
            r = 2 * np.ones(num_tasks)
        else:
            a = np.linspace(0, 2*np.pi, num_tasks+1)[:-1]
            r = 2 * np.ones(num_tasks)
        goals = np.stack((r * np.cos(a), r * np.sin(a)), axis=-1)
        tasks = [{'goal': goal} for goal in goals]
        return tasks

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
        ])
