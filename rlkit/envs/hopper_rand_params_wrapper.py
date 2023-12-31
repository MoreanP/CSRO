import numpy as np
from rand_param_envs.hopper_rand_params import HopperRandParamsEnv
import pdb

from . import register_env

def read_log_params(log_file):
    params_dict = {}
    with open(log_file) as f:
        lines = f.readlines()
    cur_key = None
    for line in lines:
        if "'" in line:
            if ")" in line:
                last_entry = line.split(")")[0].split("[")[1].split("]")[0].split(",")
                #print(last_entry)
                last_entry_float = [float(s) for s in last_entry]
                params_dict[cur_key].append(np.array(last_entry_float))
            key = line.split("'")[1]
            #print('key is %s' %key)
            cur_key = key
            params_dict[key] = []
            if "(" in line:
                first_entry = line.split("(")[1].split("[")[2].split("]")[0].split(",")
                #print(first_entry)
                first_entry_float = [float(s) for s in first_entry]
                params_dict[cur_key].append(np.array(first_entry_float))
        else:
            entry = line.split("[")[1].split("]")[0].split(",")
            entry_float = [float(s) for s in entry]
            params_dict[cur_key].append(entry_float)
    for key, value in params_dict.items():
        params_dict[key] = np.array(params_dict[key])
    return params_dict


@register_env('hopper-rand-params')
class HopperRandParamsWrappedEnv(HopperRandParamsEnv):
    def __init__(self, n_tasks=2, randomize_tasks=True, max_episode_steps=200):
        super(HopperRandParamsWrappedEnv, self).__init__()
        self.randomize_tasks = randomize_tasks
        self.tasks = self.sample_tasks(n_tasks)
        self.reset_task(0)
        self._max_episode_steps = max_episode_steps
        self.env_step = 0
    
    def sample_tasks(self, n_tasks):
        """
        Generates randomized parameter sets for the mujoco env

        Args:
            n_tasks (int) : number of different meta-tasks needed

        Returns:
            tasks (list) : an (n_tasks) length list of tasks
        """
        np.random.seed(1234)
        param_sets = []
        if self.randomize_tasks:
            for i in range(n_tasks):
            
                new_params = {}
                # body mass -> one multiplier for all body parts
                if 'body_mass' in self.rand_params:
                    body_mass_multiplyers = np.array(1.5) ** np.random.uniform(-self.log_scale_limit, self.log_scale_limit,  size=self.model.body_mass.shape)
                    new_params['body_mass'] = self.init_params['body_mass'] * body_mass_multiplyers

                # body_inertia
                if 'body_inertia' in self.rand_params:
                    body_inertia_multiplyers = np.array(1.5) ** np.random.uniform(-self.log_scale_limit, self.log_scale_limit,  size=self.model.body_inertia.shape)
                    new_params['body_inertia'] = body_inertia_multiplyers * self.init_params['body_inertia']

                # damping -> different multiplier for different dofs/joints
                if 'dof_damping' in self.rand_params:
                    dof_damping_multipliers = np.array(1.3) ** np.random.uniform(-self.log_scale_limit, self.log_scale_limit, size=self.model.dof_damping.shape)
                    new_params['dof_damping'] = np.multiply(self.init_params['dof_damping'], dof_damping_multipliers)

            	# friction at the body components
                if 'geom_friction' in self.rand_params:
                    dof_damping_multipliers = np.array(1.5) ** np.random.uniform(-self.log_scale_limit, self.log_scale_limit, size=self.model.geom_friction.shape)
                    new_params['geom_friction'] = np.multiply(self.init_params['geom_friction'], dof_damping_multipliers)

                param_sets.append(new_params)
        else:
            for i in range(n_tasks):
                # task_params = read_log_params(f"./data_copy/hopper_randparam_new/goal_idx{i}/log.txt")
                task_params = read_log_params("./data/hopper_rand_params/goal_idx{}/log.txt".format(i))

                param_sets.append(task_params)

        return param_sets
    
    def get_all_task_idx(self):
        return range(len(self.tasks))

    def reset(self):
        self.env_step = 0
        return super().reset()

    def step(self, action):
        obs, reward, done, info = super().step(action)
        self.env_step += 1
        if self.env_step >= self._max_episode_steps:
            done = True
        return obs, reward, done, info

    def reset_task(self, idx):
        self._goal_idx = idx
        self._task = self.tasks[idx]
        self._goal = idx
        self.set_task(self._task)
        self.reset()