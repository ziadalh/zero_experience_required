from typing import Optional

import habitat
from habitat import Config, Dataset
from habitat_baselines.common.baseline_registry import baseline_registry


@baseline_registry.register_env(name="NavRLEnvX")
class NavRLEnvX(habitat.RLEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        self._rl_config = config.RL
        self._core_env_config = config.TASK_CONFIG
        if len(self._rl_config.REWARD_MEASURE) > 0:
            self._reward_measure_names = [self._rl_config.REWARD_MEASURE]
            self._reward_scales = [1.0]
        else:
            self._reward_measure_names = self._rl_config.REWARD_MEASURES
            self._reward_scales = self._rl_config.REWARD_SCALES

        self._success_measure_name = self._rl_config.SUCCESS_MEASURE
        habitat.logger.info('NavRLEnvX: '
                            f'Reward Measures={self._reward_measure_names}, '
                            f'Reward Scales={self._reward_scales}, '
                            f'Success Measure={self._success_measure_name}')
        self._previous_measure = None
        self._previous_action = None
        super().__init__(self._core_env_config, dataset)

    def reset(self):
        self._previous_action = None
        observations = super().reset()
        self._previous_measure = self._get_reward_measure()
        return observations

    def _get_reward_measure(self):
        current_measure = 0.0
        for reward_measure_name, reward_scale in zip(
                self._reward_measure_names, self._reward_scales
        ):
            if "." in reward_measure_name:
                reward_measure_name = reward_measure_name.split('.')
                measure = self._env.get_metrics()[
                    reward_measure_name[0]
                ][reward_measure_name[1]]
            else:
                measure = self._env.get_metrics()[reward_measure_name]
            current_measure += measure * reward_scale
        return current_measure

    def step(self, *args, **kwargs):
        self._previous_action = kwargs["action"]
        return super().step(*args, **kwargs)

    def get_reward_range(self):
        return (
            self._rl_config.SLACK_REWARD - 1.0,
            self._rl_config.SUCCESS_REWARD + 1.0,
        )

    def get_reward(self, observations):
        reward = self._rl_config.SLACK_REWARD

        current_measure = self._get_reward_measure()

        reward += self._previous_measure - current_measure
        self._previous_measure = current_measure

        if self._episode_success():
            reward += self._episode_success() * self._rl_config.SUCCESS_REWARD

        return reward

    def _episode_success(self):
        return self._env.get_metrics()[self._success_measure_name]

    def get_done(self, observations):
        done = False
        if self._env.episode_over or self._episode_success():
            done = True
        return done

    def get_info(self, observations):
        return self.habitat_env.get_metrics()
