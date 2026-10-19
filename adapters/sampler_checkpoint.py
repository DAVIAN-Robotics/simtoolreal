"""Include the new SGS history in the existing checkpoint save/load calls."""
from isaaclab_rl.rl_games import RlGamesGpuEnv
from rl_games.common import vecenv


def register_sampler_checkpoint(env):
    class SamplerCheckpointEnv(RlGamesGpuEnv):
        def get_env_state(self):
            return dict(base=super().get_env_state(), sgs=env.sampler.state_dict())

        def set_env_state(self, state):
            if state is not None and 'sgs' in state:
                env.sampler.load_state_dict(state['sgs'])
                super().set_env_state(state['base'])
            else:
                super().set_env_state(state)

    vecenv.register('IsaacRlgWrapper',
                   lambda config_name, num_actors, **kw: SamplerCheckpointEnv(config_name, num_actors, **kw))
