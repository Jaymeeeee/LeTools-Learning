"""Keep physical SG100 transport paired with the training coordinate scale."""
import json
from pathlib import Path

def apply(policy, pre, post, path):
    settings_path=Path(__file__).resolve().parents[2]/'deployment_tuning.json'
    if not settings_path.is_file():
        return policy,pre,post,path
    settings=json.loads(settings_path.read_text())
    steps=int(settings['n_action_steps']);scale=float(settings['hand_scale'])
    assert policy.config.type=='act' and 1<=steps<=policy.config.chunk_size
    assert tuple(policy.config.input_features['observation.state'].shape)==(18,)
    assert tuple(policy.config.output_features['action'].shape)==(11,)
    from kuavo_deploy.kuavo_env.KuavoBaseRosEnv import KuavoBaseRosEnv
    offset=float(settings['right_wrist_command_offset_rad'])
    if not getattr(KuavoBaseRosEnv, '_official_wrist_offset_applied', False):
        original_step=KuavoBaseRosEnv.step
        def compensated_step(self, action):
            action=action.copy()
            action[5]+=offset
            return original_step(self,action)
        KuavoBaseRosEnv.step=compensated_step
        KuavoBaseRosEnv._official_wrist_offset_applied=True
    policy.config.n_action_steps=steps
    policy.reset()
    assert policy._action_queue.maxlen==steps
    def processor(observation):
        observation=dict(observation)
        state=observation['observation.state'].clone()
        state[...,7:18]/=scale
        observation['observation.state']=state
        return pre(observation)
    print('DEPLOYMENT_TUNING',settings,'feedback_divisor=',scale,'queue_maxlen=',policy._action_queue.maxlen,flush=True)
    return policy,processor,post,path
