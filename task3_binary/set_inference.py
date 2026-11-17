"""Change next-run deployment parameters only; stop inference before using."""
import argparse,json
from pathlib import Path
import yaml
p=argparse.ArgumentParser()
p.add_argument('--steps',type=int,required=True)
p.add_argument('--hz',type=int,default=10)
a=p.parse_args()
if not 1<=a.steps<=100 or not 1<=a.hz<=30:p.error('steps must be 1..100; hz 1..30')
root=Path(__file__).resolve().parents[1]
target=root/'deployment_tuning.json';cfg_path=root/'configs/deploy/deploy.yaml'
settings=json.loads(target.read_text());cfg=yaml.safe_load(cfg_path.read_text())
settings['n_action_steps']=a.steps;settings['ros_rate']=a.hz
cfg['env']['ros_rate']=a.hz
target.write_text(json.dumps(settings,indent=2))
cfg_path.write_text(yaml.safe_dump(cfg,sort_keys=False))
print(f'Next run: {a.hz} actions/sec; execute {a.steps} steps; replan about every {a.steps/a.hz:.2f}s. Hand scale remains {settings["hand_scale"]}.')
if a.hz!=10:print('Training data is 10Hz. No trajectory resampling is applied; motion timing changes.')
