"""Generate configurations only. Never starts training or changes source data."""
import argparse,json,math,pathlib,copy
p=argparse.ArgumentParser()
p.add_argument('--dataset-root',type=pathlib.Path,required=True,help='Binary dataset with training-only normalization stats already prepared')
p.add_argument('--reference',type=pathlib.Path,required=True)
p.add_argument('--out',type=pathlib.Path,required=True)
p.add_argument('--epochs',type=int,default=100)
a=p.parse_args()
assert not a.out.exists(),'Use a new output directory; existing configs are preserved'
c=json.loads(a.reference.read_text());root=a.dataset_root.resolve()
assert (root/'meta/info.json').is_file()
info=json.loads((root/'meta/info.json').read_text())
assert info['features']['action']['shape']==[11],'Expected binary action11 dataset'
assert info['features']['observation.state']['shape']==[18]
episodes=c['dataset']['episodes'];assert episodes and len(episodes)==len(set(episodes))
from lerobot.datasets.lerobot_dataset import LeRobotDataset
d=LeRobotDataset(c['dataset']['repo_id'],root=root,episodes=episodes,video_backend='pyav')
n=len(d);assert n>0
a.out.mkdir(parents=True)
for bs in (8,16,24,32):
 for workers in (4,8,12):
  x=copy.deepcopy(c);x['dataset']['root']=str(root);x['batch_size']=bs;x['num_workers']=workers
  x['steps']=math.ceil(n/bs)*a.epochs;x['prefetch_factor']=2;x['persistent_workers']=True
  x['resume']=False;x['checkpoint_path']=None;x['policy']['pretrained_path']=None
  x['output_dir']=str(a.out.parent/f'run_b{bs}_w{workers}');x['job_name']=f'act_binary_b{bs}_w{workers}_{a.epochs}ep'
  x['save_freq']=math.ceil(n/bs)*5;x['log_freq']=100
  (a.out/f'b{bs}_w{workers}.json').write_text(json.dumps(x,indent=2))
report={'status':'CONFIGS ONLY; no training started','training_frames':n,'train_episodes':episodes,'epochs':a.epochs,'source':str(a.reference),'dataset_root':str(root),'checks_still_required':['verify held-out split has no overlap','verify normalization stats were computed on training split only','freeze label/pose and initialization trim rules','check loader and benchmark before selecting a configuration','changing batch size changes optimizer update count; document comparison protocol']}
(a.out/'preparation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
