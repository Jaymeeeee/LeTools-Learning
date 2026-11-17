"""Boundary checks; does not publish ROS commands or move the robot."""
import sys,unittest,json,tempfile,importlib.util
from pathlib import Path
from collections import deque
from types import SimpleNamespace
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'task3_binary'))
from action_adapter import ActionAdapter
class ContractTests(unittest.TestCase):
 def test_restore_batch_and_threshold(self):
  adapter=ActionAdapter(ROOT/'task3_binary/poses_scaled157.json')
  a=np.arange(33,dtype=np.float32).reshape(3,11);a[:,7]=[.49,.5,1.]
  result=adapter.restore(a)
  np.testing.assert_equal(result[:,:7],a[:,:7]);np.testing.assert_equal(result[:,18:],a[:,8:])
  np.testing.assert_allclose(result[:,7:18],adapter.poses[[0,1,1]])
  np.testing.assert_allclose(adapter.restore_tensor(torch.tensor(a)).numpy(),result)
  self.assertEqual(result.shape,(3,21))
  with self.assertRaises(ValueError):adapter.restore(np.full(11,np.nan))
 def test_scaled_poses(self):
  original=ActionAdapter(ROOT/'task3_binary/poses.json')
  scaled=ActionAdapter(ROOT/'task3_binary/poses_scaled157.json')
  np.testing.assert_allclose(scaled.poses,original.poses*1.57)
 def test_feedback_wrist_and_queue(self):
  from kuavo_deploy.kuavo_env.KuavoBaseRosEnv import KuavoBaseRosEnv
  spec=importlib.util.spec_from_file_location('handoff_overrides',ROOT/'kuavo_deploy/utils/inference_overrides.py')
  mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
  class Policy:
   config=SimpleNamespace(type='act',chunk_size=100,input_features={'observation.state':SimpleNamespace(shape=(18,))},output_features={'action':SimpleNamespace(shape=(11,))})
   def reset(self):self._action_queue=deque(maxlen=self.config.n_action_steps)
  target=ROOT/'deployment_tuning.json'
  self.assertFalse(target.exists(),'Test needs an unused checkout; do not overwrite deployment settings')
  old=KuavoBaseRosEnv.step;captured=[]
  try:
   target.write_text((ROOT/'task3_binary/deployment_tuning.example.json').read_text())
   KuavoBaseRosEnv.step=lambda self,a:captured.append(a.copy())
   p,pre,_,_=mod.apply(Policy(),lambda x:x,None,None)
   self.assertEqual(p._action_queue.maxlen,50)
   state=torch.arange(18,dtype=torch.float32).reshape(1,18)
   result=pre({'observation.state':state})['observation.state']
   torch.testing.assert_close(result[:,:7],state[:,:7]);torch.testing.assert_close(result[:,7:],state[:,7:]/1.57)
   action=np.zeros(21,dtype=np.float32);KuavoBaseRosEnv.step(None,action)
   expected=action.copy();expected[5]=.2
   np.testing.assert_allclose(captured[-1],expected);self.assertFalse(action.any())
  finally:
   KuavoBaseRosEnv.step=old
   if hasattr(KuavoBaseRosEnv,'_official_wrist_offset_applied'):delattr(KuavoBaseRosEnv,'_official_wrist_offset_applied')
   if target.exists():target.unlink()
if __name__=='__main__':unittest.main()
