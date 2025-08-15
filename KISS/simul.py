from numpy.random import seed,standard_normal
from AIMM_simulator import *

class MyScenario(Scenario):
  def loop(self,interval=10):
    while True:
      for ue in self.sim.UEs: ue.xyz[:2]+=20*standard_normal(2)
      yield self.sim.wait(interval)

class MyLogger(Logger):
  # throughput of UE[0], UE[0] position, serving cell index
  def loop(self):
    while True:
      sc=self.sim.UEs[0].serving_cell.i
      tp=self.sim.cells[sc].get_UE_throughput(0)
      xy0=np_array_to_str(self.sim.UEs[0].xyz[:2])
      self.f.write(f'{self.sim.env.now:.2f}\t{tp:.4f}\t{xy0}\t{sc}\n')
      yield self.sim.wait(self.logging_interval)

def hetnet(n_subbands=1):
  sim=Sim()
  for i in range(9): # macros
    sim.make_cell(xyz=(500.0*(i//3),500.0*(i%3),20.0),power_dBm=30.0,n_subbands=n_subbands)
  for i in range(10): # small cells
    sim.make_cell(power_dBm=10.0,n_subbands=n_subbands)
  for i in range(20):
    sim.make_UE().attach_to_strongest_cell_simple_pathloss_model()
  sim.UEs[0].set_xyz([500.0,500.0,2.0])
  for UE in sim.UEs: UE.attach_to_strongest_cell_simple_pathloss_model()
  sim.add_logger(MyLogger(sim,logging_interval=1.0))
  sim.add_scenario(MyScenario(sim))
  sim.add_MME(MME(sim,verbosity=0,interval=50.0))
  sim.run(until=2000)

seed(3)
hetnet()