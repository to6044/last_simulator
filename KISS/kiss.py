# KISS (Keep It Simple Silly!) v3

import argparse
import dataclasses
import json
import os
import datetime
from types import NoneType
import _bisect
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from math import comb
from itertools import combinations
from pathlib import Path
import re

import numpy as np
import pandas as pd
from AIMM_simulator import *
from hexalattice.hexalattice import *
import utils_kiss


# Fix custom imports
bisect_left = _bisect.bisect_left
get_timestamp = utils_kiss.get_timestamp


def create_logfile_path(config_dict, debug_logger: bool = False):
    """Create a path for the log file based on the config parameters"""
    project_root_dir = config_dict['project_root_dir']
    script_name = config_dict['script_name']
    experiment_description = config_dict['experiment_description']
    
    # 실행 타임스탬프를 config에서 가져오기
    execution_timestamp = config_dict.get('execution_timestamp', None)
    if execution_timestamp:
        # run_kiss.py에서 전달받은 타임스탬프 사용 (YYYYMMDD_HHMMSS 형식)
        parts = execution_timestamp.split('_')
        if len(parts) == 2:
            # YYYYMMDD_HHMMSS 형식
            date_str = parts[0]  # YYYYMMDD
            time_str = parts[1]  # HHMMSS
            
            # 날짜를 YYYY_MM_DD 형식으로 변환
            if len(date_str) == 8:  # YYYYMMDD 
                date_now = f"{date_str[:4]}_{date_str[4:6]}_{date_str[6:8]}"
            else:
                date_now = date_str  # 이미 올바른 형식인 경우
                
            time_now = time_str
        else:
            # 예상치 못한 형식인 경우 기본값 사용
            date_now = get_timestamp(date_only=True)
            time_now = get_timestamp(time_only=True)
    else:
        # 기존 방식 (개별 실행 시)
        date_now = get_timestamp(date_only=True)
        time_now = get_timestamp(time_only=True)
    
    # 로그 파일명 생성 (시드별 폴더명으로도 사용)
    if debug_logger:
        folder_name = "_".join([
            time_now,
            "debug",
            script_name
        ])
    else:
        # 실험 설명에서 약어 생성
        if experiment_description.startswith("test_"):
            acronym = "test_" + ''.join(config_dict['experiment_description'])
        else:
            acronym = ''.join(word[0] for word in config_dict['experiment_description'].split('_'))
        
        if config_dict['scenario_profile'] != "switch_n_cells_off":
            # 시드와 파워를 포함한 폴더명
            # 파워 값에서 소수점을 언더스코어로 변경
            power_str = str(config_dict['variable_cell_power_dBm']).replace('.', '_')
            folder_name = "_".join([
                acronym,
                "s" + str(config_dict['seed']),
                "p" + power_str
            ])
        else:
            if config_dict['scenario_n_cells'] < 10:
                n_cells = "0" + str(config_dict['scenario_n_cells'])
            else:
                n_cells = str(config_dict['scenario_n_cells'])
            acronym = acronym[:1] + n_cells + acronym[2:]
            power_str = str(config_dict['variable_cell_power_dBm']).replace('.', '_')
            folder_name = "_".join([
                acronym,
                "s" + str(config_dict['seed']),
                "p" + power_str,
                time_now,
            ])
    
    # 기본 디렉토리 경로 구성
    if config_dict["experiment_description"].startswith("test_"):
        base_dir = os.path.join(project_root_dir, "_test", "data", "output", 
                               experiment_description, date_now, time_now)
    else:
        base_dir = os.path.join(project_root_dir, "data", "output", 
                               experiment_description, date_now, time_now)
    
    # 시드별 폴더 경로 추가
    seed_dir = os.path.join(base_dir, folder_name)
    
    # 특수 문자 제거
    seed_dir = seed_dir.replace(".", "_").replace("-", "_").replace(" ", "_").replace(":", "_")
    
    # 디렉토리 생성
    if not os.path.exists(seed_dir):
        os.makedirs(seed_dir, exist_ok=True)
    
    # 파일명은 폴더명과 동일하게
    logfile_name = folder_name
    
    # 전체 파일 경로 생성
    logfile_path = os.path.join(seed_dir, logfile_name)
    logfile_path = logfile_path.replace("-", "_").replace(" ", "_").replace(":", "_").replace(".", "_")
    
    return logfile_path



@dataclass(frozen=True)
class SmallCellParameters:
    """ Object for setting small cell base station parameters."""
    p_max_watts: float = 0.25
    p_static_watts: float = 6.8
    eta_pa: float = 0.067
    power_rf_watts: float = 1.0
    power_baseband_watts: float = 3.0
    loss_feed: float = 0.00
    loss_dc: float = 0.09
    loss_cool: float = 0.00
    loss_mains: float = 0.11
    delta_p: float = 4.0  # 10.1109/LCOMM.2013.091213.131042
    sectors: int = 1
    antennas: int = 2


@dataclass(frozen=True)
class MacroCellParameters:
    """ Object for setting macro cell base station parameters."""
    p_max_watts: float = 40.0
    p_static_watts: float = 130.0
    eta_pa: float = 0.311
    gamma_pa: float = 0.15
    power_rf_watts: float = 12.9
    power_baseband_watts: float = 29.6
    loss_feed: float = 0.5
    loss_dc: float = 0.075
    loss_cool: float = 0.10
    loss_mains: float = 0.09
    delta_p: float = 4.2  # 10.1109/LCOMM.2013.091213.131042
    sectors: int = 3
    antennas: int = 2

@dataclass(frozen=True)
class CellOffParameters:
    """ Object for setting cell base station parameters when powered OFF."""
    p_max_watts: float = 0.0
    p_static_watts: float = 0.0
    eta_pa: float = 0.0
    gamma_pa: float = 0.0
    power_rf_watts: float = 0.0
    power_baseband_watts: float = 0.0
    loss_feed: float = 0.5
    loss_dc: float = 0.0
    loss_cool: float = 0.0
    loss_mains: float = 0.0
    delta_p: float = 0.0
    sectors: int = 0
    antennas: int = 0

@dataclass(frozen=True)
class MacroCellBasicSleep:
    """ Object for setting macro cell base station parameters, when the cell is in a basic sleep mode where the PA is turned off."""
    p_max_watts: float = 0.0
    p_static_watts: float = 130.0
    eta_pa: float = 0.0
    gamma_pa: float = 0.0
    power_rf_watts: float = 12.9
    power_baseband_watts: float = 29.6
    loss_feed: float = 0.5
    loss_dc: float = 0.075
    loss_cool: float = 0.10
    loss_mains: float = 0.09
    delta_p: float = 0.0  # 10.1109/LCOMM.2013.091213.131042
    sectors: int = 3
    antennas: int = 2


class Cellv2(Cell):
    """ Class to extend original Cell class and add functionality"""

    _SLEEP_MODES = [0,1,2,3,4]

    def __init__(self, *args, sleep_mode=None, **kwargs):   # [How to use *args and **kwargs with 'peeling']
        if isinstance(sleep_mode, NoneType):
            self.sleep_mode = 0
        else:
            self.sleep_mode = sleep_mode
        # print(f'Cell[{self.i}] sleep mode is: {self.sleep_mode}')
        super().__init__(*args, **kwargs)

    def set_mcs_table(self, mcs_table_number):
        """
        Changes the lookup table used by NR_5G_standard_functions.MCS_to_Qm_table_64QAM
        """
        if mcs_table_number == 1:
            NR_5G_standard_functions.MCS_to_Qm_table_64QAM = kiss_phy_data_procedures.mcs_table_1     # same as LTE
        elif mcs_table_number == 2:
            NR_5G_standard_functions.MCS_to_Qm_table_64QAM = kiss_phy_data_procedures.mcs_table_2     # 5G NR; 256QAM
        elif mcs_table_number == 3:
            NR_5G_standard_functions.MCS_to_Qm_table_64QAM = kiss_phy_data_procedures.mcs_table_3     # 5G NR; 64QAM LowSE/RedCap (e.g. IoT devices)
        elif mcs_table_number == 4:
            NR_5G_standard_functions.MCS_to_Qm_table_64QAM = kiss_phy_data_procedures.mcs_table_4     # 5G NR; 1024QAM
        # print(f'Setting Cell[{self.i}] MCS table to: table-{mcs_table_number}')
        return
    
    def set_sleep_mode(self, mode:int):
        """
         Sets the Cell sleep mode. If set to a number between 1-4, changes behaviour of Cell and associated energy consumption.
         Default: self.sleep_mode = 0 (NO SLEEP MODE) - Cell is ACTIVE
        """
        if mode not in self._SLEEP_MODES:
            raise ValueError("Invalid sleep mode. Must be one of the following: %r." % self._SLEEP_MODES)

        # Uncomment below to debug
        # print(f'Changing Cell[{self.i}] sleep mode from {self.sleep_mode} to {mode}')
        self.sleep_mode = mode

        # # If the cell is in REAL sleep state (1-4):
        # if self.sleep_mode in self._SLEEP_MODES[1:]:  
        #     orphaned_ues = self.sim.orphaned_ues

        #     # DEBUG message
        #     print(f'Cell[{self.i}] is in SLEEP_MODE_{self.sleep_mode}')

        #     # Cellv2.power_dBm should be -inf
        #     self.power_dBm = -np.inf

        #     # ALL attached UE's should be detached (orphaned)
        #     for i in self.attached:
        #         ue = self.sim.UEs[i]
        #         orphaned_ues.append(i)
        #         ue.detach                           # This should also take care of UE throughputs.
            
        #     # Orphaned UEs should attach to cells with best RSRP
        #     self.sim.mme.attach_ues_to_n_best_rsrp(orphaned_ues)

    
    def get_sleep_mode(self):
        """
        Return the sleep_mode for the Cellv2.
        """
        return self.sleep_mode
    

    def get_cell_throughput(self):
        """
        Returns the throughput of a cell in the current timestep.
        """
        cell_throughput = 0.0
        for ue_i in self.attached:
            ue_tp_check = self.get_UE_throughput(ue_i)
            if ue_tp_check is not None:
                cell_throughput += ue_tp_check
        return cell_throughput
    
    def get_eff_bandwidth(self):
        """
        유효 대역폭 계산 (subband mask 고려)
        Beff = B * (1/n_subbands) * sum(subband_mask)
        """
        if hasattr(self, 'subband_mask') and self.subband_mask is not None:
            # Subband mask가 있는 경우
            active_subbands = np.sum(self.subband_mask)
            beff = self.bw_MHz * (active_subbands / self.n_subbands)
        else:
            # Subband mask가 없는 경우 전체 대역폭 사용
            beff = self.bw_MHz
        return beff
    
    def get_average_spectral_efficiency(self):
        """
        평균 SE 계산 (수정된 버전)
        """
        if len(self.attached) == 0:
            return 0.0
        
        from AIMM_simulator.NR_5G_standard_functions import CQI_to_64QAM_efficiency
        
        total_se = 0.0
        valid_ue_count = 0
        
        for ue_id in self.attached:
            # reports에서 직접 CQI 가져오기
            if 'cqi' in self.reports and ue_id in self.reports['cqi']:
                cqi_report = self.reports['cqi'][ue_id]
                if cqi_report is not None:
                    cqi_array = cqi_report[1] if isinstance(cqi_report, tuple) else cqi_report
                    
                    if isinstance(cqi_array, (int, float)) and not np.isnan(cqi_array):
                        # 단일 CQI 값
                        se = CQI_to_64QAM_efficiency(int(cqi_array))
                        total_se += se
                        valid_ue_count += 1
                    elif isinstance(cqi_array, np.ndarray):
                        # CQI 배열
                        se_values = []
                        for cqi_val in cqi_array:
                            if not np.isnan(cqi_val):
                                se = CQI_to_64QAM_efficiency(int(cqi_val))
                                se_values.append(se)
                        
                        if se_values:
                            ue_avg_se = np.mean(se_values)
                            total_se += ue_avg_se
                            valid_ue_count += 1
        
        # 평균 SE 반환
        if valid_ue_count > 0:
            avg_se = total_se / valid_ue_count
        else:
            # CQI 정보가 없으면 기본값 사용
            avg_se = 2.0  # 기본 SE 값 (중간 정도의 채널 품질 가정)
        
        return avg_se
            
    def get_cell_ref_capacity(self, overhead=0.15):
        """
        셀 기준 용량 계산
        C_cell(t) = B_eff(t) * SE_ref(t) * (1 - OH)
        
        Parameters:
        -----------
        overhead : float
            오버헤드 비율 (기본값 0.15 = 15%)
        
        Returns:
        --------
        float
            셀 기준 용량 (Mb/s)
        """
        beff = self.get_eff_bandwidth()  # MHz
        se_ref = self.get_average_spectral_efficiency()  # bits/Hz
        
        # 용량 계산: MHz * bits/Hz * (1-OH) = Mb/s
        capacity = beff * se_ref * (1 - overhead)
        
        return capacity    
    
    def get_cell_load(self, overhead=0.15, debug=False):
        """ 
        셀 부하 계산 (디버깅 기능 추가)
        """
        # 셀 throughput (실제로는 총 throughput)
        cell_throughput = self.get_average_throughput()
        
        # 셀에 연결된 UE 수
        n_attached = len(self.attached)
        
        # 용량 계산을 위한 요소들
        beff = self.get_eff_bandwidth()
        se_ref = self.get_average_spectral_efficiency()
        cell_capacity = beff * se_ref * (1 - overhead)
        
        # 부하 계산
        if cell_capacity > 0:
            load = cell_throughput / cell_capacity
        else:
            load = 0.0 if cell_throughput == 0 else float('inf')
        
        if debug:
            print(f"Cell Load: {load:.6f}")
            print("=" * 30)
        
        return load

    
    def loop(self):
        '''
        Main loop of Cellv2 class.
        Default: Checks if the sleep_mode flag is set and adjusts cell behaviour accordingly.
        '''
        while True:
            self.get_sleep_mode()
            if self.f_callback is not None: self.f_callback(self,**self.f_callback_kwargs)
            yield self.sim.env.timeout(self.interval)
    

class UEv2(UE):
    """ Class to extend the original UE class for extended capabilities"""

    def __init__(self, *args, **kwargs):
        self.sinr_report = {}
        self.sinr_dB = None
        # Put extra class attributes below this line
        super().__init__(*args, **kwargs)


    def get_sinr_from_cell(self, candidate_cell):
        """
        Returns the SINR for the UE from the specified cell.
        """
        if candidate_cell is None: return 0.0 # 2022-08-08 detached
        # Reset the SINR
        sinr_dB = None
        interference=from_dB(self.noise_power_dBm)*np.ones(candidate_cell.n_subbands)
        for cell in self.sim.cells:
            pl_dB=self.pathloss(cell.xyz,self.xyz)
            antenna_gain_dB=0.0
            if cell.pattern is not None:
                vector=self.xyz-cell.xyz # vector pointing from cell to UE
                angle_degrees=(180.0/math_pi)*atan2(vector[1],vector[0])
                antenna_gain_dB=cell.pattern(angle_degrees) if callable(cell.pattern) \
                else cell.pattern[int(angle_degrees)%360]
            if cell.i==candidate_cell.i: # wanted signal
                rsrp_dBm=cell.MIMO_gain_dB+antenna_gain_dB+cell.power_dBm-pl_dB
            else: # unwanted interference
                received_interference_power=antenna_gain_dB+cell.power_dBm-pl_dB
                interference+=from_dB(received_interference_power)*cell.subband_mask
        rsrp=from_dB(rsrp_dBm)
        sinr_dB=to_dB(rsrp/interference) # scalar/array
        return sinr_dB

    def get_sinr_report(self):
        """
        Scans all cells in the network and returns the SINR report for the UE.
        """
        # Reset the SINR reports
        self.sinr_report = {}
        for cell_n in self.sim.cells:
            # Store the SINR for each cell in the simulation  (key: cell.i, value: (time, SINR))
            self.sinr_report[cell_n.i] = (self.sim.env.now, 
                                          self.get_sinr_from_cell(candidate_cell=cell_n))
        return self.sinr_report



class Simv2(Sim):
    """ Class to extend original Sim class for extended capabilities from sub-classing."""
    def __init__(self, *args, **kwargs):
        self.orphaned_ues = []
        self.cell_UID = {}
        super().__init__(*args, **kwargs)
    
    def make_cellv2(self, **cell_kwargs):
        ''' 
        Convenience function: make a new Cellv2 instance and add it to the simulation; parameters as for the Cell class. Return the new Cellv2 instance.).
        '''
        self.cells.append(Cellv2(self,**cell_kwargs))
        xyz=self.cells[-1].get_xyz()
        self.cell_locations=np.vstack([self.cell_locations,xyz])
        return self.cells[-1]
    
    def make_UEv2(s,**kwargs):
        '''
        Convenience function: make a new UE instance and add it to the simulation; parameters as for the UEv2 class. Return the new UE instance.
        '''
        s.UEs.append(UEv2(s,**kwargs))
        return s.UEs[-1]
    
    def get_best_sinr_cell(self, ue_id):
        """
        Returns the cell with the best SINR for the specified UE.
        """
        j, best_sinr = None, -np.inf
        ue = self.UEs[ue_id]
        ue_sinr_reports = ue.get_sinr_report()
        for cell in self.cells:
            if cell.i not in ue_sinr_reports: continue
            time, sinr = ue_sinr_reports[cell.i]
            avg_sinr = np.average(sinr)
            if avg_sinr > best_sinr:
                j, best_sinr = cell.i, avg_sinr
        return j

class AMFv1(MME):
    """
    Adds to the basic AIMM MME and rebrands to the 5G nomenclature AMF(Access and Mobility Management Function).
    """

    def __init__(self, *args, cqi_limit:int = None, **kwargs):
        self.cqi_limit = cqi_limit
        self.poor_cqi_ues = []
        super().__init__(*args, **kwargs)

    def check_low_cqi_ue(self, ue_ids, threshold=None):
        """
        Takes a list of UE IDs and adds the UE ID to `self.poor_cqi_ues` list.
        """
        self.poor_cqi_ues.clear()
        threshold = self.cqi_limit

        for i in ue_ids:
            ue = self.sim.UEs[i]

            if isinstance(threshold, NoneType):
                return
            if isinstance(ue.cqi, NoneType):
                return
            if ue.cqi[-1] < threshold:
                self.poor_cqi_ues.append(ue.i)
                return 

    def detach_low_cqi_ue(self, poor_cqi_ues=None):
        """
        Takes a list self.poor_cqi_ues (IDs) and detaches from their serving cell.
        """
        if isinstance(poor_cqi_ues, NoneType):
            poor_cqi_ues = self.poor_cqi_ues

        for ue_id in poor_cqi_ues:
            ue = self.sim.UEs[ue_id]
            # Add the UE to the `sim.orphaned_ues` list.
            self.sim.orphaned_ues.append(ue_id)
            # Finally, detach the UE from it's serving cell.
            ue.detach()
        
        # Finally, clear the `self.poor_cqi_ues` list
        self.poor_cqi_ues.clear()

    def do_handovers(self):
        """
        Performs handovers for all UEs in the simulation.
        """
        # First, check if the strategy attribute of the class is set to 'best_sinr_cell'.
        if self.strategy == 'best_sinr_cell':
            # Iterate over all UEs in the simulation.
            for ue in self.sim.UEs:
                if ue.serving_cell is None: continue
                oldcelli=ue.serving_cell.i # 2022-08-26
                sinr_before=ue.get_sinr_from_cell(ue.serving_cell)
                CQI_before=ue.serving_cell.get_UE_CQI(ue.i)
                previous,tm=ue.serving_cell_ids[1]

                # Get the best SINR cell for the UE.
                celli = self.sim.get_best_sinr_cell(ue.i)

                if celli==ue.serving_cell.i: continue
                if self.anti_pingpong>0.0 and previous==celli:
                    if self.sim.env.now-tm<self.anti_pingpong:
                        if self.verbosity>2:
                            print(f't={float(self.sim.env.now):8.2f} handover of UE[{ue.i}] suppressed by anti_pingpong heuristic.',file=stderr)
                        continue # not enough time since we were last on this cell
                ue.detach(quiet=True)
                ue.attach(self.sim.cells[celli])
                ue.send_rsrp_reports() # make sure we have reports immediately
                ue.send_subband_cqi_report()
                CQI_after=ue.serving_cell.get_UE_CQI(ue.i)
                sinr_after = ue.get_sinr_from_cell(ue.serving_cell)
                # Print the handover event.
                print(f't={float(self.sim.env.now):.2f} handover of UE[{ue.i:3}] from Cell[{oldcelli:3}] to Cell[{ue.serving_cell.i:3}]',file=stderr,end=' ')
                print(f'CQI change {CQI_before} -> {CQI_after}',file=stderr)
                print(f'SINR change {sinr_before} -> {sinr_after}',file=stderr)

        # If the strategy is not best_sinr_cell, then perform the normal handover procedure.      
        else:
            super().do_handovers()

    def loop(self):
        '''
        Main loop of AMFv1.
        '''
        if self.sim.env.now==0.0:
            yield self.sim.env.timeout(0.5*self.interval)   # We stagger the MME startup
        print(f'MME started at {float(self.sim.env.now):.2f}, using strategy="{self.strategy}" and anti_pingpong={self.anti_pingpong:.0f}.',file=stderr)
        while True:
            self.do_handovers()
            yield self.sim.env.timeout(self.interval)
    
    def finalize(self):
        super().finalize()



class CellEnergyModel:
    """
    Defines a complete self-contained system energy model for a 5G base station (gNB).

    Parameters
    ----------
    cell : Cell
        Cell instance which this model attaches to.
    interval: float
        Time interval between CellEnergyModel updates.
    params: MacroCellParameters or SmallCellParameters
        Parameters for the energy model.
    """

    def __init__(self, cell: Cellv2, interval=1.0, params = MacroCellParameters()):
        """
        Initialize variables.
        """

        self.cell = cell
        self.cell_id = self.cell.i
        if self.cell.get_power_dBm() >= 30.0:
            self.cell_type = 'MACRO'
            self.params = params
        else:
            self.cell_type = 'SMALL'
            self.params = params

        # List of params to store
        self.p_max_watts = self.params.p_max_watts
        self.p_static_watts = self.params.p_static_watts 
        self.p_dynamic_watts = self.trx_chain_power_dynamic_watts()

        # Calculate the starting cell power
        self.cell_power_watts = self.params.sectors * self.params.antennas * (
            self.p_static_watts + self. p_dynamic_watts)
        
        
        # END of INIT

    def from_dBm_to_watts(self, x):
        """Converts dBm (decibel-milliwatt) input value to watts"""
        return from_dB(x) / 1000

    def get_power_out_per_trx_chain_watts(self, cell_power):
        """
        Takes an input value for a cell power output in dBm.
        Returns the power output of a single TRX chain in Watts.
        A TRX chain consists of an antenna, power amplifier, rf unit and baseband unit.
        """
        return self.from_dBm_to_watts(cell_power)

    def trx_chain_power_dynamic_watts(self):
        """
        Returns the power consumption (in kW), per sector / antenna.
        """


        cell_p_out_dBm = self.cell.get_power_dBm()
        cell_p_out_watts = self.from_dBm_to_watts(cell_p_out_dBm)

        if cell_p_out_watts > self.p_max_watts:
            raise ValueError('Power cannot exceed the maximum cell power!')

        if cell_p_out_watts == 0.0:
            print(f'Cell[{self.cell.i}] power is 0.0 watts. Cell is considered OFF', file=stderr)
            # Cell is OFF so zero the self.params that contribute to the dynamic power consumption
            # of the TRX chain
            # Use the CellOffParameters class to set the dynamic power consumption to zero
            self.params = CellOffParameters()

        # Get current TRX chain output power in watts
        trx_p_out_watts = self.get_power_out_per_trx_chain_watts(cell_p_out_dBm)
        
        # cell_load
        rho_raw = self.cell.get_cell_load(overhead=0.15, debug=False)
        rho = min(rho_raw, 1.0)  # Ensure rho does not exceed 1.0
        
        # 여기랑
        # Sanity check that other input values are in decimal form
        rf_nom = self.params.power_rf_watts
        bb_nom = self.params.power_baseband_watts
        rf_idle_frac = getattr(self.params, "rf_idle_frac", 0.3)  
        bb_idle_frac = getattr(self.params, "bb_idle_frac", 0.4)
        gamma_rf = getattr(self.params, "gamma_rf", 1.0)  
        gamma_bb = getattr(self.params, "gamma_bb", 1.0)
        p_rf_watts = rf_idle_frac*rf_nom + (rf_nom - rf_idle_frac*rf_nom) * (rho ** gamma_rf)
        p_bb_watts = bb_idle_frac*bb_nom + (bb_nom - bb_idle_frac*bb_nom) * (rho ** gamma_bb)
        
        # Calculate the Power Amplifier power consumption in watts
        if trx_p_out_watts == 0.0:
            p_pa_watts = 0.0
        p_pa_watts = trx_p_out_watts / (self.params.eta_pa * (1 - self.params.loss_feed))

        # Calculate the value of `P_ue_plus_C_watts` given the number of UEs multiplex by the base station
        if self.cell.get_nattached() == 0:
            p_ue_plus_C_watts = 0.0
        else:
            p_ue_plus_C_watts = trx_p_out_watts / self.cell.get_nattached()
            # FIXME: This was never implemented! 

        # Calculate power consumptions of a single TRX chain (watts)
        p_consumption_watts = p_pa_watts + p_rf_watts + p_bb_watts

        # Calculate losses (ratio)
        p_losses_ratio = (1 - self.params.loss_dc) * \
            (1 - self.params.loss_mains) * (1 - self.params.loss_cool)

        # Get the power output per TRX chain (watts)
        p_out_TRX_chain_watts = p_consumption_watts / p_losses_ratio


        # Update the instance stored value
        self.p_dynamic_watts = p_out_TRX_chain_watts

        return p_out_TRX_chain_watts
    
    def get_cell_sleep_mode(self):
        """
        Returns the cell sleep mode.
        """
        return self.cell.sleep_mode
    
    def get_cell_sleep_mode_energy_cons(self, cell_sleep_mode):
        # If the cell sleep_mode is between 1 and 4, then the cell is in sleep mode
        if cell_sleep_mode in range(1, 5):

            # Set a placeholder for sleep_time
            sleep_time = 0

            # Get the cell interval in microseconds
            time_interval = self.cell.interval * 1e6

            # Get the OFDM symbol time
            ofdm_symbol_time = calculate_ofdm_symbol_time(self.cell.bw_MHz, mu=1)

            # Calculate the maximum number of whole OFDM symbols that can be transmitted in the time interval
            n_ofdm_symbols_overall = int(time_interval / ofdm_symbol_time)

            # Determine the sleep mode and set the sleep time parameters
            mode = cell_sleep_mode
            if mode == 1:
                sleep_time = random_time_interval(self.cell.sim, min_time_us=ofdm_symbol_time, max_time_us=1000)
            elif mode == 2:
                sleep_time = random_time_interval(self.cell.sim, min_time_us=5000, max_time_us=10000)
            elif mode == 3:
                sleep_time = random_time_interval(self.cell.sim, min_time_us=50000, max_time_us=100000)
            elif mode == 4:
                sleep_time = random_time_interval(self.cell.sim, min_time_us=500000, max_time_us=1000000)

            # Calculate the active time ratio in microseconds
            ratio_active = (time_interval - sleep_time) / time_interval

            # Calculate the sleep ratio
            sleep_ratio = sleep_time / time_interval

            # Store the ratio of whole OFDM symbols that can be transmitted in the active time as a cell attribute
            self.cell.sleep_mode_ratio_active = ratio_active        
            # FIXME - this is a hack to get the ratio active to the cell. 
            # Need to find a better way to do this and scale throughput accordingly (for slep sleep modes >1)

            # The amount of static power consumed by the TRX chain is the same for all sleep modes
            static_power = self.p_static_watts

            # The ratio of active time is the amount of time that the cell is using FULL power, so there is NO SCALING for this amount of time
            active_power = self.trx_chain_power_dynamic_watts() * ratio_active

            # Sleep modes turn hardware componenets off so here we `zero` these by scaling to the ratio of active time
            # Determine the sleep mode and set the power consumption parameters accordingly
            power_rf_watts_scaled = self.params.power_rf_watts * ratio_active
            power_baseband_watts_scaled = self.params.power_baseband_watts * ratio_active
            eta_pa_scaled = self.params.eta_pa * ratio_active
            loss_dc_scaled = self.params.loss_dc * ratio_active

            if mode == 1:
                self.params = dataclasses.replace(self.params, power_rf_watts=power_rf_watts_scaled)
            elif mode == 2:
                self.params = dataclasses.replace(self.params, power_rf_watts=power_rf_watts_scaled, power_baseband_watts=power_baseband_watts_scaled)
            elif mode == 3:
                self.params = dataclasses.replace(self.params, power_rf_watts=power_rf_watts_scaled, power_baseband_watts=power_baseband_watts_scaled, eta_pa=eta_pa_scaled)
            elif mode == 4:
                self.params = dataclasses.replace(self.params, power_rf_watts=power_rf_watts_scaled, power_baseband_watts=power_baseband_watts_scaled, eta_pa=eta_pa_scaled, loss_dc=loss_dc_scaled)
            
            # The ratio of SLEEP time is the amount of time that the cell is using SCALED power, so there is LESS POWER CONSUMPTION for this amount of time
            sleep_power = self.trx_chain_power_dynamic_watts() * sleep_ratio

            # Calculate the total power consumption of the cell
            self.cell_power_watts = self.params.sectors * self.params.antennas * (static_power + active_power + sleep_power)

    def reset_energy_model_params(self, params):
        """
        Resets the energy model parameters to default values based on the __init__ power_dBm set.
        """
        params = self.params
        if self.cell_type == 'MACRO':
            self.params = params
        elif self.cell_type == 'SMALL':
            self.params = params
    

    def update_cell_power_watts(self):
        """
        Updates the cell power consumption (in watts).
        """ 
        # Reset the energy model parameters to defaults
        self.reset_energy_model_params(params=self.params)

        # Get the cell sleep mode
        cell_sleep_mode = self.get_cell_sleep_mode()
        if cell_sleep_mode > 0 and cell_sleep_mode < 5:
            return self.get_cell_sleep_mode_energy_cons(cell_sleep_mode)

        # Update the cell power as normal
        self.cell_power_watts = self.params.sectors * self.params.antennas * (
            self.p_static_watts + self.trx_chain_power_dynamic_watts())


    def get_cell_power_watts(self, time):
        """
        Returns the power consumption (in watts) of the cell at a given time.
        """
        if time == 0:
            return self.cell_power_watts
        else:
            self.update_cell_power_watts()
            return self.cell_power_watts

    def f_callback(self, x, **kwargs):
        if isinstance(x, Cellv2):
            if x.i == self.cell_id:
                self.update_cell_power_watts()
            else:
                raise ValueError(
                    'Cells can only update their own energy model instances! Check the cell_id.')
            
# End class Energy


class EXP3CellOnOff(Scenario):
    """
    Enhanced EXP3 (Exponential-weight algorithm for Exploration and Exploitation)
    for cell on/off control without real-time regret calculation.
    """
    
    def __init__(self, sim, k_cells_to_off: int, n_total_cells: int = None,
                 interval: float = 1.0, delay: float = 0.0,
                 gamma: float = 0.1, warm_up_episodes: int = 100,
                 enable_warm_up: bool = True, min_selection_guarantee: bool = True,
                 normalization_window: int = 50, event_threshold: int = 10,
                 cell_energy_models: dict = None):
        """
        Initialize Enhanced EXP3 Cell On/Off scenario.
        """
        self.sim = sim
        self.k_cells_to_off = k_cells_to_off
        self.n_total_cells = n_total_cells or len(sim.cells)
        self.interval = interval
        self.delay = delay
        self.gamma = gamma
        self.warm_up_episodes = warm_up_episodes
        self.enable_warm_up = enable_warm_up
        self.min_selection_guarantee = min_selection_guarantee
        self.normalization_window = normalization_window
        self.event_threshold = event_threshold
        
        if not cell_energy_models:
            raise ValueError("cell_energy_models parameter is required for TRX energy calculations!")
        self.cell_energy_models = cell_energy_models or {}  # ← 이 줄 추가!
        
        # Initialize EXP3 parameters
        self.n_arms = self._calculate_n_arms()
        self.weights = np.ones(self.n_arms)
        self.probabilities = np.ones(self.n_arms) / self.n_arms
        self.arm_selections = np.zeros(self.n_arms, dtype=int)
        self.arm_rewards = defaultdict(list)
        self.episode_count = 0
        self.event_count = 0
        self.current_arm = None
        self.current_cells_off = []
        self.previous_cells_off = []
        
        # Performance tracking
        self.reward_history = []
        self.cumulative_reward = 0.0
        self.efficiency_history = []
        self.throughput_history = []
        self.power_history = []
        self.arm_selection_history = []
        self.probability_history = []
        self.switching_cost_history = []
        
        # Baseline metrics
        self.baseline_power = None
        self.baseline_throughput = None
        self.baseline_efficiency = None
        self.all_on_metrics = {'power': None, 'throughput': None, 'efficiency': None}
        
        # Cell throughput tracking
        self.cell_throughput_history = defaultdict(list)
        
        # Convergence tracking
        self.probability_stability_window = 50
        self.convergence_threshold = 0.01
        self.convergence_episode = None
        self.probability_change_history = []
        
        # Normalization parameters
        self.reward_min = float('inf')
        self.reward_max = float('-inf')
        self.reward_buffer = []
        
        print(f"Enhanced EXP3CellOnOff initialized: k={k_cells_to_off}, n={self.n_total_cells}, "
              f"arms={self.n_arms}, gamma={gamma}")
    
    def _calculate_n_arms(self) -> int:
        """Calculate number of possible arms (combinations)."""
        from math import comb
        return comb(self.n_total_cells, self.k_cells_to_off)
    
    def _get_arm_combination(self, arm_idx: int) -> List[int]:
        """Convert arm index to cell combination."""
        from itertools import combinations
        all_combinations = list(combinations(range(self.n_total_cells), self.k_cells_to_off))
        return list(all_combinations[arm_idx])
    
    def _select_arm(self) -> int:
        """Select an arm based on EXP3 algorithm."""
        if self.enable_warm_up and self.episode_count < self.warm_up_episodes:
            # ε-greedy warm-up phase
            epsilon = 0.5 * (1 - self.episode_count / self.warm_up_episodes)
            if np.random.random() < epsilon:
                return np.random.choice(self.n_arms)
        
        # Ensure minimum selection guarantee
        if self.min_selection_guarantee:
            unselected = np.where(self.arm_selections == 0)[0]
            if len(unselected) > 0:
                return np.random.choice(unselected)
        
        # Normal EXP3 selection
        return np.random.choice(self.n_arms, p=self.probabilities)
    
    def _update_probabilities(self):
        """Update arm selection probabilities."""
        uniform_prob = 1.0 / self.n_arms
        weight_sum = np.sum(self.weights)
        
        old_probabilities = self.probabilities.copy()
        
        if weight_sum > 0:
            weight_prob = self.weights / weight_sum
            self.probabilities = (1 - self.gamma) * weight_prob + self.gamma * uniform_prob
        else:
            self.probabilities = np.ones(self.n_arms) / self.n_arms
        
        # Track probability changes for convergence detection
        prob_change = np.max(np.abs(self.probabilities - old_probabilities))
        self.probability_change_history.append(prob_change)
        
        # Check for convergence
        if self.convergence_episode is None and len(self.probability_change_history) >= self.probability_stability_window:
            recent_changes = self.probability_change_history[-self.probability_stability_window:]
            if np.mean(recent_changes) < self.convergence_threshold:
                self.convergence_episode = self.episode_count
                print(f"Convergence detected at episode {self.convergence_episode}")
    
    def _calculate_switching_cost(self, new_cells_off: List[int]) -> float:
        """Calculate the cost of switching configuration."""
        if not self.previous_cells_off:
            return 0.0
        
        cells_turned_on = set(self.previous_cells_off) - set(new_cells_off)
        cells_turned_off = set(new_cells_off) - set(self.previous_cells_off)
        
        # Simple switching cost model: each change has unit cost
        switching_cost = len(cells_turned_on) + len(cells_turned_off)
        
        # Normalized by maximum possible switches
        max_switches = min(len(self.previous_cells_off), len(new_cells_off)) * 2
        if max_switches > 0:
            return switching_cost / max_switches
        return 0.0
    
    def _calculate_metrics(self):
        """Calculate network metrics using only TRX energy model."""
        total_throughput = 0.0
        total_power = 0.0
        active_cells = 0
        cell_throughputs = {}
        
        # TRX 에너지 모델 존재 여부 확인
        if not self.cell_energy_models:
            raise ValueError("Cell energy models not available! Cannot calculate accurate power consumption.")
        
        print(f"=== TRX Energy Model Calculation at time {self.sim.env.now:.1f} ===")
        print(f"Available energy models: {len(self.cell_energy_models)}")
        print(f"Cells currently OFF: {self.current_cells_off}")
        
        for cell in self.sim.cells:
            cell_id = cell.i
            
            # 꺼진 셀들 처리
            if cell_id in self.current_cells_off:
                total_power += 0.001  # 1W standby power
                cell_throughputs[cell_id] = 0.0
                print(f"Cell[{cell_id}]: OFF - standby power 1W")
                continue
            
            # 활성 셀의 throughput 계산
            if hasattr(cell, 'get_average_throughput'):
                cell_tp = cell.get_average_throughput()
            elif hasattr(cell, 'attached') and len(cell.attached) > 0:
                # Calculate from attached UEs
                cell_tp = sum(cell.get_UE_throughput(ue_id).sum() 
                            for ue_id in cell.attached if hasattr(cell, 'get_UE_throughput'))
            else:
                cell_tp = 0.0
            
            cell_throughputs[cell_id] = cell_tp
            total_throughput += cell_tp if cell_tp > 0 else 0
            
            # ✅ TRX 에너지 모델만 사용 - fallback 제거!
            if cell_id not in self.cell_energy_models:
                raise KeyError(f"Cell[{cell_id}] energy model not found! Check initialization.")
            
            # TRX 에너지 모델에서 정확한 전력 계산
            cell_energy_model = self.cell_energy_models[cell_id]
            
            # 현재 시간에서의 실제 전력 소비 계산
            cell_power_watts = cell_energy_model.get_cell_power_watts(self.sim.env.now)
            total_power += cell_power_watts / 1000.0  # Convert to kW
            
            # 상세 정보 출력 (디버깅용)
            static_power = cell_energy_model.p_static_watts
            dynamic_power = cell_energy_model.p_dynamic_watts
            sectors = cell_energy_model.params.sectors
            antennas = cell_energy_model.params.antennas
            
            print(f"Cell[{cell_id}]: Total={cell_power_watts:.2f}W "
                f"(Static={static_power:.2f}W × {sectors}sec × {antennas}ant + "
                f"Dynamic={dynamic_power:.2f}W × {sectors}sec × {antennas}ant)")
            print(f"  - Cell load: {cell.get_cell_load() if hasattr(cell, 'get_cell_load') else 'N/A'}")
            print(f"  - Throughput: {cell_tp:.2f} Mbps")
            
            active_cells += 1
        
        # Calculate efficiency (bits per Joule)
        efficiency = (total_throughput * 1e6) / (total_power * 1000) if total_power > 0 else 0.0
        
        # Average cell throughput
        avg_cell_throughput = total_throughput / active_cells if active_cells > 0 else 0.0
        
        print(f"Network Summary:")
        print(f"  - Total Power: {total_power:.3f} kW")
        print(f"  - Total Throughput: {total_throughput:.2f} Mbps")
        print(f"  - Energy Efficiency: {efficiency:.2f} bits/J")
        print(f"  - Active Cells: {active_cells}")
        print("="*60)
        
        return {
            'total_throughput': total_throughput,
            'total_power': total_power,
            'efficiency': efficiency,
            'active_cells': active_cells,
            'avg_cell_throughput': avg_cell_throughput,
            'cell_throughputs': cell_throughputs
        }
    
    def _calculate_baseline_metrics(self):
        """Calculate baseline metrics with all cells on."""
        if self.baseline_power is not None:
            return  # Already calculated
        
        # Temporarily store current configuration
        temp_cells_off = self.current_cells_off.copy()
        
        # Turn all cells on
        self.current_cells_off = []
        metrics = self._calculate_metrics()
        
        self.all_on_metrics = {
            'power': metrics['total_power'],
            'throughput': metrics['total_throughput'],
            'efficiency': metrics['efficiency']
        }
        
        # Calculate random baseline (random k cells off)
        random_configs = []
        for _ in range(10):  # Sample 10 random configurations
            random_cells_off = np.random.choice(self.n_total_cells, 
                                               self.k_cells_to_off, 
                                               replace=False).tolist()
            self.current_cells_off = random_cells_off
            rand_metrics = self._calculate_metrics()
            random_configs.append(rand_metrics)
        
        # Average random baseline
        self.baseline_power = np.mean([m['total_power'] for m in random_configs])
        self.baseline_throughput = np.mean([m['total_throughput'] for m in random_configs])
        self.baseline_efficiency = np.mean([m['efficiency'] for m in random_configs])
        
        # Restore original configuration
        self.current_cells_off = temp_cells_off
    
    def _calculate_reward(self) -> float:
        """Calculate reward based on network efficiency."""
        metrics = self._calculate_metrics()
        
        # Store metrics
        self.efficiency_history.append(metrics['efficiency'])
        self.throughput_history.append(metrics['total_throughput'])
        self.power_history.append(metrics['total_power'])
        
        # Store cell throughputs
        for cell_id, tp in metrics['cell_throughputs'].items():
            self.cell_throughput_history[cell_id].append(tp)
        
        return metrics['efficiency']
    
    def _normalize_reward(self, reward: float) -> float:
        """Normalize reward to [0, 1] range."""
        self.reward_buffer.append(reward)
        
        if len(self.reward_buffer) > self.normalization_window:
            self.reward_buffer.pop(0)
        
        if len(self.reward_buffer) > 1:
            buffer_min = min(self.reward_buffer)
            buffer_max = max(self.reward_buffer)
            
            if buffer_max > buffer_min:
                normalized = (reward - buffer_min) / (buffer_max - buffer_min)
            else:
                normalized = 0.5
        else:
            normalized = 0.5
        
        return np.clip(normalized, 0.0, 1.0)
    
    def _update_weights(self, arm: int, reward: float):
        """Update weights for the selected arm without regret calculation."""
        # Normalize reward
        normalized_reward = self._normalize_reward(reward)
        
        # Store reward
        self.arm_rewards[arm].append(normalized_reward)
        self.reward_history.append(normalized_reward)
        
        # Update cumulative reward
        self.cumulative_reward += normalized_reward
        
        # Update weight for selected arm (removed regret calculation)
        if self.probabilities[arm] > 0:
            estimated_reward = normalized_reward / self.probabilities[arm]
            self.weights[arm] *= np.exp(self.gamma * estimated_reward / self.n_arms)
        
        # Prevent numerical overflow
        max_weight = np.max(self.weights)
        if max_weight > 1e10:
            self.weights /= max_weight / 1e10
    
    def _apply_cell_configuration(self, cells_to_off: List[int]):
        """Apply the cell on/off configuration."""
        self.previous_cells_off = self.current_cells_off.copy()
        
        # Calculate switching cost
        switching_cost = self._calculate_switching_cost(cells_to_off)
        self.switching_cost_history.append(switching_cost)
        
        # Turn on previously off cells
        for cell_id in self.current_cells_off:
            if cell_id not in cells_to_off:
                self.sim.cells[cell_id].set_power_dBm(43.0)  # Default power
                print(f"Time {self.sim.env.now:.1f}: Cell[{cell_id}] turned ON")
        
        # Turn off selected cells
        for cell_id in cells_to_off:
            if cell_id not in self.current_cells_off:
                cell = self.sim.cells[cell_id]
                attached_ues = list(cell.attached)
                for ue_id in attached_ues:
                    ue = self.sim.UEs[ue_id]
                    ue.detach()
                
                cell.set_power_dBm(-np.inf)
                print(f"Time {self.sim.env.now:.1f}: Cell[{cell_id}] turned OFF")
        
        self.current_cells_off = cells_to_off
    
    def get_extended_statistics(self) -> Dict:
        """Get comprehensive statistics without regret metrics."""
        stats = {
            'episode_count': self.episode_count,
            'current_arm': self.current_arm,
            'current_cells_off': self.current_cells_off,
            'cumulative_reward': self.cumulative_reward,
            'avg_reward': np.mean(self.reward_history) if self.reward_history else 0,
            'reward_variance': np.var(self.reward_history) if len(self.reward_history) > 1 else 0,
            'reward_std': np.std(self.reward_history) if len(self.reward_history) > 1 else 0,
            'probabilities': self.probabilities.copy(),
            'arm_selections': self.arm_selections.copy(),
            'convergence_episode': self.convergence_episode,
        }
        
        # Calculate energy savings
        if self.baseline_power and self.power_history:
            current_power = self.power_history[-1] if self.power_history else 0
            all_on_power = self.all_on_metrics['power'] if self.all_on_metrics['power'] else current_power
            
            stats['energy_savings_vs_all_on'] = (1 - current_power / all_on_power) * 100 if all_on_power > 0 else 0
            stats['energy_savings_vs_random'] = (1 - current_power / self.baseline_power) * 100 if self.baseline_power > 0 else 0
        
        # Average metrics
        if self.efficiency_history:
            stats['avg_efficiency'] = np.mean(self.efficiency_history)
            stats['efficiency_std'] = np.std(self.efficiency_history)
        
        if self.throughput_history:
            stats['avg_throughput'] = np.mean(self.throughput_history)
            stats['throughput_std'] = np.std(self.throughput_history)
        
        if self.power_history:
            stats['avg_power'] = np.mean(self.power_history)
            stats['power_std'] = np.std(self.power_history)
        
        # Average cell throughput
        if self.cell_throughput_history:
            all_cell_tps = []
            for cell_tps in self.cell_throughput_history.values():
                all_cell_tps.extend(cell_tps)
            stats['avg_cell_throughput'] = np.mean(all_cell_tps) if all_cell_tps else 0
            stats['cell_throughput_std'] = np.std(all_cell_tps) if all_cell_tps else 0
        
        # Switching cost
        if self.switching_cost_history:
            stats['avg_switching_cost'] = np.mean(self.switching_cost_history)
            stats['total_switching_cost'] = np.sum(self.switching_cost_history)
        
        return stats

    
    def loop(self):
        """Main loop for Enhanced EXP3 cell on/off scenario."""
        # Initial delay
        if self.delay > 0:
            yield self.sim.wait(self.delay)
        
        # Calculate baseline metrics
        self._calculate_baseline_metrics()
        
        while True:
            self.event_count += 1
            
            # Trigger arm selection after threshold events
            if self.event_count >= self.event_threshold:
                # Select new arm
                arm = self._select_arm()
                self.current_arm = arm
                self.arm_selections[arm] += 1
                self.arm_selection_history.append(arm)
                
                # Get cell combination for selected arm
                cells_to_off = self._get_arm_combination(arm)
                
                # Apply configuration
                self._apply_cell_configuration(cells_to_off)
                
                # Wait for network to stabilize
                yield self.sim.wait(self.interval * 2)
                
                # Calculate reward
                reward = self._calculate_reward()
                
                # Update weights
                self._update_weights(arm, reward)
                
                # Update probabilities
                self._update_probabilities()
                self.probability_history.append(self.probabilities.copy())
                
                # Reset event counter and increment episode
                self.event_count = 0
                self.episode_count += 1
                
                # Print progress
                if self.episode_count % 10 == 0:
                    stats = self.get_extended_statistics()
                    print(f"Episode {self.episode_count}: "
                          f"Efficiency={stats['avg_efficiency']:.2f} bits/J, "
                          f"Energy Savings={stats.get('energy_savings_vs_all_on', 0):.1f}%, "
                          f"Convergence={'Yes' if self.convergence_episode else 'No'}")
            
            yield self.sim.wait(self.interval)
    

class ChangeCellPower(Scenario):
    """
    Changes the power_dBm of the specified list of cells (default random cell ) after a specified delay time (if provided), relative to t=0.
    """

    def __init__(self, sim, interval=0.5, cells=None, n_cells=None, delay=None, new_power=None):
        """
        Initializes an instance of the ChangeCellPower class.

        Parameters:
        -----------
        sim : SimPy.Environment
            The simulation environment object.
        interval : float, optional
            The time interval between each power change. Default is 0.5.
        cells : list of int, optional
            The list of cell indices to change power. Default is random_cell (based on the seed in the simulation)
        n_cells : int, optional
            The number of random cells to change power. Default is None.
        delay : float, optional
            The delay time before changing the cell powers. Default is None.
        new_power : float, optional
            The new power_dBm to set for the specified cells. Default is None.
        """
        self.target_cells = None
        self.delay_time = delay
        self.new_power = new_power
        self.sim = sim
        self.interval = interval
        self.random_cell = self.sim.rng.integers(low=0, high=len(self.sim.cells))
        self.outer_ring = [0, 1, 2, 3, 6, 7, 11, 12, 15, 16, 17, 18]
        if self.target_cells is None:
            # If n_cells is specified, choose n_cells random cells
            if n_cells is not None and n_cells > 0:
                self.target_cells = self.sim.rng.choice(len(self.sim.cells), n_cells, replace=False)
            # If cells is specified, choose the specified cells
            elif cells is not None:
                    self.target_cells = cells
            # If neither n_cells or cells is specified, choose a random cell
            else:
                self.target_cells = self.random_cell

    def loop(self):
        """
        The main loop of the scenario, which changes the power of specified cells after a delay.
        """
        while True:
            while self.sim.env.now >= self.delay_time:
                if isinstance(self.target_cells, list) or isinstance(self.target_cells, np.ndarray):
                    for i in self.target_cells:
                        self.sim.cells[i].set_power_dBm(self.new_power)
                elif isinstance(self.target_cells, int):
                    self.sim.cells[self.target_cells].set_power_dBm(self.new_power)
                else:
                    raise ValueError(
                        'The target_cells parameter must be a list, array, or integer.')
                yield self.sim.env.timeout(self.interval)





class SwitchNCellsOff(Scenario):
    """
    Randomly selected cell from the simulation environment and remove after delay time (if provided), relative to t=0.
    """

    def __init__(self, sim, delay=None, interval=0.1, n_cells=0):
        """
        Initializes an instance of the SwitchNCellsOff class.

        Parameters:
        -----------
        sim : SimPy.Environment
            The simulation environment object.
        interval : float, optional
            The time interval between each cell removal. Default is 0.1.
        delay : float, optional
            The delay time before removing the cell. Default is None.
        """
        self.sim = sim
        self.interval = interval
        self.delay_time = delay
        self.n_cells = n_cells

    def loop(self):
        """
        The main loop of the scenario, which removes a randomly selected cell after a delay.
        """
        # Select N random cells to remove using the sim.rng object
        cell_indices = self.sim.rng.choice(len(self.sim.cells), self.n_cells, replace=False)

        # Print to stdout, the selection of the cell for removal
        for cell_index in cell_indices:
            print(f'Cell[{cell_index}] has been selected for removal.')
        while True:
            if self.sim.env.now >= self.delay_time:
                for cell_index in cell_indices:
                    for cell in self.sim.cells:
                        if cell.i == cell_index:
                            # Zero the cell's power
                            cell.set_power_dBm(-np.inf)
                            # Print to stdout, the removal of the cell
                            print(f'Cell[{cell_index}] has been `removed` from the simulation with 0.0 watts output power.')

            yield self.sim.wait(self.interval)


class SetCellSleep(Scenario):
    """
    A scenario that sets the sleep level of specified cells at specified times.
    """
    def __init__(self, sim, interval=1.0, time_cell_sleep_level_duration=None, delay=0):
        self.sleep_delay_time = delay
        self.sim = sim
        self.interval = interval
        if time_cell_sleep_level_duration is None:
            print('WARNING: No cells specified for SetCellSleep. No cells will be put to sleep.')
            self.bedtime_stories = []
        else:
            self.bedtime_stories = []
            for item in time_cell_sleep_level_duration:
                if not isinstance(item, dict):
                    raise TypeError(f'Expected dict for time_cell_sleep_level_duration, got {type(item)}')
                if {'time', 'cell', 'sleep_level', 'sleep_duration'} != item.keys():
                    raise KeyError(f'Expected keys "time", "cell", "sleep_level", and "sleep_duration" in time_cell_sleep_level_duration, got {item.keys()}')
                if not isinstance(item['time'], (int, float)):
                    raise TypeError(f'Expected int or float for time, got {type(item["time"])}')
                if not isinstance(item['cell'], int):
                    raise TypeError(f'Expected int for cell, got {type(item["cell"])}')
                if item['cell'] not in [x for x, j in enumerate(self.sim.cells)]:
                    raise KeyError(f'Cell {item["cell"]} not found in simulation environment.')
                if not isinstance(item['sleep_level'], int):
                    raise TypeError(f'Expected int for sleep_level, got {type(item["sleep_level"])}')
                if not 0 <= item['sleep_level'] <= 4:
                    raise ValueError(f'Expected sleep_level to be between 0 and 4, got {item["sleep_level"]}')
                if not isinstance(item['sleep_duration'], (int, float)):
                    raise TypeError(f'Expected int or float for sleep_duration, got {type(item["sleep_duration"])}')
                self.bedtime_stories.append(item)

    def loop(self):
        while True:
            if self.sim.env.now < self.sleep_delay_time:
                yield self.sim.wait(self.interval)
            for item in self.bedtime_stories:
                if self.sim.env.now >= item['time']:
                    self.sim.cells[item['cell']].set_sleep_mode(item['sleep_level'])
                    if item['sleep_duration'] == -1:
                        yield self.sim.wait(self.interval)
                    elif item['sleep_duration'] - self.sim.env.now <= 0: 
                        self.bedtime_stories.remove(item)
                yield self.sim.wait(self.interval)


class MyLogger(Logger):

    def __init__(self, *args,cell_energy_models=None, logfile_path=None, **kwargs):
        self.dataframe = None
        self.cell_energy_models = cell_energy_models
        self.logfile_path = logfile_path
        super().__init__(*args, **kwargs)

    def get_cqi_to_mcs(self, cqi):
        """
        Returns the MCS value for a given CQI value. Copied from `NR_5G_standard_functions.CQI_to_64QAM_efficiency`
        """
        if type(cqi) is NoneType:
            return np.nan
        else:
            return max(0,min(28,int(28*cqi/15.0)))



    def get_neighbour_cell_rsrp_rank(self, ue_id):
        """
        Returns a list of tuples containing the ID and RSRP of neighbouring cells for the given UE,
        with the highest RSRP value at position 0 for each cell in self.sim.cells.

        Parameters
        ----------
        ue_id : int
            The ID of the UE.

        Returns
        -------
        list
            A list of tuples containing the ID and RSRP of neighbouring cells for the given UE.
            
        Notes
        -----
        This function returns the neighbouring cell RSRPs, where index 0 is not the original serving cell.
        """
        # Create a place to hold the results that will contain tuples of (cell_id, rsrp) and can be efficiently sorted by rsrp
        neighbour_cell_rsrp = []
 
        # Loop through all cells
        for cell in self.sim.cells:
            cell_id = cell.i
            # Skip the serving cell
            if cell_id == self.sim.UEs[ue_id].serving_cell.i:
                continue
            # Skip cells that have no entry for the given UE
            if ue_id not in cell.reports["rsrp"]:
                continue
            # Skip cells that have no RSRP value for the given UE
            ue_rsrp = cell.reports["rsrp"][ue_id][1]
            if ue_rsrp == -np.inf:
                continue
            # Add the cell to the list if it doesn't exist already
            if not any(cell_id == cell_tuple[1] for cell_tuple in neighbour_cell_rsrp):
                # Insert (rsrp, cell_id) into the neighbour_cell_rsrp list, such that the list is sorted by rsrp in descending order
                neighbour_cell_rsrp.insert(bisect_left(neighbour_cell_rsrp, (ue_rsrp, cell_id)), (ue_rsrp, cell_id))

        return neighbour_cell_rsrp



    def get_cell_data_attached_UEs(self, cell):
        """
        Returns a list of data for each attached UE in a cell
        """
        data = []
        for attached_ue_id in cell.attached:
            UE = self.sim.UEs[attached_ue_id]
            serving_cell = UE.serving_cell
            if serving_cell.power_dBm == -np.inf or serving_cell.power_dBm <= 0.0:
                serving_cell.reports["rsrp"].clear()                        # clear the serving cell["rsrp"] reports if the serving cell is not transmitting
                                                                            # FIXME - using a dict comprehension doesn't clear the dictionary properly for some reason (e.g. {key: 0 for key in serving_cell.reports["rsrp"]}  )
                serving_cell.reports["throughput_Mbps"].clear()             # clear the serving cell["throughput"] reports if the serving cell is not transmitting
                UE.sinr_dB = 0.0                                          # clear the UE["sinr_dB"] reports if the serving cell is not transmitting
            cell_energy_model = self.cell_energy_models[serving_cell.i]
            seed = self.sim.seed
            neigh_rsrp_array = self.get_neighbour_cell_rsrp_rank(UE.i)
            tm = self.sim.env.now                                           # current time
            sc_id = serving_cell.i                                          # current UE serving_cell
            sc_sleep_mode = serving_cell.get_sleep_mode()                   # current UE serving_cell sleep mode status
            sc_xy = serving_cell.get_xyz()[:2]                              # current UE serving_cell xy position
            ue_id = UE.i                                                    # current UE ID
            ue_xy = UE.get_xyz()[:2]                                        # current UE xy position
            d2sc = np.linalg.norm(sc_xy - ue_xy)                            # current UE distance to serving_cell
            ue_tp = serving_cell.get_UE_throughput(attached_ue_id)          # current UE throughput ('fundamental')
            sc_power_dBm = serving_cell.get_power_dBm()                     # current UE serving_cell transmit power
            sc_power_watts = from_dB(sc_power_dBm) / 1e3                 # current UE serving_cell transmit power in watts
            sc_rsrp = serving_cell.get_rsrp(ue_id)                          # current UE rsrp from serving_cell
            neigh1_rsrp = neigh_rsrp_array[-1][0]    
            neigh2_rsrp = neigh_rsrp_array[-2][0]    
            noise = UE.noise_power_dBm                                      # current UE thermal noise
            sinr = UE.sinr_dB                                               # current UE sinr from serving_cell
            cqi = UE.cqi                                                    # current UE cqi from serving_cell
            mcs = self.get_cqi_to_mcs(cqi)                                  # current UE mcs for serving_cell
            cell_eff_bw = serving_cell.get_eff_bandwidth()  # current UE serving_cell effective bandwidth
            cell_avg_se = serving_cell.get_average_spectral_efficiency()  # current UE serving_cell average spectral efficiency
            cell_ref_capacity = serving_cell.get_cell_ref_capacity()  # current UE serving_cell capacity
            cell_load = serving_cell.get_cell_load()                              # current UE serving_cell load
            cell_tp = serving_cell.get_cell_throughput()                    # current UE serving_cell throughput
            cell_power_kW = cell_energy_model.get_cell_power_watts(tm) / 1e3          # current UE serving_cell power consumption
            cell_ee = (cell_tp * 1e6) / (cell_power_kW * 1e3)               # current UE serving_cell energy efficiency
            cell_se = (cell_tp * 1e6) / (serving_cell.bw_MHz * 1e6)         # current UE serving_cell spectral efficiency

            # Get the above as a list
            data_list = [seed, tm, sc_id, sc_sleep_mode, ue_id, d2sc, ue_tp, sc_power_dBm, sc_power_watts, sc_rsrp, neigh1_rsrp, neigh2_rsrp, noise, sinr, cqi, mcs, cell_eff_bw, cell_avg_se, cell_ref_capacity, cell_load, cell_tp, cell_power_kW, cell_ee, cell_se]

            # convert ndarrays to str or float
            for i, j in enumerate(data_list):
                if type(j) is np.ndarray:
                    data_list[i] = float(j)

            # Write above to `data` list
            data.append(data_list)
        return data
    
    def get_cell_data_no_UEs(self, cell):
        """
        Returns a list of data for each cell with no attached UEs.
        """
        data = []
        seed= self.sim.seed
        UE = float('nan')
        serving_cell = float('nan')
        cell_energy_model = self.cell_energy_models[cell.i]
        tm = self.sim.env.now                                       # current time
        sc_id = cell.i                                              # current cell
        sc_sleep_mode = cell.get_sleep_mode()                       # current cell sleep mode status
        sc_xy = cell.get_xyz()[:2]                                  # current cell xy position
        ue_id = None                                                # UE ID
        ue_xy = float('nan')                                        # UE xy position
        d2sc = float('nan')                                         # distance to serving_cell
        ue_tp = float('nan')                                        # UE throughput ('fundamental')
        sc_power_dBm = cell.get_power_dBm()                         # current UE serving_cell transmit power
        sc_power_watts = from_dB(sc_power_dBm) / 1e3
        sc_rsrp = float('nan')                                      # current UE rsrp from serving_cell
        neigh1_rsrp = float('nan')                                  # current UE neighbouring cell 1 rsrp
        neigh2_rsrp = float('nan')                                  # current UE neighbouring cell 2 rsrp
        noise = float('nan')                                        # current UE thermal noise
        sinr = float('nan')                                         # current UE sinr from serving_cell
        cqi = float('nan')                                          # current UE cqi from serving_cell
        mcs = float('nan')
        cell_eff_bandwidth = float('nan')
        cell_avg_se = float('nan')
        cell_ref_capacity = float('nan')
        cell_load = float('nan')# current UE mcs for serving_cell
        cell_tp = cell.get_cell_throughput()                        # current UE serving_cell throughput
        cell_power_kW = cell_energy_model.get_cell_power_watts(tm) / 1e3    # current UE serving_cell power consumption
        cell_ee = ((cell_tp * 1e6) / (cell_power_kW * 1e3)) * 1e6   # current UE serving_cell energy efficiency
        cell_se = (cell_tp * 1e6) / (cell.bw_MHz * 1e6)     # current UE serving_cell spectral efficiency

        # Get the above as a list
        data_list = [seed, tm, sc_id, sc_sleep_mode, ue_id, d2sc, ue_tp, sc_power_dBm, sc_power_watts, sc_rsrp, 
                     neigh1_rsrp, neigh2_rsrp, noise, sinr, cqi, mcs, cell_eff_bandwidth, cell_avg_se, cell_ref_capacity, cell_load, cell_tp, cell_power_kW, 
                     cell_ee, cell_se]

        # convert ndarrays to str or float
        for i, j in enumerate(data_list):
            if type(j) is np.ndarray:
                data_list[i] = float(j)

        # Write above to `data` list
        data.append(data_list)
        return data


    def get_data(self):
        # Create an empty list to store generated data
        all_data = []
        # Keep a list of column names to track
        columns = ["seed", "time", "serving_cell_id", "serving_cell_sleep_mode", "ue_id",
            "distance_to_cell(m)", "ue_throughput(Mb/s)", "sc_power(dBm)", "sc_power(watts)","sc_rsrp(dBm)", 
            "neighbour1_rsrp(dBm)", "neighbour2_rsrp(dBm)", "noise_power(dBm)", "sinr(dB)", 
            "cqi", "mcs", "cell_eff_bandwidth(MHz)", "cell_avg_ se(bits/Hz)", "cell_ref_capacity_OH(Mb/s)", "cell_load()", "cell_rr_capacity(Mb/s)", "cell_power(kW)", "cell_ee(bits/J)", 
            "cell_se(bits/Hz)"]
        for cell in self.sim.cells:
            if len(cell.attached) != 0:
                # Get data for cells with attached UEs
                all_data.append(self.get_cell_data_attached_UEs(cell))
            if len(cell.attached) == 0:
                # Get data for cells with no attached UEs
                all_data.append(self.get_cell_data_no_UEs(cell))
        # Flatten the list
        all_data = [item for sublist in all_data for item in sublist]
        # Return the column names and data
        return columns, all_data

    def add_to_dataframe(self, col_labels, new_data, ignore_index):
        if self.dataframe is None:
            self.dataframe = pd.DataFrame(new_data, columns=col_labels)
        else:
            new_data_df = pd.DataFrame(data=new_data, columns=col_labels)
            self.dataframe = pd.concat(
                [self.dataframe, new_data_df], verify_integrity=True, ignore_index=ignore_index)


    def run_routine(self, ignore_index=True):
        col_labels, new_data = self.get_data()
        self.add_to_dataframe(col_labels=col_labels,
                              new_data=new_data, ignore_index=ignore_index)

    def loop(self):
        """Main loop for logger."""
        while True:
            # Don't capture t=0
            if self.sim.env.now == 0:
                yield self.sim.wait(self.logging_interval)
            self.run_routine()
            print(f'logger time={self.sim.env.now}')
            yield self.sim.wait(self.logging_interval)


    def finalize(self):
        '''
        Function called at end of simulation, to implement any required finalization actions.
        '''

        # print(f'Finalize time={self.sim.env.now}')

        # Run routine for final time step
        self.run_routine(ignore_index=True)

        # Create a copy of the final DataFrame
        df = self.dataframe.copy()

        # Sort the data by time, UE_id then cell_id
        df.sort_values(['time','ue_id', 'serving_cell_id'], ascending=[True, True, True])
        df1 = df.copy()

        # Convert all NaN and -np.inf values to 0.0
        # df1 = df1.replace([np.inf, -np.inf], np.nan)
        # df1 = df1.fillna(0.0)

        # Find empty values in the dataframe and replace with NaN
        df1 = df1.replace(r'^\s*$', np.nan, regex=True)

        # Print df to screen
        # print(df1)

        # (DEBUGGING TOOL) 
        # Print a view of the type of value in each position
        # --------------------------------------------------
        # df_value_type = df.applymap(lambda x: type(x).__name__)
        # print(df_value_type)

        # Write the MyLogger dataframe to TSV file
        df1.to_csv(self.logfile_path, sep="\t", index=False, mode='w')

class MyLoggerWithEXP3(MyLogger):
    """
    Extended logger class with comprehensive EXP3 metrics tracking.
    Saves detailed metrics for each seed experiment.
    """
    
    def __init__(self, sim, exp3_scenario=None, cell_energy_models=None, seed=None, *args, **kwargs):
        super().__init__(sim, cell_energy_models=cell_energy_models, *args, **kwargs)
        self.exp3_scenario = exp3_scenario
        self.seed = seed if seed is not None else sim.seed
        self.exp3_metrics = []
        self.episode_metrics = []
        self.convergence_metrics = []
        
        # Extract seed number from logfile path if not provided
        if self.seed is None:
            # Try to extract seed from filename like "exp3_s1_p24_0.tsv"
            import re
            match = re.search(r'_s(\d+)_', self.logfile_path)
            if match:
                self.seed = int(match.group(1))
            else:
                self.seed = 1  # default
        
        # Create metrics file paths - use replace to maintain existing file naming pattern
        self.exp3_metrics_path = self.logfile_path.replace('.tsv', '_exp3_metrics.tsv')
        self.episode_metrics_path = self.logfile_path.replace('.tsv', '_episode_metrics.tsv')
        self.convergence_path = self.logfile_path.replace('.tsv', '_convergence_metrics.tsv')
    
    def get_exp3_detailed_data(self):
        """Collect detailed EXP3 algorithm metrics without regret."""
        if self.exp3_scenario is None:
            return None
        
        stats = self.exp3_scenario.get_extended_statistics()
        
        # Core metrics (removed regret-related fields)
        metrics = {
            'time': self.sim.env.now,
            'seed': self.seed,
            'episode': stats['episode_count'],
            'current_arm': stats['current_arm'],
            'cells_off': str(stats['current_cells_off']),
            'cumulative_reward': stats['cumulative_reward'],
            'avg_reward': stats['avg_reward'],
            'reward_variance': stats['reward_variance'],
            'reward_std': stats['reward_std'],
            'avg_efficiency': stats.get('avg_efficiency', 0),
            'efficiency_std': stats.get('efficiency_std', 0),
            'avg_throughput': stats.get('avg_throughput', 0),
            'throughput_std': stats.get('throughput_std', 0),
            'avg_power': stats.get('avg_power', 0),
            'power_std': stats.get('power_std', 0),
            'avg_cell_throughput': stats.get('avg_cell_throughput', 0),
            'cell_throughput_std': stats.get('cell_throughput_std', 0),
            'energy_savings_vs_all_on': stats.get('energy_savings_vs_all_on', 0),
            'energy_savings_vs_random': stats.get('energy_savings_vs_random', 0),
            'avg_switching_cost': stats.get('avg_switching_cost', 0),
            'total_switching_cost': stats.get('total_switching_cost', 0),
            'convergence_episode': stats.get('convergence_episode', -1)
        }
        
        # Add probability distribution for top arms
        top_k = min(10, len(stats['probabilities']))
        top_arms_idx = np.argsort(stats['probabilities'])[-top_k:][::-1]
        
        for i, arm_idx in enumerate(top_arms_idx):
            metrics[f'top{i+1}_arm'] = arm_idx
            metrics[f'top{i+1}_prob'] = stats['probabilities'][arm_idx]
            metrics[f'top{i+1}_selections'] = stats['arm_selections'][arm_idx]
            
            # Get cells for this arm
            cells_off = self.exp3_scenario._get_arm_combination(arm_idx)
            metrics[f'top{i+1}_cells_off'] = str(cells_off)
        
        return metrics
    
    def get_episode_metrics(self):
        """Get per-episode metrics for detailed analysis."""
        if not self.exp3_scenario or not self.exp3_scenario.reward_history:
            return None
        
        episode_data = {
            'seed': self.seed,
            'episode': self.exp3_scenario.episode_count,
            'time': self.sim.env.now,
            'selected_arm': self.exp3_scenario.current_arm,
            'cells_off': str(self.exp3_scenario.current_cells_off),
            'instant_reward': self.exp3_scenario.reward_history[-1] if self.exp3_scenario.reward_history else 0,
            'instant_efficiency': self.exp3_scenario.efficiency_history[-1] if self.exp3_scenario.efficiency_history else 0,
            'instant_throughput': self.exp3_scenario.throughput_history[-1] if self.exp3_scenario.throughput_history else 0,
            'instant_power': self.exp3_scenario.power_history[-1] if self.exp3_scenario.power_history else 0,
            'switching_cost': self.exp3_scenario.switching_cost_history[-1] if self.exp3_scenario.switching_cost_history else 0,
        }
        
        # Add probability changes
        if self.exp3_scenario.probability_change_history:
            episode_data['probability_change'] = self.exp3_scenario.probability_change_history[-1]
        
        return episode_data
    
    def run_routine(self, ignore_index=True):
        """Extended routine to include comprehensive EXP3 metrics."""
        # Run base routine
        super().run_routine(ignore_index)
        
        # Collect detailed EXP3 metrics
        if self.exp3_scenario:
            # Main metrics
            exp3_data = self.get_exp3_detailed_data()
            if exp3_data:
                self.exp3_metrics.append(exp3_data)
            
            # Episode metrics (only when episode changes)
            if (len(self.episode_metrics) == 0 or 
                (self.episode_metrics and 
                 self.episode_metrics[-1]['episode'] < self.exp3_scenario.episode_count)):
                episode_data = self.get_episode_metrics()
                if episode_data:
                    self.episode_metrics.append(episode_data)
    
    def save_convergence_analysis(self):
        """Analyze and save convergence metrics."""
        if not self.exp3_scenario:
            return
        
        convergence_data = []
        
        # Analyze probability stability over time
        prob_history = self.exp3_scenario.probability_history
        if prob_history and len(prob_history) > 10:
            window_size = 10
            for i in range(window_size, len(prob_history)):
                window = prob_history[i-window_size:i]
                
                # Calculate variance in probabilities over window
                prob_variance = np.var([p for probs in window for p in probs])
                
                # Find dominant arm
                avg_probs = np.mean(window, axis=0)
                dominant_arm = np.argmax(avg_probs)
                dominant_prob = avg_probs[dominant_arm]
                
                convergence_data.append({
                    'seed': self.seed,
                    'episode': i,
                    'prob_variance': prob_variance,
                    'dominant_arm': dominant_arm,
                    'dominant_prob': dominant_prob,
                    'convergence_detected': self.exp3_scenario.convergence_episode is not None and i >= self.exp3_scenario.convergence_episode
                })
        
        if convergence_data:
            df = pd.DataFrame(convergence_data)
            df.to_csv(self.convergence_path, sep='\t', index=False)
            print(f"Convergence metrics saved to: {self.convergence_path}")
    
    def finalize(self):
        """Finalization without regret summary."""
        # Run base finalization
        super().finalize()
        
        # Save main EXP3 metrics
        if self.exp3_metrics:
            exp3_df = pd.DataFrame(self.exp3_metrics)
            exp3_df.to_csv(self.exp3_metrics_path, sep='\t', index=False)
            print(f"EXP3 metrics saved to: {self.exp3_metrics_path}")
        
        # Save episode metrics
        if self.episode_metrics:
            episode_df = pd.DataFrame(self.episode_metrics)
            episode_df.to_csv(self.episode_metrics_path, sep='\t', index=False)
            print(f"Episode metrics saved to: {self.episode_metrics_path}")
        
        # Save convergence analysis
        self.save_convergence_analysis()
        
        # Print comprehensive summary (without regret)
        if self.exp3_scenario:
            print("\n" + "="*80)
            print(f"SEED {self.seed} - EXP3 SIMULATION SUMMARY")
            print("="*80)
            
            stats = self.exp3_scenario.get_extended_statistics()
            
            print(f"Total episodes: {stats['episode_count']}")
            print(f"Convergence: {'Episode ' + str(stats['convergence_episode']) if stats['convergence_episode'] else 'Not detected'}")
            print("\nPerformance Metrics:")
            print(f"  Cumulative Reward: {stats['cumulative_reward']:.4f}")
            print(f"  Average Reward: {stats['avg_reward']:.4f}")
            print(f"  Reward Std Dev: {stats['reward_std']:.4f}")
            
            print("\nNetwork Metrics:")
            print(f"  Avg Efficiency: {stats.get('avg_efficiency', 0):.4f} ± {stats.get('efficiency_std', 0):.4f} bits/J")
            print(f"  Avg Throughput: {stats.get('avg_throughput', 0):.2f} ± {stats.get('throughput_std', 0):.2f} Mb/s")
            print(f"  Avg Power: {stats.get('avg_power', 0):.2f} ± {stats.get('power_std', 0):.2f} kW")
            print(f"  Avg Cell Throughput: {stats.get('avg_cell_throughput', 0):.2f} ± {stats.get('cell_throughput_std', 0):.2f} Mb/s")
            
            print("\nEnergy Savings:")
            print(f"  vs All-On: {stats.get('energy_savings_vs_all_on', 0):.1f}%")
            print(f"  vs Random: {stats.get('energy_savings_vs_random', 0):.1f}%")
            
            print("\nSwitching Cost:")
            print(f"  Average: {stats.get('avg_switching_cost', 0):.4f}")
            print(f"  Total: {stats.get('total_switching_cost', 0):.2f}")
            
            print("\nTop 5 Selected Arms:")
            top_arms = np.argsort(stats['arm_selections'])[-5:][::-1]
            for i, arm in enumerate(top_arms):
                count = stats['arm_selections'][arm]
                prob = stats['probabilities'][arm]
                cells = self.exp3_scenario._get_arm_combination(arm)
                print(f"  {i+1}. Arm {arm}: {count} selections ({prob:.3f} prob) - Cells off: {cells}")
            
            print("="*80 + "\n")
# END MyLogger class


def calculate_ofdm_symbol_time(channel_bandwidth, mu):
    """
    Calculate the time for one OFDM symbol in 5G NR from the channel bandwidth, and subcarrier spacing.

    Parameters:
    channel_bandwidth (float): channel bandwidth in MHz
    mu (int): subcarrier spacing exponent (0, 1, 2, 3)

    Returns:
    float: the time for one OFDM symbol in microseconds
    """
    subcarrier_spacing = get_subcarrier_spacing(mu)
    num_subcarriers = channel_bandwidth * 1e6 // subcarrier_spacing
    symbol_duration = 1 / (num_subcarriers * subcarrier_spacing)
    return symbol_duration * 1e6


def get_subcarrier_spacing(mu):
    """
    Get the subcarrier spacing in Hz for a given mu.

    Parameters:
    mu (int): subcarrier spacing exponent (0, 1, 2, 3)

    Returns:
    float: the subcarrier spacing in Hz
    """
    if mu == 0:
        return 15e3
    elif mu == 1:
        return 30e3
    elif mu == 2:
        return 60e3
    elif mu == 3:
        return 120e3
    else:
        raise ValueError("Invalid mu value. Must be 0, 1, 2, or 3.")

def calculate_symbols_per_second(ofdm_symbol_duration):
    """
    Calculate the number of OFDM symbols in 1 second given the OFDM symbol duration.

    Parameters:
    ofdm_symbol_duration (float): the duration of one OFDM symbol in seconds

    Returns:
    float: the number of OFDM symbols per second
    """
    return 1 / ofdm_symbol_duration

def random_time_interval(sim, min_time_us, max_time_us):
    """
    Returns a random time interval between min_time_us and max_time_us (inclusive)
    based on the provided rng_seed value.

    Args:
        rng_seed (int): Seed value for the random number generator.
        min_time_us (float): Minimum time interval in microseconds.
        max_time_us (float): Maximum time interval in microseconds.

    Returns:
        float: Random time interval in seconds.
    """
    rng = sim.rng
    time_interval_us = rng.uniform(min_time_us, max_time_us)  # Uniform distribution between min and max
    return time_interval_us 

class PointGenerator:
    """
    A class used to generate random points within a circular region

    ...

    Attributes
    ----------
    rng : random number generator
        an instance of numpy random generator

    Methods
    -------
    generate_points(expected_pts, sim_radius, cell_centre_points=None, exclusion_radius=None)
        Generate a number of random points within a specified radius
    """

    def __init__(self, rng):
        """
        Constructs all the necessary attributes for the PointGenerator object.

        Parameters
        ----------
        rng : random number generator
            an instance of numpy random generator
        """

        self.rng = rng
    
    def generate_points(self, expected_pts, sim_radius, cell_centre_points=None, exclusion_radius=None):
            """
            Generate a number of random points within a specified radius using a variant of a Homogeneous Poisson Point Process (HPPP). If centre points and exclusion radius are provided, points within the exclusion radius of any centre point are removed. This exclusion feature makes the point process a kind of "restricted" Poisson point process or a simplified version of Matérn type III hard-core process.

            Parameters
            ----------
            expected_pts : int
                The expected number of points to generate
            sim_radius : float
                The radius within which the points will be generated
            cell_centre_points : array-like, optional
                The coordinates of centre points around which to exclude generated points (default is None)
            exclusion_radius : float, optional
                The radius around each centre point within which points will be excluded (default is None)

            Returns
            -------
            points : array-like
                The generated points in the format of a 2D numpy array where each row represents a point
                and the columns represent the x and y coordinates respectively
            """

            sim_rng = self.rng
            areaTotal = np.pi * sim_radius ** 2
            lambda0 = expected_pts / areaTotal
            points = np.empty((0, 2))

            while points.shape[0] < expected_pts:
                numbPoints = sim_rng.poisson(lambda0 * areaTotal)
                theta = 2 * np.pi * sim_rng.uniform(0, 1, numbPoints)
                rho = sim_radius * np.sqrt(sim_rng.uniform(0, 1, numbPoints))
                xx = rho * np.cos(theta)
                yy = rho * np.sin(theta)

                if cell_centre_points is not None and exclusion_radius is not None:
                    dists = np.linalg.norm(cell_centre_points - np.array([xx, yy]), axis=1)
                    indices = np.where(dists > exclusion_radius)[0]
                    xx = xx[indices]
                    yy = yy[indices]

                points = np.vstack((points, np.column_stack((xx, yy))))

            points = points[:expected_pts]
            return points


def generate_ppp_points(sim, type=None, expected_pts=100, sim_radius=500.0, cell_centre_points=None, exclusion_radius=None):

    sim_rng = sim.rng

    sim_radius = sim_radius
    xx0 = 0
    yy0 = 0
    areaTotal = np.pi * sim_radius ** 2

    lambda0 = expected_pts / areaTotal

    points = np.empty((0, 2))

    loop_count = 0
    remove_count = 0
    while points.shape[0] < expected_pts:
        loop_count += 1

        numbPoints = sim_rng.poisson(lambda0 * areaTotal)
        theta = 2 * np.pi * sim_rng.uniform(0, 1, numbPoints)
        rho = sim_radius * np.sqrt(sim_rng.uniform(0, 1, numbPoints))

        xx = rho * np.cos(theta)
        yy = rho * np.sin(theta)

        xx = xx + xx0
        yy = yy + yy0
        
        if cell_centre_points is not None and exclusion_radius is not None:
            dists = np.linalg.norm(cell_centre_points - np.array([xx, yy]), axis=1)
            indices = np.where(dists > exclusion_radius)[0]
            remove_count += (numbPoints - len(indices))
            xx = xx[indices]
            yy = yy[indices]
        
        points = np.vstack((points, np.column_stack((xx, yy))))

    points = points[:expected_pts]
    
    # kiss_debugger.debug(f"The while loop ran {loop_count} times.")
    # kiss_debugger.debug(f"{remove_count} points were removed from the exclusion zone.")
    
    return points



def hex_grid_setup(origin: tuple = (0, 0), isd: float = 500.0, sim_radius: float = 1000.0, plot: bool = False, watermark: bool = False):
    """
    Create a hexagonal grid and plot it with a dashed circle.

    Parameters
    ----------
    origin : tuple of float, optional
        The center of the simulation area, by default (0, 0)
    isd : float, optional
        The distance between two adjacent hexagons (in meters), by default 500.0
    sim_radius : float, optional
        The radius of the simulation area (in meters), by default 1000.0
    plot : bool, optional
        Whether to plot the hexagonal grid and the dashed circle, by default False
    watermark : bool, optional
        Whether to add watermark numbers to each hexagon, by default False

    Returns
    -------
    hexgrid_xy : numpy.ndarray
        A 2D array containing the x and y coordinates of the hexagonal grid.
    fig : matplotlib.figure.Figure or None
        The matplotlib Figure object containing the plot, or None if plot=False.

    Notes
    -----
    The hexagonal grid is created using the `create_hex_grid` function, and the
    resulting grid is plotted with matplotlib. The grid is centered at the origin
    and is scaled so that its diameter is `3 * isd + 500`. A dashed circle is also
    plotted with radius `sim_radius`.

    """
    if plot:
        fig, ax = plt.subplots()
    else:
        fig = None
        ax = None
    
    from hexalattice.hexalattice import create_hex_grid

    hexgrid_xy, _ = create_hex_grid(nx=5,
                                    ny=5,
                                    min_diam=isd,
                                    crop_circ=sim_radius,
                                    align_to_origin=True,
                                    edge_color=[0.75, 0.75, 0.75],
                                    h_ax=ax,
                                    do_plot=True)

    hexgrid_x = hexgrid_xy[:, 0]
    hexgrid_y = hexgrid_xy[:, 1]

    if plot:
        circle_dashed = plt.Circle(
            origin, sim_radius, fill=False, linestyle='--', color='r')

        ax.add_patch(circle_dashed)
        ax.scatter(hexgrid_x, hexgrid_y, marker='2')
        
        if watermark:
            # Add watermark numbers to each hexagon
            for i, (x, y) in enumerate(hexgrid_xy):
                ax.text(x + isd / 3 - 0.3e3, y - isd /3 + 0.33e3, str(i), fontsize=10, ha='center', va='center', color='purple', alpha=0.63, weight='bold')
        
        # Factor to set the x,y-axis limits relative to the isd value.
        ax_scaling = 3 * isd + 500
        ax.set_xlim([-ax_scaling, ax_scaling])
        ax.set_ylim([-ax_scaling, ax_scaling])
        ax.set_aspect('equal')
    
        return hexgrid_xy, fig
    else:
        if watermark:
            # Add watermark numbers to each hexagon
            for i, (x, y) in enumerate(hexgrid_xy):
                plt.text(x + isd / 3 - 0.3e3, y - isd /3 + 0.33e3, str(i), fontsize=10, ha='center', va='center', color='purple', alpha=0.63, weight='bold')
        
        return hexgrid_xy, None
    


def fig_timestamp(fig, author='', fontsize=6, color='gray', alpha=0.7, rotation=0, prespace='  '):
    """
    Add a timestamp to a matplotlib figure.

    Parameters
    ----------
    fig : matplotlib Figure
        The figure to add the timestamp to.
    author : str, optional
        The author's name to include in the timestamp. Default is ''.
    fontsize : int, optional
        The font size of the timestamp. Default is 6.
    color : str, optional
        The color of the timestamp. Default is 'gray'.
    alpha : float, optional
        The transparency of the timestamp. Default is 0.7.
    rotation : float, optional
        The rotation angle of the timestamp (in degrees). Default is 0.
    prespace : str, optional
        The whitespace to prepend to the timestamp string. Default is '  '.

    Returns
    -------
    None
    """
    date = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    fig.text(
        0.01, 0.005, f"{prespace}{author} {date}",
        ha='left', va='bottom', fontsize=fontsize, color=color,
        rotation=rotation,
        transform=fig.transFigure, alpha=alpha)


def plot_ues_fig(sim, ue_ids_start=None, ue_ids_end=None, show_labels=True, labels_start=None, labels_end=None):

    # FIXME - Add in switch to plot ues in a different color if they are in a cell
    if ue_ids_start is None and ue_ids_end is None:
            ue_ids_start = 0
            ue_ids_end = len(sim.UEs)
    if ue_ids_start is not None and ue_ids_end is None:
        ue_ids_end = ue_ids_start
    if ue_ids_start is None and ue_ids_end is not None:
        ue_ids_start = ue_ids_end
    
    if ue_ids_start == ue_ids_end:
        ue_ids = ue_ids_start
    elif ue_ids_start < ue_ids_end:
        ue_ids = list(range(ue_ids_start, ue_ids_end))
    elif ue_ids_start > ue_ids_end:
        # Throw an error
        raise ValueError("ue_ids_start must be less than ue_ids_end")
    
    if isinstance(ue_ids, int):
        ue_objs_list = [sim.UEs[ue_ids]]
    if isinstance(ue_ids, list) or isinstance(ue_ids, np.ndarray):
        ue_objs_list = [sim.UEs[i] for i in ue_ids]
    else:
        ue_objs_list = [sim.UEs[i] for i in ue_ids]
    ue_x_list = [ue.xyz[0] for ue in ue_objs_list]
    ue_y_list = [ue.xyz[1] for ue in ue_objs_list]
    ue_xy_list = [ue.xyz[:2] for ue in ue_objs_list]
    plt.scatter(x=ue_x_list, y=ue_y_list, color='red', alpha= 0.3, s=2.0)
    if show_labels:
        if labels_start is None and labels_end is None:
            labels_start = 0
            labels_end = len(ue_ids)
        if labels_end is None:
            labels_end = labels_start + 1 # Only label the first labels_start ue_ids
        for i in range(labels_start, labels_end):
            plt.annotate(text=str(ue_ids[i]), xy=ue_xy_list[i], xytext=(3,-2), textcoords='offset points',
            fontsize=8, color='red', bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3),)


def main(config_dict):
    seed = config_dict["seed"]
    sim_show_params = config_dict["sim_show_params"]
    isd = config_dict["isd"]
    sim_radius = config_dict["sim_radius"]
    power_dBm = config_dict["constant_cell_power_dBm"]
    variable_cell_power_dBm = config_dict["variable_cell_power_dBm"]
    n_variable_power_cells = config_dict["n_variable_power_cells"]
    variable_power_target_cells_list = config_dict["variable_power_target_cells_list"]
    nues = config_dict["nues"]
    until = config_dict["until"]
    base_interval = config_dict["base_interval"]
    h_BS = config_dict["h_BS"]
    h_UT = config_dict["h_UT"]
    ue_noise_power_dBm = config_dict["ue_noise_power_dBm"]
    scenario_profile = config_dict["scenario_profile"]
    scenario_n_cells = config_dict["scenario_n_cells"]
    scenario_delay = config_dict["scenario_delay"]
    SetCellSleep_bedtime_stories = config_dict["scenario_SetCellSleep_params"]["SetCellSleep_bedtime_stories"]
    mme_cqi_limit = config_dict["mme_cqi_limit"]
    mme_strategy = config_dict["mme_strategy"]
    mme_anti_pingpong = config_dict["mme_anti_pingpong"]
    mme_verbosity = config_dict["mme_verbosity"]
    plotting = config_dict["plotting"]
    plot_cell_id_watermark = config_dict["plot_cell_id_watermark"]
    plot_ues = config_dict["plot_ues"]
    plot_ues_start = config_dict["plot_ues_start"]
    plot_ues_end = config_dict["plot_ues_end"]
    plot_ues_show_labels = config_dict["plot_ues_show_labels"]
    plot_ues_labels_start = config_dict["plot_ues_show_labels_start"]
    plot_ues_labels_end = config_dict["plot_ues_show_labels_end"]
    plot_author = config_dict.get("plot_author")
    mcs_table_number = config_dict["mcs_table_number"]

    # EXP3 parameters (get from config if available)
    exp3_params = config_dict.get("scenario_exp3_params", {})
    
    # Create a simulator object
    sim = Simv2(rng_seed=seed, show_params=sim_show_params)
    sim.seed = seed

    # Create a log file path
    data_output_logfile_path = create_logfile_path(config_dict)

    # Create the 19-cell hex-grid and place Cell instance at the centre
    sim_hexgrid_centres, hexgrid_plot = hex_grid_setup(isd=isd, sim_radius=sim_radius, plot=plotting, watermark=plot_cell_id_watermark)
    for centre in sim_hexgrid_centres[:]:
        x, y = centre
        z = h_BS
        # Create the cell
        sim.make_cellv2(interval=base_interval*0.1,xyz=[x, y, z], power_dBm=power_dBm)  # interval=base_interval*0.1 (scaled down to see if rsrp dictionaries get cleared)

    # Create a dictionary of cell-specific energy models
    cell_energy_models_dict = {}
    for cell in sim.cells:
        cell_energy_models_dict[cell.i] = (CellEnergyModel(cell))
        cell.set_f_callback(cell_energy_models_dict[cell.i].f_callback(cell))

    # Initialize EXP3 scenario if selected
    exp3_scenario = None
    
    # Choose logger based on scenario
    if scenario_profile == "exp3_cell_onoff":
        # Create EXP3 scenario
        exp3_scenario = EXP3CellOnOff(
            sim=sim,
            k_cells_to_off=exp3_params.get("k_cells_to_off", 3),
            n_total_cells=exp3_params.get("n_total_cells"),
            interval=base_interval,
            delay=scenario_delay,
            gamma=exp3_params.get("gamma", 0.1),
            warm_up_episodes=exp3_params.get("warm_up_episodes", 100),
            enable_warm_up=exp3_params.get("enable_warm_up", True),
            min_selection_guarantee=exp3_params.get("min_selection_guarantee", True),
            normalization_window=exp3_params.get("normalization_window", 50),
            event_threshold=exp3_params.get("event_threshold", 10),
            cell_energy_models=cell_energy_models_dict
        )
        
        # Use extended logger with EXP3 metrics
        custom_logger = MyLoggerWithEXP3(
            sim, 
            exp3_scenario=exp3_scenario,
            logging_interval=base_interval,
            cell_energy_models=cell_energy_models_dict,
            logfile_path=".".join([data_output_logfile_path, "tsv"])
        )
    else:
        # Use standard logger for other scenarios
        custom_logger = MyLogger(
            sim,
            logging_interval=base_interval,
            cell_energy_models=cell_energy_models_dict,
            logfile_path=".".join([data_output_logfile_path, "tsv"])
        )
    
    sim.add_logger(custom_logger)


    # Define scenario options for simulation
    reduce_random_cell_power = ChangeCellPower(
        sim, 
        n_cells=n_variable_power_cells,
        delay=scenario_delay,
        new_power=variable_cell_power_dBm, 
        interval=base_interval
        )
    
    reduce_centre_cell_power = ChangeCellPower(
        sim,
        delay=scenario_delay,
        cells=variable_power_target_cells_list,
        new_power=variable_cell_power_dBm, 
        interval=base_interval
        )
    
    change_outer_ring_power = ChangeCellPower(
        sim, 
        delay=scenario_delay,
        cells=variable_power_target_cells_list, 
        new_power=variable_cell_power_dBm, 
        interval=base_interval
        )
    
    change_inner_ring_power = ChangeCellPower(
        sim, 
        delay=scenario_delay, 
        cells=variable_power_target_cells_list,
        new_power=variable_cell_power_dBm, 
        interval=base_interval
        )
    
    switch_n_cells_off = SwitchNCellsOff(
        sim, 
        delay=scenario_delay, 
        interval=base_interval,
        n_cells=scenario_n_cells
        )
    
    set_cell_sleep = SetCellSleep(
        sim, 
        interval=base_interval, 
        time_cell_sleep_level_duration=SetCellSleep_bedtime_stories
        )
    
    # Activate scenarios
    if scenario_profile == "reduce_random_cell_power":
        sim.add_scenario(scenario=reduce_random_cell_power)
    elif scenario_profile == "reduce_centre_cell_power":
        sim.add_scenario(scenario=reduce_centre_cell_power)
    elif scenario_profile == "change_outer_ring_power":
        sim.add_scenario(scenario=change_outer_ring_power)
    elif scenario_profile == "change_inner_ring_power":
        sim.add_scenario(scenario=change_inner_ring_power)
    elif scenario_profile == "switch_n_cells_off":
        sim.add_scenario(scenario=switch_n_cells_off)
    elif scenario_profile == "set_cell_sleep":
        sim.add_scenario(scenario=set_cell_sleep)
    elif scenario_profile == "exp3_cell_onoff":
        # Add EXP3 scenario
        sim.add_scenario(scenario=exp3_scenario)
        print(f"\n{'='*60}")
        print(f"EXP3 Cell On/Off Scenario Activated")
        print(f"  - Cells to turn off: {exp3_params.get('k_cells_to_off', 3)}")
        print(f"  - Gamma (exploration): {exp3_params.get('gamma', 0.1)}")
        print(f"  - Warm-up episodes: {exp3_params.get('warm_up_episodes', 100)}")
        print(f"{'='*60}\n")
    elif scenario_profile == "no_scenarios":
        pass
    else:
        raise ValueError("Scenario profile not recognised")
    
    # Add MME for handovers
    default_mme = AMFv1(sim, 
                        cqi_limit=mme_cqi_limit, 
                        interval=base_interval,
                        strategy=mme_strategy, 
                        anti_pingpong=mme_anti_pingpong,
                        verbosity=mme_verbosity)
    sim.add_MME(mme=default_mme)


    # Create instance of UMa-NLOS pathloss model
    pl_uma_nlos = UMa_pathloss(LOS=False)

    
    point_generator = PointGenerator(sim.rng)
    ue_positions = point_generator.generate_points(expected_pts=nues, sim_radius=sim_radius)
    for position in ue_positions:
        x, y = position
        ue_xyz = x, y, h_UT
        ue = sim.make_UEv2(xyz=ue_xyz, reporting_interval=base_interval, pathloss_model=pl_uma_nlos, verbosity=0)
        ue.attach_to_strongest_cell_simple_pathloss_model()



    # Generate UE positions using PPP
    ue_ppp = generate_ppp_points(sim=sim, 
                                 expected_pts=nues, 
                                 sim_radius=sim_radius,)
    
    for i in ue_ppp:
        x, y = i
        ue_xyz = x, y, h_UT
        sim.make_UEv2(xyz=ue_xyz,reporting_interval=base_interval,pathloss_model=pl_uma_nlos, verbosity=0).attach_to_strongest_cell_simple_pathloss_model()
    
    # Change the noise_power_dBm
    for ue in sim.UEs:
        ue.noise_power_dBm=ue_noise_power_dBm


    # Plot UEs if desired
    if plot_ues:
        plot_ues_fig(sim=sim, ue_ids_start=plot_ues_start, ue_ids_end=plot_ues_end ,show_labels=plot_ues_show_labels, labels_start=plot_ues_labels_start, labels_end=plot_ues_labels_end)
        fig_timestamp(fig=hexgrid_plot, author=plot_author)
        file_name = os.path.splitext(data_output_logfile_path)[0]
        fig_outfile_path = file_name + '.png'
        plt.savefig(fig_outfile_path)


    # Run simulator
    sim.run(until=until)

    # Post-simulation summary for EXP3
    if scenario_profile == "exp3_cell_onoff" and exp3_scenario:
        print("\n" + "="*60)
        print("EXP3 SIMULATION COMPLETED")
        print("="*60)
        stats = exp3_scenario.get_extended_statistics()
        print(f"Total episodes: {stats['episode_count']}")
        print(f"Cumulative reward: {stats['cumulative_reward']:.4f}")
        print(f"Average efficiency: {stats.get('avg_efficiency', 0):.4f} bits/J")
        print(f"Average throughput: {stats.get('avg_throughput', 0):.2f} Mb/s")
        print(f"Average power: {stats.get('avg_power', 0):.2f} kW")
        print(f"Energy savings (vs all-on): {stats.get('energy_savings_vs_all_on', 0):.1f}%")
        print(f"Convergence: Episode {stats['convergence_episode']}" if stats['convergence_episode'] else "Convergence: Not detected")
        print("="*60 + "\n")
    
    # Generate plots if requested
    if plotting:
        # Your existing plotting code
        pass
    
    print(f"Simulation completed. Results saved to: {data_output_logfile_path}")
    
    return sim



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run kiss.py against a specified config value.')

    parser.add_argument(
        '-c',
        '--config-file',
        type=str,
        required=True,
        default='KISS/_test/data/input/kiss/test_kiss.json'
        )
    
    args = parser.parse_args()

    # Load the config file
    config_file = args.config_file
    with open(config_file, "r") as f:
        config = json.load(f)

    # Import plt if plotting is enabled
    if config["plotting"]:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Matplotlib not installed. Plotting disabled.")
            config["plotting"] = False

    # Import kiss_phy_data_procedures if mcs_table_number is set
    if config["mcs_table_number"] is not None:
        try:
            import kiss_phy_data_procedures
        except ImportError:
            print("kiss_phy_data_procedures not installed. mcs_table_number disabled.")
            config["mcs_table_number"] = None
    
    # If line_profiler is enabled, import line_profiler
    if config["line_profiler"]:
        try:
            import line_profiler
            if callable(line_profiler):
                # Method to profile a function
                def profile_method(method):
                    def wrapper(*args, **kwargs):
                        lp = line_profiler.LineProfiler()
                        lp_wrapper = lp(method)
                        lp_wrapper(*args, **kwargs)
                        lp.print_stats()
                    return wrapper
                # Use `@profile_method` to profile a function
        except ImportError:
            print("line_profiler not installed. line_profiler disabled.")
            config["line_profiler"] = False

    # Start timer
    start_time = time()

    # Run main
    main(config_dict=config)

    # End timer
    end_time = time()


