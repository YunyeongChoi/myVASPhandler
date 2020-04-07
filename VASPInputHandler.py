#!/usr/bin/env python
# coding: utf-8

from pymatgen import MPRester, Composition
from pymatgen.io.vasp.sets import MPRelaxSet
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core import Structure
from pymatgen.io.vasp.inputs import *
from pymatgen.io.vasp import Vasprun
from pymatgen.io.cif import *
from pymatgen.analysis.ewald import EwaldSummation
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.entries.computed_entries import ComputedEntry
from pymatgen.entries.compatibility import MaterialsProjectCompatibility
from pymatgen.transformations.standard_transformations import OrderDisorderedStructureTransformation
from scipy.integrate import simps, cumtrapz
from scipy.ndimage.filters import gaussian_filter1d
from shutil import copyfile
from VASPHandyFunctions import *
from CompAnalyzer import CompAnalyzer
import argparse
import gzip
import sys
import os
import re


class VASPSetUp(object):
    
    def __init__(self, dir):
        """
        Args: 
            dir is a path to run VASP calculation
        Returns: 
            dir
        """
        self.dir = dir
        if os.path.exists(self.dir) == False:
            os.mkdir(self.dir)
        
    def incar(self, geometry_opt=False, dos=False, write_file=False, 
              functional='pbe', 
              standard_setting = {'EDIFF': 1e-05,
                                 'ISEMAR': 0,
                                 'SIGMA': 0.01,
                                 'LORBIT': 11,
                                 'ISIF': 3}, 
              addtional_option = {}, MP=False
             ):
        
        fincar = os.path.join(self.dir, 'INCAR')
        user_incar_setting = {}
        
        if MP:
            poscar = self.structure_from_dic()
            s = Structure.from_file(poscar)
            obj = MPRelaxSet(s)
            obj.incar.write_file(fincar)
            params = VASPBasicAnalysis(self.dir).params_from_incar()
            user_incar_setting = params
        
        for i in standard_setting:
            user_incar_setting[i] = standard_setting[i]
        
        if geometry_opt:
            user_incar_setting['IBRION'] = 2
            user_incar_setting['NSW'] = 200
            user_incar_setting['NELM'] = 200
            user_incar_setting['ISYM'] = 0
        else:
            user_incar_setting['IBRION'] = -1
            user_incar_setting['NSW'] = 0
            user_incar_setting['NELM'] = 5000
            user_incar_setting['LWAVE'] = '.TRUE.'
            user_incar_setting['ISYM'] = -1
            user_incar_setting['ICORELEVEL'] = 1
            
        if dos:
            user_incar_setting['NEDOS'] = 2500

        if functional == 'pbe':
            user_incar_setting['GGA'] = 'PE'
        elif functional == 'scan':
            user_incar_setting['METAGGA'] = 'SCAN'
        elif functional == 'rtpss':
            user_incar_setting['METAGGA'] = 'RTPSS'
        elif functional == 'hse':
            user_incar_setting['GGA'] = 'PE'
            user_incar_setting['LHFCALC'] = 'TRUE'
            user_incar_setting['ALGO'] = 'Damped'
            user_incar_setting['TIME'] = 0.4
            user_incar_setting['HFSCREEN'] = 0.2
        else:
            print('are you sure you want that functional?')
            user_incar_setting = np.nan

        if functional in ['scan', 'rtpss']:
            user_incar_setting['LASPH'] = 'TRUE'
            user_incar_setting['ALGO'] = 'All'
            user_incar_setting['ADDGRID'] = 'TRUE'
            user_incar_setting['ISMEAR'] = 0
            
        for i in addtional_option:
            user_incar_setting[i] = addtional_option[i]
            
        if write_file == True:
            with open(fincar, 'w') as f:
                for k in user_incar_setting:
                    f.write(' = '.join([k, str(user_incar_setting[k])])+'\n')
        else:
            return user_incar_setting

    def structure_from_mp(self, formula, spgp, write_file=False, KEY='R0CmAuNPKWzrUo8Z'):
        """
        Args: 
            formula - Chemical formula like NaCr2O4, spgp - Space group
        Returns: 
            structure object match with formula and space group
        """
        count = 0
        spgp_list = []
        poscar = os.path.join(self.dir, 'POSCAR')
        mpr = MPRester(KEY)
        mtr_list = mpr.get_materials_ids(formula)
        for mtr in mtr_list:
            a = SpacegroupAnalyzer(mpr.get_structure_by_material_id(mtr))
            spgp_list.append(a.get_space_group_symbol())
            if a.get_space_group_symbol() == spgp:
                count += 1
                structure = mpr.get_structure_by_material_id(mtr)
        if count > 1:
            print(spgp_list)
            raise ValueError("More than one structure. Check the atoms")
        elif count == 0:
            print(spgp_list)
            raise ValueError("Check the formula and space group again")
        if write_file == True:
            Poscar(structure).write_file(poscar)
        else:
            return structure

    def structure_from_id(self, mpid = 'MPID', write_file=True, KEY='R0CmAuNPKWzrUo8Z'):
        """
        Args: 
            mpid - Materials Project ID, config = [xa, xb, xc]
        Returns: 
            structure object match with Materials Project ID
        """
        poscar = os.path.join(self.dir, 'POSCAR')
        mpr = MPRester(KEY)
        mpid = self.dir.split('/')[-1].split('_')[0]
        structure = mpr.get_structure_by_material_id(mpid)
        if write_file == True:
            Poscar(structure).write_file(poscar)
        else:
            return structure
        
    def structure_from_dic(self, copy_contcar=False):
        """
        Args: 
            copy_contcar(bool) - if True, copies CONTCAR to POSCAR if 
            CONTCAR not empty
        Returns: 
            new structure file from CONTCAR
        """
        poscar = os.path.join(self.dir, 'POSCAR')
        if not os.path.exists(poscar):
            copy_contcar = True
        if not copy_contcar:
            return poscar
        contcar = os.path.join(self.dir, 'CONTCAR')
        if os.path.exists(contcar):
            with open(contcar) as f:
                contents = f.read()
            if '0' in contents:
                copyfile(contcar, poscar)
                
            return poscar
        
    def perturb_poscar(self, perturbation):
        """
        Args:
            perturbation (float) - distance in Ang to randomly perturb ions
        Returns:
            POSCAR with random displacements
        """        
        poscar = self.structure_from_dic()
        s = Structure.from_file(poscar)
        s.perturb(perturbation)
        s.to(fmt='poscar', filename=poscar)
        
        return poscar
    
    def make_supercell(self, config):
        """
        Args:
            config (tuple) - (a, b, c) to expand poscar
        Returns: 
            supercell of POSCAR
        """
        poscar = self.structure_from_dic()
        s = Structure.from_file(poscar)
        s.make_supercell(config)
        s.to(fmt='poscar', filename=poscar)
        
        return poscar
    
    def ordered_els_from_poscar(self, copy_contcar=False):
        """
        Args:
            copy_contcar (bool) - if True, copies CONTCAR to POSCAR if CONTCAR not empty  
        Returns:
            dictionary of {element (str) : number in calculated structure (int)}
        """
        poscar = self.structure_from_dic(copy_contcar)
        with open(poscar) as f:
            count = 0
            for line in f:
                if count <= 5:
                    count += 1
                if count == 6:
                    els = [v for v in line[:-1].split(' ') if v != '']
                    try:
                        num_check = [int(el) for el in els]
                        print('elements not provided in POSCAR')
                        return np.nan
                    except:
                        return els

    def change_atoms(self, atomlist, sortlength, write_file=True):
        """
        Args:
            atomlist : Dictionary that designate atoms to change.
            ex) {'Mg' : 'Mg2+', 'Si' : Si3+, 'P' : {'P3-': 0.5, 'Se2-': 0.5}}
            sortlength : length of ordered structures based on electrostatic energy
        Returns: 
            POSCAR in dicrectory 00 01 02 ...
        """
        energy_entry = {}
        min_index = []
        poscar = self.structure_from_dic()
        s = Structure.from_file(poscar)
        c = [type(atomlist[k]) for k in atomlist.keys()]

        if type({}) in c:
            for keys in atomlist.keys():
                for i in range(len(s)):
                    if s[i][keys] != 0:
                        s[i] = atomlist[keys]

            trans = OrderDisorderedStructureTransformation()
            ss = trans.apply_transformation(s, sortlength)
            matcher = StructureMatcher()
            groups = matcher.group_structures([d['structure'] for d in ss])
                
            for i in range(len(groups)):
                energy_entry[i] = {}
                for j in range(len(groups[i])):
                    a = EwaldSummation(groups[i][j])
                    energy_entry[i][j] = a.total_energy
                    if i == 0 and j == 0:
                        minimum_energy = a.total_energy
                        min_index = [i, j]
                    elif minimum_energy > a.total_energy:
                        minimum_energy = a.total_energy
                        min_index = [i, j]

            lowest_energy_structure = groups[min_index[0]][min_index[1]]        
            Poscar(lowest_energy_structure).write_file(poscar)
            
            return poscar
        
        else:
            for keys in atomlist.keys():
                if atomlist[keys].lower() == 'remove':
                    s.remove_species([keys])
                else:
                    for i in range(len(s)):
                        if s[i][keys] != 0:
                            s[i] = atomlist[keys]
            Poscar(s).write_file(poscar)
                    
            print("Something is wrong")
    
    def kpoints(self, kppa=False):
        """
        Args:
            kppa (int): auto-generated Gamma or Monkhorst-pack grid with kppa / atom ^-1
        Returns:
            writes KPOINTS file to dir
        """
        fkpoints = os.path.join(self.dir, 'KPOINTS')
        with open(fkpoints, 'w') as f:
            poscar = Poscar.from_file(self.structure_from_dic())
            s = poscar.structure
            Kpoints().automatic_density(s, kppa=kppa).write_file(fkpoints)
            
    def potcar(self, els_in_poscar=False, specific_pots=False, machine='mycom', src='gga', MP=False):
        """
        Args:
            els_in_poscar (list or False) - ordered list of elements (str) in POSCAR; if FALSE, read POSCAR
            specific_pots (bool or dict) - False to use VASP defaults; else dict of {el : which POTCAR (str)}
            machine (str) - which computer or the path to your potcars
            src (str) - 'potpaw' implies SCAN-able POTs and configuration like ELEMENT_MOD/POTCAR
        MP (bool) - if True, use MP pseudopotentials
        Returns:
            writes POTCAR file to dir
        """
        if not els_in_poscar:
            els_in_poscar = self.ordered_els_from_poscar()
        fpotcar = os.path.join(self.dir, 'POTCAR')
        if machine == 'mycom':
            path_to_pots = '/Users/yun/Downloads/MY_PSP'
        elif machine == 'cori':
            path_to_pots = '/global/homes/y/yychoi/MY_PSP'
        if src == 'gga':
            pot_dir = 'POT_GGA_PAW_PBE'
        elif src == 'gga_52':
            pot_dir = 'POT_GGA_PAW_PBE_52'
        if MP:
            specific_pots = {'Ac': 'Ac', 'Ag': 'Ag', 'Al': 'Al', 
                             'Ar': 'Ar', 'As': 'As', 'Au': 'Au', 
                             'B': 'B', 'Ba': 'Ba_sv', 'Be': 'Be_sv', 
                             'Bi': 'Bi', 'Br': 'Br', 'C': 'C', 
                             'Ca': 'Ca_sv', 'Cd': 'Cd', 'Ce': 'Ce', 
                             'Cl': 'Cl', 'Co': 'Co', 'Cr': 'Cr_pv', 
                             'Cs': 'Cs_sv', 'Cu': 'Cu_pv', 'Dy': 'Dy_3', 
                             'Er': 'Er_3', 'Eu': 'Eu', 'F': 'F', 'Fe': 'Fe_pv', 
                             'Ga': 'Ga_d', 'Gd': 'Gd', 'Ge': 'Ge_d', 'H': 'H', 
                             'He': 'He', 'Hf': 'Hf_pv', 'Hg': 'Hg', 'Ho': 'Ho_3', 
                             'I': 'I', 'In': 'In_d', 'Ir': 'Ir', 'K': 'K_sv', 
                             'Kr': 'Kr', 'La': 'La', 'Li': 'Li_sv', 'Lu': 'Lu_3', 
                             'Mg': 'Mg_pv', 'Mn': 'Mn_pv', 'Mo': 'Mo_pv', 'N': 'N', 
                             'Na': 'Na_pv', 'Nb': 'Nb_pv', 'Nd': 'Nd_3', 'Ne': 'Ne', 
                             'Ni': 'Ni_pv', 'Np': 'Np', 'O': 'O', 'Os': 'Os_pv', 
                             'P': 'P', 'Pa': 'Pa', 'Pb': 'Pb_d', 'Pd': 'Pd', 
                             'Pm': 'Pm_3', 'Pr': 'Pr_3', 'Pt': 'Pt', 'Pu': 'Pu', 
                             'Rb': 'Rb_sv', 'Re': 'Re_pv', 'Rh': 'Rh_pv', 'Ru': 'Ru_pv', 
                             'S': 'S', 'Sb': 'Sb', 'Sc': 'Sc_sv', 'Se': 'Se', 'Si': 'Si', 
                             'Sm': 'Sm_3', 'Sn': 'Sn_d', 'Sr': 'Sr_sv', 'Ta': 'Ta_pv', 
                             'Tb': 'Tb_3', 'Tc': 'Tc_pv', 'Te': 'Te', 'Th': 'Th', 
                             'Ti': 'Ti_pv', 'Tl': 'Tl_d', 'Tm': 'Tm_3', 'U': 'U', 
                             'V': 'V_pv', 'W': 'W_pv', 'Xe': 'Xe', 'Y': 'Y_sv', 
                             'Yb': 'Yb_2', 'Zn': 'Zn', 'Zr': 'Zr_sv'}
            pot_dir = 'POT_GGA_PAW_PBE'
            if src != 'gga':
                print('using GGA pots bc MP = TRUE')
        pots = os.path.join(path_to_pots, pot_dir)

        with open(fpotcar, 'w') as f:
            for el in els_in_poscar:
                if (specific_pots == False) or (el not in specific_pots):
                    pot_to_add = os.path.join(pots, el, 'POTCAR.gz')
                else:
                    pot_to_add = os.path.join(pots, specific_pots[el], 'POTCAR.gz')
                with gzip.open(pot_to_add, 'rt') as g:
                    for line in g:
                        f.write(line)
                        
    def copy_script(self, name=False, time=False, node=False, parallel=False):
        
        script = '/global/u2/y/yychoi/script'
        new_script = os.path.join(self.dir, 'script')
        copyfile(script, new_script)    
        defaultname = [v for v in self.dir.split('/') if v != '']
        
        if name:
            replace_line(new_script, 2, '#SBATCH --job-name=' + name + '\n')
        else:
            replace_line(new_script, 2, '#SBATCH --job-name=' + defaultname[-1] + '\n')
        if time:
            replace_line(new_script, 3, '#SBATCH --time=' + str(time) + '\n')
        if node:
            replace_line(new_script, 7, '#SBATCH --ntasks-per-node=' + str(node) + '\n')
        if parallel:
            replace_line(new_script, 6, '#SBATCH --nodes=' + str(parallel) + '\n')
            
    def els_to_amts(self):
        """
        Args:
            None
        Returns:
            dictionary of {element (str) : number in calculated structure (int)}
        """
        return els_to_amts(self.ordered_els_from_poscar(), os.path.join(self.dir, 'POSCAR'))
    
    def nsites(self):
        """
        Args:
            None
        Returns:
            number (int) of ions in calculated structure
        """
        return nsites(self.els_to_amts())
    
    def idxs_to_els(self):
        """
        Args:
            None
        Returns:
            Dictionary of indexs in structure to the element at that index
        """
        return idxs_to_els(self.ordered_els_from_poscar(), self.els_to_amts(), self.nsites())
    
    def els_to_idxs(self):
        """
        Args:
            None
        Returns:
            Dictionary of elemets with indexs
        """
        return els_to_idxs(self.idxs_to_els())
            
class VASPBasicAnalysis(object):
    """
    This class is for second run, convergence check, getting energy and Ehull from simulations
    """
    
    def __init__(self, dir):
        
        self.dir = dir
        
    def params_from_incar(self):
        
        incar = os.path.join(self.dir, 'INCAR')
        data = {}
        with open(incar) as f:
            for line in f:
                if ('=' in line) and (line[0] != '!'):
                    line = line[:-1].split(' = ')
                    data[line[0].strip()] = line[1].strip()
                    
        return data
    
    def params_from_outcar(self, num_params=['NKPTS', 'NKDIM', 'NBANDS', 'NEDOS',
                                             'ISPIN', 'ENCUT', 'NELM', 'EDIFF',
                                             'EDIFFG', 'NSW', 'ISIF', 'ISYM', 
                                             'NELECT', 'NUPDOWN', 'EMIN', 'EMAX', 
                                             'ISMEAR', 'SIGMA', 'AEXX'],
                                 str_params = ['PREC', 'METAGGA', 'LHFCALC', 'LEPSILON',
                                               'LRPA']):
        """
        Args:
            num_params (list) - list of numerical parameters to retrieve (str)
            str_params (list) - list of True/False or string parameters to retrieve (str)
        Returns:
            dictionary of {paramater (str) : value (str or float) for each parameter provided}
        """
        outcar = os.path.join(self.dir, 'OUTCAR')
        data = {}        
        params = num_params + str_params
        with open(outcar) as f:
            for param in params:
                f.seek(0)
                for line in f:
                    if (param in line) and ('LMODELHF' not in line) and ('INISYM' not in line):
                        val = line.split(param)[1].split('=')[1].strip().split(' ')[0].replace(';', '')
                        break
                if param in num_params:
                    data[param] = float(val)
                elif param in str_params:
                    data[param] = str(val)
        return data
        
    def ordered_els_from_outcar(self):
        """
        Args:
            None
        Returns:
            list of elements (str) as they appear in POTCAR
        """
        outcar = os.path.join(self.dir, 'OUTCAR')
        els = []
        with open(outcar) as f:
            for line in f:
                if 'VRHFIN' in line:
                    el = line.split('=')[1].split(':')[0]
                    els.append(el.strip())
                if 'POSCAR' in line:
                    break
        return els
    
    def structure(self):
        """
        Args:
            None
        Returns:
            CONTCAR directory if it exists; else POSCAR
        """
        fposcar = os.path.join(self.dir, 'POSCAR')
        if 'CONTCAR' in os.listdir(self.dir):
            fcontcar = os.path.join(self.dir, 'CONTCAR')
            with open(fcontcar) as f:
                contents = f.read()
                if len(contents) > 0:
                    return fcontcar
                else:
                    return fposcar
        else:
            return fposcar
        
    def is_converged(self):
        """
        Args:
            None
        Returns:
            True if calculation converged; else False
        """
        outcar = os.path.join(self.dir, 'OUTCAR')
        oszicar = os.path.join(self.dir, 'OSZICAR')
        if not os.path.exists(outcar):
            print('no OUTCAR file')
            return False
        try:
            with open(outcar) as f:
                contents = f.read()
                if 'Elaps' not in contents:
                    return False
        except UnicodeDecodeError:
            with open(outcar, 'rb') as f:
                contents = f.read()
                contents = str(contents)
                if 'Elaps' not in contents:
                    return False
        params = self.params_from_outcar(num_params=['NELM', 'NSW'], str_params=[])
        nelm, nsw = params['NELM'], params['NSW']
        if nsw == 0:    
            max_ionic_step = 1
        else:
            if os.path.exists(oszicar):
                with open(oszicar) as f:
                    for line in f:
                        if 'F' in line:
                            step = line.split('F')[0].strip()
                            if ' ' in step:
                                step = step.split(' ')[-1]
                            step = int(step)
            else:
                with open(outcar) as f:
                    f.seek(0)
                    for line in f:
                        if ('Iteration' in line) and ('(' in line): #)
                            step = line.split('Iteration')[1].split('(')[0].strip() #)
                            step = int(step)
            max_ionic_step = step
            if max_ionic_step == nsw:
                return False
        with open(outcar) as f:
            f.seek(0)
            for line in f:
                if ('Iteration' in line) and (str(max_ionic_step)+'(') in line: #)
                    step = line.split(str(max_ionic_step)+'(')[1].split(')')[0].strip()
                    if int(step) == nelm:
                        return False
        return True

    def second_run(self):
        """
        Args:
            directory that finished geometric optimization
        Returns:
            copy CONTCAR to POSCAR in second directory
            This is for the electroninc energy convergence after geometric optimization.
        """
        if not self.is_converged():
            print('calcuation is not converged')
            return np.nan
        
        contcar = os.path.join(self.dir, 'CONTCAR')
        stuff = [v for v in contcar.split('/' ) if v != '']
        count = [w for w in stuff[-2].split('_') if w !='']
        
        if count[-1] == 'first':
            poscar = self.dir[:-5] + 'second'
            if os.path.exists(poscar) == False:
                os.mkdir(poscar)
            copyfile(contcar, os.path.join(poscar, 'POSCAR'))
        
        obj = VASPSetUp(poscar)
        obj.incar(geometry_opt=False, dos=True, MP=True, write_file=True)
        obj.kpoints(1000)
        obj.potcar(MP=True, machine='cori')
        obj.copy_script(time='03:00:00')
    
    def els_to_amts(self):
        """
        Args:
            None
        Returns:
            dictionary of {element (str) : number in calculated structure (int)}
        """
        return els_to_amts(self.ordered_els_from_outcar(), self.structure())
    
    def nsites(self):
        """
        Args:
            None
        Returns:
            number (int) of ions in calculated structure
        """
        return nsites(self.els_to_amts())
    
    def idxs_to_els(self):
        """
        Args:
            None
        Returns:
            Dictionary of indexs in structure to the element at that index
        """
        return idxs_to_els(self.ordered_els_from_outcar(), self.els_to_amts(), self.nsites())
    
    def els_to_idxs(self):
        """
        Args:
            None
        Returns:
            Dictionary of elemets with indexs
        """
        return els_to_idxs(self.idxs_to_els())
    
    def nelect(self):
        """
        Args:
            None
        Returns:
            number of electrons in calculation (int)
        """
        return self.params_from_outcar(num_params=['NELECT'], str_params=[])['NELECT']
    
    def nbands(self):
        """
        Args:
            None
        Returns:
            number of bands in calculation (int)
        """
        return self.params_from_outcar(num_params=['NBANDS'], str_params=[])['NBANDS']

    def Etot(self, peratom = False):
        """
        Args:
            None
        Returns:
            energy (not per atom) (float) of calculated structure if converged
        """
        if not self.is_converged():
            print('calcuation is not converged')
            return np.nan
        oszicar = os.path.join(self.dir, 'OSZICAR')
        if os.path.exists(oszicar):
            with open(oszicar) as f:
                for line in f:
                    if 'F' in line:
                        E = float(line.split('F=')[1].split('E0')[0].strip())
        else:
            outcar = os.path.join(self.dir, 'OUTCAR')
            with open(outcar) as f:
                for line in reversed(list(f)):
                    if 'TOTEN' in line:
                        line = line.split('=')[1]
                        E = float(''.join([c for c in line if c not in [' ', '\n', 'e', 'V']]))
                        break
        if peratom == True:
            return float(E)/self.nsites()
        else:
            return float(E)

    def ehullmp(self, KEY='R0CmAuNPKWzrUo8Z'):
        """
        Args:
            None
        Returns:
            Energy above hull per atom of calculated structure if converged.
        """
        if not self.is_converged():
            print('calculation is not converged')
            return np.nan
        
        mpr = MPRester(KEY)
        vaspxml = os.path.join(self.dir, 'vasprun.xml')
        vasprun = Vasprun(vaspxml)
        my_entry = vasprun.get_computed_entry(inc_structure=False)
        
        return mpr.get_stability([my_entry])[0]
    
    def Efermi(self, alphabeta=True):
        """
        Args:
            None
        Returns:
            Fermi energy outputted by VASP (float)
        """
        outcar = os.path.join(self.dir, 'OUTCAR')
        with open(outcar) as f:
            for line in f:
                if 'E-fermi' in line and alphabeta == True:
                    return float(line.split(':')[1].split('XC')[0].strip()) + float(line.split(':')[3])
                elif 'E-fermi' in line and alphabeta == False:
                    return float(line.split(':')[1].split('XC')[0].strip())
                
    def Volume(self):
        """
        Args:
            None
        Returns:
            Volume/number of atoms
        """
        contcar = os.path.join(self.dir, 'CONTCAR')
        structure = Structure.from_file(contcar)
        return structure.volume/self.nsites()

class VASPDOSAnalysis(object):
    """
    This class is for converting DOSCAR to DOS.json and processing it
    """
    
    def __init__(self, calc_dir, doscar='DOSCAR'):
        '''
        Args:
            calc_dir(str) - path to VASP calculation
            doscar(str) - name of doscar file to analyze(DOSCAR or DOSCAR.lobster)
        Returns:
            path to DOSCAR to analyze
        '''
        self.calc_dir = calc_dir
        self.doscar = os.path.join(calc_dir, doscar)
        
    def doscar_to_json(self, fjson=False, remake=False):
        '''
        Args:
            fjson (str or False) - path to json to write; if False, writes to calc_dir/DOS.json
            remake (bool) - if True, regenerate json; else read json
        Returns:
            dictionary of DOS information
                first_key = energy (float)
                next_keys = ['total'] + els
                for total, keys are ['up', 'down']
                for els, keys are orbital_up, orbital_down, level_up, level_down, all_up, all_down
                    populations are as generated by vasp
        '''
        if not fjson:
            if 'lobster' not in self.doscar:
                fjson = os.path.join(self.calc_dir, 'DOS.json')
            else:
                fjson = os.path.join(self.calc_dir, 'lobDOS.json')
                
        if not os.path.exists(self.doscar) and not os.path.exists(fjson):
            print('%s doesnt exist' % self.doscar)
            return np.nan                
        
        if remake or not os.path.exists(fjson) or (read_json(fjson) == {}):
            basic_obj = VASPBasicAnalysis(self.calc_dir)
            idxs_to_els = basic_obj.idxs_to_els()
            num_params = ['ISPIN', 'NEDOS']
            d_params = basic_obj.params_from_outcar(num_params=num_params, str_params=[])
            spin, nedos = d_params['ISPIN'], int(d_params['NEDOS'])

            if 'lobster' in self.doscar:
                with open(self.doscar) as f:
                    count = 0
                    for line in f:
                        count += 1
                        if count == 6:
                            nedos = int([v for v in line.split(' ') if v != ''][2])
                            break
            if spin == 2:
                spins = ['up', 'down']
                orbital_keys = 'E,s_up,s_down,py_up,py_down,pz_up,pz_down,px_up,px_down,dxy_up,dxy_down,dyz_up,dyz_down,dzz_up,dzz_down,dxz_up,dxz_down,dxxyy_up,dxxyy_down'.split(',')
                total_keys = 'E,up,down'.split(',')
            else:
                spins = ['up']
                orbital_keys = 'E,s_up,py_up,pz_up,px_up,dxy_up,dyz_up,dzz_up,dxz_up,dxxyy_up'.split(',')
                total_keys = 'E,up'.split(',')
            dos_count = 0
            count = 0
            data = {}
            with open(self.doscar) as f:
                for line in f:
                    count += 1
                    if count == 6+(dos_count*nedos)+dos_count:
                        dos_count += 1
                        continue
                    if dos_count == 1: # DOS Count 1 means the total nedos columns in the DOSCAR file. if kept remain 1 during nedos number of loops
                        values = [v for v in line[:-1].split(' ') if v != '']
                        tmp = dict(zip(total_keys, values)) # tmp shape is {'E': ~, 'up': ~, 'down': ~}
                        data[float(tmp['E'])] = {'total' : {k : float(tmp[k]) for k in tmp if k != 'E'}}
                    elif dos_count == 0:
                        continue
                    else: # DOS Count not 1 means the elements nedos columns in the DOSCAR file in order of POSCAR. if kept remain 1 during nedos number of loops
                        values = [v for v in line[:-1].split(' ') if v != '']                        
                        el = idxs_to_els[dos_count-2] # dos count -2 is number of index in POSCAR
                        tmp = dict(zip(orbital_keys, values))
                        if el not in data[float(tmp['E'])]:
                            data[float(tmp['E'])][el] = {k : float(tmp[k]) for k in tmp if k != 'E'}
                        else:
                            for k in data[float(tmp['E'])][el]:
                                if k != 'E':
                                    data[float(tmp['E'])][el][k] += float(tmp[k])
            for E in data: # Summing p and d orbital DOS to p_up p_down d_up d_down and Summing them to all_up and all_down for each E and elements
                for el in data[E]:
                    if el != 'total':
                        for orb in ['p', 'd']:
                            for spin in spins:
                                data[E][el]['_'.join([orb, spin])] = np.sum([data[E][el][k] for k in data[E][el] if k.split('_')[0][0] == orb if k.split('_')[1] == spin])
                        for spin in spins:
                            data[E][el]['_'.join(['all', spin])] = np.sum([data[E][el]['_'.join([orb, spin])] for orb in ['s', 'p', 'd']])
            for E in data: # Summing element's total energies into key total. Why this needed again?
                data[E]['total'] = {spin : 0 for spin in spins}
                for el in data[E]:
                    if el != 'total':
                        for spin in spins:
                            data[E]['total'][spin] += data[E][el]['_'.join(['all', spin])]
            return write_json(data, fjson)
        else:
            data = read_json(fjson)
            return {float(k) : data[k] for k in data}
        
    def energies_to_populations(self, element='total', orbital='all', spin='summed', fjson=False, remake=False):
        """
        Args:
            element (str) - element or 'total' to analyze
            orbital (str) - orbital or 'all' to analyze
            spin (str) - 'up', 'down', or 'summed'
            fjson (str or False) - path to json to write; if False, writes to calc_dir/DOS.json
            remake (bool) - if True, regenerate dos_dict json; else read json         
        Returns:
            dictionary of {energies (float) : populations (float)}
        """
        dos_dict = self.doscar_to_json(fjson, remake)
        if isinstance(dos_dict, float):
            print('hmmm dos_dict is not a dict...')
            return np.nan
        energies = sorted(list(dos_dict.keys()))
        ispin = VASPBasicAnalysis(self.calc_dir).params_from_outcar(num_params=['ISPIN'], str_params=[])['ISPIN']
        if ispin == 1:
            spins = ['up']
        elif ispin == 2:
            spins = ['up', 'down']
        if element != 'total':
            if spin != 'summed':
                populations = [dos_dict[E][element]['_'.join([orbital, spin])] for E in energies]
            else:
                populations = [0 for E in energies]
                for s in range(len(spins)):
                    spin = spins[s]
                    for i in range(len(energies)):
                        E = energies[i]
                        populations[i] += dos_dict[E][element]['_'.join([orbital, spin])]
        else:
            if spin != 'summed':
                populations = [dos_dict[E][element][spin] for E in energies]
            else:
                populations = [np.sum([dos_dict[E][element][spin] for spin in spins]) for E in energies]
        return dict(zip(energies, populations))

    def min_valence_energy(self, tol=0.01, electrons='valence', fjson=False, remake=False):
        if electrons == 'all':
            nelect = VASPBasicAnalysis(self.calc_dir).nelect()
        elif electrons == 'valence':
            valence_data = atomic_valences_data()
            els_to_idxs = VASPBasicAnalysis(self.calc_dir).els_to_idxs()
            nelect = 0
            for el in els_to_idxs:
                nelect += valence_data[el]*len(els_to_idxs[el])
        dos = self.energies_to_populations(fjson=fjson, remake=remake)
        if isinstance(dos, float):
            print('hmmm energies_to_populations is not a dict...')
            return np.nan
        sorted_Es = sorted(list(dos.keys()))
        if 'lobster' in self.doscar:
            Efermi = 0
        else:
            Efermi = VASPBasicAnalysis(self.calc_dir).Efermi(alphabeta=False)
        occ_Es = [E for E in sorted_Es if E <= Efermi][::-1]

        for i in range(2, len(occ_Es)):
            int_Es = occ_Es[:i]
            int_doss = [dos[E] for E in int_Es]
            sum_dos = abs(simps(int_doss, int_Es))
            if sum_dos >= (1-tol)*nelect:
                return occ_Es[i]
            
        # Testing integral of DOS
        energy = [E for E in sorted_Es]
        energy_pop = [E * dos[E] for E in sorted_Es]
        totalE = cumtrapz(energy_pop, energy)
    
        print('DOS doesnt integrate to %.2f percent of %i electrons' % (100*(1-tol), nelect))
        return sum_dos, totalE
    
class ProcessDOS(object):
    """
    Handles generic dictionary of {energies : states}
    Used for manipulating density of states (or equivalent) and retrieving summary statistics
    """    
    def __init__(self, energies_to_populations, 
                       shift=False,
                       cb_shift=False,
                       vb_shift=False,
                       energy_limits=False, 
                       flip_sign=False,
                       min_population=False, 
                       max_population=False, 
                       abs_population=False,
                       normalization=False):
        """
        Args:
            energies_to_populations (dict) - dictionary of {energy (float) : population (float) for all energies}
            populations (array) - number of states at each energy (float)
            shift (float or False) - shift all energies
                e.g., shift = -E_Fermi would make E_fermi = 0 eV
            cb_shift (tuple or False) - shift all energies >= cb_shift[0] (float) by cb_shift[1] (float)
            vb_shift (tuple or False) - shift all energies <= vb_shift[0] (float) by vb_shift[1] (float)            
            energy_limits (list or False) - get data only for energies between (including) energy_limits[0] and energy_limits[1]
                e.g., energy_limits = [-1000, E_Fermi] would return only occupied states                
            flip_sign (True or False) - change sign of all populations
            min_population (float or False) - get data only when the population is greater than some value
                e.g., min_population = 0 would return only bonding states in the COHP (presuming flip_sign)
            max_population (float or False) - get data only when the population is less than some value
            abs_population (True or False) - make all populations >= 0
            normalization (float or False) - divide all populations by some value
            
        Returns:
            dictionary of {energy (float) : population (float)} for specified data
        """
        if isinstance(shift, float):
            energies_to_populations = {E+shift : energies_to_populations[E] for E in energies_to_populations}
        if cb_shift:
            energies_to_populations = {E+cb_shift[1] if E >= cb_shift[0] else E : energies_to_populations[E] for E in energies_to_populations}
        if vb_shift:
            energies_to_populations = {E+vb_shift[1] if E <= vb_shift[0] else E : energies_to_populations[E] for E in energies_to_populations}
        if flip_sign:
            energies_to_populations = {E : -energies_to_populations[E] for E in energies_to_populations}
        if energy_limits:
            Emin, Emax = energy_limits
            energies = [E for E in energies_to_populations if E >= Emin if E <= Emax]
            energies_to_populations = {E : energies_to_populations[E] for E in energies}
        if isinstance(min_population, float):
            energies_to_populations = {E : energies_to_populations[E] if energies_to_populations[E] >= min_population else 0. for E in energies_to_populations}
        if isinstance(max_population, float):
            energies_to_populations = {E : energies_to_populations[E] if energies_to_populations[E] <= max_population else 0. for E in energies_to_populations}
        if abs_population:
            energies_to_populations = {E : abs(energies_to_populations[E]) for E in energies_to_populations}
        if normalization:
            energies_to_populations = {E : energies_to_populations[E]/normalization for E in energies_to_populations}
        self.energies_to_populations = energies_to_populations

    def stats(self, area=True, net_area=True, energy_weighted_area=True, center=True, width=False, skewness=False, kurtosis=False):
        """
        Args:
            area (bool) - if True, compute integral of absolute populations
        net_area (bool) - if True, compute integral of populations
            energy_weighted_area (bool) - if True, compute energy-weighted integral
            center (bool) - if True, compute center
            width (bool) - if True, compute width
            skewness (bool) - if True, compute skewness
            kurtosis (bool) if True, compute kurtosis
            
        Returns:
            dictionary of {property (str) : value (float) for specified property in args}
        """
        if center:
            area, energy_weighted_area = True, True
        if width or skewness or kurtosis:
            area = True
        energies_to_populations = self.energies_to_populations
        energies = np.array(sorted(list(energies_to_populations.keys())))
        populations = np.array([energies_to_populations[E] for E in energies])
        summary = {}
        if area:
            summary['area'] = simps(abs(populations), energies)
        if net_area:
            summary['net_area'] = simps(populations, energies)
        if energy_weighted_area:
            summary['energy_weighted_area'] = simps(populations*energies, energies)
        if center:  
            summary['center'] = summary['energy_weighted_area'] / summary['area']
        if width:
            summary['width'] = simps(populations*energies**2, energies) / summary['area']
        if skewness:
            summary['skewness'] = simps(populations*energies**3, energies) / summary['area']
        if kurtosis:
            summary['kurtosis'] = simps(populations*energies**4, energies) / summary['area']
        return summary

class LOBSTERAnalysis(object):
    """
    Convert COHPCAR, COOPCAR to dictionary
    """
    
    def __init__(self, calc_dir, lobster='COHPCAR.lobster'):
        """
        Args:
            calc_dir(str) - path to calculation with LOBSTER output
            lobster(str) - COHPCAR.lobster or COOPCAR.lobster
        Returns:
            calc_dir
            path to LOBSTER output
        """
        self.calc_dir = calc_dir
        self.lobster = os.path.join(calc_dir, lobster)
        
    def pair_dict(self, fjson=False, remake=False):
        """
        Args:
            fjson (str or False) - path to json to write; if False, writes to calc_dir/lobPAIRS.json
            remake (bool) - if True, regenerate json; else read json
        Returns:
            dictionary of {pair index (int) : {'els' : (el1, el1) (str), 
                                               'sites' : (structure index for el1, structure index for el2) (int),
                                               'orbitals' : (orbital for el1, orbital for el2) (str) ('all' if all orbitals summed),
                                               'dist' : distance in Ang (float)}
                                               'energies' : [] (placeholder),
                                               'populations' : [] (placeholder)}
        """
        if not fjson:
            fjson = os.path.join(self.calc_dir, 'lobPAIRS.json')
        if remake or not os.path.exists(fjson) or (read_json(fjson) == {}):
            lobster = self.lobster
            if not os.path.exists(lobster):
                print('%s doenst exist' %lobster)
                return np.nan
            data = {}
            with open(lobster) as f:
                count = 0
                idx_count = 0
                for line in f:
                    if count < 3:
                        count += 1
                        continue
                    elif line[:3] == 'No.':
                        idx_count += 1
                        pair_idx = idx_count
                        if '[' not in line: 
                            el_site1 = line.split(':')[1].split('->')[0]
                            el_site2 = line.split(':')[1].split('->')[1].split('(')[0]
                            orb1, orb2 = 'all', 'all' 
                        else:
                            el_site1 = line.split(':')[1].split('->')[0].split('[')[0]
                            el_site2 = line.split(':')[1].split('->')[1].split('(')[0].split('[')[0]
                            orb1, orb2 = line.split('[')[1].split(']')[0], line.split('->')[1].split('[')[1].split(']')[0]  
                        dist = float(line.split('(')[1].split(')')[0])
                        el1, el2 = CompAnalyzer(el_site1).els[0], CompAnalyzer(el_site2).els[0]
                        site1, site2 = int(el_site1.split(el1)[1]), int(el_site2.split(el2)[1]) 
                        data[pair_idx] = {'els' : (el1, el2),
                                          'sites' : (site1, site2),
                                          'orbitals' : (orb1, orb2),
                                          'dist' : dist,
                                          'energies' : [],
                                          'populations' : []}
                    else:
                        return write_json(data, fjson)  
        else:
            data = read_json(fjson)
            new = {}
            for i in data:
                new[int(i)] = {}
                for j in data[i]:
                    if type(data[i][j]) == list and data[i][j] != []:
                        new[int(i)][j] = tuple(data[i][j])
                    else:
                        new[int(i)][j] = data[i][j]
            return new
        
    def detailed_dos_dict(self, fjson=False, remake=False, fjson_pairs=False):
        """
        Args:
            fjson (str) - path to json to write; if False, writes to calc_dir/COHP.json or COOP.json
            remake (bool) - if True, regenerate json; else read json   
        Returns:
            dictionary of COHP/COOP information
                first_key = energy (float)
                next_keys = each unique el1_el2 interaction and also total
                for total, value is the total population
                for each sorted(el1_el2) interaction, keys are each specific sorted(site1_site2) interaction for those elements and total
                    populations are as generated by LOBSTER
        """
        if not fjson:
            fjson = os.path.join(self.lobster.replace('CAR.lobster', '.json')) # Making COHP.json file
        if remake or not os.path.exists(fjson) or (read_json(fjson) == {}):
            lobster = self.lobster # save directory to lobster variable
            if not fjson_pairs: # if lobPAIRS.json is not exists, save lobPAIRS.json directoy to fjson_pairs variable
                fjson_pairs = os.path.join(self.calc_dir, 'lobPAIRS.json')
            data = self.pair_dict(fjson=fjson_pairs, remake=False) # make lobPAIRS.json and read
            if not isinstance(data, dict):
                return np.nan
            with open(lobster) as f:
                count = 0
                for line in f:
                    if (count < 3) or (line[:3] == 'No.'):
                        count += 1
                        continue
                    else:
                        line = line.split(' ') # check line start with energy value and save to values
                        values = [v for v in line if v != ''] 
                        energy = float(values[0]) # First value of the line is energy - check the manual 
                        for pair in data:
                            idx_up = 2 + int(pair) * 2 - 1 # line (energy, pCOHP-total(up), lpCOHP-total(up), ... ) total 4N+5
                            idx_down = idx_up + 2 * len(data) + 2
                            population_up = float(values[idx_up])
                            population_down = float(values[idx_down])     
                            population = population_up + population_down
                            data[pair]['energies'].append(energy)
                            data[pair]['populations'].append(population) # append pCOHP-1,2 ...
            new = {}
            energies = data[1]['energies'] # Get energy list
            element_combinations = list(set(['_'.join(sorted(data[pair]['els'])) for pair in data])) # Make combinaiton of elements in json file ex) O_O, O_Sn
            for i in range(len(energies)): # iteration with energies
                overall_total = 0
                tmp1 = {}
                for el_combo in element_combinations: # iterate with specific combination
                    combo_total = 0
                    pairs = [pair for pair in data if (tuple(el_combo.split('_')) == data[pair]['els'])] # find the pairs have specific combination
                    if len(pairs) == 0:
                        pairs = [pair for pair in data if (tuple(el_combo.split('_')[::-1]) == data[pair]['els'])] # For the case order is reversed
                        sites = [data[pair]['sites'][::-1] for pair in pairs]
                        orbitals = [data[pair]['orbitals'][::-1] for pair in pairs]
                    else:
                        sites = [data[pair]['sites'] for pair in pairs]
                        orbitals = [data[pair]['orbitals'] for pair in pairs]
                    tmp2 = {'_'.join([str(s) for s in site]) : {} for site in list(set(sites))} # make {'45_1': {}, ... } dictionary for each combination
                    for j in range(len(pairs)):
                        pair, site, orbital = pairs[j], sites[j], orbitals[j]
                        population = data[pair]['populations'][i]
                        if orbital == ('all', 'all'):
                            combo_total += population
                        site_key = '_'.join([str(s) for s in site])
                        orb_key = '-'.join(orbital)
                        tmp2[site_key][orb_key] = population # make {'45_1': {'all-all': population}, ... }
                    tmp2['total'] = combo_total
                    tmp1[el_combo] = tmp2 # make {'Mn-O' : {45_1': {'all-all': population}, ... }, ...}
                    overall_total += combo_total
                tmp1['total'] = overall_total
                new[energies[i]] = tmp1
            return write_json(new, fjson)
        else:
            data = read_json(fjson)
            return {float(k) : data[k] for k in data}
                    
    def energies_to_populations(self, element_pair='total', site_pair='total', orb_pair='all-all', fjson=False, remake=False):
        """
        Args:
            element_pair (str) - el1_el2 (alphabetical) or 'total'
            site_pair (str) - site1_site2 (order corresponds with el1_el2) or 'total'
            orb_pair (str) - orb1-orb2 (order corresponds with el1_el2) or 'all-all' for all orbitals
            fjson (str or False) - path to json to write; if False, writes to calc_dir/DOS.json
            remake (bool) - if True, regenerate dos_dict json; else read json            
        
        Returns:
            dictionary of {energies (float) : populations (float)} for specified subset
        """        
        dos_dict = self.detailed_dos_dict(fjson, remake)
        energies = sorted(list(dos_dict.keys()))
        if element_pair != 'total':
            if site_pair != 'total':
                populations = [dos_dict[E][element_pair][site_pair][orb_pair] for E in energies]
            else:
                populations = [dos_dict[E][element_pair][site_pair] for E in energies]
        else:
            populations = [dos_dict[E][element_pair] for E in energies]
        return dict(zip(energies, populations))