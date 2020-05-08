from VASPHandyFunctions import *
from VASPInputHandler import *
import matplotlib.pyplot as plt
import matplotlib as mpl


class Plotter(object):
    """
    This class is for plotting Voltage-Capacity curve, DOS, COHP, etc...
    """
    def __init__(self, calc_dir):
        
        self.calc_dir = calc_dir
        
    def set_rc_params(self):
        """
        Args:
            None
        Returns:
            Dictionary of settings for mpl.rcParams
        """
        params = {'axes.linewidth' : 1.5,
                  'axes.unicode_minus' : False,
                  'figure.dpi' : 300,
                  'font.size' : 40,
                  'legend.frameon' : False,
                  'legend.handletextpad' : 0.4,
                  'legend.handlelength' : 1,
                  'legend.fontsize' : 25,
                  'lines.markeredgewidth': 4,
                  'lines.linewidth': 3,
                  'lines.markersize': 15,
                  'mathtext.default' : 'regular',
                  'savefig.bbox' : 'tight',
                  'xtick.labelsize' : 30,
                  'ytick.labelsize' : 30,
                  'xtick.major.size' : 5,
                  'xtick.minor.size' : 3,
                  'ytick.major.size' : 5,
                  'ytick.minor.size' : 3,
                  'xtick.major.width' : 1,
                  'xtick.minor.width' : 0.5,
                  'ytick.major.width' : 1,
                  'ytick.minor.width' : 0.5,
                  'xtick.top' : True,
                  'ytick.right' : True,
                  'axes.edgecolor' : 'black',
                  'figure.figsize': [6, 15]}
        for p in params:
            mpl.rcParams[p] = params[p]

    def vcplotter(self, spgp, 
                  ionlist = ['Na', 'Mg'],
                  ionvalues = {'ionenergy' : {'Na' : -1.3384, 'Mg' : -1.6043, 'Li' : -1.9080},
                               'ionoxidation' : {'Na' : 1, 'Mg' : 2, 'Li' : 1},
                               'ioncolor' : {'Na' : 'g', 'Mg' : 'b', 'Li' : 'r'}},
                  xlim = (0, 1), ylim = (0.0, 5.1),
                  xlabel=r'X$_{Na/Mg}$', ylabel=r'Voltage(V)',
                  params = {'figure.figsize': [15, 10]}
                 ):
        '''        
        Args:
            spgp - Specific space group of materials
            ionlist - removed ion from structure. ['Na'] : Na removal, ['Na', 'Mg'] : Na removal and Mg insertion
            ionvalues - known energy, oxidation states, and designated color of each ions
        Returns:
            voltage-capacity curve
        '''
        
        self.set_rc_params()
        for p in params:
            mpl.rcParams[p] = params[p]
            
        totaldictionary = {}
        totaldictionary['0'] = -10000.0
        structurelist = [x[0] for x in os.walk(self.calc_dir)]

        for ion in ionlist:
            totaldictionary[ion] = {}
            for path in structurelist:
                if path.split('/')[-1].split('_')[-1] == 'second' and path.split('/')[-1].split('_')[-2] == spgp:
                    properdir = VASPBasicAnalysis(path)
                    if ion in properdir.els_to_amts().keys():
                        totaldictionary[ion][properdir.els_to_amts()[ion]] = properdir.Etot()
                    elif ion not in properdir.els_to_amts().keys() and totaldictionary['0'] < properdir.Etot():
                        totaldictionary['0'] = properdir.Etot()
        
        for key in totaldictionary:
            if type(totaldictionary[key]) == dict:
                totaldictionary[key] = dict(sorted(totaldictionary[key].items()))
                
        print(totaldictionary)
                
        for i in totaldictionary.keys():
            voltagelist = []
            if i == '0':
                continue
            else:
                for iteration, ionnum in enumerate(totaldictionary[i].keys()):
                    if iteration == 0:
                        previous_key = '0'
                        previous_value = totaldictionary[previous_key]
                        voltage = -((totaldictionary[i][ionnum] - previous_value)
                                    / (float(ionnum) - float(previous_key)) - ionvalues['ionenergy'][i])/ionvalues['ionoxidation'][i]
                        plt.plot([0, ionnum/max(totaldictionary[i].keys())], 
                                 [voltage, voltage], 
                                 c = ionvalues['ioncolor'][i], label = i)
                        voltagelist.append(voltage)
                        previous_voltage = voltage
                        previous_key = ionnum
                        previous_value = totaldictionary[i][ionnum]       
                    else:
                        voltage = -((totaldictionary[i][ionnum] - previous_value)
                                    / (float(ionnum) - float(previous_key)) - ionvalues['ionenergy'][i])/ionvalues['ionoxidation'][i]
                        plt.plot([previous_key/max(totaldictionary[i].keys()), 
                                  ionnum/max(totaldictionary[i].keys())], 
                                 [voltage, voltage], c = ionvalues['ioncolor'][i])
                        plt.plot([previous_key/max(totaldictionary[i].keys()), 
                                  previous_key/max(totaldictionary[i].keys())], 
                                 [previous_voltage, voltage], c = ionvalues['ioncolor'][i])
                        voltagelist.append(voltage)
                        previous_voltage = voltage
                        previous_key = ionnum
                        previous_value = totaldictionary[i][ionnum]
                        
                    
            plt.plot([0, 1], 
                     [sum(voltagelist)/len(voltagelist), sum(voltagelist)/len(voltagelist)], 
                     c = ionvalues['ioncolor'][i], ls='--')

        plt.legend()
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(self.calc_dir.split('/')[-1] + ' ' + spgp, y=1.01)
        plt.savefig('/global/cscratch1/sd/yychoi/JCESR/MgPostSpinels/Voltage/' + self.calc_dir.split('/')[-1] + "_" + spgp + '.png')
        plt.show()

        return
        
    def ehullplotter(self, spgp, ionlist = ['Na', 'Li', 'Mg'], ylabel=r'$E^{hull}(meV/atom)$', fjson = False, remake=False, 
                     xlim = [0, 2.05], ylim=[0, 300], xticks=(True, np.arange(0, 2.05, 0.5)), yticks=(True, np.arange(0, 301, 100)),
                     calstructure = {'CrSnO4': {'color': 'darkorange'}, 'CrTiO4': {'color': 'forestgreen'},
                                     'Fe0.5Ti1.5O4': {'color': 'dimgray'}, 'MnSnO4': {'color': 'firebrick'}, 
                                     'MnTiO4': {'color': 'darkkhaki'}},
                     params = {'figure.figsize': [12, 10]}):
        
        self.set_rc_params()
        for p in params:
            mpl.rcParams[p] = params[p]
        
        if not fjson:
            fjson = os.path.join(self.calc_dir, 'ehulldata' + spgp + '_' + ionlist[0] + '_' + ionlist[1] + '.json')
        if remake or not os.path.exists(fjson) or (read_json(fjson) == {}):        
            caldir = []
            totaldictionary = {}
            structurelist = [x[0] for x in os.walk(self.calc_dir)]
            for path in structurelist:
                if path.split('/')[-2] == 'MgPostSpinels' and path.split('/')[-1] in calstructure:
                    caldir.append(path)
            for ion in ionlist[0:2]:
                totaldictionary[ion] = {}
                for dirs in caldir:
                    structurelist = [x[0] for x in os.walk(dirs)]
                    label = dirs.split('/')[-1]
                    totaldictionary[ion][label] = {}
                    for path in structurelist:
                        if path.split('/')[-1].split('_')[-1] == 'lobster' and path.split('/')[-1].split('_')[-3] == spgp:
                            properdir = VASPBasicAnalysis(path)
                            if ion in properdir.els_to_amts().keys():
                                totaldictionary[ion][label][properdir.els_to_amts()[ion]] = properdir.ehullmp()['e_above_hull']
                            elif ionlist[0] not in properdir.els_to_amts().keys() and ionlist[1] not in properdir.els_to_amts().keys() and ionlist[2] not in properdir.els_to_amts().keys():
                                totaldictionary[ion][label][0] = properdir.ehullmp()['e_above_hull']
                    totaldictionary[ion][label] = dict(sorted(totaldictionary[ion][label].items()))
                    for key in totaldictionary[ion][label]:
                        newkey = key / max(totaldictionary[ion][label].keys())
                        totaldictionary[ion][label][newkey] = totaldictionary[ion][label].pop(key)
                        totaldictionary[ion][label][newkey] = 1000 * totaldictionary[ion][label][newkey]
            write_json(totaldictionary, fjson)
        else:
            totaldictionary = read_json(fjson)

        for i, ion in enumerate(totaldictionary):
            print(ion)
            ax = plt.subplot(int('1' + str(len(totaldictionary)) + str(i+1)))
            for struc in totaldictionary[ion]:
                print(*zip(*sorted(totaldictionary[ion][struc].items())))
                ax = plt.plot(*zip(*sorted(totaldictionary[ion][struc].items())), marker = '.', label = struc, **calstructure[struc])
            ax = plt.xlabel(r'$X_{' + ion + '}$')
            ax = plt.subplots_adjust(hspace=0.000, wspace=0.000)
            ax = plt.xlim(xlim)
            ax = plt.ylim(ylim)
            # ax = plt.xticks(xticks[1])
            ax = plt.yticks(yticks[1])
            if i != 0:
                ax = plt.gca().yaxis.set_ticklabels([])
            if i == 0:
                ax = plt.gca().invert_xaxis()
                ax = plt.ylabel(ylabel)
                ax = plt.legend(loc='lower left')
            ax = plt.plot(xlim, [0, 0], lw=1, ls='--', color='black')
        plt.suptitle('E hull difference (fd3m - pnma)', y=0.96)
        # plt.suptitle('Stability in Phase Diagram- ' + spgp, y=0.96)
        plt.savefig('/global/cscratch1/sd/yychoi/JCESR/MgPostSpinels/' + 'Ehull' + spgp + '.png')
        plt.show()
        return totaldictionary
    
    def avplotter(self, spgp, workingion1, workingion2,
                 ionenergy = {'Na' : -1.3384, 'Mg' : -1.6043, 'Li' : -1.9080},
                 ionoxidation = {'Na' : 1, 'Mg' : 2, 'Li' : 1},
                 ioncolor = {'Na' : 'darkgreen', 'Mg' : 'navy', 'Li' : 'firebrick'},
                 calstructure = {'Co0.5Sn1.5O4': {'color': 'royalblue'}, 'CrSnO4': {'color': 'darkorange'}, 'CrTiO4': {'color': 'forestgreen'}, 
                                 'Fe0.5Ti1.5O4': {'color': 'dimgray'}, 'MnSnO4': {'color': 'firebrick'}, 'Ni0.5Sn1.5O4': {'color': 'darkorchid'},
                                 'MnTiO4': {'color': 'darkkhaki'}}):
        
        self.set_rc_params()
        mpl.rcParams['figure.figsize'] = [18,18]
        mpl.rcParams['font.size'] = 60
        mpl.rcParams['legend.fontsize'] = 50
        caldir = []
        ion1volt = []
        ion2volt = []
        structurelist = [x[0] for x in os.walk(self.calc_dir)]

        for path in structurelist:
            if path.split('/')[-2] == 'MgPostSpinels' and path.split('/')[-1] in calstructure:
                caldir.append(path)
                
        for i, dirs in enumerate(caldir):
            ion1dict = {}
            ion2dict = {}
            structurelist = [x[0] for x in os.walk(dirs)]
            label = dirs.split('/')[-1]

            for path in structurelist:
                if path.split('/')[-1].split('_')[-1] == 'second' and path.split('/')[-1].split('_')[-2] == spgp:
                    a = VASPBasicAnalysis(path)
                    if workingion1 in a.els_to_amts().keys():
                        ion1dict[a.els_to_amts()[workingion1]] = a.Etot()
                    elif workingion2 in a.els_to_amts().keys():
                        ion2dict[a.els_to_amts()[workingion2]] = a.Etot()
                    elif workingion1 not in a.els_to_amts().keys() and workingion1 not in a.els_to_amts().keys():
                        ion1dict[0] = a.Etot()
                        ion2dict[0] = a.Etot()
                    else:
                        print('Check the ions properly')
                             
            ion1dict = dict(sorted(ion1dict.items()))
            ion2dict = dict(sorted(ion2dict.items()))
            voltage1 = -(ion1dict[max(ion1dict, key=int)] - ion1dict[min(ion1dict, key=int)] 
                         - max(ion1dict, key=int) * ionenergy[workingion1]) / max(ion1dict, key=int) / ionoxidation[workingion1]
            voltage2 = -(ion2dict[max(ion2dict, key=int)] - ion2dict[min(ion2dict, key=int)] 
                         - max(ion2dict, key=int) * ionenergy[workingion2]) / max(ion2dict, key=int) / ionoxidation[workingion2]
            if i == 0:
                plt.bar(label, voltage1, width=0.5, color=ioncolor[workingion1], label=workingion1)
                plt.bar(label, voltage2-voltage1, width=0.5, bottom=voltage1, color=ioncolor[workingion2], label=workingion2)
            else:
                plt.bar(label, voltage1, width=0.5, color=ioncolor[workingion1])
                plt.bar(label, voltage2-voltage1, width=0.5, bottom=voltage1, color=ioncolor[workingion2])
            print(voltage1)
            print(voltage2)
        
        plt.legend()
        plt.tick_params(axis='x', which='major', labelsize=24) # Move to method input
        plt.title('Average Voltage' + "_" + spgp, y=1.02)
        plt.ylabel('Voltage(V)')
        plt.savefig('/global/cscratch1/sd/yychoi/JCESR/MgPostSpinels/Voltage/' + self.calc_dir.split('/')[-1] + "Voltages_" + spgp + '.png')
        
        return

    def dos(self, spgp,
            what_to_plot={'total' : {'spins' : ['summed'], 'orbitals' : ['all']}},
            colors_and_labels = {'total-summed-all' : {'color' : 'black', 'label' : 'total'}},
            xlim=(0, 0.1), ylim=(-10, 4), 
            xticks=(False, [0, 0.1]), yticks=(False, [-10, 4]), 
            xlabel=r'$DOS/e^-$', ylabel=r'$E-E_F\/(eV)$',
            legend=True,
            smearing=0.2,
            shift='Fermi', normalization='electron',
            cb_shift=False,
            vb_shift=False,
            show=False,
            doscar='DOSCAR.lobster'
           ):
        
        self.set_rc_params()    

        # Generating paths that have specific space group and second run.
        caldir = []
        structurelist = [x[0] for x in os.walk(self.calc_dir)]
        for path in structurelist:
            if path.split('/')[-1].split('_')[-1] == 'second' and path.split('/')[-1].split('_')[-2] == spgp:
                caldir.append(path)            
        
        # Ordering directories to Na Na0.5 0 Mg0.5 Mg
        recaldir = {}
        for i in range(len(caldir)):
            if re.findall('[A-Z][^A-Z]*', caldir[i].split('/')[-1].split('_')[0])[0] == "Na":
                recaldir[0] = caldir[i]
            if re.findall('[A-Z][^A-Z]*', caldir[i].split('/')[-1].split('_')[0])[0] == "Na0.5":
                recaldir[1] = caldir[i]
            if re.findall('[A-Z][^A-Z]*', caldir[i].split('/')[-1].split('_')[0])[0] == "Mg":
                recaldir[4] = caldir[i]
            if re.findall('[A-Z][^A-Z]*', caldir[i].split('/')[-1].split('_')[0])[0] == "Mg0.5":
                recaldir[3] = caldir[i]     
            if re.findall('[A-Z][^A-Z]*', caldir[i].split('/')[-1].split('_')[0])[0] == re.findall('[A-Z][^A-Z]*', self.calc_dir.split('/')[-1])[0]:
                recaldir[2] = caldir[i]      
        recaldir = dict(sorted(recaldir.items()))
        caldir = list(recaldir.values())
        print(caldir)
        
        if show == True:
            mpl.rcParams['font.size'] = 20
            mpl.rcParams['xtick.labelsize'] = 30
            mpl.rcParams['ytick.labelsize'] = 30
            mpl.rcParams['legend.fontsize'] = 15
            mpl.rcParams['figure.figsize'] = [20,12]
            
        for i, j in enumerate(caldir):
            ax = plt.subplot(int('1' + str(len(caldir)) + str(i+1)))
            if 'lobster' in doscar:
                Efermi = 0.
            else:
                Efermi = VASPBasicAnalysis(j).Efermi(alphabeta=False)
            if shift == 'Fermi':                
                shift_value = -VASPBasicAnalysis(j).Efermi(alphabeta=False)
            if normalization == 'electron':
                normalization = VASPBasicAnalysis(j).params_from_outcar(num_params=['NELECT'], str_params=[])['NELECT']
            elif normalization == 'atom':
                normalization = VASPBasicAnalysis(j).nsites()
            occupied_up_to = Efermi + shift_value
            print(occupied_up_to, Efermi, shift_value)
            dos_lw = 1    
            for element in what_to_plot:
                for spin in what_to_plot[element]['spins']:
                    for orbital in what_to_plot[element]['orbitals']:
                        tag = '-'.join([element, spin, orbital])
                        color = colors_and_labels[tag]['color']
                        label = colors_and_labels[tag]['label']
                        d = VASPDOSAnalysis(j, doscar=doscar).energies_to_populations(element=element, orbital=orbital, spin=spin)
                        if spin == 'down':
                            flip_sign = True
                        else:
                            flip_sign = False
                        d = ProcessDOS(d, shift=shift_value, 
                                        flip_sign=flip_sign,
                                        normalization=normalization,
                                        cb_shift=cb_shift,
                                        vb_shift=vb_shift).energies_to_populations
                               
                        energies = sorted(list(d.keys()))
                        populations = [d[E] for E in energies]
                        occ_energies = [E for E in energies if E <= occupied_up_to]
                        occ_populations = [d[E] for E in occ_energies]
                        unocc_energies = [E for E in energies if E > occupied_up_to]
                        unocc_populations = [d[E] for E in unocc_energies]    
                        if smearing:
                            occ_populations = gaussian_filter1d(occ_populations, smearing)
                            unocc_populations = gaussian_filter1d(unocc_populations, smearing)
                        ax = plt.plot(occ_populations, occ_energies, color=color, label=label, alpha=0.9, lw=dos_lw)
                        ax = plt.plot(unocc_populations, unocc_energies, color=color, label='__nolegend__', alpha=0.9, lw=dos_lw)                
                        ax = plt.fill_betweenx(occ_energies, occ_populations, color=color, alpha=0.2, lw=0)                               
            ax = plt.xticks(xticks[1])
            ax = plt.yticks(yticks[1])
            ax = plt.subplots_adjust(hspace=0.000, wspace=0.000)
            if xticks[0] == False:
                ax = plt.gca().xaxis.set_ticklabels([])      
            if yticks[0] == False or i!=0:
                ax = plt.gca().yaxis.set_ticklabels([])
            if i == 2:
                ax = plt.xlabel(xlabel)
            if i == 0:
                ax = plt.ylabel(ylabel)
            ax = plt.xlim(xlim)
            ax = plt.ylim(ylim)    
            if legend:
                ax = plt.legend(loc='upper right')
                ax = plt.title(j.split('/')[-1].split('_')[0], y=1.01)
        if show:
            plt.savefig('/global/cscratch1/sd/yychoi/JCESR/MgPostSpinels/Voltage/' + 'DOStest' + '.png')
            plt.show()
        return ax
    
    def cohp(self, structure = 'MnTiO4',
             pairs_to_plot=['total'],
             colors_and_labels = {'total' : {'color' : 'black','label' : 'total'}},                                  
             xlim=(-0.5, 0.5), ylim=(-10, 0), 
             xticks=(False, np.arange(-0.5, 0.5, 0.2)), yticks=(False, np.arange(-10, 0.1, 2)),
             xlabel=r'$-COHP/bond$', ylabel=r'$E-E_F\/(eV)$',
             legend=True,
             smearing=1,
             shift=0, normalization='bonds',
             show=False,
             zero_line='horizontal'):
        """
        Args:
            calc_dir (str) - path to calculation with DOSCAR
            pairs_to_plot (list) - list of 'el1_el2' to plot and/or 'total'
            colors_and_labels (dict) - {pair (str) : {'color' : color (str), 'label' : label (str)}}
            tdos (str or bool) - if not False, 'DOSCAR' or 'DOSCAR.losbter' to retrieve tDOS from
            xlim (tuple) - (xmin (float), xmax (float))
            ylim (tuple) - (ymin (float), ymax (float))
            xticks (tuple) - (bool to show label or not, (xtick0, xtick1, ...))
            yticks (tuple) - (bool to show label or not, (ytick0, ytick1, ...))
            xlabel (str) - x-axis label
            ylabel (str) - y-axis label
            legend (bool) - include legend or not
            smearing (float or False) - std. dev. for Gaussian smearing of DOS or False for no smearing
            shift (float or 'Fermi') - if 'Fermi', make Fermi level 0; else shift energies by shift
            normalization ('electron', 'atom', or False) - divide populations by number of electrons, number of atoms, or not at all
            show (bool) - if True, show figure; else just return ax
            zero_line (str) - if 'horizontal', 'vertical', 'both', or False
                       
        Returns:
            matplotlib axes object
        """
        self.set_rc_params()
        caldir = [self.calc_dir + '/' + structure + '_pnma_second_lobster', self.calc_dir + '/' + structure + '_fd3m_second_lobster']
        if show == True:
            mpl.rcParams['font.size'] = 10
            mpl.rcParams['xtick.labelsize'] = 10
            mpl.rcParams['ytick.labelsize'] = 10
            mpl.rcParams['legend.fontsize'] = 10
            fig = plt.figure(figsize=(5, 7))
        
        for i, j in enumerate(caldir):
            ax = plt.subplot(int('1' + str(len(caldir)) + str(i+1)))         
            occupied_up_to = shift
            dos_lw = 1
            for count, pair in enumerate(pairs_to_plot):
                color = colors_and_labels[pair]['color']
                label = colors_and_labels[pair]['label']
                if normalization == 'electron':
                    normalization = VASPBasicAnalysis(j).params_from_outcar(num_params=['NELECT'], str_params=[])['NELECT']
                elif normalization == 'atom':
                    normalization = VASPBasicAnalysis(j).nsites
                else:
                    normalization = LOBSTERAnalysis(j).count_bonds()[label]
                    print(normalization)
                d = LOBSTERAnalysis(j).energies_to_populations(element_pair=pair)
                bond = LOBSTERAnalysis(j).bond_strength(element_pair=pair, energy = list(ylim))
                totalbond = LOBSTERAnalysis(j).bond_strength(element_pair=pair, average = False, energy = list(ylim))/normalization
                flip_sign = True
                d = ProcessDOS(d, shift=0, 
                               flip_sign=flip_sign,
                               normalization=normalization).energies_to_populations
                energies = sorted(list(d.keys()))
                populations = [d[E] for E in energies]
                occ_energies = [E for E in energies if E <= occupied_up_to]
                occ_populations = [d[E] for E in occ_energies]
                unocc_energies = [E for E in energies if E > occupied_up_to]
                unocc_populations = [d[E] for E in unocc_energies]
                if smearing:
                    occ_populations = gaussian_filter1d(occ_populations, smearing)
                    unocc_populations = gaussian_filter1d(unocc_populations, smearing)
                ax = plt.plot(occ_populations, occ_energies, color=color, label=label, alpha=0.9, lw=dos_lw)
                ax = plt.plot(unocc_populations, unocc_energies, color=color, label='__nolegend__', alpha=0.9, lw=dos_lw) 
                ax = plt.fill_betweenx(occ_energies, occ_populations, color=color, alpha=0.2, lw=0)
                ax = plt.plot(xlim, [-bond, -bond], lw=1, ls='--', color=color)
                ax = plt.annotate(label + ' Total = ' + str(-totalbond)[0:6] + 'eV', xy=(0.15, 0.05 + count/20), xytext=(-15,2), 
                            ha='left', va='top', xycoords='axes fraction', textcoords='offset points', fontsize=10, color = color)
            ax = plt.xticks(xticks[1])
            ax = plt.yticks(yticks[1])
            ax = plt.subplots_adjust(hspace=0.000, wspace=0.000)
            if xticks[0]:
                ax = plt.gca().xaxis.set_ticklabels([])      
            if yticks[0] or i != 0:
                ax = plt.gca().yaxis.set_ticklabels([])
            ax = plt.xlabel(xlabel)
            if i == 0:
                ax = plt.ylabel(ylabel)
                ax = plt.title(structure + '_pnma', y=1.03)
            else:
                ax = plt.title(structure + '_fd3m', y=1.03)
            ax = plt.xlim(xlim)
            ax = plt.ylim(ylim)
            if zero_line in ['horizontal', 'both']:
                ax = plt.plot(xlim, [0, 0], lw=1, ls='--', color='black')
            if zero_line in ['vertical', 'both']:
                ax = plt.plot([0, 0], ylim, lw=1, ls='--', color='black')
            if legend:
                ax = plt.legend(loc='upper right')
        if show:
            plt.savefig('/global/cscratch1/sd/yychoi/JCESR/MgPostSpinels/' + structure + '_COHP' + '.png')
            plt.show()
        return ax