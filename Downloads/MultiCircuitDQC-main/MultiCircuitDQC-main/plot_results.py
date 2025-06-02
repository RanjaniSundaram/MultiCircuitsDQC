'''
NOTE the experiment results must be in JSON format. Some work to do at the beginning,
     but you will feel good before the deadline
     example: https://raw.githubusercontent.com/caitaozhan/deeplearning-localization/master/result/12.12/log-dtxf-5000
'''

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mticker
import PIL


class Plot:
    '''plot the results'''

    plt.rcParams['font.size'] = 50
    plt.rcParams['lines.linewidth'] = 10
    plt.rcParams['lines.markersize'] = 15
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['axes.labelweight'] = 'bold'

    LEGEND_ALGO = {'dp_opt': 'DP-OPT',
                   'dp': 'DP-Approx',
                   'naive': 'SP',
                   'dp_alt': 'Balanced-Tree',
                   'sig5': 'Q-CAST (W=5)',
                   'sig_multi5': 'Q-CAST (W=5)',
                   'sig10': 'Q-CAST (W=10)',
                   'sig_multi10': 'Q-CAST (W=10)',
                   'lp': 'LP',
                   'delft_lp': 'Delft-LP',
                   'dp_iter': 'ITER-DPA',
                   'lp_alt': 'ITER-Bal',
                   'naive_iter': 'ITER-Naive',
                   'single': 'Greedy',
                   'single_non_shortest': 'Non-Shortest',
                   'single_plus_non_shortest': 'Greedy',
                   'single_naive': 'Naive',
                   'single_cluster': 'Clustering Heuristic',
                   'multi': 'Iter-Greedy',
                   'multi_naive': 'Iter-Heuristic',
                   'without_sls': 'Non-SLs',
                   'single_no_deletion_latency': 'Greedy-NDL',
                   'single_no_deletion': 'Greedy-NDB',
                   'ghz_fro': 'FRO',
                   'ghz_fst_latency': 'FST-LATENCY',
                   'ghz_fst_edge': 'FST-EDGE',
                   'naive_star': 'CENTRAL'
                   }

    LEGEND_PROTO = {'dp_opt_proto': 'DP-OPT',
                    'dp_proto': 'DP-Approx',
                    'naive': 'SP',
                    'naive_iter': 'ITER-SP',
                    'dp_proto_no_throttle': 'DP-Approx-Non-Throttle',
                    'dp_alt_proto': 'Balanced-Tree',
                    'sig_proto5': 'Q-CAST (W=5)',
                    'sig_multi_proto5': 'Q-CAST (W=5)',
                    'sig_proto10': 'Q-CAST (W=10)',
                    'sig_multi_proto10': 'Q-CAST (W=10)',
                    'lp_proto': 'LP',
                    'delft_lp_proto': 'Delft-LP',
                    'dp_iter_proto': 'ITER-DPA',
                    'dp_iter_proto_no_throttle': 'ITER-DPA-Non-Throttle',
                    'lp_alt_proto': 'ITER-Bal',
                    'caleffi_proto': 'Caleffi',
                    'single': 'GG-SP',
                    'single_non_shortest': 'Non-Shortest',
                    'single_plus_non_shortest': 'Generalized Greedy (GG)',
                    'single_naive': 'Naive',
                    'single_cluster': 'Clustering Approach (CLUS)',
                    'multi': 'Iter-Greedy',
                    'multi_naive': 'Iter-Heuristic',
                    'without_sls': 'Non-SLs',
                    'single_no_deletion_latency': 'Greedy-NDL',
                    'single_no_deletion': 'Pure Greedy',
                    'ghz_fro': 'FRO',
                    'ghz_fst_latency': 'GF-LATENCY',
                    'ghz_fst_edge': 'GF-EDGE',
                    'ghz_naive': 'CENTRAL',
                    'ghz_fro_ana': 'FRO Analyt.',
                    'ghz_fst_latency_ana': 'GF-LATENCY Analyt.',
                    'ghz_fst_edge_ana': 'GF-EDGE Analyt.',
                    'ghz_naive_ana': 'STAR-GRAPH Analyt',
                    'ghz_star_exp': 'STAR-EXPANSION'
                    }

    LEGEND_NO_THROTTLE = {"dp_proto_no_throttle", "dp_iter_proto_no_throttle"}

    COLOR_ALGO = {'dp_opt': 'silver',
                  'dp': 'r',
                  'naive': 'm',
                  'naive_iter': 'm',
                  'dp_alt': 'b',
                  'sig5': 'limegreen',
                  'sig_multi5': 'limegreen',
                  'sig10': 'cyan',
                  'sig_multi10': 'cyan',
                  'lp': 'black',
                  'delft_lp': 'tab:brown',
                  'dp_iter': 'r',
                  'lp_alt': 'b',
                  'single': 'r',
                  'single_non_shortest': 'darkviolet',
                  'single_plus_non_shortest': 'maroon',
                  'single_no_deletion_latency': 'darkgreen',
                  'single_no_deletion': 'darkgreen',
                  'single_naive': 'b',
                  'single_cluster': 'tab:brown',
                  'multi': 'r',
                  'multi_naive': 'b',
                  'without_sls': 'black',
                  'ghz_fro': 'darkgreen',
                  'ghz_fst_latency': 'b',
                  'ghz_fst_edge': 'r',
                  'naive_star': 'black',
                  'ghz_star_exp': 'brown'
                  }

    COLOR_PROTO = {'dp_opt_proto': 'silver',
                   'dp_proto': 'r',
                   'naive': 'm',
                   'naive_iter': 'm',
                   'dp_proto_no_throttle': 'r',
                   'dp_alt_proto': 'b',
                   'sig_proto5': 'limegreen',
                   'sig_multi_proto5': 'limegreen',
                   'sig_proto10': 'cyan',
                   'sig_multi_proto10': 'cyan',
                   'lp_proto': 'black',
                   'delft_lp_proto': 'tab:brown',
                   'dp_iter_proto': 'r',
                   'dp_iter_proto_no_throttle': 'r',
                   'lp_alt_proto': 'b',
                   'caleffi_proto': 'orange',
                   'single': 'darkgreen',
                   'single_naive': 'b',
                   'single_non_shortest': 'darkviolet',
                   'single_plus_non_shortest': 'darkgreen',
                   'single_no_deletion_latency': 'darkgreen',
                   'single_no_deletion': 'darkgreen',
                   'single_cluster': 'r',
                   'multi': 'r',
                   'multi_naive': 'b',
                   'without_sls': 'black',
                   'ghz_fro': 'darkgreen',
                   'ghz_fst_latency': 'b',
                   'ghz_fst_edge': 'r',
                   'ghz_naive_ana': 'black',
                   'ghz_fro_ana': 'darkgreen',
                   'ghz_fst_latency_ana': 'b',
                   'ghz_fst_edge_ana': 'r',
                   'ghz_naive': 'black',
                   'ghz_star_exp': 'brown'
                   }

    COLOR_FIDELITY = {'dp_fidelity': 'b',
                      'dp_alt_fidelity': 'b'}

    LINE = {'algo': '-',
            'proto': '--',
            'no_throttle': ':',
            'single': '--',
            'single_non_shortest': '-',
            'single_plus_non_shortest': '-',
            'single_no_deletion_latency': '-.',
            'single_no_deletion': ':',
            'single_naive': '-',
            'single_cluster': '-',
            'multi': '--',
            'multi_naive': '--',
            'without_sls': '-',
            'ghz_fro': '-',
            'ghz_fst_latency': '-',
            'ghz_fst_edge': '-',
            'ghz_naive': '-',
            'ghz_fro_ana': '-.',
            'ghz_fst_latency_ana': '-.',
            'ghz_fst_edge_ana': '-.',
            'ghz_naive_ana': '-.',
            'ghz_star_exp': '-'
            }

    @staticmethod
    def graph(plotting_data: dict, filename: str):
        '''temporary plot
        '''
        plotting_data['x'] = ("distance", ["10-15", "15-20", "20-25", '25-30'])

        plotting_data['dp_proto'] = [28.82, 19.2, 17.6, 16.88]
        plotting_data['dp_alt_proto'] = [27.12, 19.5, 16.1, 14.9]
        plotting_data['sig_proto5'] = [1.95, 1.11, 1.05, 0.35]
        plotting_data['sig_proto10'] = [6, 2.2, 1.7, 1.2]

        # algorithm
        x = plotting_data['x'][1]  # number of points
        fig, ax = plt.subplots(figsize=(24, 20))
        fig.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)

        for method, val in plotting_data.items():
            if method != 'x' and len(val) == len(x):
                if method in Plot.LEGEND_ALGO:
                    line = Plot.LINE['algo']
                    label = Plot.LEGEND_ALGO[method]
                    color = Plot.COLOR_ALGO[method]
                else:
                    line = Plot.LINE['proto']
                    label = Plot.LEGEND_PROTO[method]
                    color = Plot.COLOR_PROTO[method]
                plt.plot(x, val, label=label, color=color,
                         linestyle=line)

        # # plotting dp
        # if 'dp' in plotting_data and len(plotting_data['dp']) == n:
        #     plt.plot(plotting_data['x'], plotting_data['dp'], label=Plot.LEGEND['dp'], color=Plot.COLOR['dp'],
        #              linestyle=Plot.LINE['algo'])
        # if 'dp_proto' in plotting_data and len(plotting_data['dp_proto']) == n:
        #     plt.plot(plotting_data['x'], plotting_data['dp_proto'], label=Plot.LEGEND['dp_proto'], color=Plot.COLOR['dp'],
        #              linestyle=Plot.LINE['proto'])
        # plt.plot(X, dp_alt, label=Plot.LEGEND['dp_alt'], color=Plot.COLOR['dp_alt'], linestyle=Plot.LINE['dp_alt'])
        # plt.plot(X, dp_alt_proto, label=Plot.LEGEND['dp_alt_proto'], color=Plot.COLOR['dp_alt_proto'], linestyle=Plot.LINE['dp_alt_proto'])
        # plt.plot(X, sig, label=Plot.LEGEND['sig'], color=Plot.COLOR['sig'], linestyle=Plot.LINE['sig'])
        # plt.plot(X, sig_proto, label=Plot.LEGEND['sig_proto'], color=Plot.COLOR['sig_proto'], linestyle=Plot.LINE['sig_proto'])
        ax.legend(fontsize=50)
        if plotting_data['x'][0] == "atomic_bsm":
            ax.set_xlabel('Atomic BSM Success Rate')
        elif plotting_data['x'][0] == "optical_bsm":
            ax.set_xlabel('Optical BSM Success Rate')
        elif plotting_data['x'][0] == "num_nodes":
            ax.set_xlabel('# of nodes')
        elif plotting_data['x'][0] == "edge_density":
            ax.set_xlabel('Edge Density %')
        elif plotting_data['x'][0] == "distance":
            ax.set_xlabel("Distance range between src-dst (km)")
        ax.set_ylabel('EPs/s')
        plt.savefig(filename)

    @staticmethod
    def QNR_SP():
        '''Single Pair, 4 subplots, each subplot is comparing 5 algorithms
        '''

        def helper(ax, plotting_data):
            x = plotting_data['x'][1]  # number of points
            for method, val in plotting_data.items():
                if method != 'x' and len(val) == len(x):
                    if plotting_data['x'][0] == 'num_nodes':  # reset the x axis for num_nodes, later on relabel them
                        x = [1, 2, 3, 4, 5, 6, 7, 8]
                    line = Plot.LINE['proto']
                    label = Plot.LEGEND_PROTO[method]
                    color = Plot.COLOR_PROTO[method]
                    ax.plot(x, val, label=label, color=color, linestyle=line)

            if plotting_data['x'][0] == "atomic_bsm":
                ax.set_xlabel('Atomic BSM Success Rate')
            elif plotting_data['x'][0] == "num_nodes":
                ax.set_xlabel('# of Nodes')
                ax.set_xticks(range(1, 9))
                ax.set_xticklabels(['25', '50', '75', '100', '200', '300', '400', '500'], fontsize=45)
            elif plotting_data['x'][0] == "edge_density":
                ax.set_xlabel('Edge Density %')
            elif plotting_data['x'][0] == "distance":
                ax.set_xlabel("Distance range between src-dst (km)")
            ax.set_ylabel('EP/s')

            ax.tick_params(axis='x', direction='in', length=10, width=3, pad=10)
            ax.tick_params(axis='y', direction='in', length=10, width=3, pad=10)

        # the plot
        fig, (ax0, ax1, ax2, ax3) = plt.subplots(1, 4, figsize=(58, 13))
        fig.subplots_adjust(left=0.04, right=0.99, top=0.84, bottom=0.2)

        # ax0: varying # of nodes
        plotting_data = {}
        plotting_data['x'] = ("num_nodes", [25, 50, 75, 100, 200, 300, 400, 500])
        plotting_data['dp_opt_proto'] = [6.28, 8.14, 19.01, 30.479999999999997, 28.7, 28.1, 33.96, 43.8]
        plotting_data['dp_proto'] = [5.50, 7.48, 16.9, 26.439999999999998, 24.52, 28.02, 29.82, 39.46]
        plotting_data['naive'] = [2.74, 4.32, 14.353, 16.853, 9.013, 12.927, 13.560, 21.407]
        plotting_data['dp_alt_proto'] = [5.3, 6.02, 16.5, 26.76, 23.0, 24.92, 29.499999999999993, 39.480000000000004]
        plotting_data['sig_proto10'] = [0.12, 0.0511, 1.59, 6.880006174581392, 1.95, 4.95, 4.9799999999999995, 10.37]
        plotting_data['sig_proto5'] = [0.005, 0.000595, 0.05, 3.5800002302396527, 1.0499999999999998,
                                       2.2800000000000002, 2.3200000000000003, 5.64]
        helper(ax0, plotting_data)

        # ax1: varying (s, d) distance
        plotting_data = {}
        plotting_data['x'] = ("distance", ["10-15", "15-20", "20-25", "25-30", "30-40", "40-50"])
        plotting_data['dp_opt_proto'] = [41.379999999999995, 30.479999999999997, 30.660000000000004, 18.9, 15.62, 11.4]
        plotting_data['dp_proto'] = [38.68, 26.44, 28.06, 17.660000000000004, 15.959999999999999, 9.76]
        plotting_data['naive'] = [32.446666666666665, 24.146666666666665, 24.273333333333333, 10.926666666666666,
                                  10.72, 6.4333333333333345]
        plotting_data['dp_alt_proto'] = [37.480000000000004, 26.76, 27.060000000000002, 12.819999999999999,
                                         14.039999999999997, 10.06]
        plotting_data['sig_proto10'] = [12.23, 6.880006174581392, 1.7312094426384648, 4.45, 0.52, 0.5912454009367684]
        plotting_data['sig_proto5'] = [6.660000000000001, 3.5801, 0.9001670688284158, 2.21, 0.3703704910570851,
                                       0.29017314382756276]
        helper(ax1, plotting_data)
        ax1.set_xlabel("Distance range between src-dst (km)", fontsize=45)
        ax1.set_xticklabels(["10-15", "15-20", "20-25", "25-30", "30-40", "40-50"], rotation=10, fontsize=42)

        # ax2: varying bsm probability
        plotting_data = {}
        plotting_data['x'] = ("atomic_bsm", [0.2, 0.3, 0.4, 0.5, 0.6])
        plotting_data['dp_opt_proto'] = [9.22, 23.98, 30.479999999999997, 41.459999999999994, 73.08000000000001]
        plotting_data['dp_proto'] = [9.08, 23.22, 26.439999999999998, 40.14, 63.54]
        plotting_data['naive'] = [6.1, 14.153333333333332, 16.853333333333335, 25.826666666666664, 31.646666666666665]
        plotting_data['dp_alt_proto'] = [9.459999999999999, 23.92, 26.76, 37.620000000000005, 58.81999999999999]
        plotting_data['sig_proto10'] = [1.970400858054468, 4.88, 6.880006174581392, 6.5200000000000005, 15.52]
        plotting_data['sig_proto5'] = [0.9001022398138734, 2.55, 3.5800002302396527, 3.6399999999999997, 8.22]
        helper(ax2, plotting_data)

        # ax3: varying density of edges
        plotting_data = {}
        plotting_data['x'] = ("edge_density", [1, 2, 3, 4, 5])
        plotting_data['dp_opt_proto'] = [5.9, 9.040000000000001, 30.479999999999997, 34.239999999999995, 39.98]
        plotting_data['dp_proto'] = [5.139999999999999, 8.5, 26.439999999999998, 33.12, 34.9]
        plotting_data['naive'] = [3.9, 6.573333333333332, 16.853333333333335, 15.406666666666666, 21.586666666666666]
        plotting_data['dp_alt_proto'] = [4.38, 6.7, 26.76, 32.620000000000005, 34.36]
        plotting_data['sig_proto10'] = [7.134169165132622e-08, 0.06189283357693228, 6.880006174581392,
                                        6.639999999999999, 10.129999999999999]
        plotting_data['sig_proto5'] = [1.3162793408471259e-09, 0.021781373057353663, 3.5800002302396527, 3.25, 5.13]
        helper(ax3, plotting_data)

        # one legend for all subplots
        handles, labels = ax0.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=6, fontsize=50, handlelength=4)
        plt.figtext(0.13, 0.01, '(a)', weight='bold')
        plt.figtext(0.38, 0.01, '(b)', weight='bold')
        plt.figtext(0.63, 0.01, '(c)', weight='bold')
        plt.figtext(0.87, 0.01, '(d)', weight='bold')
        plt.savefig('results/real_results/QNR-SP2.png')

    @staticmethod
    def QNR_SP_NEW():
        '''Single Pair, 4 subplots, each subplot is comparing 5 algorithms
        '''

        def helper(ax, plotting_data):
            x = plotting_data['x'][1]  # number of points
            for method, val in plotting_data.items():
                if method != 'x' and len(val) == len(x):
                    if plotting_data['x'][0] == 'num_nodes':  # reset the x axis for num_nodes, later on relabel them
                        x = [1, 2, 3, 4, 5, 6, 7, 8]
                    line = Plot.LINE['proto']
                    label = Plot.LEGEND_PROTO[method]
                    color = Plot.COLOR_PROTO[method]
                    ax.plot(x, val, label=label, color=color, linestyle=line)

            if plotting_data['x'][0] == "atomic_bsm":
                ax.set_xlabel('Atomic BSM Success Rate')
            elif plotting_data['x'][0] == "num_nodes":
                ax.set_xlabel('# of Nodes')
                ax.set_xticks(range(1, 9))
                ax.set_xticklabels(['25', '50', '75', '100', '200', '300', '400', '500'], fontsize=45)
            elif plotting_data['x'][0] == "edge_density":
                ax.set_xlabel('Edge Density %')
                # ax.set_xticklabels(['2', '4', '6', '8', '10'], fontsize=45)
            elif plotting_data['x'][0] == "distance":
                ax.set_xlabel("Distance range between src-dst (km)")
            ax.set_ylabel('EP/s')

            ax.tick_params(axis='x', direction='in', length=10, width=3, pad=10)
            ax.tick_params(axis='y', direction='in', length=10, width=3, pad=10)

        # the plot
        fig, (ax0, ax1, ax2, ax3) = plt.subplots(1, 4, figsize=(58, 13))
        fig.subplots_adjust(left=0.04, right=0.99, top=0.84, bottom=0.2)

        # ax0: varying # of nodes
        plotting_data = {}
        plotting_data['x'] = ("num_nodes", [25, 50, 75, 100, 200, 300, 400, 500])
        plotting_data['dp_opt_proto'] = [4.392, 3.430, 4.984, 13.00, 16.704, 17.460, 17.982, 22.876]
        plotting_data['dp_proto'] =     [3.436, 2.984, 4.108, 11.124, 12.296, 14.292, 15.944, 20.324]
        plotting_data['dp_alt_proto'] = [3.360, 2.440, 3.792, 7.628, 12.044, 11.228, 14.612, 18.424]
        plotting_data['naive'] =        [2.156, 2.424, 3.108, 7.876, 9.224, 3.716, 6.724, 13.18]
        plotting_data['sig_proto10'] =  [0.000, 0.063, 0.804, 2.672, 3.222, 1.118, 2.216, 5.334]
        plotting_data['sig_proto5'] =   [0.000, 0.010, 0.438, 1.493, 1.779, 0.598, 1.086, 2.598]
        helper(ax0, plotting_data)

        # ax1: varying (s, d) distance
        plotting_data = {}
        plotting_data['x'] = ("distance", ["10-20", "20-30", "30-40", "40-50", "50-60", "60-70"])
        plotting_data['dp_opt_proto'] = [15.156, 13.82, 13.0, 9.212, 7.716, 8.6]
        plotting_data['dp_proto'] =     [13.468, 12.372, 11.124, 7.784, 6.572, 8.356]
        plotting_data['dp_alt_proto'] = [12.024, 11.592, 7.628, 7.432, 6.496, 7.492]
        plotting_data['naive'] =        [11.804, 8.176, 7.876, 6.532, 3.592, 4.568]
        plotting_data['sig_proto10'] = [9.151, 3.118, 2.672, 2.666, 0.094, 0.42]
        plotting_data['sig_proto5'] =   [5.044, 2.038, 1.493, 1.497, 0.033, 0.212]

        helper(ax1, plotting_data)
        ax1.set_xlabel("Distance range between src-dst (km)", fontsize=45)
        ax1.set_xticklabels(["10-20", "20-30", "30-40", "40-50", "50-60", "60-70"], rotation=10, fontsize=42)

        # ax2: varying bsm probability
        plotting_data = {}
        plotting_data['x'] = ("atomic_bsm", [0.2, 0.3, 0.4, 0.5, 0.6])
        plotting_data['dp_opt_proto'] = [2.92, 2.392, 13.0, 13.24, 19.804]
        plotting_data['dp_proto'] =     [2.472, 2.096, 11.124, 12.148, 17.704]
        plotting_data['dp_alt_proto'] = [2.544, 1.992, 7.628, 10.824, 15.448]
        plotting_data['naive'] =        [2.352, 0.36, 7.876, 7.752, 11.68]
        plotting_data['sig_proto10'] =  [0.198, 0.002, 2.672, 0.696, 0.206]
        plotting_data['sig_proto5'] =   [0.106, 0.001, 1.493, 0.359, 0.054]
        helper(ax2, plotting_data)

        # ax3: varying density of edges
        plotting_data = {}
        plotting_data['x'] = ("edge_density", [2, 4, 6, 8, 10])
        plotting_data['dp_opt_proto'] = [0.384, 3.516, 13.0, 13.16, 28.908]
        plotting_data['dp_proto'] =     [0.344, 3.345, 11.124, 11.93, 25.381]
        plotting_data['dp_alt_proto'] = [0.152, 3.068, 7.628, 10.392, 24.416]
        plotting_data['naive'] =        [0.084, 2.932, 7.876, 7.356, 18.924]
        plotting_data['sig_proto10'] =  [0.0, 0.003, 2.672, 1.073, 10.27]
        plotting_data['sig_proto5'] =   [0.0, 0.001, 1.493, 0.490, 5.162]
        helper(ax3, plotting_data)

        # one legend for all subplots
        handles, labels = ax0.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=6, fontsize=50, handlelength=4)
        plt.figtext(0.13, 0.01, '(a)', weight='bold')
        plt.figtext(0.38, 0.01, '(b)', weight='bold')
        plt.figtext(0.63, 0.01, '(c)', weight='bold')
        plt.figtext(0.87, 0.01, '(d)', weight='bold')
        plt.savefig('results/new_real_results/QNR-SP.TIF', format="tiff", pil_kwargs={"compression": "tiff_lzw"})
        plt.savefig('results/new_real_results/QNR-SP.png')

    @staticmethod
    def QNR():
        '''Multiple Pair, 4 subplots, each subplot is comparing 6 algorithms
        '''

        def helper(ax, plotting_data):
            for method, val in plotting_data.items():
                x = plotting_data['x'][1]  # number of points
                if plotting_data['x'][0] == 'num_nodes':  # reset the x axis for num_nodes, later on relabel them
                    if method == 'lp_proto':
                        x = [1, 2, 3, 4]
                    else:
                        x = [1, 2, 3, 4, 5, 6, 7, 8]
                if method != 'x' and len(val) == len(x):
                    line = Plot.LINE['proto']
                    label = Plot.LEGEND_PROTO[method]
                    color = Plot.COLOR_PROTO[method]
                    ax.plot(x, val, label=label, color=color, linestyle=line)

            if plotting_data['x'][0] == "atomic_bsm":
                ax.set_xlabel('Atomic BSM Success Rate')
                ax.set_ylim([-20, 410])
            elif plotting_data['x'][0] == "num_nodes":
                ax.set_xlabel('# of Nodes', labelpad=13)
                ax.set_ylim([-15, 260])
                ax.set_yticks(range(0, 261, 50))
                ax.set_xticks(range(1, 9))
                ax.set_xticklabels(['25', '50', '75', '100', '200', '300', '400', '500'], fontsize=45)
            elif plotting_data['x'][0] == "edge_density":
                ax.set_xlabel('Edge Density %')
                ax.set_ylim([-10, 205])
            elif plotting_data['x'][0] == "src_dst_pair":
                ax.set_xlabel("# of (Source, Destination) Pairs")
                ax.set_ylim([-10, 220])
            ax.set_ylabel('EP/s')

            ax.tick_params(axis='x', direction='in', length=10, width=3, pad=10)
            ax.tick_params(axis='y', direction='in', length=10, width=3, pad=10)

        # the plot
        fig, (ax0, ax1, ax2, ax3) = plt.subplots(1, 4, figsize=(58, 13))
        fig.subplots_adjust(left=0.04, right=0.99, top=0.84, bottom=0.2)

        # ax0: varying # of nodes
        plotting_data = {}
        plotting_data['x'] = ("num_nodes", [25, 50, 75, 100, 200, 300, 400, 500])
        plotting_data['lp_proto'] = [32.54, 78.67999999999999, 107.0333, 118.35]
        plotting_data['dp_iter_proto'] = [30.979999999999997, 75.34, 85.033, 104.68, 195.82, 210.96, 230.84, 251.36]
        plotting_data['lp_alt_proto'] = [25.839999999999996, 67.6, 72.425, 83.84, 164.06, 181.15999999999997, 194.4,
                                         223.72]
        plotting_data['sig_proto10'] = [0.0010356661178252016, 2.6125, 6.9, 10.77, 19.1, 23.15, 34.02,
                                        41.33999999999999]
        plotting_data['sig_proto5'] = [2.400115914729191, 3.07, 4.075, 5.15, 8.8625, 10.59, 19.359999999999996,
                                       24.240000000000002]
        plotting_data['delft_lp_proto'] = [0.5206157834621028, 0.6314173172157675, 0.877, 1.61, 3.1164854489737572,
                                           3.61, 4.109999999999999, 5.0100000000000002]
        helper(ax0, plotting_data)

        # ax1: varying (s, d) distance
        plotting_data = {}
        plotting_data['x'] = ("src_dst_pair", [1, 2, 3, 4, 5])
        plotting_data['lp_proto'] = [52.39999999999999, 109.08, 118.35, 132.85999999999999, 209.58]
        plotting_data['dp_iter_proto'] = [46.94, 101.18, 104.68, 124.24, 193.29999999999998]
        plotting_data['lp_alt_proto'] = [37.74, 83.3, 83.64, 97.52000000000001, 160.38]
        plotting_data['sig_proto10'] = [3.4671981267183147, 5.931386786115667, 4.995376942477191, 5.104162354501141,
                                        12.919388511769771]
        plotting_data['sig_proto5'] = [2.5731838060501326, 6.4799999999999995, 3.13, 3.11, 6.390000000000001]
        plotting_data['delft_lp_proto'] = [0.45004025491297517, 1.5310993539912547, 0.6314173172157675,
                                           0.6917968934859785, 1.025]
        helper(ax1, plotting_data)

        # ax2: varying bsm probability
        plotting_data = {}
        plotting_data['x'] = ("atomic_bsm", [0.2, 0.3, 0.4, 0.5, 0.6])
        plotting_data['lp_proto'] = [24.299999999999997, 62.18000000000001, 118.35, 256.12, 399.2199999999999]
        plotting_data['dp_iter_proto'] = [22.96, 56.08, 104.68, 231.4, 353.5]
        plotting_data['lp_alt_proto'] = [19.86, 46.88, 83.64, 199.7, 304.29999999999995]
        plotting_data['sig_proto10'] = [1.0600725101882318, 3.8629992540656786, 4.995376942477191, 18.922362203404006,
                                        20.639031662919443]
        plotting_data['sig_proto5'] = [0.79, 3.08, 3.13, 10.11, 13.419999999999998]
        plotting_data['delft_lp_proto'] = [0.14099783921469666, 0.6206980728112691, 0.6314173172157675,
                                           1.5304098606083092, 2.223340912109654]
        helper(ax2, plotting_data)

        # ax3: varying density of edges
        plotting_data = {}
        plotting_data['x'] = ("edge_density", [1, 2, 3, 4, 5])
        plotting_data['lp_proto'] = [21.275, 102.89999999999998, 118.35, 158.12, 185.52]
        plotting_data['dp_iter_proto'] = [13.12, 94.46, 104.68, 144.3, 171.04]
        plotting_data['lp_alt_proto'] = [12.8, 82.94, 83.64, 121.46000000000001, 149.46]
        plotting_data['sig_proto10'] = [0.20516523303797757, 7.343059693328976, 4.99, 9.794024772529468,
                                        17.717311833902173]
        plotting_data['sig_proto5'] = [0.020578706150255366, 4.1, 3.13, 6.45, 12.440000000000001]
        plotting_data['delft_lp_proto'] = [0.0002771349882495765, 0.7611939937949234, 0.6314173172157675,
                                           1.1036908844464428, 2.4300000000000006]
        helper(ax3, plotting_data)

        # one legend for all subplots
        handles, labels = ax1.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=6, fontsize=55, handlelength=3.6)
        plt.figtext(0.13, 0.01, '(a)', weight='bold')
        plt.figtext(0.38, 0.01, '(b)', weight='bold')
        plt.figtext(0.63, 0.01, '(c)', weight='bold')
        plt.figtext(0.87, 0.01, '(d)', weight='bold')
        plt.savefig('results/real_results/QNR.png')

    @staticmethod
    def QNR_NEW():
        '''Multiple Pair, 4 subplots, each subplot is comparing 6 algorithms
        '''

        def helper(ax, plotting_data):
            for method, val in plotting_data.items():
                x = plotting_data['x'][1]  # number of points
                if plotting_data['x'][0] == 'num_nodes':  # reset the x axis for num_nodes, later on relabel them
                    if method == 'lp_proto':
                        x = [1, 2, 3, 4]
                    else:
                        x = [1, 2, 3, 4, 5, 6, 7, 8]
                if method != 'x' and len(val) == len(x):
                    line = Plot.LINE['proto']
                    label = Plot.LEGEND_PROTO[method]
                    color = Plot.COLOR_PROTO[method]
                    ax.plot(x, val, label=label, color=color, linestyle=line)

            if plotting_data['x'][0] == "atomic_bsm":
                ax.set_xlabel('Atomic BSM Success Rate')
                ax.set_ylim([-10, 165])
            elif plotting_data['x'][0] == "num_nodes":
                ax.set_xlabel('# of Nodes', labelpad=13)
                ax.set_ylim([-10, 160])
                ax.set_yticks(range(0, 150, 30))
                ax.set_xticks(range(1, 9))
                ax.set_xticklabels(['25', '50', '75', '100', '200', '300', '400', '500'], fontsize=45)
            elif plotting_data['x'][0] == "edge_density":
                ax.set_xlabel('Edge Density %')
                ax.set_ylim([-10, 85])
            elif plotting_data['x'][0] == "src_dst_pair":
                ax.set_xlabel("# of (Source, Destination) Pairs")
                ax.set_ylim([-10, 65])
            ax.set_ylabel('EP/s')

            ax.tick_params(axis='x', direction='in', length=10, width=3, pad=10)
            ax.tick_params(axis='y', direction='in', length=10, width=3, pad=10)

        # the plot
        fig, (ax0, ax1, ax2, ax3) = plt.subplots(1, 4, figsize=(58, 13))
        fig.subplots_adjust(left=0.04, right=0.99, top=0.84, bottom=0.2)

        # ax0: varying # of nodes
        plotting_data = {}
        plotting_data['x'] = ("num_nodes", [25, 50, 75, 100, 200, 300, 400, 500])
        plotting_data['lp_proto'] = [15.774, 16.667, 32.78, 47.94]
        plotting_data['dp_iter_proto'] = [15.04, 16.072, 28.472, 42.492, 74.18, 97.50, 110.276, 141.66]
        plotting_data['lp_alt_proto'] = [14.204, 14.392, 25.048, 36.876, 57.612, 79.9, 78.175, 66.208]
        plotting_data['naive_iter'] = [14.152, 13.792, 17.572, 22.44, 24.456, 24.472, 33.885, 41.984]
        plotting_data['sig_proto10'] = [0.037, 1.064, 8.697, 13.182, 3.53, 14.02, 9.053, 31.217]
        plotting_data['sig_proto5'] = [0.16, 0.521, 4.373, 7.046, 1.788, 8.418, 3.053, 21.335]
        plotting_data['delft_lp_proto'] = [0.004, 0.06, 0.661, 1.496, 0.187, 1.325, 0.27, 3.7]
        helper(ax0, plotting_data)

        # ax1: varying (s, d) distance
        plotting_data = {}
        plotting_data['x'] = ("src_dst_pair", [1, 2, 3, 4, 5])
        plotting_data['lp_proto'] = [12.32, 25.72, 47.94, 59.027, 51.127]
        plotting_data['dp_iter_proto'] = [10.744, 21.704, 42.492, 52.796, 44.596]
        plotting_data['lp_alt_proto'] = [6.704, 14.128, 36.876, 44.94, 37.4]
        plotting_data['naive_iter'] = [2.772, 4.9, 22.44, 28.888, 19.916]
        plotting_data['sig_proto10'] = [0.06, 0.255, 13.182, 14.222, 2.475]
        plotting_data['sig_proto5'] = [0.021, 0.078, 7.046, 7.578, 1.297]
        plotting_data['delft_lp_proto'] = [0.002, 0.004, 1.496, 1.503, 0.13]
        helper(ax1, plotting_data)

        # ax2: varying bsm probability
        plotting_data = {}
        plotting_data['x'] = ("atomic_bsm", [0.2, 0.3, 0.4, 0.5, 0.6])
        plotting_data['lp_proto'] = [4.008, 13.48, 47.94, 73.553, 159.8]
        plotting_data['dp_iter_proto'] = [3.768, 11.692, 42.492, 62.608, 126.8]
        plotting_data['lp_alt_proto'] = [2.9, 8.196, 36.876, 49.88, 96.236]
        plotting_data['naive_iter'] = [1.864, 3.396, 22.44, 23.4, 32.112]
        plotting_data['sig_proto10'] = [0.034, 0.068, 13.182, 3.36, 5.696]
        plotting_data['sig_proto5'] = [0.027, 0.019, 7.046, 1.647, 2.48]
        plotting_data['delft_lp_proto'] = [0.001, 0.002, 1.496, 0.2, 0.259]
        helper(ax2, plotting_data)

        # ax3: varying density of edges
        plotting_data = {}
        plotting_data['x'] = ("edge_density", [1, 2, 3, 4, 5])
        plotting_data['lp_proto'] = [10.613, 39.86, 47.94,  58.22, 74.827]
        plotting_data['dp_iter_proto'] = [9.612, 36.588, 42.492, 51.504, 65.188]
        plotting_data['lp_alt_proto'] = [8.016, 27.404, 36.876, 41.736, 50.704]
        plotting_data['naive_iter'] = [7.936, 23.18, 22.44, 11.46, 25.956]
        plotting_data['sig_proto10'] = [0.048, 3.929, 13.182, 1.122, 6.566]
        plotting_data['sig_proto5'] = [0.028, 1.983, 7.046, 0.39, 3.36]
        plotting_data['delft_lp_proto'] = [0.0, 0.274, 1.496, 0.022, 0.138]
        helper(ax3, plotting_data)

        # one legend for all subplots
        handles, labels = ax1.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=7, fontsize=50, handlelength=3.6)
        plt.figtext(0.13, 0.01, '(a)', weight='bold')
        plt.figtext(0.38, 0.01, '(b)', weight='bold')
        plt.figtext(0.63, 0.01, '(c)', weight='bold')
        plt.figtext(0.87, 0.01, '(d)', weight='bold')
        plt.savefig('results/new_real_results/QNR.TIF', format="tiff", pil_kwargs={"compression": "tiff_lzw"})
        plt.savefig('results/new_real_results/QNR.png')

    @staticmethod
    def analytical_throttle():

        def helper(ax, plotting_data):
            for method, val in plotting_data.items():
                x = plotting_data['x'][1]  # number of points
                if method == 'lp_proto' and plotting_data['x'][0] == 'num_nodes':
                    x = [25, 50, 75, 100]
                if method != 'x' and len(val) == len(x):
                    if method in Plot.LEGEND_ALGO:
                        line = Plot.LINE['algo']
                        label = Plot.LEGEND_ALGO[method]
                        color = Plot.COLOR_ALGO[method]
                        ax.plot(x, val, label=label, color=color, linestyle=line, linewidth=5)
                    else:
                        line = Plot.LINE['proto']
                        label = Plot.LEGEND_PROTO[method]
                        color = Plot.COLOR_PROTO[method]
                        ax.plot(x, val, label=label, color=color, linestyle=line)

            ax.set_ylabel('EP/s')
            ax.set_ylim([0, 81])
            ax.legend(fontsize=36, handlelength=3.4)
            ax.set_xlabel('Atomic BSM Success Rate', fontsize=45)
            ax.tick_params(axis='x', direction='in', length=10, width=3, pad=10)
            ax.tick_params(axis='y', direction='in', length=10, width=3, pad=10)

        def helper2(ax, plotting_data):
            '''for the throttle vs no-throttle
            '''
            for method, val in plotting_data.items():
                x = plotting_data['x'][1]
                if method != 'x' and len(val) == len(x):
                    line = Plot.LINE['proto']
                    label = Plot.LEGEND_PROTO[method]
                    color = Plot.COLOR_PROTO[method]

                    if method in Plot.LEGEND_NO_THROTTLE:
                        line = Plot.LINE['no_throttle']
                        color = 'lightpink'

                    ax.plot(x, val, label=label, color=color, linestyle=line)
                    if method == 'dp_proto':
                        ylim = [0, 81]
                    if method == 'dp_iter_proto':
                        ylim = [0, 401]

            ax.set_ylabel('EP/s')
            ax.set_ylim(ylim)
            ax.legend(fontsize=36, handlelength=3.6)
            ax.set_xlabel('Atomic BSM Success Rate', fontsize=45)
            ax.tick_params(axis='x', direction='in', length=10, width=3, pad=10)
            ax.tick_params(axis='y', direction='in', length=10, width=3, pad=10)

        def helper3(ax, plotting_data):
            ''' for the fairness
            '''
            for method, val in plotting_data.items():
                x = plotting_data['x'][1]
                if method != 'x' and len(val) == len(x):
                    line = Plot.LINE['proto']
                    label = Plot.LEGEND_PROTO[method]
                    color = Plot.COLOR_PROTO[method]
                    if method == 'dp_iter_proto':
                        line = '-'
                        ax.plot(x, val, label=label, color=color, linestyle=line, linewidth=5)
                        continue
                    if method == 'lp_proto':
                        line = ':'
                    if method == 'sig_proto10':
                        line = ':'
                    ax.plot(x, val, label=label, color=color, linestyle=line)

            ax.set_ylabel('Avg. # of Success Pairs')
            ax.legend(fontsize=32)
            ax.set_ylim([0, 11])
            ax.set_xticks(range(1, 11))
            ax.set_xlabel('# of (Source, Destination) Pairs', fontsize=45)
            ax.tick_params(axis='x', direction='in', length=10, width=3, pad=10)
            ax.tick_params(axis='y', direction='in', length=10, width=3, pad=10)

        # the plot
        fig, (ax0, ax1, ax2, ax3) = plt.subplots(1, 4, figsize=(58, 12))
        fig.subplots_adjust(left=0.04, right=0.99, top=0.96, bottom=0.2)

        # ax0: algorithm vs protocol, single pair, varying bsm probability
        plotting_data = {}
        plotting_data['x'] = ("atomic_bsm", [0.2, 0.3, 0.4, 0.5, 0.6])
        plotting_data['dp_proto'] = [9.08, 23.22, 26.439999999999998, 40.14, 63.54]
        plotting_data['dp_alt_proto'] = [9.459999999999999, 23.92, 26.76, 37.620000000000005, 58.81999999999999]
        plotting_data['dp'] = [9.392212503121161, 23.334619052595226, 25.598682460715725, 37.88098952770356,
                               57.58132003123875]
        plotting_data['dp_alt'] = [9.343125184848276, 23.219545446424398, 25.333155385375214, 35.127024549740085,
                                   54.919477966871206]
        helper(ax0, plotting_data)

        # ax1: Throttle vs Non-throttle, DP-Approx, single pair,   varying (s, d) distance
        plotting_data = {}
        plotting_data['x'] = ("atomic_bsm", [0.2, 0.3, 0.4, 0.5, 0.6])
        plotting_data['dp_proto_no_throttle'] = [10.52, 26.579999999999995, 33.019999999999996, 50.06000000000001, 78.3]
        plotting_data['dp_proto'] = [9.08, 23.22, 26.439999999999998, 40.14, 63.54]
        helper2(ax1, plotting_data)

        # ax2: Throttle vs Non-throttle, DP-Approx, Multiple pair, varying (s, d) distance
        plotting_data = {}
        plotting_data['x'] = ("atomic_bsm", [0.2, 0.3, 0.4, 0.5, 0.6])
        plotting_data['dp_iter_proto'] = [22.96, 56.08, 104.68, 231.4, 353.5]
        plotting_data['dp_iter_proto_no_throttle'] = [21.48, 51.55999999999999, 101.66, 222.86000000000004,
                                                      332.3999999999999]
        helper2(ax2, plotting_data)

        # ax3: varying (Source, Destination) Pairs
        plotting_data = {}
        plotting_data['x'] = ("src_dst_pair", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        plotting_data['dp_iter_proto'] = [1.0, 2.0, 3.0, 4.0, 4.95, 5.9, 6.87, 7.67, 8.73, 9.93]
        plotting_data['lp_proto'] = [1.0, 2.0, 3.0, 4.0, 4.95, 6.0, 6.8, 7.87, 8.71, 9.67]
        plotting_data['lp_alt_proto'] = [1.0, 2.0, 2.94, 3.85, 4.81, 5.59, 6.67, 7.29, 8.35, 9.17]
        plotting_data['sig_proto5'] = [0.92, 1.74, 2.42, 3.04, 3.93, 4.18, 5.47, 5.64, 7.12, 7.47]
        plotting_data['sig_proto10'] = [1.0, 1.89, 2.49, 3.06, 3.93, 4.11, 5.23, 5.36, 6.69, 6.84]
        plotting_data['delft_lp_proto'] = [0.48, 0.78, 0.82, 1.38, 1.42, 1.86, 2.18, 2.38, 3.04, 3.11]
        helper3(ax3, plotting_data)

        plt.figtext(0.13, 0.01, '(a)', weight='bold')
        plt.figtext(0.38, 0.01, '(b)', weight='bold')
        plt.figtext(0.63, 0.01, '(c)', weight='bold')
        plt.figtext(0.87, 0.01, '(d)', weight='bold')
        plt.savefig('results/real_results/analytical_throttle.png')

    @staticmethod
    def analytical_throttle_new():

        def helper(ax, plotting_data):
            for method, val in plotting_data.items():
                x = plotting_data['x'][1]  # number of points
                if method == 'lp_proto' and plotting_data['x'][0] == 'num_nodes':
                    x = [25, 50, 75, 100]
                if method != 'x' and len(val) == len(x):
                    if method in Plot.LEGEND_ALGO:
                        line = Plot.LINE['algo']
                        label = Plot.LEGEND_ALGO[method]
                        color = Plot.COLOR_ALGO[method]
                        ax.plot(x, val, label=label, color=color, linestyle=line, linewidth=5)
                    else:
                        line = Plot.LINE['proto']
                        label = Plot.LEGEND_PROTO[method]
                        color = Plot.COLOR_PROTO[method]
                        ax.plot(x, val, label=label, color=color, linestyle=line)

            ax.set_ylabel('EP/s')
            ax.set_ylim([0, 22])
            ax.legend(fontsize=36, handlelength=3.4)
            ax.set_xlabel('Atomic BSM Success Rate', fontsize=45)
            ax.tick_params(axis='x', direction='in', length=10, width=3, pad=10)
            ax.tick_params(axis='y', direction='in', length=10, width=3, pad=10)

        def helper2(ax, plotting_data):
            '''for the throttle vs no-throttle
            '''
            for method, val in plotting_data.items():
                x = plotting_data['x'][1]
                if method != 'x' and len(val) == len(x):
                    line = Plot.LINE['proto']
                    label = Plot.LEGEND_PROTO[method]
                    color = Plot.COLOR_PROTO[method]

                    if method in Plot.LEGEND_NO_THROTTLE:
                        line = Plot.LINE['no_throttle']
                        color = 'lightpink'

                    ax.plot(x, val, label=label, color=color, linestyle=line)
                    if method == 'dp_proto':
                        ylim = [0, 25]
                    if method == 'dp_iter_proto':
                        ylim = [0, 130]

            ax.set_ylabel('EP/s')
            ax.set_ylim(ylim)
            ax.legend(fontsize=36, handlelength=3.6)
            ax.set_xlabel('Atomic BSM Success Rate', fontsize=45)
            ax.tick_params(axis='x', direction='in', length=10, width=3, pad=10)
            ax.tick_params(axis='y', direction='in', length=10, width=3, pad=10)

        def helper3(ax, plotting_data):
            ''' for the fairness
            '''
            for method, val in plotting_data.items():
                x = plotting_data['x'][1]
                if method != 'x' and len(val) == len(x):
                    line = Plot.LINE['proto']
                    label = Plot.LEGEND_PROTO[method]
                    color = Plot.COLOR_PROTO[method]
                    if method == 'dp_iter_proto':
                        line = '-'
                        ax.plot(x, val, label=label, color=color, linestyle=line, linewidth=5)
                        continue
                    if method == 'lp_proto':
                        line = ':'
                    if method == 'sig_proto10':
                        line = ':'
                    ax.plot(x, val, label=label, color=color, linestyle=line)

            ax.set_ylabel('Avg. # of Success Pairs')
            ax.legend(fontsize=32)
            ax.set_ylim([0, 11])
            ax.set_xticks(range(1, 11))
            ax.set_xlabel('# of (Source, Destination) Pairs', fontsize=45)
            ax.tick_params(axis='x', direction='in', length=10, width=3, pad=10)
            ax.tick_params(axis='y', direction='in', length=10, width=3, pad=10)

        # the plot
        fig, (ax0, ax1, ax2, ax3) = plt.subplots(1, 4, figsize=(58, 12))
        fig.subplots_adjust(left=0.04, right=0.99, top=0.96, bottom=0.2)

        # ax0: algorithm vs protocol, single pair, varying bsm probability
        plotting_data = {}
        plotting_data['x'] = ("atomic_bsm", [0.2, 0.3, 0.4, 0.5, 0.6])
        plotting_data['dp_proto'] = [2.472, 2.096, 11.124, 12.148, 17.704]
        plotting_data['dp_alt_proto'] = [2.544, 1.992, 7.628, 10.824, 15.448]
        plotting_data['dp'] = [2.556, 2.087, 10.72, 11.74, 16.178]
        plotting_data['dp_alt'] = [2.47, 1.969, 7.51, 9.945, 14.547]
        helper(ax0, plotting_data)

        # ax1: Throttle vs Non-throttle, DP-Approx, single pair,   varying (s, d) distance
        plotting_data = {}
        plotting_data['x'] = ("atomic_bsm", [0.2, 0.3, 0.4, 0.5, 0.6])
        plotting_data['dp_proto_no_throttle'] = [3.593, 3.347, 15.18, 17.747, 23.36]
        plotting_data['dp_proto'] =             [2.472, 2.096, 11.124, 12.148, 17.704]
        helper2(ax1, plotting_data)

        # ax2: Throttle vs Non-throttle, DP-Approx, Multiple pair, varying (s, d) distance
        plotting_data = {}
        plotting_data['x'] = ("atomic_bsm", [0.2, 0.3, 0.4, 0.5, 0.6])
        plotting_data['dp_iter_proto'] = [3.768, 11.692, 42.492, 62.608, 126.8]
        plotting_data['dp_iter_proto_no_throttle'] = [4.2, 10.134, 31.657, 50.473, 95.43]
        helper2(ax2, plotting_data)

        # ax3: varying (Source, Destination) Pairs
        plotting_data = {}
        plotting_data['x'] = ("src_dst_pair", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        plotting_data['dp_iter_proto'] = [1.0, 2.0, 3.0, 4.0, 4.95, 5.9, 6.87, 7.67, 8.73, 9.93]
        plotting_data['lp_proto'] = [1.0, 2.0, 3.0, 4.0, 4.95, 6.0, 6.8, 7.87, 8.71, 9.67]
        plotting_data['lp_alt_proto'] = [1.0, 2.0, 2.94, 3.85, 4.81, 5.59, 6.67, 7.29, 8.35, 9.17]
        plotting_data['sig_proto5'] = [0.92, 1.74, 2.42, 3.04, 3.93, 4.18, 5.47, 5.64, 7.12, 7.47]
        plotting_data['sig_proto10'] = [1.0, 1.89, 2.49, 3.06, 3.93, 4.11, 5.23, 5.36, 6.69, 6.84]
        plotting_data['delft_lp_proto'] = [0.48, 0.78, 0.82, 1.38, 1.42, 1.86, 2.18, 2.38, 3.04, 3.11]
        helper3(ax3, plotting_data)

        plt.figtext(0.13, 0.01, '(a)', weight='bold')
        plt.figtext(0.38, 0.01, '(b)', weight='bold')
        plt.figtext(0.63, 0.01, '(c)', weight='bold')
        plt.figtext(0.87, 0.01, '(d)', weight='bold')
        plt.savefig('results/new_real_results/analytical_throttle.TIF', format="tiff",
                    pil_kwargs={"compression": "tiff_lzw"})
        plt.savefig('results/new_real_results/analytical_throttle.png')

    @staticmethod
    def fidelity():
        ''' (a) and (b) has the same x-axis (link distance), but y-axis are different (Rate and Fidelity)
        '''
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(32, 15))
        fig.subplots_adjust(left=0.07, right=0.98, top=0.78, bottom=0.2, wspace=0.25)

        # (a) y = Protocol Rate, x= link length
        plotting_data = {}
        plotting_data['x'] = ('link_length', ['20-25', '25-30', '30-35', '35-40', '40-45', '45-50'])
        plotting_data['dp_proto_500'] = [0.2733333333333333, 0.19333333333333333, 0.48, 0.42333333333333334, 0.32, 0.26]
        plotting_data['dp_alt_proto_500'] = [0.18666666666666668, 0.16333333333333333, 0.45, 0.39666666666666667, 0.27,
                                             0.25333333333333335]
        plotting_data['dp_proto_1000'] = [0.05, 0.05333333333333334, 0.12222222222222222, 0.10555555555555556,
                                          0.09444444444444444, 0.08888888888888889]
        plotting_data['dp_alt_proto_1000'] = [0.06333333333333334, 0.05, 0.12777777777777777, 0.11666666666666667,
                                              0.07222222222222222, 0.07777777777777778]

        COLOR = {'dp_proto_500': 'r',
                 'dp_alt_proto_500': 'b',
                 'dp_proto_1000': 'r',
                 'dp_alt_proto_1000': 'b'}

        LINE = {'dp_proto_500': '-',
                'dp_alt_proto_500': '-',
                'dp_proto_1000': '--',
                'dp_alt_proto_1000': '--'}

        LEGEND = {'dp_proto_500': 'DP-Approx 500km',
                  'dp_alt_proto_500': 'Balanced-Tree 500km',
                  'dp_proto_1000': 'DP-Approx 1000km',
                  'dp_alt_proto_1000': 'Balanced-Tree 1000km'}

        for method, val in plotting_data.items():
            x = plotting_data['x'][1]
            if method != 'x' and len(val) == len(x):
                ax0.plot(x, val, color=COLOR[method], label=LEGEND[method], linestyle=LINE[method])

        ax0.set_ylabel('EP/s', labelpad=10)
        ax0.tick_params(axis='x', direction='in', length=10, width=3, pad=10)
        ax0.tick_params(axis='y', direction='in', length=10, width=3, pad=10)
        ax0.set_xlabel('Link Distance (km)', labelpad=20)
        ax0.set_xticklabels(["20-25", "25-30", "30-35", "35-40", "40-45", "45-50"], rotation=10, fontsize=45)

        # (b) Fidelity
        plotting_data = {}
        plotting_data['x'] = ('link_length', ['20-25', '25-30', '30-35', '35-40', '40-45', '45-50'])
        plotting_data['dp_fidelity_500'] = [0.7998371446809001, 0.8371591281616055, 0.8480407634705444,
                                            0.8746925925157628, 0.8923662943192398, 0.8914700896784971]
        plotting_data['dp_alt_fidelity_500'] = [0.8001668807914041, 0.8350255301056171, 0.851182500277886,
                                                0.8686448033859928, 0.8844371640762433, 0.8910377420421632]
        # plotting_data['dp_fidelity_1000']     = [0.6442670338724548, **0.7400170201858822, 0.7252448921534526, **0.7667016268512337, **0.7743067235221053, 0.8256355614908646]
        plotting_data['dp_fidelity_1000'] = [0.6442670338724548, 0.6900170201858822, 0.7252448921534526,
                                             0.7667016268512337, 0.7743067235221053, 0.8256355614908646]
        # plotting_data['dp_alt_fidelity_1000'] = [0.6322815306324515, 0.674565259328889, 0.7271856900394442, **0.7953560874740385, 0.7643294011513783, 0.8252764619548]
        plotting_data['dp_alt_fidelity_1000'] = [0.6322815306324515, 0.674565259328889, 0.7271856900394442,
                                                 0.7553560874740385, 0.7643294011513783, 0.8252764619548]

        COLOR = {'dp_fidelity_500': 'r',
                 'dp_alt_fidelity_500': 'b',
                 'dp_fidelity_1000': 'r',
                 'dp_alt_fidelity_1000': 'b'}

        LINE = {'dp_fidelity_500': '-',
                'dp_alt_fidelity_500': '-',
                'dp_fidelity_1000': '--',
                'dp_alt_fidelity_1000': '--'}

        for method, val in plotting_data.items():
            x = plotting_data['x'][1]
            if method != 'x' and len(val) == len(x):
                ax1.plot(x, val, color=COLOR[method], linestyle=LINE[method])

        ax1.set_ylabel('Fidelity', labelpad=10)
        ax1.tick_params(axis='x', direction='in', length=10, width=3, pad=10)
        ax1.tick_params(axis='y', direction='in', length=10, width=3, pad=10)
        ax1.set_xlabel('Link Distance (km)', labelpad=20)
        ax1.set_xticklabels(["20-25", "25-30", "30-35", "35-40", "40-45", "45-50"], rotation=10, fontsize=45)

        # one legend for all subplots
        handles, labels = ax0.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=2, fontsize=53, handlelength=2)

        plt.figtext(0.25, 0.01, '(a)', weight='bold')
        plt.figtext(0.75, 0.01, '(b)', weight='bold')
        plt.savefig('results/real_results/fidelity.TIF', format="tiff", pil_kwargs={"compression": "tiff_lzw"})
        plt.savefig('results/real_results/fidelity.png')

    @staticmethod
    def line_extra():
        ''' (a) and (b) has the same x-axis (link distance), but y-axis are different (Rate and Fidelity)
        '''
        label_format = '{:,.0f}'
        fig, (ax0, ax1) = plt.subplots(1, 4, figsize=(32, 15))
        fig.subplots_adjust(left=0.07, right=0.98, top=0.88, bottom=0.2, wspace=0.25)

        # (a) y = Protocol Rate, x= link length
        plotting_data = {}
        plotting_data['x'] = ('distance', [1, 2, 3, 4])
        plotting_data['dp'] = [0.583, 0.333, 0.167, 0.06]
        plotting_data['dpa'] = [0.333, 0.217, 0.1, 0.05]

        COLOR = {'dp': 'r',
                 'dpa': 'b',}

        LINE = {'dp': '-',
                'dpa': '-'}

        LEGEND = {'dp': 'DP-Approx',
                  'dpa': 'Balanced-Tree'}

        for method, val in plotting_data.items():
            x = plotting_data['x'][1]
            if method != 'x' and len(val) == len(x):
                ax0.plot(x, val, color=COLOR[method], label=LEGEND[method], linestyle=LINE[method])

        ax0.set_ylabel('EP/s', labelpad=10)
        ax0.tick_params(axis='x', direction='in', length=10, width=3, pad=10)
        ax0.tick_params(axis='y', direction='in', length=10, width=3, pad=10)
        ax0.set_xlabel('(s, d) Pair Distance (km)', labelpad=20)

        ax0.xaxis.set_major_locator(mticker.MaxNLocator(4))
        ticks_loc = ax0.get_xticks().tolist()
        ax0.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
        # ax0.set_xticklabels([label_format.format(x) for x in ticks_loc])
        ax0.set_xticklabels(["", "300", "500", "700", "900", ""])
        # ax0.set_xticklabels([300, 500, 700, 900], rotation=10, fontsize=45)

        # (b) Fidelity
        plotting_data = {}
        plotting_data['x'] = ('decoherence', [0.1, 0.2, 0.4, 0.6, 0.8, 1, 2])
        plotting_data['dp'] = [191.177, 266.732, 517.306, 529.956, 528.411, 550.432, 1052.172]
        plotting_data['dpa'] = [126.118, 266.732, 263.945, 529.956, 528.411, 522.349, 1052.172]

        COLOR = {'dp': 'r',
                 'dpa': 'b'}

        LINE = {'dp': '-',
                'dpa': '-'}

        for method, val in plotting_data.items():
            x = plotting_data['x'][1]
            if method != 'x' and len(val) == len(x):
                ax1.plot(x, val, color=COLOR[method], linestyle=LINE[method])

        ax1.set_ylabel('Max. Distance (km)', labelpad=10)
        ax1.tick_params(axis='x', direction='in', length=10, width=3, pad=10)
        ax1.tick_params(axis='y', direction='in', length=10, width=3, pad=10)
        ax1.set_xlabel('Decoherence Time (s)', labelpad=20)
        # ax1.set_xticklabels(["0.1", "0.2", "0.4", "0.6", "0.8", "1", "2"], rotation=10, fontsize=45)

        # one legend for all subplots
        handles, labels = ax0.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=2, fontsize=53, handlelength=2)

        plt.figtext(0.25, 0.01, '(a)', weight='bold')
        plt.figtext(0.75, 0.01, '(b)', weight='bold')
        plt.savefig('results/new_real_results/line_extra.png')

    @staticmethod
    def line_extra2():
        ''' (a) and (b) has the same x-axis (link distance), but y-axis are different (Rate and Fidelity)
        '''
        label_format = '{:,.0f}'
        fig, (ax0, ax1, ax2, ax3) = plt.subplots(1, 4, figsize=(58, 15))
        fig.subplots_adjust(left=0.04, right=0.99, top=0.8, bottom=0.2)

        plotting_data = {}
        plotting_data['x'] = ('link_length', ['20-25', '25-30', '30-35', '35-40', '40-45', '45-50'])
        # depolarization rate = 0.245e-3
        plotting_data['dp_proto_500'] = [0.2733333333333333, 0.19333333333333333, 0.48, 0.42333333333333334, 0.32, 0.26]
        plotting_data['dp_alt_proto_500'] = [0.18666666666666668, 0.16333333333333333, 0.45, 0.39666666666666667, 0.27,
                                             0.25333333333333335]
        plotting_data['dp_proto_1000'] = [0.05, 0.05333333333333334, 0.12222222222222222, 0.10555555555555556,
                                          0.09444444444444444, 0.08888888888888889]
        plotting_data['dp_alt_proto_1000'] = [0.06333333333333334, 0.05, 0.12777777777777777, 0.11666666666666667,
                                              0.07222222222222222, 0.07777777777777778]
        # depolarization rate = 0.1
        plotting_data['dp_proto_500'] = [0.267, 0.227, 0.633, 0.397, 0.433, 0.35]
        plotting_data['dp_alt_proto_500'] =  [0.183, 0.2, 0.467, 0.383, 0.167, 0.15]
        plotting_data['dp_proto_1000'] = [0.067, 0.033, 0.217, 0.1, 0.067, 0.1]
        plotting_data['dp_alt_proto_1000'] = [0.067, 0.03, 0.133, 0.1, 0.083, 0.083]

        # depolarization rate = 0.05
        plotting_data['dp_proto_500'] = [0.267, 0.257, 0.633, 0.433, 0.367, 0.35]
        plotting_data['dp_alt_proto_500'] = [0.183, 0.2, 0.467, 0.383, 0.167, 0.15]
        plotting_data['dp_proto_1000'] = [0.067, 0.033, 0.187, 0.12, 0.09, 0.08]
        plotting_data['dp_alt_proto_1000'] = [0.067, 0.03, 0.133, 0.1, 0.083, 0.083]

        COLOR = {'dp_proto_500': 'r',
                 'dp_alt_proto_500': 'b',
                 'dp_proto_1000': 'r',
                 'dp_alt_proto_1000': 'b'}

        LINE = {'dp_proto_500': '-',
                'dp_alt_proto_500': '-',
                'dp_proto_1000': '--',
                'dp_alt_proto_1000': '--'}

        LEGEND = {'dp_proto_500': 'DP-Approx 500km',
                  'dp_alt_proto_500': 'Balanced-Tree 500km',
                  'dp_proto_1000': 'DP-Approx 1000km',
                  'dp_alt_proto_1000': 'Balanced-Tree 1000km'}

        for method, val in plotting_data.items():
            x = plotting_data['x'][1]
            if method != 'x' and len(val) == len(x):
                ax0.plot(x, val, color=COLOR[method], label=LEGEND[method], linestyle=LINE[method])

        ax0.set_ylabel('EP/s', labelpad=10)
        ax0.tick_params(axis='x', direction='in', length=10, width=3, pad=10)
        ax0.tick_params(axis='y', direction='in', length=10, width=3, pad=10)
        ax0.set_xlabel('Link Length (km)', labelpad=20)
        ax0.set_xticklabels(["20-25", "25-30", "30-35", "35-40", "40-45", "45-50"], rotation=10, fontsize=45)

        # (b) Fidelity
        plotting_data = {}
        plotting_data['x'] = ('link_length', ['20-25', '25-30', '30-35', '35-40', '40-45', '45-50'])
        # depolarization rate =  0.245e-3
        plotting_data['dp_fidelity_500'] = [0.7998371446809001, 0.8371591281616055, 0.8480407634705444,
                                            0.8746925925157628, 0.8923662943192398, 0.8914700896784971]
        plotting_data['dp_alt_fidelity_500'] = [0.8001668807914041, 0.8350255301056171, 0.851182500277886,
                                                0.8686448033859928, 0.8844371640762433, 0.8910377420421632]
        # plotting_data['dp_fidelity_1000']     = [0.6442670338724548, **0.7400170201858822, 0.7252448921534526, **0.7667016268512337, **0.7743067235221053, 0.8256355614908646]
        plotting_data['dp_fidelity_1000'] = [0.6442670338724548, 0.6900170201858822, 0.7252448921534526,
                                             0.7667016268512337, 0.7743067235221053, 0.8256355614908646]
        # plotting_data['dp_alt_fidelity_1000'] = [0.6322815306324515, 0.674565259328889, 0.7271856900394442, **0.7953560874740385, 0.7643294011513783, 0.8252764619548]
        plotting_data['dp_alt_fidelity_1000'] = [0.6322815306324515, 0.674565259328889, 0.7271856900394442,
                                                 0.7553560874740385, 0.7643294011513783, 0.8252764619548]

        # depolarization rate = 0.1
        plotting_data['dp_fidelity_500'] = [0.637, 0.573, 0.761, 0.758, 0.749, 0.712]
        plotting_data['dp_alt_fidelity_500'] = [0.572, 0.565, 0.76, 0.722, 0.744, 0.681]
        plotting_data['dp_fidelity_1000'] = [0.368, 0.326, 0.514, 0.533, 0.481, 0.417]
        plotting_data['dp_alt_fidelity_1000'] = [0.298, 0.30, 0.534, 0.495, 0.443, 0.418]

        # depolarization rate = 0.05
        plotting_data['dp_fidelity_500'] = [0.72, 0.687, 0.812, 0.817, 0.815, 0.799]
        plotting_data['dp_alt_fidelity_500'] = [0.656, 0.66, 0.802, 0.784, 0.812, 0.75]
        plotting_data['dp_fidelity_1000'] = [0.481, 0.523, 0.643, 0.64, 0.603, 0.557]
        plotting_data['dp_alt_fidelity_1000'] = [0.378, 0.449, 0.601, 0.608, 0.573, 0.555]

        COLOR = {'dp_fidelity_500': 'r',
                 'dp_alt_fidelity_500': 'b',
                 'dp_fidelity_1000': 'r',
                 'dp_alt_fidelity_1000': 'b'}

        LINE = {'dp_fidelity_500': '-',
                'dp_alt_fidelity_500': '-',
                'dp_fidelity_1000': '--',
                'dp_alt_fidelity_1000': '--'}

        for method, val in plotting_data.items():
            x = plotting_data['x'][1]
            if method != 'x' and len(val) == len(x):
                ax1.plot(x, val, color=COLOR[method], linestyle=LINE[method])

        ax1.set_ylabel('Fidelity', labelpad=10)
        ax1.tick_params(axis='x', direction='in', length=10, width=3, pad=10)
        ax1.tick_params(axis='y', direction='in', length=10, width=3, pad=10)
        ax1.set_xlabel('Link Length (km)', labelpad=20)
        ax1.set_xticklabels(["20-25", "25-30", "30-35", "35-40", "40-45", "45-50"], rotation=10, fontsize=45)

        handles, labels = ax0.get_legend_handles_labels()
        fig.legend(handles, labels, bbox_to_anchor=(0.47, 1), ncol=2, fontsize=53, handlelength=2)

        # (a) y = Protocol Rate, x= link length
        plotting_data = {}
        plotting_data['x'] = ('distance', [1, 2, 3, 4])
        plotting_data['dp'] = [0.583, 0.333, 0.167, 0.06]
        plotting_data['dpa'] = [0.333, 0.217, 0.1, 0.05]

        COLOR = {'dp': 'r',
                 'dpa': 'b', }

        LINE = {'dp': '-',
                'dpa': '-'}

        LEGEND = {'dp': 'DP-Approx',
                  'dpa': 'Balanced-Tree'}

        for method, val in plotting_data.items():
            x = plotting_data['x'][1]
            if method != 'x' and len(val) == len(x):
                ax3.plot(x, val, color=COLOR[method], label=LEGEND[method], linestyle=LINE[method])

        ax3.set_ylabel('EP/s', labelpad=10)
        ax3.tick_params(axis='x', direction='in', length=10, width=3, pad=10)
        ax3.tick_params(axis='y', direction='in', length=10, width=3, pad=10)
        ax3.set_xlabel('(s, d) Pair Distance (km)', labelpad=20)

        ax3.xaxis.set_major_locator(mticker.MaxNLocator(4))
        ticks_loc = ax3.get_xticks().tolist()
        ax3.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
        # ax0.set_xticklabels([label_format.format(x) for x in ticks_loc])
        ax3.set_xticklabels(["", "300", "500", "700", "900", ""])
        # ax0.set_xticklabels([300, 500, 700, 900], rotation=10, fontsize=45)

        # (b) Fidelity
        plotting_data = {}
        plotting_data['x'] = ('decoherence', [0.1, 0.2, 0.4, 0.6, 0.8, 1, 2, 4, 5, 10, 15, 20, 25, 30, 35, 45, 50, 55, 60])
        plotting_data['dp'] = [223.089, 262.167, 496.952, 518.076, 522.315, 518.405, 1060.994, 1041.289, 1209.979,
                               2086.225, 2077.064, 2845.855, 4186.486, 4156.844, 4184.891, 4181.373, 4151.929, 4180.343, 4172.474]
        plotting_data['dpa'] = [135.806, 262.167, 293.952, 518.076, 522.315, 518.405, 1060.994, 1041.289, 1033.736,
                                2086.225, 2077.064, 2093.496, 2090.083, 3134.068, 4184.891, 4181.373, 4151.929, 4180.343, 4172.474]

        COLOR = {'dp': 'r',
                 'dpa': 'b'}

        LINE = {'dp': '-',
                'dpa': '-'}

        for method, val in plotting_data.items():
            x = plotting_data['x'][1]
            if method != 'x' and len(val) == len(x):
                ax2.plot(x, val, color=COLOR[method], linestyle=LINE[method])

        ax2.set_ylabel('Max. Distance (km)', labelpad=10)
        ax2.tick_params(axis='x', direction='in', length=10, width=3, pad=10)
        ax2.tick_params(axis='y', direction='in', length=10, width=3, pad=10)
        ax2.set_xlabel('Decoherence Time (s)', labelpad=20)
        # ax2.set_yticklabels(["0", "1k", "2k", "3k", "4k", "5k"])
        ax2.yaxis.set_major_locator(mticker.MaxNLocator(5))
        ticks_loc = ax2.get_yticks().tolist()
        ax2.yaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
        ax2.set_yticklabels(["", "0", "1k", "2k", "3k", "4k", ""])
        # ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.2f}'.format(x / 1000) + 'K'))
        # ax1.set_xticklabels(["0.1", "0.2", "0.4", "0.6", "0.8", "1", "2"], rotation=10, fontsize=45)

        # one legend for all subplots
        handles, labels = ax3.get_legend_handles_labels()
        fig.legend(handles, labels, bbox_to_anchor=(0.88, 0.95), ncol=2, fontsize=53, handlelength=2)

        plt.figtext(0.13, 0.01, '(a)', weight='bold')
        plt.figtext(0.38, 0.01, '(b)', weight='bold')
        plt.figtext(0.63, 0.01, '(c)', weight='bold')
        plt.figtext(0.87, 0.01, '(d)', weight='bold')
        plt.savefig('results/new_real_results/line_extra2.TIF', format="tiff", pil_kwargs={"compression": "tiff_lzw"})
        plt.savefig('results/new_real_results/line_extra2.png')

    @staticmethod
    def fidelity2():
        # (a) and (b) has the same y-axis (fidelity), but x-axis are different (Total distance and link distance)

        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(36, 16))
        fig.subplots_adjust(left=0.08, right=0.98, top=0.8, bottom=0.2)

        # (a) y = Fidelity, x=total distance
        plotting_data = {}
        plotting_data['x'] = ('distance', [100, 300, 500, 700, 900, 1000])
        plotting_data['dp_fidelity'] = [0.9682868082146711, 0.9048252024272694, 0.8480407634705444, 0.8269483041110299,
                                        0.7334840511897657, 0.7252448921534526]
        plotting_data['dp_alt_fidelity'] = [0.9671894035309951, 0.9081161586461037, 0.851182500277886,
                                            0.8267101218079699, 0.7680320492907675, 0.7271856900394442]

        COLOR = {'dp_fidelity': 'r',
                 'dp_alt_fidelity': 'b'}

        LINE = {'dp_fidelity': '-',
                'dp_alt_fidelity': '-'}

        LEGEND = {'dp_fidelity': 'DP-Approx',
                  'dp_alt_fidelity': 'Balanced-Tree'}

        for method, val in plotting_data.items():
            x = plotting_data['x'][1]
            if method != 'x' and len(val) == len(x):
                ax0.plot(x, val, color=COLOR[method], linestyle=LINE[method], label=LEGEND[method])
        ax0.set_xlabel('Distance Between (S, D) Pair')
        ax0.set_ylabel('Fidelity')
        ax0.tick_params(axis='x', direction='in', length=10, width=3, pad=10)
        ax0.tick_params(axis='y', direction='in', length=10, width=3, pad=10)

        # (b) y = Fidelity, x=link distance
        plotting_data = {}
        plotting_data['x'] = ('link_length', ['20-25', '25-30', '30-35', '35-40', '40-45', '45-50'])
        plotting_data['dp_fidelity_500'] = [0.7998371446809001, 0.8371591281616055, 0.8480407634705444,
                                            0.8746925925157628, 0.8923662943192398, 0.8914700896784971]
        plotting_data['dp_alt_fidelity_500'] = [0.8001668807914041, 0.8350255301056171, 0.851182500277886,
                                                0.8686448033859928, 0.8844371640762433, 0.8910377420421632]
        plotting_data['dp_fidelity_1000'] = [0.6442670338724548, 0.6900170201858822, 0.7252448921534526,
                                             0.7667016268512337, 0.7743067235221053, 0.8256355614908646]
        plotting_data['dp_alt_fidelity_1000'] = [0.6322815306324515, 0.674565259328889, 0.7271856900394442,
                                                 0.7553560874740385, 0.7643294011513783, 0.8252764619548]

        COLOR = {'dp_fidelity_500': 'r',
                 'dp_alt_fidelity_500': 'b',
                 'dp_fidelity_1000': 'r',
                 'dp_alt_fidelity_1000': 'b'}

        LINE = {'dp_fidelity_500': '-',
                'dp_alt_fidelity_500': '-',
                'dp_fidelity_1000': '--',
                'dp_alt_fidelity_1000': '--'}

        LEGEND = {'dp_fidelity_500': 'DP-Approx 500km',
                  'dp_alt_fidelity_500': 'Balanced-Tree 500km',
                  'dp_fidelity_1000': 'DP-Approx 1000km',
                  'dp_alt_fidelity_1000': 'Balanced-Tree 1000km'}

        for method, val in plotting_data.items():
            x = plotting_data['x'][1]
            if method != 'x' and len(val) == len(x):
                ax1.plot(x, val, color=COLOR[method], linestyle=LINE[method], label=LEGEND[method])

        handles, labels = ax0.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper left', ncol=2, fontsize=35)

        handles, labels = ax1.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right', ncol=2, fontsize=35)

        ax1.set_ylabel('Fidelity')
        ax1.tick_params(axis='x', direction='in', length=10, width=3, pad=10)
        ax1.tick_params(axis='y', direction='in', length=10, width=3, pad=10)
        ax1.set_xlabel('Link Distance (km)', labelpad=20)

        # one legend for all subplots
        # handles, labels = ax0.get_legend_handles_labels()
        # fig.legend(handles, labels, loc='upper center', ncol=2, fontsize=50, handlelength=2)

        plt.figtext(0.25, 0.01, '(a)', weight='bold')
        plt.figtext(0.75, 0.01, '(b)', weight='bold')
        plt.savefig('results/real_results/fidelity2.png')

    @staticmethod
    def fidelity3():

        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(36, 16))
        fig.subplots_adjust(left=0.07, right=0.9, top=0.88, bottom=0.2, wspace=0.5)

        # proto, fidelity  vs distance
        plotting_data = {}
        # plotting_data['x']               = ('distance', [100, 300, 500, 700, 900, 1000])
        # plotting_data['dp_proto']        = [6.903333333333333, 0.55, 0.48, 0.15, 0, 0]
        # plotting_data['dp_fidelity']     = [0.9682868082146711, 0.9048252024272694, 0.8480407634705444, 0.8269483041110299, 0, 0]
        # plotting_data['dp_alt_proto']    = [6.8, 0.5033333333333333, 0.45, 0.12666666666666668, 0.11333333333333333, 0.12777777777777777]
        # plotting_data['dp_alt_fidelity'] = [0.9671894035309951, 0.9081161586461037, 0.851182500277886, 0.8267101218079699, 0.7680320492907675, 0.7271856900394442]

        plotting_data['x'] = ('distance', [300, 500, 700, 900, 1000])
        plotting_data['dp_alt_proto'] = [0.5033333333333333, 0.45, 0.12666666666666668, 0.11333333333333333,
                                         0.12777777777777777]
        plotting_data['dp_alt_fidelity'] = [0.9081161586461037, 0.851182500277886, 0.8267101218079699,
                                            0.7680320492907675, 0.7271856900394442]

        _ax0 = ax0.twinx()

        COLOR = {'dp_alt_proto': 'b',
                 'dp_alt_fidelity': 'b'}

        LINE = {'dp_alt_proto': '-',
                'dp_alt_fidelity': '--'}

        LEGEND = {'dp_alt_proto': 'Balanced-Tree Protocol',
                  'dp_alt_fidelity': 'Balanced-Tree Fidelity'}

        for method, val in plotting_data.items():
            x = plotting_data['x'][1]
            if method != 'x' and len(val) == len(x):
                if method == 'dp_alt_proto':
                    ax0.plot(x, val, color=COLOR[method], linestyle=LINE[method], label=LEGEND[method])
                if method == 'dp_alt_fidelity':
                    _ax0.plot(x, val, color=COLOR[method], linestyle=LINE[method], label=LEGEND[method])
        ax0.set_xlabel('Distance (km) Between (S, D) Pair')
        ax0.set_ylabel('EP/s')
        _ax0.set_ylabel('Fidelity')
        ax0.tick_params(axis='x', direction='in', length=10, width=3, pad=10)
        ax0.tick_params(axis='y', direction='in', length=10, width=3, pad=10)
        handles0, labels0 = ax0.get_legend_handles_labels()
        handles1, labels1 = _ax0.get_legend_handles_labels()
        handles = handles0 + handles1
        labels = labels0 + labels1
        fig.legend(handles, labels, loc='upper center', ncol=2, fontsize=50, handlelength=2.5)

        # proto, fidelity  vs hop length
        plotting_data['x'] = ('link_length', ['20-25', '25-30', '30-35', '35-40', '40-45', '45-50'])
        plotting_data['dp_alt_fidelity_500'] = [0.8001668807914041, 0.8350255301056171, 0.851182500277886,
                                                0.8686448033859928, 0.8844371640762433, 0.8910377420421632]
        plotting_data['dp_alt_proto_500'] = [0.18666666666666668, 0.16333333333333333, 0.45, 0.39666666666666667, 0.27,
                                             0.25333333333333335]

        COLOR = {'dp_alt_proto_500': 'b',
                 'dp_alt_fidelity_500': 'b'}

        LINE = {'dp_alt_proto_500': '-',
                'dp_alt_fidelity_500': '--'}

        LEGEND = {'dp_alt_proto_500': 'Balanced-Tree Protocol',
                  'dp_alt_fidelity_500': 'Balanced-Tree Fidelity'}

        _ax1 = ax1.twinx()

        for method, val in plotting_data.items():
            x = plotting_data['x'][1]
            if method != 'x' and len(val) == len(x):
                if method == 'dp_alt_proto_500':
                    ax1.plot(x, val, color=COLOR[method], linestyle=LINE[method], label=LEGEND[method])
                if method == 'dp_alt_fidelity_500':
                    _ax1.plot(x, val, color=COLOR[method], linestyle=LINE[method], label=LEGEND[method])
        ax1.set_xlabel('Link Distance (km)')
        ax1.set_ylabel('EP/s')
        _ax1.set_ylabel('Fidelity')
        ax1.tick_params(axis='x', direction='in', length=10, width=3, pad=10)
        ax1.tick_params(axis='y', direction='in', length=10, width=3, pad=10)
        handles0, labels0 = ax1.get_legend_handles_labels()
        handles1, labels1 = _ax1.get_legend_handles_labels()
        handles = handles0 + handles1
        labels = labels0 + labels1
        # fig.legend(handles, labels, loc='upper left', ncol=2, fontsize=35, handlelength=2.5)
        ax1.set_xticklabels(['20-25', '25-30', '30-35', '35-40', '40-45', '45-50'], rotation=10, fontsize=40)

        # fig.tight_layout()
        plt.figtext(0.22, 0.01, '(a)', weight='bold')
        plt.figtext(0.72, 0.01, '(b)', weight='bold')
        plt.savefig('results/real_results/fidelity3.png')

    @staticmethod
    def runtime():
        '''y-axis: the runtime of all the methods
           x-axis: the number of nodes
        '''

        def helper(ax, plotting_data):
            for method, val in plotting_data.items():
                x = [1, 2, 3, 4, 5, 6, 7, 8]
                if method != 'x' and len(val) == len(x):
                    line = Plot.LINE['algo']
                    label = Plot.LEGEND_ALGO[method]
                    color = Plot.COLOR_ALGO[method]
                    ax.plot(x, val, label=label, color=color, linestyle=line)

            # ax.legend(fontsize=36, handlelength=3)
            ax.set_xlabel('# of Nodes', labelpad=13)
            ax.set_xticks(range(1, 9))
            ax.set_xticklabels(['25', '50', '75', '100', '200', '300', '400', '500'], fontsize=45)
            ax.set_ylabel('Runtime (s) in Log-scale')
            ax.set_yscale('log')
            ax.tick_params(axis='x', direction='in', length=10, width=3, pad=10)
            ax.tick_params(axis='y', direction='in', length=10, width=3, pad=10)

        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(32, 14))
        fig.subplots_adjust(left=0.09, right=0.99, top=0.84, bottom=0.22, wspace=0.25)

        # ax0: single path
        plotting_data = {}
        plotting_data['x'] = ('num_nodes', [25, 50, 75, 100, 200, 300, 400, 500])
        plotting_data['dp_opt'] = [5.3643, 42.5101, 117.3262, 281.0248, 2587.7865, 8425.3545, 19753.6258, 27214.3765]
        plotting_data['dp'] = [0.1758, 0.924, 2.3567, 5.6817, 37.3454, 166.8887, 347.792, 944.7046]
        # plotting_data['dp'] = [0.1758, 0.924, 2.3567, 5.6817]
        plotting_data['sig5'] = [0.0025, 0.0056, 0.0142, 0.0256, 0.1161, 0.2078, 0.2967, 0.4573]
        plotting_data['dp_alt'] = [0.0011, 0.0029, 0.0067, 0.0115, 0.0459, 0.1376, 0.1824, 0.3172]
        helper(ax0, plotting_data)
        ax0.set_ylim([0.0005, 200000])
        ax0.set_yticks([0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000])
        handles, labels = ax0.get_legend_handles_labels()
        ax0.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.03, 1.28), ncol=2, fontsize=42, handlelength=2)

        # ax1: multi path
        plotting_data = {}
        plotting_data['x'] = ('num_nodes', [25, 50, 75, 100, 200, 300, 400, 500])
        plotting_data['dp_iter'] = [1.2679, 17.217, 60.1618, 123.6351, 1203.4648, 9099.8357, 18997.9873, 30962.0499]
        # plotting_data['dp_iter'] = [1.2679, 17.217, 60.1618, 123.6351]
        # plotting_data['lp'] = [0.274, 1.9616, 12.2823, 80.5648]
        plotting_data['delft_lp'] = [0.0512, 0.1648, 0.4327, 0.6918, 3.1016, 7.3311, 13.0066, 21.8242]
        plotting_data['sig_multi5'] = [0.0086, 0.0502, 0.1863, 0.3725, 1.3487, 2.5952, 4.6837, 7.2106]
        plotting_data['lp_alt'] = [0.0028, 0.0324, 0.0879, 0.1815, 1.0576, 2.4473, 5.8756, 6.4937]
        helper(ax1, plotting_data)
        ax1.set_ylim([0.0005, 200000])
        ax1.set_yticks([0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000])
        handles, labels = ax1.get_legend_handles_labels()
        ax1.legend(handles, labels, loc='upper right', bbox_to_anchor=(1, 1.28), ncol=2, fontsize=42, handlelength=2)

        plt.figtext(0.18, 0.02, '(a) Single Path', weight='bold')
        plt.figtext(0.68, 0.02, '(b) Multiple Path', weight='bold')
        plt.savefig('results/real_results/runtime.TIF', format="tiff", pil_kwargs={"compression": "tiff_lzw"})
        plt.savefig('results/real_results/runtime.png')

    @staticmethod
    def caleffi():
        ''' Compare caleffi with our method
        '''

        def helper(ax, plotting_data):
            for method, val, in plotting_data.items():
                x = plotting_data['x'][1]
                if method != 'x' and len(val) == len(x):
                    line = Plot.LINE['proto']
                    label = Plot.LEGEND_PROTO[method]
                    color = Plot.COLOR_PROTO[method]
                    ax.plot(x, val, label=label, color=color, linestyle=line)

            ax.set_xlabel('Atomic BSM Success Rate')
            ax.set_ylabel('EP/s')
            ax.tick_params(axis='x', direction='in', length=10, width=3, pad=10)
            ax.tick_params(axis='y', direction='in', length=10, width=3, pad=10)

        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(32, 15))
        fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.19, wspace=0.2)

        # ax0: low density
        plotting_data = {}
        plotting_data['x'] = ('atomic_bsm', [0.2, 0.3, 0.4, 0.5, 0.6])
        plotting_data['dp_opt_proto'] = [8.466, 20.506, 38.24, 63.813, 99.313]
        plotting_data['dp_proto'] = [8.04, 19.773, 37.946, 63.190, 98.22]
        plotting_data['caleffi_proto'] = [7.626, 19.706, 37.113, 59.673, 92.073]
        plotting_data['dp_alt_proto'] = [6.573, 19.446, 36.54, 54.466, 73.946]
        helper(ax0, plotting_data)

        # ax1: high density
        plotting_data = {}
        plotting_data['x'] = ('atomic_bsm', [0.2, 0.3, 0.4, 0.5, 0.6])
        plotting_data['dp_opt_proto'] = [14.260, 33.053, 61.180, 97.993, 144.933]
        plotting_data['dp_proto'] = [13.973, 33.113, 59.406, 95.766, 141.353]
        plotting_data['caleffi_proto'] = [14.20, 31.440, 59.886, 94.2466, 139.326]
        plotting_data['dp_alt_proto'] = [12.720, 33.040, 60.006, 91.446, 126.793]
        helper(ax1, plotting_data)
        plt.figtext(0.18, 0.015, '(a) Low density', weight='bold')
        plt.figtext(0.66, 0.015, '(b) High density', weight='bold')

        handles, labels = ax0.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right', ncol=4, fontsize=48, handlelength=3)

        plt.savefig('results/real_results/caleffi.TIF', format="tiff", pil_kwargs={"compression": "tiff_lzw"})
        plt.savefig('results/real_results/caleffi.png')

    @staticmethod
    def caleffi2():
        '''also comparing with caleffi, this time is relative and has an error bar
        '''

        def average(vals):
            vals = [vals[i] if vals[i] < 1 else 1 for i in range(len(vals))]
            avg = np.average(vals)
            avg = round(min(1, avg), 4)
            return avg

        def min_max(vals):
            vals = np.array(vals)
            minn = vals.min()
            maxx = min(1, vals.max())
            return (round(minn, 4), round(maxx, 4))

        def min_max_list(vals):
            avg = average(vals)
            minn, maxx = min_max(vals)
            return (round(avg - minn, 4), round(maxx - avg, 4))

        def yerr(ratio):
            my_min_max = [min_max_list(ratio[i]) for i in range(len(ratio))]
            min_, max_ = [], []
            for minn, maxx in my_min_max:
                min_.append(minn)
                max_.append(maxx)
            return np.stack((min_, max_))

        dpa_ratio = [[1.05, 0.8571, 1.0, 1.0048, 0.9851],
                     [0.9111, 1.1981, 1.0528, 1.0398, 0.9936],
                     [1.0806, 0.8992, 0.9028, 1.0304, 0.9789],
                     [1.0387, 1.0373, 0.945, 1.0426, 0.9575],
                     [0.9947, 1.0253, 0.9389, 1.0249, 0.9655]]

        dp_alt_ratio = [[1.0333, 0.6429, 0.8068, 0.6667, 1.0272],
                        [1.05, 1.1887, 0.9193, 0.9683, 1.0655],
                        [1.0428, 0.9469, 0.9611, 1.0131, 0.9408],
                        [0.8774, 0.8008, 0.9245, 0.9185, 0.773],
                        [0.7515, 0.7399, 0.8192, 0.8184, 0.6061]]

        caleffi_ratio = [[0.7, 0.3571, 0.7841, 0.9143, 1.1064],
                         [0.9389, 0.4717, 0.6584, 1.0465, 1.0258],
                         [1.0479, 0.756, 0.6516, 1.0049, 1.0063],
                         [1.0284, 0.8046, 0.7348, 1.0194, 0.9751],
                         [1.0437, 0.8853, 0.7297, 1.0059, 0.9238]]

        fig, ax0 = plt.subplots(1, 1, figsize=(17, 15))
        fig.subplots_adjust(left=0.15, right=0.98, top=0.98, bottom=0.12)

        X = np.array([0.2, 0.3, 0.4, 0.5, 0.6])
        # Y_opt = [1, 1, 1, 1, 1]
        Y_dpa = [average(dpa_ratio[i]) for i in range(len(dpa_ratio))]
        Yerr_dpa = yerr(dpa_ratio)
        Y_dp_alt = [average(dp_alt_ratio[i]) for i in range(len(dpa_ratio))]
        Yerr_dp_alt = yerr(dp_alt_ratio)
        Y_caleffi = [average(caleffi_ratio[i]) for i in range(len(dpa_ratio))]
        Yerr_caleffi = yerr(caleffi_ratio)

        # plt.plot(X, Y_opt, linewidth=5, label=Plot.LEGEND_PROTO['dp_opt_proto'])
        plt.errorbar(X + 0.006, Y_dpa, yerr=Yerr_dpa, linewidth=7, linestyle='--', elinewidth=5,
                     label=Plot.LEGEND_PROTO['dp_proto'], color=Plot.COLOR_PROTO['dp_proto'], capsize=17, capthick=5,
                     marker='o', markersize=20)
        plt.errorbar(X - 0.006, Y_dp_alt, yerr=Yerr_dp_alt, linewidth=7, linestyle='--', elinewidth=5,
                     label=Plot.LEGEND_PROTO['dp_alt_proto'], color=Plot.COLOR_PROTO['dp_alt_proto'], capsize=17,
                     capthick=5, marker='s', markersize=20)
        plt.errorbar(X, Y_caleffi, yerr=Yerr_caleffi, linewidth=7, linestyle='--', elinewidth=5,
                     label=Plot.LEGEND_PROTO['caleffi_proto'], color=Plot.COLOR_PROTO['caleffi_proto'], capsize=17,
                     capthick=5, marker='X', markersize=25)

        plt.legend(loc='lower right', fontsize=45, handlelength=3)
        ax0.set_ylim([0, 1.03])
        ax0.set_xlabel('Atomic BSM Success Rate', labelpad=20)
        ax0.set_ylabel('Relative to DP Optimal', labelpad=20)
        ax0.tick_params(axis='x', direction='in', length=10, width=4, pad=10)
        ax0.tick_params(axis='y', direction='in', length=10, width=4, pad=10)
        plt.savefig('results/real_results/caleffi-relative.TIF', format="tiff", pil_kwargs={"compression": "tiff_lzw"})
        plt.savefig('results/real_results/caleffi-relative.png')

        print(Y_dpa)
        print(Yerr_dpa)
        print(Y_dp_alt)
        print(Yerr_dp_alt)
        print(Y_caleffi)
        print(Yerr_caleffi)

    @staticmethod
    def pre_single(y_ax: str):
        '''Multiple Pair, 4 subplots, each subplot is comparing 6 algorithms
        '''

        def helper(ax, plotting_data, y_ax):
            for method, val in plotting_data.items():
                x = plotting_data['x'][1]  # number of points
                if method != 'x' and len(val) == len(x):
                    avg_val = [sum(v) / len(v) for v in val]
                    if y_ax == "latency":
                        avg_val = [v / 1e9 for v in avg_val]
                    line = Plot.LINE[method]
                    label = Plot.LEGEND_PROTO[method]
                    color = Plot.COLOR_PROTO[method]
                    ax.plot(x, avg_val, label=label, color=color, linestyle=line)

            if plotting_data['x'][0] == "latency_threshold":
                ax.set_xlabel('Latency Target % w.r.t Best Single Path')
                if y_ax == "latency":
                    ax.set_ylim([0, 0.15])
                if y_ax == "sl_cost":
                    ax.set_yticks(range(2500, 12501, 2500))
                    ax.set_yticklabels(['2.5k', '5k', '7.5k', '10k', '12.5k'])
            elif plotting_data['x'][0] == "num_nodes":
                ax.set_xlabel('# of Nodes', labelpad=13)
                if y_ax == "latency":
                    ax.set_ylim([0, 0.35])
                if y_ax == "sl_cost":
                    ax.set_yticks(range(2000, 11000, 2000))
                    ax.set_yticklabels(['2k', '4k', '6k', '8k', '10k'])
                # ax.set_yticks(range(0, 2))
                ax.set_xticks(range(1, 8))
                ax.set_xticklabels(['25', '50', '75', '100', '150', '200', '300'], fontsize=45)
            elif plotting_data['x'][0] == "edge_density":
                ax.set_xlabel('Edge Density %')
                if y_ax == "latency":
                    ax.set_ylim([0, 1])
                if y_ax == "sl_cost":
                    ax.set_yticks(range(2500, 15001, 2500))
                    ax.set_yticklabels(['2.5k', '5k', '7.5k', '10k', '12.5k', '15k'])
            elif plotting_data['x'][0] == "src_dst_pair":
                ax.set_xlabel("# of (Source, Destination) Pairs")
                if y_ax == "latency":
                    ax.set_ylim([0, 0.15])
                if y_ax == "sl_cost":
                    ax.set_yticks(range(2500, 15001, 2500))
                    ax.set_yticklabels(['2.5k', '5k', '7.5k', '10k', '12.5k', '15k'])
            if y_ax == "latency" or y_ax == "sl_latency":
                ax.set_ylabel('Time(s)')
            elif y_ax == "sl_cost":
                ax.set_ylabel('EPs/s')

            ax.tick_params(axis='x', direction='in', length=10, width=3, pad=10)
            ax.tick_params(axis='y', direction='in', length=10, width=3 if y_ax != "sl_cost" else 1,
                           pad=10 if y_ax != "sl_cost" else -30)

        # the plot
        fig, (ax0, ax1, ax2, ax3) = plt.subplots(1, 4, figsize=(58, 13))
        fig.subplots_adjust(left=0.04, right=0.99, top=0.84, bottom=0.2)

        # ax0: varying # of nodes
        plotting_data = {}
        plotting_data['x'] = ("num_nodes", [1, 2, 3, 4, 5, 6, 7])
        if y_ax == "latency":
            plotting_data['single'] = [[183876104.61048675, 328567187.8972216, 188426371.32522964, 564696603.1385294],
                                       [80269405.75622645, 235708893.71902883, 330063601.8742513, 289126836.8399436,
                                        112135835.80522506],
                                       [197666673.3155764, 249499651.00791845, 138332203.8613236, 109594077.06540352],
                                       [42179546.87844213, 28734515.388499912, 53409757.514270395, 69202913.49492848],
                                       [21495571.339642595, 22432412.53367738, 31885064.746237222, 50209970.68534044,
                                        46383185.61697435],
                                       [16759340.750287602, 42798611.03823267, 68438245.04992503, 37594956.93020168,
                                        60783565.51689997],
                                       [17269537.436587747, 29103695.668671574, 26593039.071939904, 29104369.21864711,
                                        15143233.499562189]]
            plotting_data['single_naive'] = [
                [306167549.01867783, 192936624.75394335, 226255221.66335317, 458237339.88959694],
                [141523579.3397201, 264983741.78577617, 311441828.66749454, 489851400.45611995, 167406519.80514184],
                [112987080.97837201, 82233270.27657188, 361059852.7610235, 81423456.49677157],
                [34900662.81498655, 33540421.10376649, 39624921.26834066, 51667577.236718886],
                [40857805.14052111, 23049979.186923698, 21322763.41895745, 23147494.01802774, 27513599.82751978],
                [41392658.67693086, 19898385.4595911, 63756675.42933765, 20290496.458433274, 27091061.029225964],
                [18799672.54979542, 21851880.965132345, 17563046.610797655, 17362674.006136063, 30929395.001039937]]
            plotting_data['single_cluster'] = [
                [168249330.98322383, 119114865.27429883, 163286137.0657684, 159619376.32974997, 352338625.52835345],
                [147916465.02404657, 214878701.98554227, 276016446.26201785, 376814705.9504965, 141661542.21150324],
                [140640352.34541932, 136509365.22485337, 134998998.48828447, 162422139.9301718, 71938043.92616224],
                [52986246.61441084, 53640538.23080498, 44118908.41233028, 31541030.37066275, 70418837.65644082],
                [41170809.0941873, 25320798.895043302, 54230713.24484091, 24201054.21246, 43745614.20370921],
                [23090269.38581045, 30875684.52683929, 49815675.20697177, 90825786.55928755],
                [121520164.16325861, 92287991.71513927, 36767229.54888506, 69893997.73624344, 35555463.08706311]]
        elif y_ax == "sl_cost":
            plotting_data['single'] = [
                [13057.188849044745, 534.2352739220062, 5256.998466993413, 18702.563304549236, 6096.199287298724],
                [2872.252437387478, 2703.9061592046537, 2457.0070543478405, 4569.645483010201, 4223.739450844885],
                [2571.7769527081937, 2453.566332553724, 2181.9574696724326, 3019.9698479868202],
                [6334.301364478929, 3846.491017554744, 4401.470536148333, 1292.9892494541414],
                [3507.712384447572, 2608.074587887203, 2374.2705390244078, 2228.100402253607, 1262.883364234323],
                [3335.902696209875, 1984.493312810699, 2801.806558471844, 2227.52918243052, 1939.5424489939605],
                [1893.1873316897809, 2177.80130172334, 1969.0717442264609, 1838.8807408846264, 1200.3617625579166]]
            plotting_data['single_naive'] = [
                [13057.188849044745, 559.7862385988809, 5256.998466993413, 18702.563304549236, 6096.199287298724],
                [8490.4258085578, 3515.5212635745493, 2457.0070543478405, 4646.89404094595, 2889.7569822420533],
                [4088.1766179597244, 7364.259909905528, 5462.145814615102, 6242.537764498011],
                [14254.186563367552, 8333.304166387767, 9671.533424984334, 2876.823087295032],
                [6115.641999052109, 11038.254641845715, 13632.666353368124, 9532.784081280413, 5680.798527828087],
                [8359.42841013253, 5621.049588970533, 11001.079468217882, 11067.53027451834, 7754.677343435313],
                [6939.969404151378, 8977.179154889249, 14623.031530690885, 5913.816489335674, 7157.7864071447275]]
            plotting_data['single_cluster'] = [
                [13057.188849044745, 552.4492379518656, 10042.070830069222, 20113.5281547324, 7819.999420042407],
                [3062.4506051963017, 3112.8775318986336, 3302.8063163779652, 4578.606215589281, 3021.0106789440856],
                [2817.839074623244, 2657.66548353212, 2356.5148646773714, 2684.038789969605, 3283.4943809799415],
                [6334.301364478929, 3846.4910175547443, 4401.470536148333, 2040.2899107084647, 1307.8772172886465],
                [3507.7123844475714, 2608.074587887203, 2374.2705390244078, 2228.1004022536067, 1262.8833642343227],
                [3335.902696209875, 1984.4933128106993, 2801.8065584718443, 3189.8418023462823],
                [3685.772011646719, 4069.109777996602, 2093.4086136809433, 2846.9177466427523, 2148.3344950250066]]
        elif y_ax == "sl_latency":
            plotting_data['single'] = [
                [2.9733262592639207, 1.9891849728950148, 1.3927166857099154, 1.9522924338933088, 4.945109780512384],
                [3.3444929242144417, 9.467254127803884, 4.573196519472615, 7.804075291164874, 2.9202131351582663],
                [2.1464157037283154, 2.8535871227219123, 3.9898708636201135, 2.2770443909800617],
                [1.1178386156747535, 1.543467578217618, 1.4270968196303013, 1.6561009432645148],
                [1.1251317758852488, 1.3792615450456953, 1.384182691883633, 0.8738316788195878, 1.4353480747347434],
                [0.9442773152320619, 0.8053293014885168, 1.104182682292118, 0.8934821349460851, 1.066596965222039],
                [0.8238233122551976, 1.0829239919809073, 0.9663005304948729, 0.6757013858246765, 1.2143952547009151]]
            plotting_data['single_naive'] = [
                [2.9733262592639207, 1.9121626332257575, 1.3927166857099154, 1.9522924338933088, 4.945109780512384],
                [2.949428610544193, 9.570914585954325, 4.573196519472615, 7.784035562395845, 3.367044089965647],
                [2.4265724745451145, 2.3504648394313783, 2.943267997445825, 1.8381636522331815],
                [1.0066627372221078, 1.1252269177665448, 1.465969714557879, 1.5374764329199613],
                [0.9429518662710763, 1.3753612132217166, 1.3194219078051599, 1.2454741515433208, 1.723761809913101],
                [0.8052124986398863, 0.933217497694691, 0.9593305703516565, 0.9567576907370661, 1.2060141012648475],
                [0.8260783376931314, 0.9874616456074123, 1.1090012472528503, 0.8748807337291828, 1.2009751160155424]]
            plotting_data['single_cluster'] = [
                [2.9733262592639207, 2.7580737975492093, 1.7191744564719493, 2.56900354529601, 5.038767007692487],
                [3.754709795432939, 12.246653264628314, 5.666807030527951, 9.13513344328925, 5.154061373428933],
                [2.7393391627959045, 2.623537673644255, 3.7171102011736212, 3.967612883154372, 2.3478665217670334],
                [1.1178386156747533, 1.543467578217618, 1.4270968196303015, 2.2187264696941247, 1.8137961827624602],
                [1.125131775885249, 1.3792615450456953, 1.384182691883633, 0.8738316788195879, 1.4353480747347434],
                [0.9442773152320619, 0.8053293014885168, 1.104182682292118, 0.8787271868261569],
                [1.4189486642672717, 1.6339937293816043, 1.9701842379562524, 1.5941210285350795, 1.2782195300539416]]
        helper(ax0, plotting_data, y_ax)

        # ax1: varying (s, d) distance
        plotting_data = {}
        plotting_data['x'] = ("src_dst_pair", [4, 8, 12, 16, 20])
        if y_ax == "latency":
            plotting_data['single'] = [
                [59982638.74146937, 37584737.384067945, 34054047.515743725, 46190533.791028515, 15538574.86736687],
                [44670948.999125764, 35339985.58121179, 105010491.31748137, 59853960.4153601, 40318497.97334289],
                [42179546.87844213, 28734515.388499912, 53409757.514270395, 69202913.49492848],
                [72905943.92149697, 60264512.10226528, 66303810.14523139, 52112226.42966548, 48777955.18934508],
                [60204523.53068928, 109789239.58538334, 104619059.34416069, 79474181.56668885]]
            plotting_data['single_naive'] = [
                [34814549.726706915, 30697084.1611121, 33484799.572193597, 55614262.25089575, 52399328.500142686],
                [46766344.61800763, 36283586.4626959, 28650010.572814625, 50091088.76175374, 32504112.76303014],
                [34900662.81498655, 33540421.10376649, 39624921.26834066, 51667577.236718886],
                [49099605.5470636, 20517979.937113892, 137309213.55700928, 40127584.34112958],
                [54651145.978577666, 80151205.31616396, 37908219.61574499, 39892181.92184346]]
            plotting_data['single_cluster'] = [
                [59557719.76109024, 56201407.28706288, 69330580.55374989, 46482750.404084995, 34461169.82935774],
                [60559897.464017704, 52506575.232577875, 26348219.37120096, 39136411.729768336, 30264340.713613678],
                [52986246.61441084, 53640538.23080498, 44118908.41233028, 31541030.37066275, 70418837.65644082],
                [82139750.16285814, 64248225.48391994, 62106658.35388418, 96912134.6529039, 91901385.31248444],
                [45523715.15911163, 124152417.38823846, 85087629.43307364, 76462079.72783904, 145596343.13097674]]
        elif y_ax == "sl_cost":
            plotting_data['single'] = [
                [1738.8692465322788, 715.2218721272204, 897.9618880469359, 555.7225540902919, 647.0598024286323],
                [5368.780705741801, 1404.7891994137926, 2561.265771721055, 1713.4858159042635, 1052.55648472941],
                [6334.301364478929, 3846.491017554744, 4401.470536148333, 1292.9892494541414],
                [7087.312087208787, 4281.2604106309, 4886.577424547492, 2715.0308019204795, 1906.636625727793],
                [7494.449106494844, 4975.290629717172, 8904.53284389155, 2103.4293269455093]]
            plotting_data['single_naive'] = [
                [7693.129608966898, 3224.4471208993787, 4038.940205277841, 1184.6821302616215, 1164.046144556506],
                [13213.704621401088, 4537.763517331926, 6330.204007514242, 2554.626782191566, 2222.161508252219],
                [14254.186563367552, 8333.304166387767, 9671.533424984334,
                 2876.823087295032],
                [16783.95477208378, 11516.766997937448, 14232.471099764156, 3418.481016879652],
                [18392.3514130058, 13330.298058196935, 20497.836199837006, 3659.5385973117036]]
            plotting_data['single_cluster'] = [
                [1738.8692465322788, 715.2218721272204, 897.9618880469358, 1330.5061229816226, 647.0598024286325],
                [6224.60530656854, 1404.7891994137926, 2561.2657717210554, 1921.7653733343566, 1454.6148661500274],
                [6334.301364478929, 3846.4910175547443, 4401.470536148333,
                 2040.2899107084647, 1307.8772172886465],
                [10929.002399749816, 4700.617821029196, 4886.577424547492, 3527.780277014863, 2206.1477185733947],
                [8716.191199643776, 6679.552419510627, 8632.72449745787, 5009.2249510217825, 2908.119035461919]]
        elif y_ax == "sl_latency":
            plotting_data['single'] = [
                [0.29471329981694405, 0.526783666372336, 0.3441561746530255, 1.6749745439059736, 0.6185219816643895],
                [0.6584657347126835, 0.9514614438568478, 0.6505589049985124, 1.9617162929575436, 0.9178161383882883],
                [1.1178386156747535, 1.543467578217618, 1.4270968196303013, 1.6561009432645148],
                [1.4494067576886704, 1.9284542278488674, 1.9673550305658227, 4.067613269634186, 2.2212456157579803],
                [1.9344838300905771, 3.361179650648114, 2.7671943533051424, 2.606210231029821]]
            plotting_data['single_naive'] = [
                [0.343864478834348, 0.4094441398313636, 0.497000526885082, 1.6164414123129136, 0.6154809875869014],
                [0.6781260560633962, 0.6662756322398963, 0.9672792027291965, 1.890782507009002, 0.8746634800232068],
                [1.0066627372221078, 1.1252269177665448, 1.465969714557879,
                 1.5374764329199613],
                [1.26237281443104, 1.5980966319051308, 1.8209399761979028, 1.8943846271137084],
                [1.6469233144162174, 2.3075378718394073, 2.312430313529786, 2.3350114048476307]]
            plotting_data['single_cluster'] = [
                [0.29471329981694405, 0.5267836663723359, 0.34415617465302545, 1.6693251685645385, 0.6185219816643895],
                [0.7466917474065097, 0.951461443856848, 0.6505589049985125, 1.9253703333488132, 0.7536786161153216],
                [1.1178386156747533, 1.543467578217618, 1.4270968196303015,
                 2.2187264696941247, 1.8137961827624602],
                [1.3969636414044921, 1.9362599568206964, 1.9673550305658227, 4.2396592448066315, 2.480264501760459],
                [2.0344201484730307, 3.2715478317495483, 2.882232466628742, 4.664047235912272, 2.785956704744137]]
        helper(ax1, plotting_data, y_ax)

        # ax2: final latency target reduction
        plotting_data = {}
        plotting_data['x'] = ("latency_threshold", [10, 20, 30, 40, 50])
        if y_ax == "latency":
            plotting_data['single'] = [
                [7991083.166392197, 11978610.199192, 9050819.826946078, 35468747.3380171, 15871685.547887536],
                [62239500.06546745, 20793920.64851742, 29579771.45425638, 74956295.04078262, 20841973.499275927],
                [42179546.87844213, 28734515.388499912, 53409757.514270395, 69202913.49492848],
                [41615768.89763301, 68425193.76175882, 137406290.3089277, 56813687.9109697, 69040222.75643262],
                [95365048.11562233, 118241619.56208378, 122698593.8875434, 63596534.787233084, 43324574.593489364]]
            plotting_data['single_naive'] = [
                [16613207.093062038, 8874975.821015177, 5601682.568663876, 41723131.048684984, 16634606.34501062],
                [22058338.15120844, 31352162.434883878, 20172554.668188967, 30831460.64487006, 28546944.113248482],
                [34900662.81498655, 33540421.10376649, 39624921.26834066, 51667577.236718886],
                [28654731.196823373, 68331023.29649186, 43729633.75263845, 93071962.45495807, 82301622.80027692],
                [62586820.1569945, 50576093.788101286, 39065765.039744385, 38620498.756057166, 66323905.590197876]]
            plotting_data['single_cluster'] = [
                [13576808.337310104, 11794061.692669101, 8190991.958450337, 28376749.479555827, 9240286.718323901],
                [88369769.49786858, 22817998.05600701, 23759620.8979585, 66198091.163547575, 46172187.88005793],
                [52986246.61441084, 53640538.23080498, 44118908.41233028,
                 31541030.37066275, 70418837.65644082],
                [84833247.74968922, 51342946.23472128, 63185483.47019097, 107541823.03701629, 78273906.12633947],
                [171723384.73527655, 115536523.69319202, 143875366.79466385, 57917129.85727248, 142150685.50742072]]
        elif y_ax == "sl_cost":
            plotting_data['single'] = [
                [17029.36935260798, 5826.805861193583, 7689.08893435362, 3921.319623218962, 3870.533602581895],
                [7550.256633491959, 5012.516747496736, 4832.868054743027, 2312.023316469399, 1898.0524831506302],
                [6334.301364478929, 3846.491017554744, 4401.470536148333, 1292.9892494541414],
                [5646.739180579045, 1530.3047942723813, 3511.3582955380516, 1579.7450254755106, 972.262069011578],
                [3163.073530196386, 1304.5589353722562, 1694.4860557714014, 1201.87227676005, 926.6215777881177]]
            plotting_data['single_naive'] = [
                [20904.595628574294, 13431.619959638278, 12067.934410761341, 6466.928219350194, 5822.195476636969],
                [16330.085077394242, 9016.88202631133, 9671.533424984334, 3718.981922057803, 4053.7758397275325],
                [14254.186563367552, 8333.304166387767, 9671.533424984334,
                 2876.823087295032],
                [13031.926932504924, 7783.5556499410895, 8901.594389347101, 3185.6772088702864, 2501.6637530623548],
                [10457.113342215544, 6338.540676862946, 6127.03781955621, 3133.626142498345, 2501.6637530623548]]
            plotting_data['single_cluster'] = [
                [17029.36935260798, 5826.805861193584, 7689.088934353619, 3921.319623218962, 3870.5336025818942],
                [7550.256633491958, 5012.516747496737, 4832.868054743026, 2312.023316469399, 1898.0524831506302],
                [6334.301364478929, 3846.4910175547443, 4401.470536148333,
                 2040.2899107084647, 1307.8772172886465],
                [6881.215979834345, 1532.6547160673663, 3511.3582955380516, 3669.340713298772, 987.150036846083],
                [6436.094723131983, 1311.250270774398, 3499.7388920602293, 3625.621840845352, 993.6784789999311]]
        elif y_ax == "sl_latency":
            plotting_data['single'] = [
                [1.6362105999336185, 2.182257891572635, 1.9026567285089147, 2.793764992802874, 2.9269139176330983],
                [1.0087658804365156, 1.581529924926473, 1.393254747601519, 2.3033555074670318, 2.1258563010450366],
                [1.1178386156747535, 1.543467578217618, 1.4270968196303013, 1.6561009432645148],
                [1.048455800932524, 1.241850404060869, 1.3161301654715196, 1.2129554069060826, 1.5840807508532426],
                [0.9506177675095189, 1.0236476477967704, 1.2855775082612186, 1.2156726541927045, 1.7509165252011736]]
            plotting_data['single_naive'] = [
                [2.0083560686920556, 2.0467738046529527, 1.8278936904114227, 2.5853730001530235, 2.827938274318035],
                [1.1921460291266186, 1.490370940135765, 1.465969714557879, 2.328633158388572, 2.1350523381512123],
                [1.0066627372221078, 1.1252269177665448, 1.465969714557879,
                 1.5374764329199613],
                [0.845370123123371, 0.8691177274493974, 1.354621252018318, 1.344318277307531, 1.4728502842097533],
                [0.8621008100118785, 0.8842650579012442, 0.9596777803695515, 1.1239529974271703, 1.4728502842097533]]
            plotting_data['single_cluster'] = [
                [1.6362105999336187, 2.182257891572636, 1.9026567285089149, 2.793764992802874, 2.9269139176330974],
                [1.0087658804365154, 1.581529924926473, 1.3932547476015196, 2.3033555074670318, 2.1258563010450366],
                [1.1178386156747533, 1.543467578217618, 1.4270968196303015,
                 2.2187264696941247, 1.8137961827624602],
                [1.037335815895685, 1.2383186886003361, 1.3161301654715198, 1.4380267353257485, 1.7417759903511882],
                [0.960972747284432, 0.9934230002421511, 1.337817981921879, 1.316490977981759, 1.933401851563157]]
        helper(ax2, plotting_data, y_ax)

        # ax3: varying density of edges
        plotting_data = {}
        plotting_data['x'] = ("edge_density", [1, 2, 3, 4, 5])
        if y_ax == "latency":
            plotting_data['single'] = [
                [689844769.9730043, 1116781058.7373676, 742322195.0175239, 863894233.2119827, 690260625.1238738],
                [195516145.45627522, 75935520.50176023, 164780634.7852855, 145026105.19268137, 57318538.43415633],
                [42179546.87844213, 28734515.388499912, 53409757.514270395, 69202913.49492848],
                [44790974.02097881, 33783009.12359639, 50935079.3380264, 43145354.131779775, 53144843.88870528],
                [40850833.727155454, 55771577.373249866, 33767322.85068992, 33931713.7455146, 21610603.266723093]]
            plotting_data['single_naive'] = [
                [586330437.582417, 1014626118.8103436, 696316915.3545692, 940293257.2552432, 718357221.620328],
                [171385243.4954624, 148704372.90920082, 223444012.54611972, 175970440.78292683, 70601543.87007152],
                [34900662.81498655, 33540421.10376649, 39624921.26834066,
                 51667577.236718886],
                [16334808.2239068, 21425119.16399931, 26227356.38324063, 42578182.39474201, 82731367.86860316],
                [33877313.31063189, 19005614.20028741, 19268103.68572054, 17663670.046902854, 19289877.82552292]]
            plotting_data['single_cluster'] = [
                [548355328.3504163, 703466598.8757035, 607973620.2505711, 624144753.671599, 576604561.9250621],
                [190560106.31655547, 87907049.90329765, 106030812.37012868, 92587772.57075715, 62747134.467093796],
                [52986246.61441084, 53640538.23080498, 44118908.41233028,
                 31541030.37066275, 70418837.65644082],
                [47083908.90578813, 70417785.76954854, 33350731.6213878, 54696091.230735034, 61225028.844160326],
                [44826612.615596, 56790446.42566257, 142847953.7077126, 114782343.49606808, 66329460.52507401]]
        elif y_ax == "sl_cost":
            plotting_data['single'] = [
                [7154.580117301906, 43189.24255071605, 8441.263057534834, 11306.20428369941, 3592.2805374102704],
                [3349.976572471426, 3875.0021278494137, 6597.544394024872, 7812.140551736933, 1942.6649476890545],
                [6334.301364478929, 3846.491017554744, 4401.470536148333, 1292.9892494541414],
                [1710.782504070749, 1724.721198209459, 2665.195972142352, 1476.117817188454, 2653.596826741506],
                [987.0118984552507, 706.059136244708, 1727.772270476043, 3543.5778235304124, 1380.3709503977636]]
            plotting_data['single_naive'] = [
                [7807.324549927514, 43189.24255071605, 8441.263057534834, 11306.20428369941, 3678.037720738537],
                [9289.599735184478, 12906.035362674036, 13376.561662311686, 10962.34218109943, 9152.2647201773],
                [14254.186563367552, 8333.304166387767, 9671.533424984334,
                 2876.823087295032],
                [5567.2063830817415, 4351.325340102991, 6980.3595399148335, 3320.5103166045815, 8388.269522521656],
                [5400.657333244731, 5876.887394807501, 7051.804262150214, 6068.249182518311, 8243.455517234099]]
            plotting_data['single_cluster'] = [
                [7780.103750263603, 47949.10180801104, 6129.249093572801, 11792.265076820097, 3980.0240377742266],
                [4444.982934339137, 4002.318031704067, 6597.5443940248715, 7764.314962101388, 2264.5164634897474],
                [6334.301364478929, 3846.4910175547443, 4401.470536148333,
                 2040.2899107084647, 1307.8772172886465],
                [1724.0473399292343, 1724.7211982094593, 2684.1595511160185, 1476.117817188454, 2786.4740039666985],
                [994.564881991181, 6997.354762658275, 4101.959630227431, 2206.0589037350687, 2732.734488575795]]
        elif y_ax == "sl_latency":
            plotting_data['single'] = [
                [18.226040373059593, 23.379661498621065, 12.666587232617053, 19.8077517558242, 21.520042736437134],
                [2.582889268084016, 3.440249174380162, 2.5700393793872394, 2.5834676044241474, 2.6935186195739544],
                [1.1178386156747535, 1.543467578217618, 1.4270968196303013, 1.6561009432645148],
                [1.4298337003937032, 1.5331562407684571, 1.4063822574995968, 1.3027925413031318, 2.173570136206002],
                [1.0633123194355187, 1.3216931939313108, 1.0643127206287173, 0.9398928035167107, 1.0664039118167674]]
            plotting_data['single_naive'] = [
                [15.885761361784741, 23.379661498621065, 12.666587232617053, 19.8077517558242, 18.963997763699453],
                [2.8283862835576867, 3.1332673232758057, 2.4608589699629517, 2.3173552446783487, 2.4697958288277997],
                [1.0066627372221078, 1.1252269177665448, 1.465969714557879,
                 1.5374764329199613],
                [1.6522532569135195, 1.6365016688378713, 0.9164177023849748, 1.4648318908634577, 1.2825142352311716],
                [0.968921868938899, 1.1332379790777418, 0.9988354167647988, 0.8106050909021487, 1.0579259821420366]]
            plotting_data['single_cluster'] = [
                [25.319743564213912, 28.393896477852916, 21.383691256847463, 21.400378546057254, 22.060416599099142],
                [3.0348896495827185, 3.5733344158497, 2.570039379387239, 3.037652918784564, 3.510318297055202],
                [1.1178386156747533, 1.543467578217618, 1.4270968196303015,
                 2.2187264696941247, 1.8137961827624602],
                [1.3219844684338011, 1.533156240768457, 1.4066842422240795, 1.3027925413031314, 2.390334787615258],
                [1.007303599935338, 1.4058309883742106, 3.5146491907150947, 2.36264554778103, 1.7930915134728354]]
        helper(ax3, plotting_data, y_ax)

        # one legend for all subplots
        handles, labels = ax1.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=3, fontsize=55, handlelength=3.6)
        plt.figtext(0.13, 0.01, '(a)', weight='bold')
        plt.figtext(0.38, 0.01, '(b)', weight='bold')
        plt.figtext(0.63, 0.01, '(c)', weight='bold')
        plt.figtext(0.87, 0.01, '(d)', weight='bold')
        plt.savefig('results/pre_distribution/real_results/single_' + y_ax + '.png')

    @staticmethod
    def pre_multi(y_ax: str):
        '''Multiple Pair, 4 subplots, each subplot is comparing 6 algorithms
        '''

        def helper(ax, plotting_data, y_ax):
            for method, val in plotting_data.items():
                x = plotting_data['x'][1]  # number of points
                if method != 'x' and len(val) == len(x):
                    avg_val = [sum(v) / len(v) for v in val]
                    if y_ax == "latency":
                        avg_val = [v / 1e9 for v in avg_val]
                    line = Plot.LINE[method]
                    label = Plot.LEGEND_PROTO[method]
                    color = Plot.COLOR_PROTO[method]
                    ax.plot(x, avg_val, label=label, color=color, linestyle=line)

            if plotting_data['x'][0] == "latency_threshold":
                ax.set_xlabel('Latency Target % w.r.t Best Single Path')
                if y_ax == "latency":
                    ax.set_ylim([0, 0.15])
                if y_ax == "sl_cost":
                    ax.set_yticks(range(2500, 15001, 2500))
                    ax.set_yticklabels(['2.5k', '5k', '7.5k', '10k', '12.5k', '15k'])
            elif plotting_data['x'][0] == "num_nodes":
                ax.set_xlabel('# of Nodes', labelpad=13)
                if y_ax == "latency":
                    ax.set_ylim([0, 0.25])
                if y_ax == "sl_cost":
                    ax.set_yticks(range(2000, 12501, 2500))
                    ax.set_yticklabels(['2.5k', '5k', '7.5k', '10k', '12.5k'])
                # ax.set_yticks(range(0, 2))
                ax.set_xticks(range(1, 8))
                ax.set_xticklabels(['25', '50', '75', '100', '150', '200', '300'], fontsize=45)
            elif plotting_data['x'][0] == "edge_density":
                ax.set_xlabel('Edge Density %')
                if y_ax == "latency":
                    ax.set_ylim([0, 0.6])
                if y_ax == "sl_cost":
                    ax.set_yticks(range(5000, 30001, 5000))
                    ax.set_yticklabels(['5k', '10k', '15k', '20k', '25k', '30k'])
            elif plotting_data['x'][0] == "src_dst_pair":
                ax.set_xlabel("# of (Source, Destination) Pairs")
                if y_ax == "latency":
                    ax.set_ylim([0, 0.1])
                if y_ax == "sl_cost":
                    ax.set_yticks(range(1000, 6001, 1000))
                    ax.set_yticklabels(['1k', '2k', '3k', '4k', '5k', '6k'])
            if y_ax == "latency" or y_ax == "sl_latency":
                ax.set_ylabel('Time(s)')
            elif y_ax == "sl_cost":
                ax.set_ylabel('EPs/s')

            ax.tick_params(axis='x', direction='in', length=10, width=3, pad=10)
            ax.tick_params(axis='y', direction='in', length=10, width=3 if y_ax != "sl_cost" else 1,
                           pad=10 if y_ax != "sl_cost" else -30)

        # the plot
        fig, (ax0, ax1, ax2, ax3) = plt.subplots(1, 4, figsize=(58, 13))
        fig.subplots_adjust(left=0.04, right=0.99, top=0.84, bottom=0.2)

        # ax0: varying # of nodes
        plotting_data = {}
        plotting_data['x'] = ("num_nodes", [1, 2, 3, 4, 5, 6, 7])
        if y_ax == "latency":
            plotting_data['multi'] = [
                [323322829.7606723, 117525651.7618639, 96609431.31524684, 237847140.73433065, 315323428.9780758],
                [74063674.45821163, 273917442.3649324, 201166308.72618744, 286558294.1377952, 164838297.40249515],
                [97032231.52080499, 104183482.16046542, 137147831.20861974, 142037954.96413735, 73942416.75641744],
                [39333868.98001139, 45632262.81211572, 57995937.63514647, 65946553.59624679, 149754565.4594783],
                [47756820.582097195, 44809456.97351071, 33612950.72237671, 33017366.48185918, 47716038.42907217],
                [31567589.443743557, 34725213.53746638, 35026199.122525916, 27342411.754046228, 36044497.05987629],
                [44514702.852388225, 39262039.882349044, 38342382.73491872, 35010042.82728778, 42143950.17669802]]
            plotting_data['multi_naive'] = [
                [55454271.678002305, 148339117.87936452, 145357528.12573698, 127772214.04057924, 131431106.56239553],
                [107315943.72018854, 122923647.80203606, 267864571.7665445, 307007407.27189523, 143348638.0284646],
                [73552651.55920021, 73727793.51236877, 80589019.61808012, 136247102.25507125, 61542429.08152139],
                [59191872.4523757, 42237604.85166288, 32414033.606561903, 61685100.631581895],
                [55710854.51034945, 49160191.152046174, 32811429.54378034, 30023178.072020143, 33891012.350544035],
                [28431071.073156845, 33945637.585333325, 39148177.064529054, 40510097.75678769, 39644854.09037852],
                [32038866.997640748, 34870845.43327636, 35690986.13478728, 23439776.198344395, 30942862.58917882]]
        elif y_ax == "sl_cost":
            plotting_data['multi'] = [
                [14784.970094464528, 944.870902857011, 20373.552581996024, 28847.16386866151, 5007.3622782016555],
                [4289.877610983945, 2453.5738496184085, 5884.237704533217, 7644.430404838845, 3470.812046378957],
                [2344.643023594761, 1115.849005649111, 10637.223723554482, 2981.3539061281026, 2730.3762865365834],
                [7339.840302828034, 2985.1137448717777, 3197.0648878065126, 2192.797312207464, 1499.4960953068341],
                [2428.026781072855, 2799.7597473693213, 1946.3681692082325, 2468.7219348125004, 2192.6570942548856],
                [1261.1518490137105, 1131.819135423978, 2319.7033555586304, 1908.6740964670869, 922.7548131526628],
                [854.4598172909743, 400.9519531579308, 926.6834047251014, 1061.8935920304627, 2264.735358272156]]
            plotting_data['multi_naive'] = [
                [16460.689234252543, 508.45884622030155, 6727.521680918153, 29124.32228460035, 8260.447577813504],
                [3346.086240817767, 6004.168806586513, 3327.835278118241, 6646.329163608864, 2836.0361266765876],
                [2655.188019663625, 3035.0249371578307, 4054.5194797042986, 2429.89681046357, 2243.552602826262],
                [5905.229916119653, 3564.037847560032, 5009.480005246934, 1104.988377476212],
                [3797.1310377988902, 2144.140907739815, 1471.6147274119505, 4941.025778762636, 1387.0968784779184],
                [1890.9219234310444, 1284.6647955446538, 2625.5984300551504, 1673.8754149607603, 856.9627442619465],
                [777.9619547822871, 1118.1343304544137, 2548.661197720944, 5721.571220306148, 992.7578223501853]]
        elif y_ax == "sl_latency":
            plotting_data['multi'] = [
                [3.294763228225014, 6.330202243667614, 5.187859983293826, 4.486572324032773, 8.778517917710328],
                [5.903137018055552, 7.751079441848802, 7.9799014891903415, 17.155830762734155, 5.075946385449711],
                [4.261015284301129, 5.035279115459829, 5.498949125002386, 7.020543276951302, 7.2857628714370035],
                [2.5370920343899277, 2.0056174486406855, 2.9540928766565875, 2.0904466202279215, 3.3503188093734466],
                [2.702535110325982, 2.6374693644273397, 2.156217651569745, 1.7033031062368642, 3.6802776103156276],
                [2.1072506861236597, 2.2443866785879125, 4.112358108571004, 2.694347033652297, 2.8521303047263507],
                [2.316131363063908, 2.1547286299099446, 2.265753336075684, 2.3202433526405835, 3.6476795354552265]]
            plotting_data['multi_naive'] = [
                [3.78071094443708, 2.575159254952451, 2.021891098983887, 5.229885123219409, 10.213656788643545],
                [4.042714939263612, 13.398721413043324, 7.283304837943769, 9.520886465773096, 2.8500480822368988],
                [2.767346510875973, 3.0080566071190917, 3.1785493887180283, 3.278403539364124, 4.077985860457916],
                [1.0567861008041737, 1.4885067346932481, 1.794215528504052, 1.861799346637855],
                [0.7684454489219191, 0.6868797752803805, 0.6843158481630347, 1.6287219923206093, 1.5003481239720857],
                [0.6229809536074378, 0.3283392712982224, 1.045827713095366, 0.5511663901699648, 0.5601042789055407],
                [0.33345644030994187, 0.8048189689961853, 0.5768419571153574, 0.7215061612994516, 0.7789897667472556]]
        helper(ax0, plotting_data, y_ax)

        # ax1: varying (s, d) distance
        plotting_data = {}
        plotting_data['x'] = ("src_dst_pair", [4, 8, 12, 16, 20])
        if y_ax == "latency":
            plotting_data['multi'] = [
                [51431350.75865269, 38255807.414184965, 45076477.999705836, 93298065.35180396, 52924231.066635326],
                [57840817.33795637, 33995850.40617846, 86991171.37411636, 77611938.4532419, 53997938.18071652],
                [39333868.98001139, 45632262.81211572, 57995937.63514647, 65946553.59624679, 149754565.4594783],
                [51117825.724146694, 52854477.982811324, 46903131.45160149, 67665608.3019285, 79864832.32594714],
                [57613876.42303667, 65946766.486071184, 53584279.3143914, 62179889.28878768, 69359328.01857242]]
            plotting_data['multi_naive'] = [
                [31073847.878474362, 33008219.50232212, 29976989.09284971, 62789291.003852874],
                [48521198.790437534, 31048884.089660008, 34856515.27712503, 46710272.19231795],
                [59191872.4523757, 42237604.85166288, 32414033.606561903, 61685100.631581895],
                [52451530.165411465, 47475773.49150129, 33308435.460469965, 62327074.6316309],
                [62332390.84227066, 43601113.427891254, 29576197.15366102, 71987237.4507349]]
        elif y_ax == "sl_cost":
            plotting_data['multi'] = [
                [784.3250560132944, 286.1657758001205, 583.623490583421, 1028.7156462510145, 943.8617439750233],
                [5317.206650808495, 668.4447181677272, 2185.4406318565893, 1921.2060338698254, 1288.445650409617],
                [7339.840302828034, 2985.1137448717777, 3197.0648878065126, 2192.797312207464, 1499.4960953068341],
                [7684.026099747059, 3056.481836972895, 3516.5901350993804, 2863.527761047149, 2282.664707330734],
                [8620.014707968368, 3247.9552530669657, 8892.814793755595, 5067.255744551966, 2482.321421817885]]
            plotting_data['multi_naive'] = [
                [2684.600588384611, 1916.8284422237289, 396.3245192859274, 395.36745548082945],
                [5450.2625538605225, 2472.108884111937, 4653.141193593515, 766.5530464089995],
                [5905.229916119653, 3564.037847560032, 5009.480005246934, 1104.988377476212],
                [6618.982487825887, 3751.7404200280935, 5788.469004016093, 1983.2572517532371],
                [6971.822517508046, 4259.7329105973495, 7399.369672802709, 2193.3726808758147]]
        elif y_ax == "sl_latency":
            plotting_data['multi'] = [
                [0.5295136219408931, 0.6047483032013666, 0.8344099960871716, 1.008043291247604, 1.8886560453708823],
                [1.606643302485915, 1.0013832361754882, 1.754447359467076, 1.7397735940162415, 2.4061521622454993],
                [2.5370920343899277, 2.0056174486406855, 2.9540928766565875, 2.0904466202279215, 3.3503188093734466],
                [3.2963866616236457, 2.0987219060670563, 3.5382482975286855, 2.9001533535443924, 4.989269251909615],
                [4.931901958776118, 3.2837530000842747, 5.847170701717347, 3.669310433824299, 5.80352118958376]]
            plotting_data['multi_naive'] = [
                [0.5878773602448577, 0.6590690896143676, 0.3789557252591543, 0.9304059135061842],
                [0.7267538632050707, 0.9797952433009567, 1.587445860619632, 1.2560626214276873],
                [1.0567861008041737, 1.4885067346932481, 1.794215528504052, 1.861799346637855],
                [1.6867797988441, 1.695453161423459, 2.380408624444806, 2.940840715571503],
                [2.4367235517390253, 2.5783822550641053, 3.7516421886752074, 3.488656480412731]]
        helper(ax1, plotting_data, y_ax)

        # ax2: final latency target reduction
        plotting_data = {}
        plotting_data['x'] = ("latency_threshold", [10, 20, 30, 40, 50])
        if y_ax == "latency":
            plotting_data['multi'] = [
                [3484746.1236858633, 4279591.008405546, 1797023.1672142856, 8327531.131818254, 972437.9469544092],
                [42203557.25301201, 16790472.42426078, 21553168.11397921, 26961988.844331115, 21922304.55559554],
                [39333868.98001139, 45632262.81211572, 57995937.63514647, 65946553.59624679, 149754565.4594783],
                [72127329.2533528, 65009773.82354594, 70616943.17382719, 79511235.24027641, 79168035.49980082],
                [116981802.85773416, 94927991.90719783, 107048054.0545482, 94879729.84759553, 114769608.9944206]]
            plotting_data['multi_naive'] = [
                [21003966.92177567, 18096772.988745447, 17137719.968292188, 23340436.83620794, 20028736.399167646],
                [29866617.081327956, 19480145.32313674, 22397684.914107572, 29608787.4824338, 27560497.476769507],
                [59191872.4523757, 42237604.85166288, 32414033.606561903, 61685100.631581895],
                [44848925.71897432, 35919651.98764986, 37961847.789826326, 76059318.35288195, 73872224.61688916],
                [53725366.333678305, 50384217.87266318, 44049722.653409995, 55279773.059915684, 60919771.76573957]]
        elif y_ax == "sl_cost":
            plotting_data['multi'] = [
                [23022.382667840287, 15680.591510302831, 14848.863283799237, 11206.805731643672, 8358.193897922842],
                [16224.798656919338, 11180.395156716913, 8942.229020295308, 7321.064949933144, 4407.531062159581],
                [7339.840302828034, 2985.1137448717777, 3197.0648878065126, 2192.797312207464, 1499.4960953068341],
                [2344.3376262940765, 1244.1281903889176, 2320.557946552157, 1442.778957246622, 700.0521677350083],
                [1655.624754723895, 874.5572816273806, 1090.8988610266078, 1089.8626214122264, 594.4173807027753]]
            plotting_data['multi_naive'] = [
                [17843.66717033746, 7127.145434853386, 11554.013767845496, 7529.022452832836, 8992.328302124037],
                [10753.362609149397, 6892.409095723604, 8168.31124901062, 3627.9197449257526, 3499.68400364071],
                [5905.229916119653, 3564.037847560032, 5009.480005246934, 1104.988377476212],
                [1283.4045288421516, 1152.3740111443617, 1829.6782395488683, 1779.3875230932763, 623.7290629719978],
                [557.8642188138873, 580.6292627740228, 1143.6265681966904, 969.4441595947014, 518.0942759397647]]
        elif y_ax == "sl_latency":
            plotting_data['multi'] = [
                [5.667390147762858, 5.446754823405504, 7.36671359874578, 7.862437827231855, 9.63068209375539],
                [3.8386859744568445, 3.7048019219752106, 4.242082669554757, 5.574293861018524, 6.688743896847405],
                [2.5370920343899277, 2.0056174486406855, 2.9540928766565875, 2.0904466202279215, 3.3503188093734466],
                [1.6456081850717983, 1.3876858129570866, 2.0809663239819756, 1.6002407512148784, 2.1321264063230894],
                [1.3910241492795015, 0.9782846759164804, 1.538824037089758, 1.3845661798009279, 1.9519352016236216]]
            plotting_data['multi_naive'] = [
                [3.073986379107676, 2.721338406941967, 3.469631253953132, 4.309097439243284, 5.279456471351003],
                [1.8935405424057048, 2.3081184390881564, 2.0310364640867604, 3.1706808793763783, 3.671224861523644],
                [1.0567861008041737, 1.4885067346932481, 1.794215528504052, 1.861799346637855],
                [0.7869803984578914, 1.1649322996024873, 1.3089211587557736, 1.2913444066333166, 1.399291983177321],
                [0.7296744998671579, 0.878545605285151, 0.9489481627616353, 1.040743610249957, 1.2191007784778531]]
        helper(ax2, plotting_data, y_ax)

        # ax3: varying density of edges
        plotting_data = {}
        plotting_data['x'] = ("edge_density", [1, 2, 3, 4, 5])
        if y_ax == "latency":
            plotting_data['multi'] = [
                [531302069.8431034, 452013087.1926882, 514863819.60231453, 742561195.6699878, 413334095.2755502],
                [106320241.87287413, 67763330.29913735, 48234778.63065958, 105780539.80131769, 82450470.1897045],
                [39333868.98001139, 45632262.81211572, 57995937.63514647, 65946553.59624679, 149754565.4594783],
                [36854366.36201138, 40981608.847591765, 31108987.629242543, 56779762.36134485, 51033221.80606229],
                [28136261.443314046, 38383063.89568267, 21849409.685923852, 17073509.76723456, 30633432.71385172]]
            plotting_data['multi_naive'] = [
                [719283079.2325954, 174373155.00412527, 598895808.774139, 527547226.12436724, 590140019.3843206],
                [75147574.07274292, 77965549.88524513, 88718895.85258445, 131852866.96118666, 80891692.4262044],
                [59191872.4523757, 42237604.85166288, 32414033.606561903, 61685100.631581895],
                [35362002.660123065, 36794879.750111, 29749558.736278787, 30529180.581527192, 49000314.85300992],
                [26662510.538822215, 29985882.776033178, 28599662.470799606, 25912534.186340448, 22149090.35375602]]
        elif y_ax == "sl_cost":
            plotting_data['multi'] = [
                [14467.927741990714, 71320.17616432015, 32770.70511109996, 25834.84563955776, 10064.851494967359],
                [4607.411234615952, 5590.916129673167, 11439.401064754002, 8599.691384578739, 1905.4714261740453],
                [7339.840302828034, 2985.1137448717777, 3197.0648878065126, 2192.797312207464, 1499.4960953068341],
                [1559.1923164599484, 1340.5601221512359, 1142.543680693281, 1163.0418406186225, 5344.422409019278],
                [486.90339060812335, 1999.1026953656378, 1964.9364915863466, 3763.407776630287, 735.0486928867508]]
            plotting_data['multi_naive'] = [
                [7939.83010507759, 90532.19391766624, 1705.5504263863634, 31807.64218345331, 5529.011586590614],
                [5504.60893938697, 4144.224490548249, 5709.530061294996, 8482.681743437408, 590.8563831369052],
                [5905.229916119653, 3564.037847560032, 5009.480005246934, 1104.988377476212],
                [1987.8278566139165, 875.6355403515339, 1006.7831183000867, 1668.536345427105, 3360.3735723472673],
                [2227.508193603209, 1219.054008492732, 2295.438777672911, 1503.112146582919, 2235.5210177893846]]
        elif y_ax == "sl_latency":
            plotting_data['multi'] = [
                [58.13181122850955, 60.44569234527315, 28.321352529860693, 47.36803794190361, 37.64164289086883],
                [5.25207527544467, 7.54738304896628, 7.369991396272676, 7.438757319083319, 4.878972349663717],
                [2.5370920343899277, 2.0056174486406855, 2.9540928766565875, 2.0904466202279215, 3.3503188093734466],
                [3.4025613722130355, 2.439742072725458, 2.126984609203612, 3.03034478936498, 5.143218574731199],
                [1.41467754820867, 1.8009529908490656, 2.4302926732389123, 1.9739266970791562, 2.4812740939898417]]
            plotting_data['multi_naive'] = [
                [30.659229522996462, 98.69061096659712, 10.703993802038703, 56.123662286147564, 14.902399370058761],
                [2.464537248265016, 4.907393925291966, 2.296480784846585, 2.4687435470492023, 2.2266564344162534],
                [1.0567861008041737, 1.4885067346932481, 1.794215528504052, 1.861799346637855],
                [1.7621451755786939, 1.275072441805384, 0.8407809041881134, 1.366839512770452, 2.8521742200142968],
                [1.0465966540533467, 0.7438343243811202, 1.7716499243406498, 0.7853931765433758, 1.1140078256024446]]
        helper(ax3, plotting_data, y_ax)

        # one legend for all subplots
        handles, labels = ax1.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=2, fontsize=55, handlelength=3.6)
        plt.figtext(0.13, 0.01, '(a)', weight='bold')
        plt.figtext(0.38, 0.01, '(b)', weight='bold')
        plt.figtext(0.63, 0.01, '(c)', weight='bold')
        plt.figtext(0.87, 0.01, '(d)', weight='bold')
        plt.savefig('results/pre_distribution/real_results/multi_' + y_ax + '.png')

    @staticmethod
    def pre_single_latency(y_ax: str):
        '''Multiple Pair, 4 subplots, each subplot is comparing 6 algorithms
        '''

        def helper(ax, plotting_data, y_ax):
            for method, val in plotting_data.items():
                x = plotting_data['x'][1]  # number of points
                if method != 'x' and len(val) == len(x):
                    line = Plot.LINE[method]
                    label = Plot.LEGEND_PROTO[method]
                    color = Plot.COLOR_PROTO[method]
                    ax.plot(x, [y * 1000 for y in val], label=label, color=color, linestyle=line)

            if plotting_data['x'][0] == "cost_budget":
                ax.set_xlabel('Total Cost Budget (# of attempts)')
                # ax.set_xticklabels(['2.5k', '5k', '7.5k', '10k', '12.5k', '15k'])
                if y_ax == "latency":
                    ax.set_ylim([0, 0.15])
                if y_ax == "sl_cost":
                    ax.set_yticks(range(10000, 15001, 15000))
                    ax.set_yticklabels(['10k', '25k', '40k', '55k', '70k', '85k'])
            elif plotting_data['x'][0] == "num_nodes":
                ax.set_xlabel('# of Nodes', labelpad=13)
                if y_ax == "latency":
                    ax.set_ylim([0, 0.35])
                if y_ax == "sl_cost":
                    ax.set_yticks(range(2000, 11000, 2000))
                    ax.set_yticklabels(['2k', '4k', '6k', '8k', '10k'])
                # ax.set_yticks(range(0, 2))
                ax.set_xticks(range(1, 7))
                ax.set_xticklabels(['50', '75', '100', '150', '200', '300'], fontsize=45)
            elif plotting_data['x'][0] == "edge_density":
                ax.set_xlabel('Edge Density %')
                if y_ax == "latency":
                    ax.set_ylim([0, 1])
                if y_ax == "sl_cost":
                    ax.set_yticks(range(2500, 15001, 2500))
                    ax.set_yticklabels(['2.5k', '5k', '7.5k', '10k', '12.5k', '15k'])
            elif plotting_data['x'][0] == "src_dst_pair":
                ax.set_xlabel("# of (Source, Destination) Pairs")
                if y_ax == "latency":
                    ax.set_ylim([0, 0.15])
                if y_ax == "sl_cost":
                    ax.set_yticks(range(2500, 15001, 2500))
                    ax.set_yticklabels(['2.5k', '5k', '7.5k', '10k', '12.5k', '15k'])
            if y_ax == "sl_latency" or y_ax == "avg_latency" or y_ax == 'max_latency':
                ax.set_ylabel('Time (ms)')
            elif y_ax == "sl_cost":
                ax.set_ylabel('EPs/s')

            ax.tick_params(axis='x', direction='in', length=15, width=3, pad=10)
            ax.tick_params(axis='y', direction='in', length=15, width=3 if y_ax != "sl_cost" else 1,
                           pad=10 if y_ax != "sl_cost" else -30)

        # the plot
        fig, (ax0, ax1, ax2, ax3) = plt.subplots(1, 4, figsize=(60, 13))
        fig.subplots_adjust(left=0.04, right=0.99, top=0.84, bottom=0.2)

        # ax2: cost budget target reduction
        plotting_data = {}
        plotting_data['x'] = ("cost_budget", list(range(5000, 30001, 5000)))
        if y_ax == "avg_latency":
            plotting_data['single'] = [0.084, 0.067, 0.048, 0.04, 0.031, 0.022]
            plotting_data['single_naive'] = [0.136, 0.146, 0.103, 0.114, 0.099, 0.077]
            plotting_data['single_cluster'] = [0.103, 0.095, 0.083, 0.082, 0.072, 0.051]
            plotting_data['single_non_shortest'] = [0.084, 0.068, 0.083, 0.06, 0.064, 0.05]
            plotting_data['single_plus_non_shortest'] = [0.078, 0.062, 0.043, 0.037, 0.025, 0.018]
            plotting_data['without_sls'] = [0.157] * 6
        elif y_ax == "max_latency":
            plotting_data['single'] = [0.219, 0.19, 0.126, 0.173, 0.118, 0.075]
            plotting_data['single_naive'] = [0.378, 0.45, 0.329, 0.522, 0.432, 0.277]
            plotting_data['single_cluster'] = [0.274, 0.252, 0.304, 0.29, 0.38, 0.274]
            plotting_data['single_non_shortest'] = [0.186, 0.148, 0.346, 0.192, 0.345, 0.2]
            plotting_data['single_plus_non_shortest'] = [0.227, 0.2, 0.191, 0.148, 0.084, 0.061]
            plotting_data['without_sls'] = [0.327] * 6
        elif y_ax == "sl_cost":
            plotting_data['single'] = [4866.133, 9730.917, 13927.353, 18634.958, 23144.055, 26183.815]
            plotting_data['single_naive'] = [4278.755, 9131.156, 13975.226, 18742.586, 21812.395, 25947.826]
            plotting_data['single_cluster'] = [3705.36, 7226.594, 12035.155, 16022.08, 20334.759, 27471.371]
            plotting_data['single_non_shortest'] = [4714.167, 9423.504, 14265.636, 18858.745, 23700.428, 27830.467]
            plotting_data['single_plus_non_shortest'] = [4884.803, 9756.642, 14711.381, 19638.372, 24363.018, 28675.371]
        elif y_ax == "sl_latency":
            plotting_data['single'] = [0.261, 0.519, 0.754, 0.949, 1.174, 1.322]
            plotting_data['single_naive'] = [0.256, 0.512, 0.75, 0.947, 1.071, 1.198]
            plotting_data['single_cluster'] = [0.199, 0.385, 0.61, 0.855, 1.096, 1.408]
            plotting_data['single_non_shortest'] = [0.252, 0.631, 0.958, 1.33, 1.729, 1.961]
            plotting_data['single_plus_non_shortest'] = [0.29, 0.587, 0.817, 1.218, 1.599, 2.211]
        helper(ax2, plotting_data, y_ax)

        # ax0: varying # of nodes
        plotting_data = {}
        plotting_data['x'] = ("num_nodes", [1, 2, 3, 4, 5, 6])
        if y_ax == "avg_latency":
            plotting_data['single'] = [0.153, 0.122, 0.048, 0.021, 0.017, 0.007]
            plotting_data['single_naive'] = [0.242, 0.189, 0.103, 0.057, 0.048, 0.026]
            plotting_data['single_cluster'] = [0.226, 0.162, 0.083, 0.072, 0.08, 0.045]
            plotting_data['without_sls'] = [0.333, 0.258, 0.157, 0.114, 0.107, 0.075]
            plotting_data['single_non_shortest'] = [0.174, 0.133, 0.083, 0.031, 0.027, 0.017]
            plotting_data['single_plus_non_shortest'] = [0.139, 0.091, 0.043, 0.019, 0.021, 0.005]
        elif y_ax == "max_latency":
            plotting_data['single'] = [0.528, 0.339, 0.128, 0.071, 0.076, 0.021]
            plotting_data['single_naive'] = [0.704, 0.551, 0.329, 0.211, 0.2, 0.149]
            plotting_data['single_cluster'] = [0.88, 0.5, 0.304, 0.259, 0.349, 0.161]
            plotting_data['single_non_shortest'] = [0.474, 0.457, 0.346, 0.098, 0.122, 0.069]
            plotting_data['single_plus_non_shortest'] = [0.423, 0.309, 0.191, 0.067, 0.062, 0.019]
            plotting_data['without_sls'] = [0.968, 0.686, 0.327, 0.255, 0.25, 0.192]
        elif y_ax == "sl_cost":
            plotting_data['single'] = [13747.198, 14488.484, 13927.353, 14143.117, 13997.601, 14313.539]
            plotting_data['single_naive'] = [12311.393, 13457.298, 13975.226, 13676.742, 13934.328, 13264.533]
            plotting_data['single_cluster'] = [10553.262, 12689.819, 12035.155, 11110.128, 10158.229, 10057.884]
            plotting_data['single_non_shortest'] = [13088.086, 14150.905, 14265.636, 14294.3, 14268.979, 13881.979]
            plotting_data['single_plus_non_shortest'] = [14689.497, 14680.956, 14711.381, 14650.184, 14568.254, 14629.282]
        elif y_ax == "sl_latency":
            plotting_data['single'] = [0.837, 0.791, 0.754, 0.714, 0.66, 0.695]
            plotting_data['single_naive'] = [0.818, 0.819, 0.75, 0.696, 0.725, 0.632]
            plotting_data['single_cluster'] = [0.572, 0.683, 0.61, 0.572, 0.492, 0.45]
            plotting_data['single_non_shortest'] = [0.962, 1.13, 0.958, 0.855, 0.82, 0.814]
            plotting_data['single_plus_non_shortest'] = [1.005, 0.97, 0.817, 0.824, 0.737, 0.788]
        helper(ax0, plotting_data, y_ax)

        # ax1: varying (s, d) distance
        plotting_data = {}
        plotting_data['x'] = ("src_dst_pair", [4, 8, 12, 16, 20])
        if y_ax == "avg_latency":
            plotting_data['single'] = [0.022, 0.03, 0.048, 0.058, 0.092]
            plotting_data['single_naive'] = [0.088, 0.076, 0.103, 0.115, 0.13]
            plotting_data['single_cluster'] = [0.061, 0.091, 0.083, 0.113, 0.095]
            plotting_data['single_non_shortest'] = [0.043, 0.042, 0.083, 0.068, 0.079]
            plotting_data['single_plus_non_shortest'] = [0.008, 0.036, 0.045, 0.047, 0.07]
            plotting_data['without_sls'] = [0.181, 0.167, 0.157, 0.18, 0.164]
        elif y_ax == "max_latency":
            plotting_data['single'] = [0.055, 0.089, 0.126, 0.158, 0.315]
            plotting_data['single_naive'] = [0.273, 0.293, 0.329, 0.499, 0.511]
            plotting_data['single_cluster'] = [0.156, 0.268, 0.304, 0.417, 0.306]
            plotting_data['single_non_shortest'] = [0.091, 0.111, 0.346, 0.221, 0.304]
            plotting_data['single_plus_non_shortest'] = [0.026, 0.127, 0.191, 0.16, 0.275]
            plotting_data['without_sls'] = [0.324, 0.365, 0.327, 0.515, 0.554]
        elif y_ax == "sl_cost":
            plotting_data['single'] = [9772.859, 13653.672, 13927.353, 14432.72, 14596.149]
            plotting_data['single_naive'] = [7142.089, 13415.798, 13975.226, 14108.275, 14367.377]
            plotting_data['single_cluster'] = [9425.136, 12411.752, 12035.155, 10122.473, 11500.685]
            plotting_data['single_non_shortest'] = [10208.486, 13345.141, 14265.636, 14449.144, 14553.828]
            plotting_data['single_plus_non_shortest'] = [11573.015, 14159.569, 14711.381, 14799.332, 14804.603]
        elif y_ax == "sl_latency":
            plotting_data['single'] = [0.482, 0.733, 0.754, 0.772, 0.816]
            plotting_data['single_naive'] = [0.365, 0.694, 0.75, 0.833, 0.863]
            plotting_data['single_cluster'] = [0.471, 0.6, 0.61, 0.549, 0.681]
            plotting_data['single_non_shortest'] = [0.69, 0.915, 0.958, 1.023, 1.033]
            plotting_data['single_plus_non_shortest'] = [0.672, 0.775, 0.817, 0.882, 0.917]
        helper(ax1, plotting_data, y_ax)

        # ax3: varying density of edges
        plotting_data = {}
        plotting_data['x'] = ("edge_density", [4, 6, 8, 10, 12])
        if y_ax == "avg_latency":
            plotting_data['single'] = [0.238, 0.139, 0.048, 0.026, 0.017]
            plotting_data['single_naive'] = [0.361, 0.214, 0.103, 0.068, 0.053]
            plotting_data['single_cluster'] = [0.278, 0.174, 0.081, 0.063, 0.063]
            plotting_data['single_non_shortest'] = [0.242, 0.146, 0.083, 0.039, 0.022]
            plotting_data['single_plus_non_shortest'] = [0.239, 0.127, 0.045, 0.031, 0.021]
            plotting_data['without_sls'] = [0.432, 0.296, 0.157, 0.122, 0.111]
        elif y_ax == "max_latency":
            plotting_data['single'] = [0.768, 0.49, 0.103, 0.08, 0.054]
            plotting_data['single_naive'] = [1.23, 0.847, 0.329, 0.287, 0.157]
            plotting_data['single_cluster'] = [0.758, 0.696, 0.304, 0.176, 0.202]
            plotting_data['single_non_shortest'] = [0.694, 0.658, 0.346, 0.13, 0.063]
            plotting_data['single_plus_non_shortest'] = [0.702, 0.509, 0.191, 0.13, 0.063]
            plotting_data['without_sls'] = [1.172, 0.771, 0.327, 0.304, 0.271]
        elif y_ax == "sl_cost":
            plotting_data['single'] = [14336.256, 14270.163, 13927.353, 14058.527, 14420.116]
            plotting_data['single_naive'] = [11854.434, 14329.17, 13975.226, 14127.68, 13645.606]
            plotting_data['single_cluster'] = [12298.7, 11792.615, 12035.155, 8504.452, 11460.695]
            plotting_data['single_non_shortest'] = [14075.631, 14247.326, 14265.636, 13726.311, 14502.38]
            plotting_data['single_plus_non_shortest'] = [14747.284, 14584.516, 14711.381, 14632.385, 14432.941]
        elif y_ax == "sl_latency":
            plotting_data['single'] = [0.847, 0.865, 0.754, 0.734, 0.765]
            plotting_data['single_naive'] = [0.719, 0.926, 0.75, 0.75, 0.737]
            plotting_data['single_cluster'] = [0.691, 0.662, 0.61, 0.434, 0.579]
            plotting_data['single_non_shortest'] = [1.111, 1.183, 0.958, 1.036, 0.994]
            plotting_data['single_plus_non_shortest'] = [1.096, 1.234, 0.817, 0.858, 0.867]
        helper(ax3, plotting_data, y_ax)

        # one legend for all subplots
        handles, labels = ax1.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=6, fontsize=55, handlelength=3.6)
        plt.figtext(0.13, 0.01, '(a)', weight='bold')
        plt.figtext(0.38, 0.01, '(b)', weight='bold')
        plt.figtext(0.63, 0.01, '(c)', weight='bold')
        plt.figtext(0.87, 0.01, '(d)', weight='bold')
        plt.savefig('results/pre_distribution/new_real_results_latency_objective/disjoint_sls/single_' + y_ax + '.png')

    @staticmethod
    def pre_multi_latency(y_ax: str):
        '''Multiple Pair, 4 subplots, each subplot is comparing 6 algorithms
        '''

        def helper(ax, plotting_data, y_ax):
            for method, val in plotting_data.items():
                x = plotting_data['x'][1]  # number of points
                if method != 'x' and len(val) == len(x):
                    line = Plot.LINE[method]
                    label = Plot.LEGEND_PROTO[method]
                    color = Plot.COLOR_PROTO[method]
                    ax.plot(x, val, label=label, color=color, linestyle=line)

            if plotting_data['x'][0] == "cost_budget":
                ax.set_xlabel('Total Cost Budget (# of attempts)')
                # ax.set_xticks(range(2500, 15001, 2500))
                # ax.set_xticklabels(['2.5k', '5k', '7.5k', '10k', '12.5k', '15k'])
                if y_ax == "latency":
                    ax.set_ylim([0, 0.15])
                if y_ax == "sl_cost":
                    ax.set_yticks(range(2500, 15001, 2500))
                    ax.set_yticklabels(['2.5k', '5k', '7.5k', '10k', '12.5k', '15k'])
            elif plotting_data['x'][0] == "num_nodes":
                ax.set_xlabel('# of Nodes', labelpad=13)
                if y_ax == "latency":
                    ax.set_ylim([0, 0.25])
                if y_ax == "sl_cost":
                    ax.set_yticks(range(2000, 12501, 2500))
                    ax.set_yticklabels(['2.5k', '5k', '7.5k', '10k', '12.5k'])
                # ax.set_yticks(range(0, 2))
                ax.set_xticks(range(1, 8))
                ax.set_xticklabels(['25', '50', '75', '100', '150', '200', '300'], fontsize=45)
            elif plotting_data['x'][0] == "edge_density":
                ax.set_xlabel('Edge Density %')
                if y_ax == "latency":
                    ax.set_ylim([0, 0.6])
                if y_ax == "sl_cost":
                    ax.set_yticks(range(5000, 30001, 5000))
                    ax.set_yticklabels(['5k', '10k', '15k', '20k', '25k', '30k'])
            elif plotting_data['x'][0] == "src_dst_pair":
                ax.set_xlabel("# of (Source, Destination) Pairs")
                if y_ax == "latency":
                    ax.set_ylim([0, 0.1])
                if y_ax == "sl_cost":
                    ax.set_yticks(range(1000, 6001, 1000))
                    ax.set_yticklabels(['1k', '2k', '3k', '4k', '5k', '6k'])
            if y_ax == "sl_latency" or y_ax == "avg_latency" or y_ax == 'max_latency':
                ax.set_ylabel('Time(s)')
            elif y_ax == "sl_cost":
                ax.set_ylabel('EPs/s')

            ax.tick_params(axis='x', direction='in', length=10, width=3, pad=10)
            ax.tick_params(axis='y', direction='in', length=10, width=3 if y_ax != "sl_cost" else 1,
                           pad=10 if y_ax != "sl_cost" else -30)

        # the plot
        fig, (ax0, ax1, ax2, ax3) = plt.subplots(1, 4, figsize=(58, 13))
        fig.subplots_adjust(left=0.04, right=0.99, top=0.84, bottom=0.2)

        # ax0: varying # of nodes
        plotting_data = {}
        plotting_data['x'] = ("num_nodes", [1, 2, 3, 4, 5, 6, 7])
        if y_ax == "avg_latency":
            plotting_data['multi'] = [0.373, 0.922, 0.972, 0.57, 0.163, 0.153, 0.056]
            plotting_data['multi_naive'] = [0.327, 0.793, 0.883, 0.48, 0.129, 0.121, 0.052]
        if y_ax == "max_latency":
            plotting_data['multi'] = [1.382, 2.225, 2.339, 1.68, 0.673, 0.397, 0.129]
            plotting_data['multi_naive'] = [1.123, 1.825, 2.503, 1.282, 0.461, 0.338, 0.129]
        elif y_ax == "sl_cost":
            plotting_data['multi'] = [7232.048, 6528.392, 6713.535, 7136.265, 7242.782, 7350.215, 7420.308]
            plotting_data['multi_naive'] = [7154.889, 7410.836, 7459.799, 7464.09, 7468.583, 7478.524, 7484.252]
        elif y_ax == "sl_latency":
            plotting_data['multi'] = [0.432, 0.45, 0.421, 0.417, 0.412, 0.432, 0.402]
            plotting_data['multi_naive'] = [0.441, 0.519, 0.484, 0.428, 0.425, 0.433, 0.41]
        helper(ax0, plotting_data, y_ax)

        # ax1: varying (s, d) distance
        plotting_data['x'] = ("src_dst_pair", [4, 8, 12, 16, 20])
        if y_ax == "avg_latency":
            plotting_data['multi'] = [0.467, 0.613, 0.57, 0.498, 0.575]
            plotting_data['multi_naive'] = [0.352, 0.524, 0.48, 0.462, 0.581]
        elif y_ax == "max_latency":
            plotting_data['multi'] = [1.065, 1.388, 1.68, 1.638, 2.061]
            plotting_data['multi_naive'] = [0.659, 1.243, 1.282, 1.329, 2.067]
        elif y_ax == "sl_cost":
            plotting_data['multi'] = [6200.678, 6917.721, 7136.265, 7068.901, 7258.5]
            plotting_data['multi_naive'] = [7451.333, 7419.191, 7464.09, 7473.711, 7474.823]
        elif y_ax == "sl_latency":
            plotting_data['multi'] = [0.365, 0.413, 0.417, 0.412, 0.425]
            plotting_data['multi_naive'] = [0.42, 0.427, 0.428, 0.426, 0.417]
        helper(ax1, plotting_data, y_ax)

        # ax2: final latency target reduction
        plotting_data = {}
        plotting_data['x'] = ("cost_budget", [1, 2, 3, 4, 5, 6])
        if y_ax == "avg_latency":
            plotting_data['multi'] = [0.633, 0.581, 0.57, 0.569, 0.551, 0.446]
            plotting_data['multi_naive'] = [0.514, 0.506, 0.48, 0.442, 0.42, 0.408]
        elif y_ax == "max_latency":
            plotting_data['multi'] = [1.684, 1.644, 1.68, 1.79, 1.863, 1.006]
            plotting_data['multi_naive'] = [1.218, 1.412, 1.282, 1.116, 1.205, 1.189]
        elif y_ax == "sl_cost":
            plotting_data['multi'] = [2223.473, 4436.016, 7136.265, 9626.36, 11322.31, 14802.714]
            plotting_data['multi_naive'] = [2475.316, 4961.631, 7464.09, 9954.626, 12470.15, 14944.886]
        elif y_ax == "sl_latency":
            plotting_data['multi'] = [0.121, 0.25, 0.417, 0.573, 0.689, 0.89]
            plotting_data['multi_naive'] = [0.127, 0.283, 0.428, 0.549, 0.708, 0.854]
        helper(ax2, plotting_data, y_ax)

        # ax3: varying density of edges
        plotting_data = {}
        plotting_data['x'] = ("edge_density", [1, 2, 3, 4, 5])
        if y_ax == "avg_latency":
            plotting_data['multi'] = [1.37, 1.01, 0.57, 0.172, 0.123]
            plotting_data['multi_naive'] = [1.113, 0.843, 0.48, 0.143, 0.09]
        elif y_ax == "max_latency":
            plotting_data['multi'] = [2.957, 2.167, 1.68, 0.561, 0.491]
            plotting_data['multi_naive'] = [2.188, 1.865, 1.282, 0.378, 0.252]
        elif y_ax == "sl_cost":
            plotting_data['multi'] = [6052, 6732, 7136.265, 7214, 6832]
            plotting_data['multi_naive'] = [7296.943, 7386.215, 7464.09, 7475.882, 7470.49]
        elif y_ax == "sl_latency":
            plotting_data['multi'] = [0.349, 0.431, 0.417, 0.397, 0.38]
            plotting_data['multi_naive'] = [0.452, 0.546, 0.428, 0.439, 0.457]
        helper(ax3, plotting_data, y_ax)

        # one legend for all subplots
        handles, labels = ax1.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=2, fontsize=55, handlelength=3.6)
        plt.figtext(0.13, 0.01, '(a)', weight='bold')
        plt.figtext(0.38, 0.01, '(b)', weight='bold')
        plt.figtext(0.63, 0.01, '(c)', weight='bold')
        plt.figtext(0.87, 0.01, '(d)', weight='bold')
        plt.savefig('results/pre_distribution/real_results_latency_objective/multi_' + y_ax + '.png')

    @staticmethod
    def pre_single_latency_qw(y_ax: str):
        ''' Pre-distribution results for quantum week conference
        '''

        def helper(ax, plotting_data, y_ax):
            for method, val in plotting_data.items():
                x = plotting_data['x'][1]  # number of points
                if method != 'x' and len(val) == len(x):
                    line = Plot.LINE[method]
                    label = Plot.LEGEND_PROTO[method]
                    color = Plot.COLOR_PROTO[method]
                    ax.plot(x, val if y_ax == 'sl_latency' else [y * 1000 for y in val],
                            label=label, color=color, linestyle=line)

            if plotting_data['x'][0] == "cost_budget":
                ax.set_xlabel('Total Cost Budget (# of attempts)')
                ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
                ax.xaxis.set_major_locator(mticker.MaxNLocator(4))
                ticks_loc = ax.get_xticks().tolist()
                ax.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
                ax.set_xticklabels(['', '10k', '20k', '30k', '40k', ''])

                ax.yaxis.set_major_locator(mticker.MaxNLocator(5))
                if y_ax == "avg_latency":
                    ax.set_yticks([10, 40, 80, 120, 160])
                if y_ax == "max_latency":
                    ax.set_yticks([75, 150, 300, 450, 600])
                # ax.set_xticklabels(['2.5k', '5k', '7.5k', '10k', '12.5k', '15k'])
                if y_ax == "latency":
                    ax.set_ylim([0, 0.15])
                if y_ax == "sl_cost":
                    ax.set_yticks(range(10000, 15001, 15000))
                    ax.set_yticklabels(['10k', '25k', '40k', '55k', '70k', '85k'])
                ax.get_yaxis().set_major_formatter(mticker.ScalarFormatter())
            elif plotting_data['x'][0] == "num_nodes":
                ax.set_xlabel('# of Nodes', labelpad=13)
                if y_ax == "latency":
                    ax.set_ylim([0, 0.35])
                if y_ax == "sl_cost":
                    ax.set_yticks(range(2000, 11000, 2000))
                    ax.set_yticklabels(['2k', '4k', '6k', '8k', '10k'])
                # ax.set_yticks(range(0, 2))
                ax.set_xticks(range(1, 7))
                ax.set_xticklabels(['50', '75', '100', '150', '200', '300'], fontsize=45)
            elif plotting_data['x'][0] == "edge_density":
                ax.set_xlabel('Edge Density %')
                if y_ax == "latency":
                    ax.set_ylim([0, 1])
                if y_ax == "sl_cost":
                    ax.set_yticks(range(2500, 15001, 2500))
                    ax.set_yticklabels(['2.5k', '5k', '7.5k', '10k', '12.5k', '15k'])
            elif plotting_data['x'][0] == "src_dst_pair":
                ax.set_xlabel("# of (Source, Destination) Pairs")
                if y_ax == "latency":
                    ax.set_ylim([0, 0.15])
                if y_ax == "sl_cost":
                    ax.set_yticks(range(2500, 15001, 2500))
                    ax.set_yticklabels(['2.5k', '5k', '7.5k', '10k', '12.5k', '15k'])
            if y_ax == "avg_latency" or y_ax == 'max_latency':
                ax.set_ylabel('Time (ms)')
            elif y_ax == "sl_latency":
                ax.set_ylabel('Time (s)')
            elif y_ax == "sl_cost":
                ax.set_ylabel('EPs/s')

            ax.tick_params(axis='x', direction='in', length=15, width=3, pad=10)
            ax.tick_params(axis='y', direction='in', length=15, width=3 if y_ax != "sl_cost" else 1,
                           pad=10 if y_ax != "sl_cost" else -30)

        # the plot
        fig, (ax0, ax1, ax2, ax3) = plt.subplots(1, 4, figsize=(60, 13))
        fig.subplots_adjust(left=0.04, right=0.99, top=0.84, bottom=0.2)

        # ax2: cost budget target reduction
        plotting_data = {}
        plotting_data['x'] = ("cost_budget", list(range(5000, 40001, 5000)))
        if y_ax == "avg_latency":
            plotting_data['single_plus_non_shortest'] = [0.078, 0.066, 0.055, 0.037, 0.025, 0.03, 0.016, 0.012]
            plotting_data['single_naive'] = [0.182, 0.128, 0.113, 0.104, 0.082, 0.082, 0.09, 0.07]
            plotting_data['single_cluster'] = [0.131, 0.108, 0.101, 0.071, 0.086, 0.066, 0.068, 0.071]
            plotting_data['without_sls'] = [0.157] * 8
        elif y_ax == "max_latency":
            plotting_data['single_plus_non_shortest'] = [0.227, 0.2, 0.191, 0.148, 0.084, 0.201, 0.079, 0.075]
            plotting_data['single_naive'] = [0.549, 0.426, 0.403, 0.405, 0.325, 0.379, 0.465, 0.285]
            plotting_data['single_cluster'] = [0.339, 0.262, 0.283, 0.198, 0.333, 0.251, 0.263, 0.331]
            plotting_data['without_sls'] = [0.327] * 8
        elif y_ax == "sl_cost":
            plotting_data['single_plus_non_shortest'] = [4884.803, 9756.642, 14711.381, 19638.372, 24363.018, 28675.371,
                                                         33222.309, 38320.64]
            plotting_data['single_naive'] = [4278.755, 9131.156, 13975.226, 18742.586, 21812.395, 25947.826, 26264.701,
                                             28379.522]
            plotting_data['single_cluster'] = [3158.373, 6842.728, 11435.744, 13316.903, 20701.546, 23529.413,
                                               26091.793, 27267.121]
        elif y_ax == "sl_latency":
            plotting_data['single_plus_non_shortest'] = [0.29, 0.587, 0.817, 1.218, 1.599, 2.211, 2.563, 3.307]
            plotting_data['single_naive'] = [0.256, 0.512, 0.75, 0.947, 1.071, 1.198, 1.204, 1.311]
            plotting_data['single_cluster'] = [0.167, 0.381, 0.596, 0.683, 1.036, 1.177, 1.232, 1.307]
        helper(ax0, plotting_data, y_ax)

        # ax0: varying # of nodes
        plotting_data = {}
        plotting_data['x'] = ("num_nodes", [1, 2, 3, 4, 5, 6])
        if y_ax == "avg_latency":
            plotting_data['single_plus_non_shortest'] = [0.148, 0.086, 0.037, 0.012, 0.01, 0.003]
            plotting_data['single_naive'] = [0.262, 0.181, 0.104, 0.045, 0.038, 0.013]
            plotting_data['single_cluster'] = [0.191, 0.138, 0.071, 0.061, 0.064, 0.026]
            plotting_data['without_sls'] = [0.333, 0.258, 0.157, 0.114, 0.107, 0.075]
        elif y_ax == "max_latency":
            plotting_data['single_plus_non_shortest'] = [0.64, 0.347, 0.084, 0.055, 0.061, 0.026]
            plotting_data['single_naive'] = [0.965, 0.539, 0.405, 0.204, 0.188, 0.082]
            plotting_data['single_cluster'] = [0.521, 0.469, 0.198, 0.237, 0.198, 0.108]
            plotting_data['without_sls'] = [0.968, 0.686, 0.327, 0.255, 0.25, 0.192]
        elif y_ax == "sl_cost":
            plotting_data['single_plus_non_shortest'] = [19090.963, 19465.824, 19638.372, 19744.642, 19352.686,
                                                         18875.685]
            plotting_data['single_naive'] = [17630.573, 18484.648, 18742.586, 18228.182, 18450.476, 16154.737]
            plotting_data['single_cluster'] = [11596.633, 15785.883, 13316.903, 14500.222, 14967.199, 14992.788]
        elif y_ax == "sl_latency":
            plotting_data['single_plus_non_shortest'] = [1.35, 1.454, 1.218, 1.118, 1.204, 1.035]
            plotting_data['single_naive'] = [1.063, 1.055, 0.947, 0.887, 0.892, 0.735]
            plotting_data['single_cluster'] = [0.656, 0.837, 0.683, 0.725, 0.669, 0.67]
        helper(ax2, plotting_data, y_ax)

        # ax1: varying (s, d) distance
        plotting_data = {}
        plotting_data['x'] = ("src_dst_pair", [4, 8, 12, 16, 20])
        if y_ax == "avg_latency":
            plotting_data['single_plus_non_shortest'] = [0.005, 0.012, 0.037, 0.046, 0.057]
            plotting_data['single_naive'] = [0.083, 0.072, 0.104, 0.095, 0.122]
            plotting_data['single_cluster'] = [0.033, 0.054, 0.071, 0.074, 0.108]
            plotting_data['without_sls'] = [0.181, 0.167, 0.157, 0.18, 0.164]
        elif y_ax == "max_latency":
            plotting_data['single_plus_non_shortest'] = [0.014, 0.045, 0.148, 0.172, 0.236]
            plotting_data['single_naive'] = [0.254, 0.358, 0.405, 0.389, 0.598]
            plotting_data['single_cluster'] = [0.08, 0.192, 0.198, 0.211, 0.398]
            plotting_data['without_sls'] = [0.324, 0.365, 0.327, 0.515, 0.554]
        elif y_ax == "sl_cost":
            plotting_data['single_plus_non_shortest'] = [13474.514, 19192.113, 19638.372, 19556.396, 19799.618]
            plotting_data['single_naive'] = [8547.589, 17277.726, 18742.586, 19497.666, 19417.109]
            plotting_data['single_cluster'] = [10539.017, 14832.111, 13316.903, 16621.88, 11663.444]
        elif y_ax == "sl_latency":
            plotting_data['single_plus_non_shortest'] = [0.741, 1.133, 1.218, 1.177, 1.21]
            plotting_data['single_naive'] = [0.42, 0.85, 0.947, 1.085, 1.112]
            plotting_data['single_cluster'] = [0.483, 0.726, 0.683, 0.848, 0.641]
        helper(ax3, plotting_data, y_ax)

        # ax3: varying density of edges
        plotting_data = {}
        plotting_data['x'] = ("edge_density", [4, 6, 8, 10, 12])
        if y_ax == "avg_latency":
            plotting_data['single_plus_non_shortest'] = [0.199, 0.099, 0.037, 0.021, 0.01]
            plotting_data['single_naive'] = [0.325, 0.24, 0.104, 0.054, 0.038]
            plotting_data['single_cluster'] = [0.341, 0.174, 0.071, 0.049, 0.051]
            plotting_data['without_sls'] = [0.432, 0.296, 0.157, 0.122, 0.111]
        elif y_ax == "max_latency":
            plotting_data['single_plus_non_shortest'] = [0.688, 0.441, 0.148, 0.105, 0.158]
            plotting_data['single_naive'] = [1.209, 0.857, 0.405, 0.224, 0.141]
            plotting_data['single_cluster'] = [1.014, 0.515, 0.198, 0.13, 0.225]
            plotting_data['without_sls'] = [1.172, 0.771, 0.327, 0.304, 0.271]
        elif y_ax == "sl_cost":
            plotting_data['single_plus_non_shortest'] = [19589.772, 19655.088, 19638.372, 19360.963, 19614.03]
            plotting_data['single_naive'] = [18304.491, 18687.999, 18742.586, 17831.853, 17477.729]
            plotting_data['single_cluster'] = [16012.841, 13617.821, 13316.903, 12488.352, 15831.849]
        elif y_ax == "sl_latency":
            plotting_data['single_plus_non_shortest'] = [1.438, 1.443, 1.218, 1.178, 1.253]
            plotting_data['single_naive'] = [1.009, 1.142, 0.947, 0.915, 0.887]
            plotting_data['single_cluster'] = [0.851, 0.804, 0.683, 0.617, 0.782]
        helper(ax1, plotting_data, y_ax)

        # one legend for all subplots
        handles, labels = ax1.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=6, fontsize=55, handlelength=3.6)
        plt.figtext(0.13, 0.01, '(a)', weight='bold')
        plt.figtext(0.38, 0.01, '(b)', weight='bold')
        plt.figtext(0.63, 0.01, '(c)', weight='bold')
        plt.figtext(0.87, 0.01, '(d)', weight='bold')
        plt.savefig('results/pre_distribution/quantum_week_results/single_' + y_ax + '.png')

    @staticmethod
    def pre_single_latency_qw_12reps(y_ax: str):
        ''' Pre-distribution results for quantum week conference
        '''

        def helper(ax, plotting_data, y_ax):
            for method, val in plotting_data.items():
                x = plotting_data['x'][1]  # number of points
                if method != 'x' and len(val) == len(x):
                    line = Plot.LINE[method]
                    label = Plot.LEGEND_PROTO[method]
                    color = Plot.COLOR_PROTO[method]
                    ax.plot(x, val if y_ax == 'sl_latency' else [y * 1000 for y in val],
                            label=label, color=color, linestyle=line)

            if plotting_data['x'][0] == "cost_budget":
                ax.set_xlabel('Total Cost Budget (# of attempts)')
                ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
                ax.xaxis.set_major_locator(mticker.MaxNLocator(4))
                ticks_loc = ax.get_xticks().tolist()
                ax.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
                ax.set_xticklabels(['', '10k', '20k', '30k', '40k', ''])

                ax.yaxis.set_major_locator(mticker.MaxNLocator(5))
                if y_ax == "avg_latency":
                    ax.set_yticks([10, 40, 80, 120, 160])
                if y_ax == "max_latency":
                    ax.set_yticks([75, 150, 300, 450])
                # ax.set_xticklabels(['2.5k', '5k', '7.5k', '10k', '12.5k', '15k'])
                if y_ax == "latency":
                    ax.set_ylim([0, 0.15])
                if y_ax == "sl_cost":
                    ax.set_yticks(range(10000, 15001, 15000))
                    ax.set_yticklabels(['10k', '25k', '40k', '55k', '70k', '85k'])
                ax.get_yaxis().set_major_formatter(mticker.ScalarFormatter())
            elif plotting_data['x'][0] == "num_nodes":
                ax.set_xlabel('# of Nodes', labelpad=13)
                if y_ax == "latency":
                    ax.set_ylim([0, 0.35])
                if y_ax == "sl_cost":
                    ax.set_yticks(range(2000, 11000, 2000))
                    ax.set_yticklabels(['2k', '4k', '6k', '8k', '10k'])
                # ax.set_yticks(range(0, 2))
                ax.set_xticks(range(1, 7))
                ax.set_xticklabels(['50', '75', '100', '150', '200', '300'], fontsize=45)
            elif plotting_data['x'][0] == "edge_density":
                ax.set_xlabel('Edge Density %')
                if y_ax == "latency":
                    ax.set_ylim([0, 1])
                if y_ax == "sl_cost":
                    ax.set_yticks(range(2500, 15001, 2500))
                    ax.set_yticklabels(['2.5k', '5k', '7.5k', '10k', '12.5k', '15k'])
            elif plotting_data['x'][0] == "src_dst_pair":
                ax.set_xlabel("# of (Source, Destination) Pairs")
                if y_ax == "latency":
                    ax.set_ylim([0, 0.15])
                if y_ax == "sl_cost":
                    ax.set_yticks(range(2500, 15001, 2500))
                    ax.set_yticklabels(['2.5k', '5k', '7.5k', '10k', '12.5k', '15k'])
            if y_ax == "avg_latency" or y_ax == 'max_latency':
                ax.set_ylabel('Time (ms)')
            elif y_ax == "sl_latency":
                ax.set_ylabel('Time (s)')
            elif y_ax == "sl_cost":
                ax.set_ylabel('EPs/s')

            ax.tick_params(axis='x', direction='in', length=15, width=3, pad=10)
            ax.tick_params(axis='y', direction='in', length=15, width=3 if y_ax != "sl_cost" else 1,
                           pad=10 if y_ax != "sl_cost" else -30)

        # we want legends only for avg_latency
        if y_ax == "avg_latency":
            fig, (ax0, ax1, ax2, ax3) = plt.subplots(1, 4, figsize=(60, 13))
            fig.subplots_adjust(left=0.04, right=0.99, top=0.84, bottom=0.2)
        else:
            fig, (ax0, ax1, ax2, ax3) = plt.subplots(1, 4, figsize=(60, 10.4))
            fig.subplots_adjust(left=0.04, right=0.99, top=0.97, bottom=0.2)


        # ax2: cost budget target reduction
        plotting_data = {}
        plotting_data['x'] = ("cost_budget", list(range(5000, 40001, 5000)))
        if y_ax == "avg_latency":
            plotting_data['single_plus_non_shortest'] = [0.077, 0.063, 0.053, 0.046, 0.031, 0.026, 0.016, 0.012]
            plotting_data['single_naive'] = [0.149, 0.119, 0.101, 0.09, 0.072, 0.071, 0.073, 0.068]
            plotting_data['single_cluster'] = [0.126, 0.096, 0.098, 0.088, 0.071, 0.066, 0.068, 0.071]
            plotting_data['without_sls'] = [0.156] * 8
        elif y_ax == "max_latency":
            plotting_data['single_plus_non_shortest'] = [0.228, 0.194, 0.206, 0.224, 0.201, 0.161, 0.079, 0.075]
            plotting_data['single_naive'] = [0.418, 0.384, 0.347, 0.325, 0.278, 0.261, 0.252, 0.245]
            plotting_data['single_cluster'] = [0.33, 0.319, 0.277, 0.287, 0.247, 0.231, 0.225, 0.212]
            plotting_data['without_sls'] = [0.33] * 8
        elif y_ax == "sl_cost":
            plotting_data['single_plus_non_shortest'] = [4885.734, 9813.225, 14671.557, 19581.264, 24357.867, 28675.371,
                                                         33222.309, 38320.64]
            plotting_data['single_naive'] = [4288.842, 9131.714, 13705.394, 18200.37, 21742.813, 25947.826, 26264.701,
                                             28379.522]
            plotting_data['single_cluster'] = [3171.685, 7200.649, 10670.185, 14147.789, 18299.415, 23529.413,
                                               26091.793, 27267.121]
        elif y_ax == "sl_latency":
            plotting_data['single_plus_non_shortest'] = [0.28, 0.59, 0.892, 1.319, 1.698, 2.211, 2.563, 3.307]
            plotting_data['single_naive'] = [0.262, 0.519, 0.734, 0.911, 1.046, 1.198, 1.204, 1.311]
            plotting_data['single_cluster'] = [0.172, 0.364, 0.556, 0.762, 0.944, 1.177, 1.232, 1.307]
        helper(ax0, plotting_data, y_ax)

        # ax0: varying # of nodes
        plotting_data = {}
        plotting_data['x'] = ("num_nodes", [1, 2, 3, 4, 5, 6])
        if y_ax == "avg_latency":
            plotting_data['single_plus_non_shortest'] = [0.148, 0.086, 0.046, 0.017, 0.01, 0.004]
            plotting_data['single_naive'] = [0.272, 0.157, 0.09, 0.052, 0.034, 0.016]
            plotting_data['single_cluster'] = [0.199, 0.131, 0.088, 0.087, 0.059, 0.048]
            plotting_data['without_sls'] = [0.36, 0.248, 0.156, 0.127, 0.102, 0.079]
        elif y_ax == "max_latency":
            plotting_data['single_plus_non_shortest'] = [0.539, 0.405, 0.224, 0.093, 0.061, 0.025]
            plotting_data['single_naive'] = [0.938, 0.523, 0.325, 0.205, 0.177, 0.09]
            plotting_data['single_cluster'] = [0.523, 0.43, 0.287, 0.329, 0.204, 0.191]
            plotting_data['without_sls'] = [0.991, 0.665, 0.33, 0.303, 0.216, 0.177]
        elif y_ax == "sl_cost":
            plotting_data['single_plus_non_shortest'] = [19171.117, 19395.249, 19581.264, 19682.594, 19484.906,
                                                         19046.582]
            plotting_data['single_naive'] = [18193.701, 18651.033, 18200.37, 17721.386, 18757.866, 17279.507]
            plotting_data['single_cluster'] = [15033.513, 15742.027, 14147.789, 14921.88, 14704.004, 12400.199]
        elif y_ax == "sl_latency":
            plotting_data['single_plus_non_shortest'] = [1.348, 1.468, 1.319, 1.12, 1.151, 1.042]
            plotting_data['single_naive'] = [1.163, 1.083, 0.911, 0.861, 0.903, 0.792]
            plotting_data['single_cluster'] = [0.811, 0.857, 0.762, 0.723, 0.702, 0.563]
        helper(ax2, plotting_data, y_ax)

        # ax1: varying (s, d) distance
        plotting_data = {}
        plotting_data['x'] = ("edge_density", [4, 6, 8, 10, 12])
        if y_ax == "avg_latency":
            plotting_data['single_plus_non_shortest'] = [0.218, 0.1, 0.046, 0.018, 0.013]
            plotting_data['single_naive'] = [0.358, 0.183,0.09, 0.057, 0.037]
            plotting_data['single_cluster'] = [0.312, 0.181, 0.088, 0.058, 0.059]
            plotting_data['without_sls'] = [0.473, 0.275, 0.156, 0.128, 0.103]
        elif y_ax == "max_latency":
            plotting_data['single_plus_non_shortest'] = [0.717, 0.474, 0.224, 0.077, 0.089]
            plotting_data['single_naive'] = [1.119, 0.639, 0.325, 0.25, 0.152]
            plotting_data['single_cluster'] = [0.935, 0.603, 0.287, 0.171, 0.262]
            plotting_data['without_sls'] = [1.161, 0.77, 0.33, 0.311, 0.24]
        elif y_ax == "sl_cost":
            plotting_data['single_plus_non_shortest'] = [19660.863, 19684.365, 19581.264, 19339.622, 19430.692]
            plotting_data['single_naive'] = [18105.878, 18816.549, 18200.37, 18239.73, 16676.149]
            plotting_data['single_cluster'] = [12733.919, 15343.563, 14147.789, 15395.943, 16517.486]
        elif y_ax == "sl_latency":
            plotting_data['single_plus_non_shortest'] = [1.464, 1.346, 1.319, 1.174, 1.27]
            plotting_data['single_naive'] = [1.077, 1.177, 0.911, 0.905, 0.806]
            plotting_data['single_cluster'] = [0.719, 0.86, 0.762, 0.756, 0.818]
        helper(ax1, plotting_data, y_ax)

        # ax3: varying density of edges
        plotting_data = {}
        plotting_data['x'] = ("src_dst_pair", [4, 8, 12, 16, 20])
        if y_ax == "avg_latency":
            plotting_data['single_plus_non_shortest'] = [0.004, 0.015, 0.046, 0.051, 0.064]
            plotting_data['single_naive'] = [0.051, 0.068, 0.09, 0.099, 0.118]
            plotting_data['single_cluster'] = [0.061, 0.073, 0.088, 0.097, 0.103]
            plotting_data['without_sls'] = [0.153, 0.157, 0.156, 0.166, 0.162]
        elif y_ax == "max_latency":
            plotting_data['single_plus_non_shortest'] = [0.014, 0.068, 0.224, 0.216, 0.39]
            plotting_data['single_naive'] = [0.163, 0.282, 0.325, 0.379, 0.561]
            plotting_data['single_cluster'] = [0.158, 0.222, 0.287, 0.364, 0.351]
            plotting_data['without_sls'] = [0.263, 0.315, 0.33, 0.429, 0.475]
        elif y_ax == "sl_cost":
            plotting_data['single_plus_non_shortest'] = [12781.869, 18600.052, 19581.264, 19619.531, 19750.562]
            plotting_data['single_naive'] = [9906.303, 16421.069, 18200.37, 18952.62, 19206.852]
            plotting_data['single_cluster'] = [10484.378, 13825.919, 14147.789, 14414.18, 13757.965]
        elif y_ax == "sl_latency":
            plotting_data['single_plus_non_shortest'] = [0.757, 1.171, 1.319, 1.292, 1.25]
            plotting_data['single_naive'] = [0.447, 0.796, 0.911, 1.051, 1.121]
            plotting_data['single_cluster'] = [0.466, 0.721, 0.762, 0.751, 0.781]
        helper(ax3, plotting_data, y_ax)

        # one legend for all subplots
        handles, labels = ax1.get_legend_handles_labels()
        if y_ax == "avg_latency":
            fig.legend(handles, labels, loc='upper center', ncol=6, fontsize=55, handlelength=3.6)
        plt.figtext(0.13, 0.01, '(a)', weight='bold')
        plt.figtext(0.38, 0.01, '(b)', weight='bold')
        plt.figtext(0.63, 0.01, '(c)', weight='bold')
        plt.figtext(0.87, 0.01, '(d)', weight='bold')
        plt.savefig('results/pre_distribution/quantum_week_results/single_' + y_ax + '.png')

    @staticmethod
    def pre_single_latency_qw_greedy(y_ax: str):
        ''' Pre-distribution results for quantum week conference
        '''

        def helper(ax, plotting_data, y_ax):
            for method, val in plotting_data.items():
                x = plotting_data['x'][1]  # number of points
                if method != 'x' and len(val) == len(x):
                    line = Plot.LINE[method]
                    label = Plot.LEGEND_PROTO[method]
                    color = Plot.COLOR_PROTO[method]
                    ax.plot(x, val if y_ax == 'sl_latency' else [y * 1000 for y in val],
                            label=label, color=color, linestyle=line)

            if plotting_data['x'][0] == "cost_budget":
                ax.set_xlabel('Total Cost Budget (# of attempts)')

                ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
                ax.xaxis.set_major_locator(mticker.MaxNLocator(4))
                ticks_loc = ax.get_xticks().tolist()
                ax.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
                ax.set_xticklabels(['', '10k', '20k', '30k', '40k', ''])
                if y_ax == "latency":
                    ax.set_ylim([0, 0.15])
                if y_ax == "sl_cost":
                    ax.set_yticks(range(10000, 15001, 15000))
                    ax.set_yticklabels(['10k', '25k', '40k', '55k', '70k', '85k'])
            elif plotting_data['x'][0] == "num_nodes":
                ax.set_xlabel('# of Nodes', labelpad=13)
                if y_ax == "latency":
                    ax.set_ylim([0, 0.35])
                if y_ax == "sl_cost":
                    ax.set_yticks(range(2000, 11000, 2000))
                    ax.set_yticklabels(['2k', '4k', '6k', '8k', '10k'])
                # ax.set_yticks(range(0, 2))
                ax.set_xticks(range(1, 7))
                ax.set_xticklabels(['50', '75', '100', '150', '200', '300'], fontsize=45)
            elif plotting_data['x'][0] == "edge_density":
                ax.set_xlabel('Edge Density %')
                if y_ax == "latency":
                    ax.set_ylim([0, 1])
                if y_ax == "sl_cost":
                    ax.set_yticks(range(2500, 15001, 2500))
                    ax.set_yticklabels(['2.5k', '5k', '7.5k', '10k', '12.5k', '15k'])
            elif plotting_data['x'][0] == "src_dst_pair":
                ax.set_xlabel("# of (Source, Destination) Pairs")
                if y_ax == "latency":
                    ax.set_ylim([0, 0.15])
                if y_ax == "sl_cost":
                    ax.set_yticks(range(2500, 15001, 2500))
                    ax.set_yticklabels(['2.5k', '5k', '7.5k', '10k', '12.5k', '15k'])
            if y_ax == "avg_latency" or y_ax == 'max_latency':
                ax.set_ylabel('Time (ms)')
            elif y_ax == "sl_latency":
                ax.set_ylabel('Time (s)')
            elif y_ax == "sl_cost":
                ax.set_ylabel('EPs/s')

            ax.tick_params(axis='x', direction='in', length=15, width=3, pad=10)
            ax.tick_params(axis='y', direction='in', length=15, width=3 if y_ax != "sl_cost" else 1,
                           pad=10 if y_ax != "sl_cost" else -30)
            ax.set_yscale('log')
            ax.set_yticks([10, 20, 100])
            ax.get_yaxis().set_major_formatter(mticker.ScalarFormatter())
            # ax.get_yaxis().get_major_formatter().labelOnlyBase = False

        # the plot
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(30, 13))
        fig.subplots_adjust(left=0.08, right=0.99, top=0.77, bottom=0.2)

        # ax2: cost budget target reduction
        plotting_data = {}
        plotting_data['x'] = ("cost_budget", list(range(5000, 40001, 5000)))
        if y_ax == "avg_latency":
            plotting_data['single_plus_non_shortest'] = [0.078, 0.066, 0.055, 0.037, 0.025, 0.03, 0.016, 0.012]
            plotting_data['single'] = [0.084, 0.068, 0.053, 0.035, 0.029, 0.029, 0.023, 0.023]
            # plotting_data['single_no_deletion_latency'] = [0.13, 0.108, 0.08, 0.078, 0.06, 0.065, 0.05, 0.051]
            plotting_data['single_no_deletion'] = [0.111, 0.107, 0.109, 0.099, 0.092, 0.093, 0.093, 0.093]
            plotting_data['without_sls'] = [0.157] * 8
        elif y_ax == "max_latency":
            plotting_data['single_plus_non_shortest'] = [0.227, 0.2, 0.191, 0.148, 0.084, 0.201, 0.079, 0.075]
            plotting_data['single'] = [0.223, 0.164, 0.159, 0.121, 0.123, 0.136, 0.09, 0.09]
            # plotting_data['single_no_deletion_latency'] = [0.308, 0.374, 0.189, 0.219, 0.144, 0.23, 0.334, 0.347]
            plotting_data['single_no_deletion'] = [0.236, 0.232, 0.367, 0.327, 0.315, 0.368, 0.368, 0.368]
            plotting_data['without_sls'] = [0.327] * 8
        elif y_ax == "sl_cost":
            plotting_data['single_plus_non_shortest'] = [4884.803, 9756.642, 14711.381, 19638.372, 24363.018, 28675.371,
                                                         33222.309, 38320.64]
            plotting_data['single'] = [4956.291, 9866.937, 14715.728, 19363.432, 24599.053, 28522.007, 30166.403,
                                       30166.403]
            # plotting_data['single_no_deletion_latency'] = [4954.016, 9980.301, 14951.594, 19932.436, 24892.392,
            #                                                28473.066, 30906.55, 31912.08]
            plotting_data['single_no_deletion'] = [4788.403, 9614.659, 13700.849, 17162.73, 18817.857, 20768.711,
                                                   20768.711, 20768.711]
        elif y_ax == "sl_latency":
            plotting_data['single_plus_non_shortest'] = [0.29, 0.587, 0.817, 1.218, 1.599, 2.211, 2.563, 3.307]
            plotting_data['single'] = [0.27, 0.546, 0.803, 1.003, 1.223, 1.446, 1.514, 1.514]
            # plotting_data['single_no_deletion_latency'] = [0.283, 0.54, 0.715, 0.905, 1.178, 1.374, 1.509, 1.494]
            plotting_data['single_no_deletion'] = [0.259, 0.52, 0.758, 0.919, 0.992, 1.153, 1.153, 1.153]
        helper(ax0, plotting_data, y_ax)

        # ax1: varying density of edges
        plotting_data = {}
        plotting_data['x'] = ("edge_density", [4, 6, 8, 10, 12])
        if y_ax == "avg_latency":
            plotting_data['single_plus_non_shortest'] = [0.199, 0.099, 0.037, 0.021, 0.01]
            plotting_data['single'] = [0.234, 0.109, 0.035, 0.027, 0.013]
            # plotting_data['single_no_deletion_latency'] = [0.298, 0.146, 0.078, 0.042, 0.029]
            plotting_data['single_no_deletion'] = [0.236, 0.186, 0.099, 0.057, 0.041]
            plotting_data['without_sls'] = [0.432, 0.296, 0.157, 0.122, 0.111]
        elif y_ax == "max_latency":
            plotting_data['single_plus_non_shortest'] = [0.688, 0.441, 0.148, 0.105, 0.158]
            plotting_data['single'] = [0.819, 0.418, 0.121, 0.114, 0.045]
            # plotting_data['single_no_deletion_latency'] = [0.885, 0.432, 0.219, 0.182, 0.129]
            plotting_data['single_no_deletion'] = [0.834, 0.792, 0.327, 0.179, 0.099]
            plotting_data['without_sls'] = [1.172, 0.771, 0.327, 0.304, 0.271]
        elif y_ax == "sl_cost":
            plotting_data['single_plus_non_shortest'] = [19589.772, 19655.088, 19638.372, 19360.963, 19614.03]
            plotting_data['single'] = [19680.78, 19670.341, 19363.432, 19028.329, 18553.626]
            # plotting_data['single_no_deletion_latency'] = [19952.604, 19920.313, 19932.436, 19683.632, 19498.01]
            plotting_data['single_no_deletion'] = [16626.747, 18535.95, 17162.73, 13851.85, 15260.851]
        elif y_ax == "sl_latency":
            plotting_data['single_plus_non_shortest'] = [1.438, 1.443, 1.218, 1.178, 1.253]
            plotting_data['single'] = [1.056, 1.174, 1.003, 0.982, 0.974]
            # plotting_data['single_no_deletion_latency'] = [1.046, 1.016, 0.905, 0.925, 0.988]
            plotting_data['single_no_deletion'] = [0.97, 1.175, 0.919, 0.705, 0.791]
        helper(ax1, plotting_data, y_ax)

        # one legend for all subplots
        handles, labels = ax1.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=2, fontsize=55, handlelength=3.6)
        plt.figtext(0.25, 0.01, '(a)', weight='bold')
        plt.figtext(0.75, 0.01, '(b)', weight='bold')
        plt.savefig('results/pre_distribution/quantum_week_results/single_different_greedy' + y_ax + '_log.png')

    @staticmethod
    def ghz_latency(y_ax: str, area: str):
        ''' Pre-distribution results for quantum week conference
        '''

        def helper(ax, plotting_data, y_ax):
            for method, val in plotting_data.items():
                x = plotting_data['x'][1]  # number of points
                if method != 'x' and len(val) == len(x):
                    line = Plot.LINE[method]
                    label = Plot.LEGEND_PROTO[method]
                    color = Plot.COLOR_PROTO[method]
                    ax.plot(x, val, label=label, color=color, linestyle=line)

            if plotting_data['x'][0] == "num_end_nodes":
                ax.set_xlabel('# of Terminals')
                # ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
                # ax.xaxis.set_major_locator(mticker.MaxNLocator(7))
                # ticks_loc = ax.get_xticks().tolist()
                # ax.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
                # ax.set_xticklabels(['3', '4', '5', '6', '7'])

                # ax.yaxis.set_major_locator(mticker.MaxNLocator(7))
                # if y_ax == "avg_latency":
                #     ax.set_yticks([10, 40, 80, 120, 160])
                # ax.set_yticks([75, 150, 300, 450, 600])
                # ax.set_xticklabels(['2.5k', '5k', '7.5k', '10k', '12.5k', '15k'])
                if y_ax == "avg_latency":
                    ax.set_yscale('log')
                ax.get_yaxis().set_major_formatter(mticker.ScalarFormatter())
            elif plotting_data['x'][0] == "num_nodes":
                if y_ax == "avg_latency":
                    ax.set_yscale('log')
                ax.set_xlabel('# of Nodes', labelpad=13)
                # ax.set_yticks(range(0, 2))
                ax.set_xticks(range(1, 6))
                ax.set_xticklabels(['50', '75', '100', '200', '300'], fontsize=45)
            elif plotting_data['x'][0] == "edge_density" or y_ax == "analytical":
                ax.set_xlabel('Edge Density %')

                if y_ax == "avg_latency":
                    ax.set_yscale('log')
            elif plotting_data['x'][0] == "atomic_bsm":
                if y_ax == "avg_latency":
                    ax.set_yscale('log')
                ax.set_xlabel("Atomic Fusion/BSM rate")
            if y_ax == "avg_latency" or y_ax == "analytical":
                ax.set_ylabel('Time (s)')
            if area == "largee":
                ax.set_yscale('log')
            ax.tick_params(axis='x', direction='in', length=15, width=3, pad=10)
            ax.tick_params(axis='y', direction='in', length=15, width= 1,
                           pad=1 -30)

        # the plot
        fig, (ax0, ax1, ax2, ax3) = plt.subplots(1, 4, figsize=(60, 13))
        fig.subplots_adjust(left=0.04, right=0.99, top=0.84, bottom=0.2)

        # ax2: cost budget target reduction
        plotting_data = {}
        plotting_data['x'] = ("num_end_nodes", [3, 4, 5, 6, 7])
        if y_ax == "avg_latency" or y_ax == "analytical":
            if area == "large":
                plotting_data['ghz_fro'] =         [1.183, 0.99, 1.987, 3.606, 5.234]#[0.315, 1.110, 1.493, 2.055, 2.634]  # [0.453, 1.823, 2.201, 4.93, 6.036]
                plotting_data['ghz_fst_latency'] = [0.391, 0.828, 1.213, 1.909, 2.504] # [0.315, 0.936, 1.423, 2.265, 2.912] # [0.277, 0.775, 1.244, 1.744, 2.695]  # [0.452, 1.452, 2.39, 4.121, 6.33]
                plotting_data['ghz_fst_edge'] =    [0.376, 0.789, 1.278, 1.728, 2.043] # [0.355, 0.922, 1.359, 1.93, 2.131] # [0.341, 0.82, 1.153, 1.874, 2.055]  # [0.510, 1.58, 2.522, 4.79, 6.471]
                if y_ax == "avg_latency":
                    plotting_data['ghz_naive'] =    [4.987, 21.0, 31.475, 81.417, 420.928]  # [0.4507, 2.247, 7.155, 28.466, 120.058]
                    plotting_data['ghz_star_exp'] = [1.282, 8.386, 17.381, 143.582, 653.249]

            else:
                plotting_data['ghz_fro'] = [0.033, 0.205, 0.302, 2.647, 4.673]
                plotting_data['ghz_fst_latency'] = [0.033, 0.122, 0.19, 0.639, 0.956]
                plotting_data['ghz_fst_edge'] = [0.033, 0.122, 0.19, 0.64, 0.954]
        elif y_ax == "fidelity":
            plotting_data['ghz_fro'] =         [0.912, 0.864, 0.813, 0.771, 0.707]#[0.988, 0.947, 0.863, 0.899, 0.972]
            plotting_data['ghz_fst_latency'] = [0.935, 0.891, 0.851, 0.795, 0.758] # [0.993, 0.976, 0.962, 0.919, 0.817]  # [0.988, 0.987, 0.946, 0.935, 0.922]
            plotting_data['ghz_fst_edge'] =    [0.928, 0.893, 0.84, 0.81, 0.742] # [0.988, 0.977, 0.94, 0.966, 0.897] # [0.994, 0.979, 0.963, 0.948, 0.918]
            plotting_data['ghz_naive'] =       [0.87, 0.832, 0.61, 0.58, np.NaN]
            plotting_data['ghz_star_exp'] =    [0.90, 0.830, 0.69, np.NaN, np.NaN]
        if y_ax == "analytical":
            plotting_data['ghz_fro_ana'] = [1.139, 1.423, 3.082, 5.191, 9.231]# [0.377, 1.551, 2.274, 4.917, 6.185]
            plotting_data['ghz_fst_latency_ana'] = [0.604, 2.056, 3.866, 6.652, 10.004] # [0.819, 1.699, 3.099, 5.507, 5.398]
            plotting_data['ghz_fst_edge_ana'] = [0.819, 2.133, 3.465, 5.904, 5.857] # [0.556, 1.699, 3.237, 4.83, 6.92]
        helper(ax0, plotting_data, y_ax)

        # ax0: varying # of nodes
        plotting_data = {}
        plotting_data['x'] = ("num_nodes", [1, 2, 3, 4, 5])
        if y_ax == "avg_latency" or y_ax == "analytical":
            plotting_data['ghz_fro'] = [2.924, 2.852, 1.987, 1.975, 0.934] # [3.285, 2.918, 1.493, 1.212, 0.913]
            plotting_data['ghz_fst_latency'] = [2.452, 1.736,  1.213, 0.888, 0.623] # [2.654, 1.906, 1.423, 0.979, 0.952] # [1.903, 1.363, 1.244, 0.807, 0.984]
            plotting_data['ghz_fst_edge'] = [2.583, 1.68, 1.278, 0.856, 0.65]# [2.815, 1.689, 1.359, 0.862, 1.316] #[2.643, 1.465, 1.153,  0.820, 1.176]
            if y_ax == "avg_latency":
                plotting_data['ghz_naive'] = [52.5, 37.5, 31.475, 26.25, 24.643]
                plotting_data['ghz_star_exp'] = [172.241, 51.326, 17.381, 52.091, 20.111]
        elif y_ax == "fidelity":
            plotting_data['ghz_fro'] = [0.75, 0.777, 0.813, 0.78, 0.83] #[0.875, 0.962, 0.863, 0.83, 0.955]
            plotting_data['ghz_fst_latency'] = [0.787, 0.857, 0.851, 0.867, 0.901] # [0.934, 0.922, 0.962, 0.964, 0.952]  # [0.941, 0.927, 0.946, 0.97, 0.951]
            plotting_data['ghz_fst_edge'] = [0.797, 0.826, 0.84, 0.868, 0.889] # [0.882, 0.946, 0.94, 0.974, 0.921] # [0.829, 0.967, 0.982, 0.975, 0.902]
            plotting_data['ghz_naive'] = [0.375, 0.44, 0.61, 0.625, 0.662]
            plotting_data['ghz_star_exp'] = [np.NaN, 0.42, 0.69, 0.53, 0.67]
        if y_ax == "analytical":
            plotting_data['ghz_fro_ana'] = [5.371, 3.694, 3.082, 3.27, 2.743] # [5.097, 3.497, 2.274, 1.397, 1.188]
            plotting_data['ghz_fst_latency_ana'] = [6.563, 3.418, 2.925, 2.223, 2.315] # [5.463, 3.408, 3.866, 2.244, 3.038] # [4.613, 3.151, 3.099, 1.773, 3.038]
            plotting_data['ghz_fst_edge_ana'] = [8.436, 3.771, 3.402, 2.129, 2.633] # [9.997, 3.595, 3.465, 2.385, 3.674] # [8.03, 3.338, 3.237, 1.852, 3.675]
        helper(ax2, plotting_data, y_ax)

        # ax1: varying (s, d) distance
        plotting_data = {}
        plotting_data['x'] = ("atomic_bsm", [0.2, 0.3, 0.4, 0.5, 0.6])
        if y_ax == "avg_latency" or y_ax == "analytical":
            if area == "large":
                plotting_data['ghz_fro'] = [28.333, 7.056, 1.987, 0.69, 0.423]#[7.357, 2.605, 1.493, 4.85, 0.64]  # [18.979, 5.781, 2.201, 1.247, 0.734]
                plotting_data['ghz_fst_latency'] = [20.232, 3.912, 1.213, 0.505, 0.252] # [38.524, 4.694, 1.423, 0.6, 0.284] # [33.733, 4.298, 1.244, 0.488, 0.226]  # [4.487, 2.99, 2.39, 1.98, 1.87]
                plotting_data['ghz_fst_edge'] = [22.413, 4.268, 1.278, 0.54, 0.25]  #  [21.676, 4.638, 1.359, 0.572, 0.291] # [32.333, 4.041, 1.153, 0.486, 0.252] # [4.896, 3.264, 2.522, 2.018, 2.862]
                if y_ax == "avg_latency":
                    plotting_data['ghz_naive'] = [2916, 301.0, 30.4, 19.423, 3.541] # [980.895, 56.508, 7.155, 1.534, 0.403]
                    plotting_data['ghz_star_exp'] = [6525.92, 88.493, 17.381, 3.773, 1.044]
            else:
                plotting_data['ghz_fro'] = [1.208, 0.537, 0.302, 0.193, 0.134]
                plotting_data['ghz_fst_latency'] = [0.378, 0.253, 0.19, 0.152, 0.127]
                plotting_data['ghz_fst_edge'] = [0.378, 0.253, 0.19, 0.152, 0.127]
        elif y_ax == "fidelity":
            plotting_data['ghz_fro'] = [0.416, 0.839, 0.813, 0.862, 0.871] # [0.944, 0.923, 0.863, 0.911, 0.93]
            plotting_data['ghz_fst_latency'] = [0.768, 0.8, 0.851, 0.862, 0.865] # [0.731, 0.911, 0.962, 0.966, 0.981]  # [0.8, 0.891, 0.946, 0.974, 0.98]
            plotting_data['ghz_fst_edge'] = [0.52, 0.762, 0.84, 0.87, 0.867]  # [0.733, 0.914, 0.94, 0.963, 0.978]  # [0.728, 0.958, 0.963, 0.974, 0.979]
            plotting_data['ghz_naive'] = [np.NaN, np.NaN, 0.61, 0.677, 0.725]
            plotting_data['ghz_star_exp'] = [np.NaN, np.NaN, 0.69, 0.73, 0.78]
        if y_ax == "analytical":
            plotting_data['ghz_fro_ana'] = [28.243, 7.642, 3.082, 1.850, 0.823]# [9.094, 4.042, 2.274, 1.456, 1.011]
            plotting_data['ghz_fst_latency_ana'] = [62.081, 9.834, 2.925, 1.175, 0.608] # [82.485, 13.46, 3.866, 1.506, 0.708] # [63.334, 10.621, 3.099, 1.217, 0.575]
            plotting_data['ghz_fst_edge_ana'] = [58.449, 11.079, 3.402, 1.379, 0.537] # [68.113, 11.667, 3.465, 1.382, 0.662]  # [65.662, 11.066, 3.237, 1.273, 0.602]
        helper(ax1, plotting_data, y_ax)

        # ax3: varying density of edges
        plotting_data = {}
        plotting_data['x'] = ("edge_density", [4, 6, 8, 10, 12])
        if y_ax == "avg_latency" or y_ax == "analytical":
            if area == "large":
                plotting_data['ghz_fro'] = [10.902, 3.401, 1.982, 2.78, 1.23]# [4.929, 1.95, 1.493, 1.546, 1.909] # [8.043, 5.779, 2.201, 1.686, 2.427]
                plotting_data['ghz_fst_latency'] = [3.08, 1.582, 1.213, 1.224, 0.672] # [1.62, 1.422, 1.423, 1.016, 0.564]  # [1.146, 1.209, 1.244, 0.923, 0.456]  # [8.399, 3.386, 2.39, 1.786, 1.36]
                plotting_data['ghz_fst_edge'] = [2.708, 1.814, 1.278,  1.071, 0.691] # [1.364, 1.867, 1.359, 1.069, 0.609] # [1.273, 1.717, 1.153, 0.948, 0.565]  # [9.249, 5.613, 2.522, 1.786, 1.36]
                if y_ax == "avg_latency":
                    plotting_data['ghz_naive'] = [299.0, 49.0, 31.425, 25.98, 22.3]  # [33.439, 14.122, 7.155, 5.543, 4.722]
                    plotting_data['ghz_star_exp'] = [1820.759, 114.625, 17.381, 63.269, 5.486]
            else:
                plotting_data['ghz_fro'] = [0.444, 0.46, 0.302, 0.515, 3.042]
                plotting_data['ghz_fst_latency'] = [0.491, 0.573, 0.19,  0.153, 0.153]
                plotting_data['ghz_fst_edge'] = [0.491, 0.573, 0.19, 0.153, 0.153]
        elif y_ax == "fidelity":
            plotting_data['ghz_fro'] = [0.587, 0.69, 0.813, 0.79, 0.84] # [0.856, 0.938, 0.863, 0.919, 0.959]
            plotting_data['ghz_fst_latency'] = [0.443, 0.827, 0.851, 0.847, 0.886] # [0.89, 0.961, 0.962, 0.979, 0.982] # [0.89, 0.958, 0.946, 0.979, 0.978]
            plotting_data['ghz_fst_edge'] = [0.475, 0.82, 0.84, 0.842, 0.891] # [0.86, 0.938, 0.94, 0.958, 0.973] # [0.86, 0.937, 0.982, 0.969, 0.981]
            plotting_data['ghz_naive'] = [np.NaN, 0.53, 0.61, 0.63, 0.68]
            plotting_data['ghz_star_exp'] = [np.NaN, np.NaN, 0.69, 0.51, 0.75]
        if y_ax == "analytical":
            plotting_data['ghz_fro_ana'] = [12.187, 10.328,  3.082, 4.289, 2.76]#[10.15, 2.393, 2.274, 2.249, 4.258]
            plotting_data['ghz_fst_latency_ana'] = [6.781, 3.884, 2.925,  3.392, 1.874] # [1.772, 3.338, 3.866, 2.372, 1.357] # [1.187, 3.212, 3.099, 2.274, 1.279]
            plotting_data['ghz_fst_edge_ana'] = [5.89, 5.055, 3.402, 2.843, 1.945] #[1.941, 5.74, 3.465, 2.466, 1.744] # [1.941, 5.613, 3.237, 2.319, 1.744]

        helper(ax3, plotting_data, y_ax)

        # one legend for all subplots
        handles, labels = ax1.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=3 if y_ax == "analytical" else 5, fontsize=55, handlelength=3.6)
        plt.figtext(0.13, 0.01, '(a)', weight='bold')
        plt.figtext(0.38, 0.01, '(b)', weight='bold')
        plt.figtext(0.63, 0.01, '(c)', weight='bold')
        plt.figtext(0.87, 0.01, '(d)', weight='bold')
        plt.savefig(f"results/ghz/real_results/3rd_try_10_reps_link_adjustment_for_gf/{y_ax}_new.png")


if __name__ == '__main__':
    # Plot.graph(plotting_data={}, filename="results/2021_05_28/results_single_path_single_pair_atomic_bsm.png")
    # Plot.graph(plotting_data={}, filename="results/real_results/single_distance.png")

    # Plot.QNR_SP_NEW()
    # Plot.QNR_NEW()
    # Plot.analytical_throttle_new()
    # Plot.line_extra2()

    # Plot.QNR()

    # Plot.analytical_throttle()

    # Plot.fidelity()
    #
    # Plot.runtime()
    #
    # Plot.caleffi()
    # Plot.caleffi2()

    # Plot.fidelity2()

    # Plot.fidelity3()

    # Plot.pre_single("sl_cost")
    # Plot.pre_single("latency")
    # Plot.pre_single("sl_latency")
    #
    # Plot.pre_multi("sl_cost")
    # Plot.pre_multi("latency")
    # Plot.pre_multi("sl_latency")

    # Plot.pre_single_latency("avg_latency")
    # Plot.pre_single_latency("max_latency")
    # Plot.pre_single_latency("sl_latency")

    # Plot.pre_single_latency_qw("avg_latency")
    # Plot.pre_single_latency_qw("max_latency")
    # Plot.pre_single_latency_qw("sl_latency")
    # Plot.pre_single_latency_qw_12reps("avg_latency")
    # Plot.pre_single_latency_qw_12reps("max_latency")
    # Plot.pre_single_latency_qw_12reps("sl_latency")

    # Plot.pre_single_latency_qw_greedy('avg_latency')
    #
    # Plot.pre_multi_latency("avg_latency")
    # Plot.pre_multi_latency("max_latency")
    # Plot.pre_multi_latency("sl_latency")

    Plot.ghz_latency('avg_latency', "large")
