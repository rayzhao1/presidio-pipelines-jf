"""pr05_stageone_ieeg_biomarker.py
"""

import sys
from multiprocessing import Pool
from glob import glob
import presidio_pipelines as prespipe

import numpy as np
import pandas as pd
import scipy.stats as sp_stats

import matplotlib
matplotlib.use('wxAgg')
import matplotlib.pyplot as plt
import seaborn as sns


def _helper(minput):
    h5_power_feats = prespipe.ieeg.modules.apply_reader(minput[0], prespipe.ieeg.stage_one_wavelettransform.HDF5WaveletData)
    h5_kernel_info = h5_power_feats['morlet_kernel_data'].axes[0]['kernel_axis'][...]
    h5_power_data = h5_power_feats['spectrogram_data']


if __name__ == '__main__':

    df_sxsurv = pd.read_csv(sys.argv[2])
    df_sxsurv['survey_start'] = pd.to_datetime(df_sxsurv['survey_start'])
    print(df_sxsurv.columns)

    #
    for score_type in ['vas_d']:
        plt.figure(figsize=(9,6))
        ax = plt.subplot(2,1,1)
        ax = sns.histplot(x=score_type, data=df_sxsurv, ax=ax)
        ax.set_xlabel('{} Score'.format(score_type))
        ax.set_ylabel('Survey Counts')
        ax.set_title('Distribution of Symptoms Scores During Biomarker Recordings')

        #
        rv, pv = sp_stats.pearsonr(df_sxsurv[score_type], (df_sxsurv['survey_start']-df_sxsurv['survey_start'].iloc[0]).dt.total_seconds())
        ax = plt.subplot(2,1,2)
        ax = sns.scatterplot(x='survey_start', y=score_type, data=df_sxsurv, ax=ax)
        ax = sns.lineplot(x='survey_start', y=score_type, data=df_sxsurv.set_index('survey_start').resample('6h').mean().rolling('6h').mean().reset_index(), ax=ax)
        ax.set_ylim([0, 100])
        ax.set_xlabel('Date')
        ax.set_ylabel('Survey Score')
        ax.set_title('r={:0.2f}    |    p={:0.3e}'.format(rv, pv))
        plt.setp( ax.xaxis.get_majorticklabels(), rotation=60)
        ax.grid(True, linestyle='--', color=[0.75, 0.75, 0.75])
        plt.tight_layout()
        plt.show()


    fig = plt.figure(figsize=(20,4))
    for i, score_pair in enumerate([((df_sxsurv['vas_d'], df_sxsurv['vas_a']), ('VAS-Depression', 'VAS-Anxiety')),
                                    ((df_sxsurv['vas_d'], 100-df_sxsurv['vas_e']), ('VAS-Depression', 'VAS-LowEnergy')),
                                    ((df_sxsurv['vas_d'], df_sxsurv['vas_fogg']), ('VAS-Depression', 'VAS-Fogginess')),
                                    ((df_sxsurv['vas_d'], df_sxsurv['vas_irr']), ('VAS-Depression', 'VAS-Irritability')),
                                    ((df_sxsurv['vas_d'], df_sxsurv['vas_app']), ('VAS-Depression', 'VAS-Appetite'))]):

        nan_ix = np.isnan(score_pair[0][0]) | np.isnan(score_pair[0][1])

        ax = plt.subplot(1,5,i+1)
        ax.scatter(score_pair[0][0][~nan_ix], score_pair[0][1][~nan_ix])
        ax.set_xlabel(score_pair[1][0])
        ax.set_ylabel(score_pair[1][1])
        ax.set_xlim([0, 100])
        ax.set_ylim([0, 100])
        rv, pv = sp_stats.pearsonr(score_pair[0][0][~nan_ix], score_pair[0][1][~nan_ix])
        ax.text(70, 80, 'r={:0.2f}\np={:0.3e}'.format(rv,pv))
    plt.tight_layout()
    plt.show()

    #
    h5_input_dir = sys.argv[1]
    h5_files = sorted(glob(h5_input_dir))
    print(h5_files)
    h5_powers = []
    for h5_f in h5_files:
        temp_data = prespipe.ieeg.modules.apply_reader(h5_f, prespipe.ieeg.stage_one_wavelettransform.HDF5WaveletData)
        h5_powers.append(temp_data["spectrogram_data"][...])
    h5_powers = np.array(h5_powers).astype(float)
    print(h5_powers.shape)
    
    #
    corrs = np.nan*np.zeros((h5_powers.shape[1], h5_powers.shape[-1]))
    pvals = np.nan*np.zeros((h5_powers.shape[1], h5_powers.shape[-1]))
    for fq in range(corrs.shape[0]):
        for ch in range(corrs.shape[1]):
            corrs[fq, ch], pvals[fq, ch] = sp_stats.pearsonr(h5_powers[:,fq,0,ch], df_sxsurv['vas_d'])
    df_corrs = pd.DataFrame(corrs, index=temp_data['morlet_kernel_data'].axes[0]['kernel_axis']['CFreq'], columns=temp_data['spectrogram_data'].axes[2]['channellabel_axis'][:corrs.shape[1],0]).T
    df_corrs.columns = np.log10(df_corrs.columns)

    df_pvals = pd.DataFrame(pvals, index=temp_data['morlet_kernel_data'].axes[0]['kernel_axis']['CFreq'], columns=temp_data['spectrogram_data'].axes[2]['channellabel_axis'][:corrs.shape[1],0]).T
    df_pvals.columns = np.log10(df_pvals.columns)

    df_corrs = df_corrs.iloc[:, ::-1]
    df_pvals = df_pvals.iloc[:, ::-1]

    ch_grp, ch_idx = np.unique(df_corrs.index, return_index=True)
    ch_grp = ch_grp.astype(str)
    ch_all = [str('{}{}'.format(ch[0], ch[1])) for ch in temp_data['spectrogram_data'].axes[2]['channellabel_axis'][:corrs.shape[1]].astype(str)]
    print(ch_grp)

    freq_list = np.array([1, 4, 8, 15, 30, 70, 100, 250])

    plt.figure(figsize=(12,16))
    ax = plt.subplot(111)
    ax = sns.heatmap(df_corrs, vmin=-1, vmax=1, cmap='RdBu_r', cbar_kws={'label': 'PearsonR'}, ax=ax)
    ax.set_xticks(np.log10(freq_list) * 50/df_corrs.columns.max())
    ax.set_xticklabels(freq_list)
    #ax.set_yticks(ch_idx)
    #ax.set_yticklabels(ch_grp)
    ax.set_yticks(np.arange(len(df_corrs.index)))
    ax.set_yticklabels(np.array(ch_all), fontsize=6)
    ax.set_ylabel('Bipolar Channels per Lead')
    ax.set_xlabel('Spectral Frequency (Hz)')
    ax.grid(True, linestyle='--', color=[0.25, 0.25, 0.25])
    plt.tight_layout()
    plt.show()

    df_corrs_thresh = df_corrs.copy()
    df_corrs_thresh[df_pvals > 0.01] = np.nan
    plt.figure(figsize=(9,9))
    ax = plt.subplot(111)
    ax = sns.heatmap(df_corrs_thresh, vmin=-1, vmax=1, cmap='RdBu_r', cbar_kws={'label': 'PearsonR'}, ax=ax)
    ax.set_xticks(np.log10(freq_list) * 50/df_corrs.columns.max())
    ax.set_xticklabels(freq_list)
    ax.set_yticks(ch_idx)
    ax.set_yticklabels(ch_grp)
    ax.set_ylabel('Bipolar Channels per Lead')
    ax.set_xlabel('Spectral Frequency (Hz)')
    ax.grid(True, linestyle='--', color=[0.25, 0.25, 0.25])
    plt.tight_layout()
    plt.show()

    band = [7, 15]
    band_ix = np.flatnonzero((temp_data['morlet_kernel_data'].axes[0]['kernel_axis']['CFreq'] > band[0]) & 
                             (temp_data['morlet_kernel_data'].axes[0]['kernel_axis']['CFreq'] < band[1]))
    print(band_ix)
    print(temp_data['morlet_kernel_data'].axes[0]['kernel_axis']['CFreq'][band_ix])
    for chan in ['vLA', 'vLH', 'vRA', 'vRH']:
        chan_lbl = temp_data['spectrogram_data'].axes[2]['channellabel_axis'][:h5_powers.shape[-1],0].astype(str)
        plt.figure(figsize=(16,9))
        cnt = 0
        for ii, lbl in enumerate(chan_lbl):
            if chan not in lbl:
                continue
            ax = plt.subplot(3,5,cnt+1)
            ax.scatter(np.log10(h5_powers[:, :, :, :][:, band_ix, 0, ii].mean(axis=1)), df_sxsurv['vas_d'], s=5)
            rv, pv = sp_stats.pearsonr(np.log10(h5_powers[:, band_ix, 0, ii].mean(axis=1)), df_sxsurv['vas_d'])
            ax.set_xlabel('log Spectral Power (beta/low-gamma band)')
            ax.set_ylabel('VAS Depression')
            ax.set_ylim([0, 100])
            ax.set_title('{} | {:0.2f} | {:0.3e}'.format(lbl, rv, pv))
            cnt += 1
        plt.tight_layout()
        plt.show()
