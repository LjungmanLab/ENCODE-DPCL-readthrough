# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 10:14:24 2023

@author: abmcs
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import re
import json
from PIL import Image
import copy
import fiftyone as fo
import fiftyone.utils.utils3d as fou3d
import fiftyone.brain as fob
from fiftyone import ViewField as F
import numpy as np
import statistics
from sklearn.decomposition import PCA


def load_gene_list(listpath, nspath, sspath):
    
    with open(listpath) as f:
        lines = f.read().splitlines()
    
    header = ['chr', 'tes_start', 'tes_end', 'gene', 'score', 'strand', 'gene_name', 'gene_id', 'biotype', 'start', 'end', 'ol_chr', 'ol_start', 'ol_end', 'ol_gene', 'ol_score', 'ol_strand', 'ol_biotype', 'distance']
    ns = pd.read_csv(nspath, names=header, sep='\t', engine='python')
    ss = pd.read_csv(sspath, names=header, sep='\t', engine='python')
    ns_sub = ns[['gene', 'distance']].drop_duplicates(keep='first', ignore_index=True)
    ss_sub = ss[['gene', 'distance']].drop_duplicates(keep='first', ignore_index=True)
    closest = ns_sub.merge(ss_sub, on='gene', how='inner', suffixes=['_ns', '_ss'])
    gene_df = closest[closest['gene'].isin(lines)].copy().drop_duplicates(subset=['gene'], keep='first', ignore_index=True)
    
    return gene_df


def load_file(filepath, genes):
    
    header = ['chr', 'start', 'end', 'bin_id', 'score', 'strand', 'class', 'original_seg', 'seg_diff', 'class_match', 'gene', 'gene_coord', 'gene_type', 'bin_len', 'counts']
    df = pd.read_csv(filepath, sep='\t', names = header, engine='python')
    # df_sub = df[df['class'].str.contains('III|IVd|IVe|IVf|,V') == False].copy()
    df_filter = df[df['gene'].isin(genes['gene'])].copy()
    df_filter[['RT_ID', 'position']] = df_filter['bin_id'].str.split('_', n=1, expand=True)
    df_filter[['counts', 'position']] = df_filter[['counts', 'position']].apply(pd.to_numeric)
    df_meta = df_filter[['RT_ID', 'score', 'strand', 'class', 'original_seg', 'seg_diff', 'class_match', 'gene', 'gene_coord', 'gene_type']].drop_duplicates(ignore_index = True)
    df_meta = df_meta.merge(genes, how='left', on='gene')
    
    return df_filter, df_meta


def sum_counts(df, cell):
    
    cell_cols = list(df.columns[df.columns.str.contains(cell)])
    df_sub = df[['chr', 'start', 'end', 'gene', 'GENE_strand', 'biotype', 'featureLength']+cell_cols].copy()
    df_sub[cell+'_sum_counts'] = df[cell_cols].sum(axis=1)
    df_out = df_sub[['chr', 'start', 'end', 'gene', 'GENE_strand', 'biotype', 'featureLength', cell+'_sum_counts']]
    
    return df_out


def mean_rpkm(df, cell):
    
    cell_cols = list(df.columns[df.columns.str.contains(cell)])
    df_sub = df[['chr', 'start', 'end', 'gene', 'GENE_strand', 'biotype', 'featureLength']+cell_cols].copy()
    df_sub[cell+'_mean_rpkm'] = df[cell_cols].mean(axis=1)
    df_out = df_sub[['chr', 'start', 'end', 'gene', 'GENE_strand', 'biotype', 'featureLength', cell+'_mean_rpkm']]
    
    return df_out


def load_rpkm(filepath, cell_list):
    
    '''
    Description
    -----------
    Creates merged replicate rpkm dataframe
    
    Parameters
    ----------
    cell_list : list of cell lines to loop through

    Returns
    -------
    final_rpkm_0h : df containing merged replicate (mean) rpkm for all cell lines from Bru-seq data
    '''
    
    rpkm = pd.read_csv(filepath, sep='\t', engine='python')
    rpkm.rename(columns={'geneName/geneID': 'gene'}, inplace=True)
    
    merge_rpkm = pd.DataFrame(columns=['chr', 'start', 'end', 'gene', 'GENE_strand', 'biotype', 'featureLength'])
    for cell in tqdm(cell_list, desc='Create merged count/rpkm files'):
        # merge sample columns for cell line (sum counts, mean rpkm)
        r_sub = mean_rpkm(rpkm, cell)
        # concat dfs to get final version
        merge_rpkm = merge_rpkm.merge(r_sub, how='outer')
    # remove the weird PAR_Y ensembl_ids to prevent duplicates (these do not have counts anyway)
    final_rpkm = merge_rpkm[~merge_rpkm['gene'].str.contains('_PAR_Y')].copy()
        
    return final_rpkm


def create_counts_dict(df, bins=None):
    
    id_set = set(df['RT_ID'])
    
    ct_dict = {}
    for seg in tqdm(id_set, desc='Building count_dict'):
        df_sub = df.loc[df['RT_ID'] == seg]
        # remove segments that are less than 250 bp (i.e. have no main seg counts)
        if not any(df_sub['position'] == 0):
            continue
        # remove duplicate entries (occur if 2 genes share a TES, this occurs a handful of times)
        if any(df_sub['bin_id'].duplicated()):
            continue
        # remove entries with fewer bins than fixed region
        if bins is not None and len(df_sub) < bins:
            continue
        df_sorted = df_sub.sort_values(by = ['position'])
        ct_list = list(df_sorted['counts'])
        ct_dict[seg] = ct_list
        
    return ct_dict


def scale_binned_counts(count_dict):
    frac_dict = {}
    for seg, count_list in count_dict.items():
        gene_counts = count_list[0:4]
        # gene_counts = count_list[0:8]
        gene_max = max(gene_counts)
        if gene_max == 0:
            continue
        frac_list = []
        for count in count_list[4:]:
            frac = count / gene_max
            frac_list.append(frac)
        frac_dict[seg] = frac_list
    
    return frac_dict


def find_frac_signal(count_dict, metadata, fractions):
    
    # create new metadata columns
    metadata['less_than_gene'] = np.nan
    metadata['all_greater_than_gene'] = np.nan
    metadata['most_greater_than_gene'] = np.nan
    metadata['nbins'] = np.nan
    for frac in fractions:
        metadata['nbins_'+str(frac)] = np.nan
        metadata['first_bin_'+str(frac)] = np.nan
        metadata['last_bin_'+str(frac)] = np.nan
        metadata['test_bin_'+str(frac)] = np.nan
        metadata['bin_frac_list_'+str(frac)] = np.nan
    # get column values
    for seg, count_list in tqdm(count_dict.items(), desc='Updating metadata columns with fractions'):
        if all(x <= 1 for x in count_list):
            sig_check = True
        else:
            sig_check = False
        if all(x > 1 for x in count_list):
            sig_check2 = True
        else:
            sig_check2 = False
        if (sum(x > 1 for x in count_list) / len(count_list)) > 0.5:
            sig_check3 = True
        else:
            sig_check3 = False
        # add this info to metadata df
        metadata.loc[metadata['RT_ID'] == seg, 'nbins'] = len(count_list)
        metadata.loc[metadata['RT_ID'] == seg, 'less_than_gene'] = sig_check
        metadata.loc[metadata['RT_ID'] == seg, 'all_greater_than_gene'] = sig_check2
        metadata.loc[metadata['RT_ID'] == seg, 'most_greater_than_gene'] = sig_check3
        for frac in fractions:
            first_bin = None
            last_bin = None
            test_bin = None
            marker = None
            counter = 0
            bin_list = []
            for idx, value in enumerate(count_list): 
                if value <= frac:
                    bin_list.append(idx)
                    counter += 1
                    if counter == 4 and not test_bin:
                        if not last_bin:
                            test_bin = first_bin
                        else:
                            test_bin = last_bin
                    if marker is None:
                        if first_bin is None:
                            first_bin = idx
                            marker = idx
                        else:
                            last_bin = idx
                            marker = idx
                    else:
                        continue
                if value > frac:
                    counter = 0
                    if marker is not None:
                        marker = None
            if not last_bin:
                last_bin = first_bin
            if not test_bin:
                test_bin = last_bin
            # add this info to metadata df
            metadata.loc[metadata['RT_ID'] == seg, 'nbins_'+str(frac)] = len(bin_list)
            metadata.loc[metadata['RT_ID'] == seg, 'first_bin_'+str(frac)] = first_bin
            metadata.loc[metadata['RT_ID'] == seg, 'last_bin_'+str(frac)] = last_bin
            metadata.loc[metadata['RT_ID'] == seg, 'test_bin_'+str(frac)] = test_bin
            metadata.loc[metadata['RT_ID'] == seg, 'bin_frac_list_'+str(frac)] = str(bin_list)
    
    return metadata


def mean_subtraction(frac_dict, subset, dataset_dict):
    
    frac_dict_sub = {k: copy.deepcopy(frac_dict[k]) for k in subset if k in frac_dict}
    sig_arr = np.array(list(frac_dict_sub.values()))
    mean_sub_sig = sig_arr - sig_arr.mean(axis=0)
    
    # add mean sub counts to datasets
    mean_sub_dict = dict(zip(frac_dict_sub.keys(), mean_sub_sig.tolist()))
    mean_sub_dict1 = {k: {'mean_sub_counts': v} for k, v in mean_sub_dict.items()}
    {k: v.update(mean_sub_dict1[k]) for k, v in dataset_dict.items() if k in mean_sub_dict1}
        
    return dataset_dict


def median_subtraction(frac_dict, subset, dataset_dict):
    
    frac_dict_sub = {k: copy.deepcopy(frac_dict[k]) for k in subset if k in frac_dict}
    sig_arr = np.array(list(frac_dict_sub.values()))
    median_sub_sig = sig_arr - np.median(sig_arr, axis=0)
    
    # add mean sub counts to datasets
    median_sub_dict = dict(zip(frac_dict_sub.keys(), median_sub_sig.tolist()))
    median_sub_dict1 = {k: {'median_sub_counts': v} for k, v in median_sub_dict.items()}
    {k: v.update(median_sub_dict1[k]) for k, v in dataset_dict.items() if k in median_sub_dict1}
        
    return dataset_dict


def mean_med_subtraction(frac_dict, subset, dataset_dict):
    
    frac_dict_sub = {k: copy.deepcopy(frac_dict[k]) for k in subset if k in frac_dict}
    sig_arr = np.array(list(frac_dict_sub.values()))
    mean_sub_sig = sig_arr - sig_arr.mean(axis=0)
    median_sub_sig = sig_arr - np.median(sig_arr, axis=0)
    
    # add mean and median sub counts to datasets
    mean_sub_dict = dict(zip(frac_dict_sub.keys(), mean_sub_sig.tolist()))
    mean_sub_dict1 = {k: {'mean_sub_counts': v} for k, v in mean_sub_dict.items()}
    {k: v.update(mean_sub_dict1[k]) for k, v in dataset_dict.items() if k in mean_sub_dict1}
    
    median_sub_dict = dict(zip(frac_dict_sub.keys(), median_sub_sig.tolist()))
    median_sub_dict1 = {k: {'median_sub_counts': v} for k, v in median_sub_dict.items()}
    {k: v.update(median_sub_dict1[k]) for k, v in dataset_dict.items() if k in median_sub_dict1}
    
    return dataset_dict


def output_pcd_files(dataset_dict, cellline, filepath):
    
    os.makedirs(filepath, exist_ok = True)
    
    for key, value in tqdm(dataset_dict.items(), desc='Output .pcd files'):
        # if len(value) < 3:
        #     continue
        pcd_filename = filepath + cellline + '_' + key.replace(':', '_') + '.pcd'
        region_counts = [[i-4, value['region_counts'][i]] for i in range(0, len(value['region_counts']))]
        scaled_counts = [[i, value['scaled_counts'][i]] for i in range(0, len(value['scaled_counts']))]
        segment_counts = [[i-4, value['segment_counts'][i]] for i in range(0, len(value['segment_counts']))]
        if 'median_sub_counts' in value.keys():
            mean_sub_counts = [[i, value['mean_sub_counts'][i]] for i in range(0, len(value['mean_sub_counts']))]
            median_sub_counts = [[i, value['median_sub_counts'][i]] for i in range(0, len(value['median_sub_counts']))]
            pcd_dict = {'scaled_counts': {'data': scaled_counts, 'legend_selected': True, 'yAxis_name': 'Scaled_binned_counts'},
                        'median_sub_counts': {'data': median_sub_counts, 'legend_selected': True, 'yAxis_name': 'Scaled_binned_counts'},
                        'mean_sub_counts': {'data': mean_sub_counts, 'legend_selected': False, 'yAxis_name': 'Scaled_binned_counts'},
                        'region_counts': {'data': region_counts, 'legend_selected': False, 'yAxis_name': 'Binned_counts'},
                        'segment_counts': {'data': segment_counts, 'legend_selected': False, 'yAxis_name': 'Binned_counts'}}
        else:
            pcd_dict = {'scaled_counts': {'data': scaled_counts, 'legend_selected': True, 'yAxis_name': 'Scaled_binned_counts'},
                        'region_counts': {'data': region_counts, 'legend_selected': False, 'yAxis_name': 'Binned_counts'},
                        'segment_counts': {'data': segment_counts, 'legend_selected': False, 'yAxis_name': 'Binned_counts'}}
        # convert dict to json
        with open(pcd_filename, "w") as outfile: 
            json.dump(pcd_dict, outfile)
    
    return


def minimal_class_list(rt_class):
    
    I = False
    II = False
    III = False
    IV = False
    if re.search(r'\b(?:Ia|Ib)\b', rt_class):
        I = True
    if 'II' in rt_class:
        II = True
    if 'III' in rt_class:
        III = True
    if 'IV' in rt_class:
        IV = True
    
    classes = np.array(['I', 'II', 'III', 'IV'])
    fil = np.array([I, II, III, IV])
    rt_class_simple = ','.join(list(classes[fil]))
    
    return rt_class_simple


def points_to_polyline(points, h, w, max_h, min_h, max_w, min_w):
    poly_points = []
    num_points = len(points)-1
    x_step = (max_w - min_w) / (num_points)
    for i, p in enumerate(points):
        x = (i * x_step) + min_w
        y = 1 - ((p * (max_h - min_h)) + min_h)
        poly_points.append((x,y))
        
    return fo.Polyline(points=[poly_points], closed=False, filled=False)


def counts_to_polylines(dataset, points_field, poly_field):
    dataset.compute_metadata()
    polylines = []
    max_h = 249/288.
    min_h = 34/288.
    max_w = 389/432.
    min_w = 56/432.

    for points in tqdm(dataset.values(points_field), desc='Counts to polylines'):
        h = 288
        w = 432
        if not points:
            polylines.append(None)
            continue
        polyline = points_to_polyline(points, h, w, max_h, min_h, max_w, min_w)
        polylines.append(polyline)
    dataset.set_values(poly_field, polylines)


def create_fo_dataset(dataset, pcd_dir, dataset_dict, updated_metadata, rpkm, cellline, region_length, thumbnail_info):
    
    ### NOTES: try and get plots y axes to line up at zero and move legend
    
    samples = []
    for filename in tqdm(os.listdir(pcd_dir)):
        sample = fo.Sample(filepath=os.path.join(pcd_dir, filename))
        seg = filename.split('_', maxsplit=1)[1].split('.')[0].replace('_', ':')
        # get data
        count_dict = dataset_dict[seg]
        row = updated_metadata.loc[updated_metadata['RT_ID'] == seg]
        # add base fields
        sample['orthographic_projection_metadata'] = thumbnail_info
        sample['RT_ID'] = seg
        sample['cell_line'] = cellline
        sample['fixed_region_length'] = region_length
        sample['fixed_region_bins'] = len(count_dict['region_counts'])
        seg_coord_list = re.split('-|:', seg)
        sample['segment_length'] = int(seg_coord_list[2]) - int(seg_coord_list[1])
        sample['log_segment_length'] = np.log10(sample['segment_length']+0.01)
        sample['segment_bins'] = len(count_dict['segment_counts'])
        # add counts fields
        sample['region_counts'] = count_dict['region_counts']
        sample['scaled_counts'] = count_dict['scaled_counts']
        sample['segment_counts'] = count_dict['segment_counts']
        if 'median_sub_counts' in count_dict.keys():
            sample['median_sub_counts'] = count_dict['median_sub_counts']
        if 'mean_sub_counts' in count_dict.keys():
            sample['mean_sub_counts'] = count_dict['mean_sub_counts']
        # add metadata fields
        sample['gene'] = row['gene'].values[0]
        sample['strand'] = row['strand'].values[0]
        gene_coord_list = re.split('-|:', row['gene_coord'].values[0])
        sample['gene_length'] = int(gene_coord_list[2]) - int(gene_coord_list[1])
        sample['log_gene_length'] = np.log10(sample['gene_length']+0.01)
        sample['gene_biotype'] = row['gene_type'].values[0]
        sample['gene_rpkm'] = rpkm.loc[rpkm['gene'] == row['gene'].values[0], cellline+'_mean_rpkm'].values[0]
        sample['log_gene_rpkm'] = np.log10(sample['gene_rpkm']+0.01)
        sample['RT_seg_class_string'] = row['class'].values[0]
        sample['RT_seg_class'] = sample['RT_seg_class_string'].split(',')
        sample['RT_seg_class_minimal'] = minimal_class_list(sample['RT_seg_class_string'])
        sample['original_segment'] = row['original_seg'].values[0]
        segdiff = row['seg_diff'].values[0]
        sample['segment_len_diff'] = None if np.isnan(float(segdiff)) else float(segdiff)
        sample['RT_seg_classes_match'] = str(row['class_match'].values[0])
        sample['nearest_ds_gene_ss'] = row['distance_ss'].values[0]
        sample['nearest_ds_gene_ns'] = row['distance_ns'].values[0]
        sample['sig_less_than_gene'] = str(row['less_than_gene'].values[0])
        sample['sig_greater_than_gene'] = str(row['all_greater_than_gene'].values[0])
        sample['most_greater_than_gene'] = str(row['most_greater_than_gene'].values[0])
        sample['nbins_50'] = row['nbins_0.5'].values[0]
        sample['end_bin_50'] = row['test_bin_0.5'].values[0]
        sample['first_bin_50'] = row['first_bin_0.5'].values[0]
        sample['last_bin_50'] = row['last_bin_0.5'].values[0]
        sample['nbins_10'] = row['nbins_0.1'].values[0]
        sample['end_bin_10'] = row['test_bin_0.1'].values[0]
        sample['first_bin_10'] = row['first_bin_0.1'].values[0]
        sample['last_bin_10'] = row['last_bin_0.1'].values[0]
        sample['nbins_1'] = row['nbins_0.01'].values[0]
        sample['end_bin_1'] = row['test_bin_0.01'].values[0]
        sample['first_bin_1'] = row['first_bin_0.01'].values[0]
        sample['last_bin_1'] = row['last_bin_0.01'].values[0]
        sample['nbins_0'] = row['nbins_0'].values[0]
        sample['end_bin_0'] = row['test_bin_0'].values[0]
        sample['first_bin_0'] = row['first_bin_0'].values[0]
        sample['last_bin_0'] = row['last_bin_0'].values[0]
        # add sample to dataset
        samples.append(sample)
    
    dataset.add_samples(samples)
        
    return


def load_dataset(dataset_name):
    
    if dataset_name not in fo.list_datasets():
        dataset = fo.Dataset()
        dataset.persistent = True
        dataset.name = dataset_name
    else:
        dataset = fo.load_dataset(dataset_name)
    
    return dataset


def create_datasets(cell_list, suffix, filepath, genes, rpkm):
    
    dataset_names = fo.list_datasets()
        
    if len(dataset_names) == 5:
        # load datasets and metadata
        dataset0 = load_dataset('readthrough_signals')
        dataset1 = load_dataset('readthrough_signals_sub1_ss')
        dataset2 = load_dataset('readthrough_signals_sub2_ns')
        dataset1s = load_dataset('readthrough_signals_sub1_ss_ltg')
        dataset2s = load_dataset('readthrough_signals_sub2_ns_ltg')
        metadata_dict = {}
        for cellline in cell_list:
            print(cellline)
            # load in metadata files
            m_filepath = 'C:/Users/abmcs/Desktop/shared_biosamples/read-through/fused_dec/final_output/binning_test/' + cellline + '.' + suffix.rsplit('.', maxsplit=3)[1] + '.metadata.txt'
            updated_metadata = pd.read_csv(m_filepath, sep='\t', engine='python')
            metadata_dict[cellline] = updated_metadata
    else:
        # create datasets
        image_filepath = filepath+'scaled_axes.png'
        if not os.path.exists(image_filepath):
            plt.plot()
            plt.xlim(left=0, right=69)
            plt.ylim(bottom=0, top=1)
            plt.savefig(image_filepath)
            im = Image.open(image_filepath)
            width, height = im.size
        else:
            im = Image.open(image_filepath)
            width, height = im.size
    
        thumbnail = fou3d.OrthographicProjectionMetadata(min_bound=None, max_bound=None, width=width, height=height, filepath = os.path.abspath(image_filepath))
        # thumbnail.filepath = os.path.abspath(image_filepath)
            
        metadata_dict = {}
        
        for cellline in cell_list:
            
            print(cellline)
            
            if cellline in fo.list_datasets():
                continue
            
            file = cellline + suffix
            
            region_length = int(suffix.rsplit('.', maxsplit=3)[1].rsplit('_', maxsplit=1)[1])
            
            df, metadata = load_file(filepath+file, genes)
            
            # create or load fixed region counts dict
            npy_filepath = 'C:/Users/abmcs/Desktop/shared_biosamples/read-through/fused_dec/final_output/binning_test/' + cellline + suffix.rsplit('.', maxsplit=1)[0] + '_dict.npy'
            if not os.path.exists(npy_filepath):
                bins = np.floor(region_length / 250) + 4
                count_dict = create_counts_dict(df, bins)
                np.save(npy_filepath, np.array(count_dict))
            else:
                count_dict = np.load(npy_filepath, allow_pickle=True).tolist()
            
            # create or load segment counts dict
            npy_filepath_seg = 'C:/Users/abmcs/Desktop/shared_biosamples/read-through/fused_dec/final_output/binning_test/' + cellline + '.RTsegments.merged.bin_250.sum_counts_dict.npy'
            if not os.path.exists(npy_filepath_seg):
                file_seg = cellline + '.RTsegments.merged.bin_250.sum_counts.bed'
                df_seg, metadata_seg = load_file(filepath+file_seg, genes)
                count_dict_seg = create_counts_dict(df_seg)
                np.save(npy_filepath_seg, np.array(count_dict_seg))
            else:
                count_dict_seg = np.load(npy_filepath_seg, allow_pickle=True).tolist()
            
            # scale counts by max gene signal
            npy_filepath_scaled = 'C:/Users/abmcs/Desktop/shared_biosamples/read-through/fused_dec/final_output/binning_test/' + cellline + suffix.rsplit('.', maxsplit=2)[0] + '.scaled_counts_dict.npy'
            if not os.path.exists(npy_filepath_scaled):
                frac_dict = scale_binned_counts(count_dict)
                np.save(npy_filepath_scaled, np.array(frac_dict))
            else:
                frac_dict = np.load(npy_filepath_scaled, allow_pickle=True).tolist()
            
            # find bins with 50%, 10%, 1%, 0% signal (first and last) and add to metadata_df
            # also add col to describe if any RT signal is greater than gene sig (>1)
            m_filepath = 'C:/Users/abmcs/Desktop/shared_biosamples/read-through/fused_dec/final_output/binning_test/' + cellline + '.' + suffix.rsplit('.', maxsplit=3)[1] + '.metadata.txt'
            if not os.path.exists(m_filepath):
                frac_list = [0.50, 0.10, 0.01, 0]
                updated_metadata = find_frac_signal(frac_dict, metadata.copy(), frac_list)
                updated_metadata.to_csv(m_filepath, sep='\t', header=True, index=False)
            else:
                updated_metadata = pd.read_csv(m_filepath, sep='\t', engine='python')
            
            metadata_dict[cellline] = updated_metadata
            
            # combine these counts files into nested dictionary
            dataset_dict0 = {}
            for key in set(list(count_dict.keys()) + list(count_dict_seg.keys())):
                try:
                    dataset_dict0.setdefault(key,{}).setdefault('region_counts',count_dict[key])
                except KeyError:
                    pass
                
                try:
                    dataset_dict0.setdefault(key,{}).setdefault('scaled_counts',frac_dict[key])          
                except KeyError:
                    pass
                
                try:
                    dataset_dict0.setdefault(key,{}).setdefault('segment_counts',count_dict_seg[key])          
                except KeyError:
                    pass
                
                # if key does not have all counts values, remove key
                if len(dataset_dict0[key]) < 3:
                    del dataset_dict0[key]
                
            # subset dictionary into 3 dataset versions
            subset1 = list(updated_metadata.loc[(updated_metadata['distance_ss'] > region_length), 'RT_ID'])
            subset2 = list(updated_metadata.loc[(updated_metadata['distance_ns'] > region_length), 'RT_ID'])
            subset1s = list(updated_metadata.loc[(updated_metadata['distance_ss'] > region_length) & (updated_metadata['less_than_gene'] == True), 'RT_ID'])
            subset2s = list(updated_metadata.loc[(updated_metadata['distance_ns'] > region_length) & (updated_metadata['less_than_gene'] == True), 'RT_ID'])
            
            dataset_dict1 = {k: copy.deepcopy(dataset_dict0[k]) for k in subset1 if k in dataset_dict0}
            dataset_dict2 = {k: copy.deepcopy(dataset_dict0[k]) for k in subset2 if k in dataset_dict0}
            dataset_dict1s = {k: copy.deepcopy(dataset_dict0[k]) for k in subset1s if k in dataset_dict0}
            dataset_dict2s = {k: copy.deepcopy(dataset_dict0[k]) for k in subset2s if k in dataset_dict0}
            
            # perform mean and median subtraction of scaled counts for relevant datasets
            dataset_dict1 = mean_med_subtraction(frac_dict, subset1, dataset_dict1)
            dataset_dict2 = mean_med_subtraction(frac_dict, subset2, dataset_dict2)
            dataset_dict1s = mean_med_subtraction(frac_dict, subset1s, dataset_dict1s)
            dataset_dict2s = mean_med_subtraction(frac_dict, subset2s, dataset_dict2s)
            
            # create .pcd (json) files for each segment/gene
            output_pcd_files(dataset_dict0, cellline, filepath+'pcd_all/'+cellline+'/')
            output_pcd_files(dataset_dict1, cellline, filepath+'pcd_ss_subset/'+cellline+'/')
            output_pcd_files(dataset_dict2, cellline, filepath+'pcd_ns_subset/'+cellline+'/')
            output_pcd_files(dataset_dict1s, cellline, filepath+'pcd_ss_subset_ltg/'+cellline+'/')
            output_pcd_files(dataset_dict2s, cellline, filepath+'pcd_ns_subset_ltg/'+cellline+'/')
            
            # create fiftyone dataset from .pcd files
            dataset0 = load_dataset('readthrough_signals')
            dataset1 = load_dataset('readthrough_signals_sub1_ss')
            dataset2 = load_dataset('readthrough_signals_sub2_ns')
            dataset1s = load_dataset('readthrough_signals_sub1_ss_ltg')
            dataset2s = load_dataset('readthrough_signals_sub2_ns_ltg')
            
            create_fo_dataset(dataset0, filepath+'pcd_all/'+cellline+'/', dataset_dict0, updated_metadata, rpkm, cellline, region_length, thumbnail)
            create_fo_dataset(dataset1, filepath+'pcd_ss_subset/'+cellline+'/', dataset_dict1, updated_metadata, rpkm, cellline, region_length, thumbnail)
            create_fo_dataset(dataset2, filepath+'pcd_ns_subset/'+cellline+'/', dataset_dict2, updated_metadata, rpkm, cellline, region_length, thumbnail)
            create_fo_dataset(dataset1s, filepath+'pcd_ss_subset_ltg/'+cellline+'/', dataset_dict1s, updated_metadata, rpkm, cellline, region_length, thumbnail)
            create_fo_dataset(dataset2s, filepath+'pcd_ns_subset_ltg/'+cellline+'/', dataset_dict2s, updated_metadata, rpkm, cellline, region_length, thumbnail)
            
        # plot polylines for scaled y-axis
        for dataset in [dataset0, dataset1, dataset2, dataset1s, dataset2s]:
            if 'scaled_polylines' not in dataset.get_field_schema():
                counts_to_polylines(dataset, 'scaled_counts', 'scaled_polylines')
        
    return dataset0, dataset1, dataset2, dataset1s, dataset2s, metadata_dict


def mean_plot(arr, dataset, plots_dir, string, ylab, classI = False):
    
    plt.plot(arr.mean(axis=0))
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.xlabel('nBins (250bp)')
    plt.ylabel(ylab)
    if classI:
        plt.title('Mean ' + string + ' signal (class I; n=' + str(len(arr)) + ')')
        plt.savefig(plots_dir + dataset.name + '.mean_' + string + '_signal_classI' + '.png')
    else:
        plt.title('Mean ' + string + ' signal (n=' + str(len(arr)) + ')')
        plt.savefig(plots_dir + dataset.name + '.mean_' + string + '_signal' + '.png')
    plt.show()


def cell_plot(msd, dataset, plots_dir, string, ylab, classI = False):
    
    for cellline, arr in msd.items():
        plt.plot(arr.mean(axis=0), label=cellline)
        plt.xlim(left=0)
        plt.ylim(bottom=0)
        plt.xlabel('nBins (250bp)')
        plt.ylabel(ylab)
        if classI:
            plt.title('Mean ' + string + ' signal (class I): ' + cellline)
            plt.savefig(plots_dir + dataset.name + '.mean_' + string + '_signal_classI.' + cellline + '.png')
        else:
            plt.title('Mean ' + string + ' signal: ' + cellline)
            plt.savefig(plots_dir + dataset.name + '.mean_' + string + '_signal.' + cellline + '.png')
        plt.show()


def combo_plot(msd, mean_arr, dataset, plots_dir, string, xlimit, ylimit, ylab, classI = False):
    
    for cellline, arr in msd.items():
        plt.plot(arr.mean(axis=0), label=cellline)
        plt.xlim(left=0, right=xlimit)
        plt.ylim(bottom=0, top=ylimit)
    plt.plot(mean_arr.mean(axis=0), ls='--', color='black', label='All')
    plt.xlabel('nBins (250bp)')
    plt.ylabel(ylab)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    if classI:
        plt.title('Mean ' + string + ' signal for cell lines (class I)')
        plt.savefig(plots_dir + dataset.name + '.mean_' + string + '_signal_classI.allCL.png', bbox_inches = 'tight')
    else:
        plt.title('Mean ' + string + ' signal for cell lines')
        plt.savefig(plots_dir + dataset.name + '.mean_' + string + '_signal.allCL.png', bbox_inches = 'tight')
    plt.show()


def combo_plot2(msd, med_arr, dataset, plots_dir, string, xlimit, ylimit, ylab, classI = False):
    
    for cellline, arr in msd.items():
        plt.plot(np.median(arr, axis=0), label=cellline)
        plt.xlim(left=0, right=xlimit)
        plt.ylim(bottom=0, top=ylimit)
    plt.plot(np.median(med_arr, axis=0), ls='--', color='black', label='All')
    plt.xlabel('nBins (250bp)')
    plt.ylabel(ylab)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    if classI:
        plt.title('Median ' + string + ' signal for cell lines (class I)')
        plt.savefig(plots_dir + dataset.name + '.median_' + string + '_signal_classI.allCL.png', bbox_inches = 'tight')
    else:
        plt.title('Median ' + string + ' signal for cell lines')
        plt.savefig(plots_dir + dataset.name + '.median_' + string + '_signal.allCL.png', bbox_inches = 'tight')
    plt.show()


def median_plot(arr, dataset, plots_dir, string, ylab, classI = False):
    
    plt.plot(np.median(arr, axis=0))
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.xlabel('nBins (250bp)')
    plt.ylabel(ylab)
    if classI:
        plt.title('Median ' + string + ' signal (class I; n=' + str(len(arr)) + ')')
        plt.savefig(plots_dir + dataset.name + '.median_' + string + '_signal_classI' + '.png')
    else:
        plt.title('Median ' + string + ' signal (n=' + str(len(arr)) + ')')
        plt.savefig(plots_dir + dataset.name + '.median_' + string + '_signal' + '.png')
    plt.show()


def create_mean_sig_dict(dataset, value, cell_list, classI = False):
    
    mean_sig_dict = {}
    for cellline in tqdm(cell_list):
        if classI:
            arr = np.array(dataset.match((F("cell_line")==cellline) & ((F('RT_seg_class_string') == 'Ia') | (F('RT_seg_class_string') == 'Ib'))).values(value))
        else:
            arr = np.array(dataset.match(F("cell_line")==cellline).values(value))
        mean_sig_dict[cellline] = arr
    
    return mean_sig_dict


def plot_aggregate_signals(dataset, cell_list, plots_dir):
    
    os.makedirs(plots_dir, exist_ok = True)
    
    # get signal arrays
    print('Creating signal arrays...')
    arr_all = np.array(dataset.values('scaled_counts'))
    arr_all_reg = np.array(dataset.values('region_counts'))
    # arr_all_seg = np.array(dataset.values('segment_counts'))
    arr_all_c1 = np.array(dataset.match((F('RT_seg_class_string') == 'Ia') | (F('RT_seg_class_string') == 'Ib')).values('scaled_counts'))
    arr_all_reg_c1 = np.array(dataset.match((F('RT_seg_class_string') == 'Ia') | (F('RT_seg_class_string') == 'Ib')).values('region_counts'))
    # arr_all_seg_c1 = np.array(dataset.match((F('RT_seg_class_string') == 'Ia') | (F('RT_seg_class_string') == 'Ib')).values('segment_counts'))
    
    # # plot mean signal
    # print('Plotting mean signal...')
    # mean_plot(arr_all, dataset, plots_dir, 'scaled', 'Mean scaled counts')
    # mean_plot(arr_all_reg, dataset, plots_dir, 'region', 'Mean binned counts')
    # # mean_plot(arr_all_seg, dataset, plots_dir, 'segment', 'Mean binned counts')
    # mean_plot(arr_all, dataset, plots_dir, 'scaled', 'Mean scaled counts', classI=True)
    # mean_plot(arr_all_reg_c1, dataset, plots_dir, 'region', 'Mean binned counts', classI=True)
    # # mean_plot(arr_all_seg_c1, dataset, plots_dir, 'segment', 'Mean binned counts', classI=True)
    
    # # plot median signal
    # print('Plotting median signal...')
    # median_plot(arr_all, dataset, plots_dir, 'scaled', 'Median scaled counts')
    # median_plot(arr_all_reg, dataset, plots_dir, 'region', 'Median binned counts')
    # median_plot(arr_all, dataset, plots_dir, 'scaled', 'Median scaled counts', classI=True)
    # median_plot(arr_all_reg_c1, dataset, plots_dir, 'region', 'Median binned counts', classI=True)
    
    # create cell line mean signal dicts
    print('Creating mean signal dictionaries per cell line...')
    msd = create_mean_sig_dict(dataset, 'scaled_counts', cell_list)
    msd_reg = create_mean_sig_dict(dataset, 'region_counts', cell_list)
    # msd_seg = create_mean_sig_dict(dataset, 'segment_counts', cell_list)
    msd_c1 = create_mean_sig_dict(dataset, 'scaled_counts', cell_list, classI=True)
    msd_reg_c1 = create_mean_sig_dict(dataset, 'region_counts', cell_list, classI=True)
    # msd_seg_c1 = create_mean_sig_dict(dataset, 'segment_counts', cell_list, classI=True)
    
    # # plot mean signal for cell lines individually
    # print('Plotting mean cell line signal...')
    # cell_plot(msd, dataset, plots_dir, 'scaled', 'Mean scaled counts')
    # cell_plot(msd_reg, dataset, plots_dir, 'region', 'Mean binned counts')
    # # cell_plot(msd_seg, dataset, plots_dir, 'segment', 'Mean binned counts')
    # cell_plot(msd_c1, dataset, plots_dir, 'scaled', 'Mean scaled counts', classI=True)
    # cell_plot(msd_reg_c1, dataset, plots_dir, 'region', 'Mean binned counts', classI=True)
    # # cell_plot(msd_seg_c1, dataset, plots_dir, 'segment', 'Mean binned counts', classI=True)
    
    # plot all cell lines and mean on same plot
    print('Plotting cell line and mean signals...')
    combo_plot(msd, arr_all, dataset, plots_dir, 'scaled', 69, 1, 'Mean scaled counts')
    combo_plot(msd_reg, arr_all_reg, dataset, plots_dir, 'region', 73, 100, 'Mean binned counts')
    # combo_plot(msd_seg, arr_all_seg, dataset, plots_dir, 'segment', 'Mean binned counts')
    combo_plot(msd_c1, arr_all_c1, dataset, plots_dir, 'scaled', 69, 1, 'Mean scaled counts', classI=True)
    combo_plot(msd_reg_c1, arr_all_reg_c1, dataset, plots_dir, 'region', 73, 100, 'Mean binned counts', classI=True)
    # combo_plot(msd_seg_c1, arr_all_seg_c1, dataset, plots_dir, 'segment', 'Mean binned counts', classI=True)
    
    # plot all cell lines and median on same plot
    print('Plotting cell line and median signals...')
    combo_plot2(msd, arr_all, dataset, plots_dir, 'scaled', 69, 1, 'Median scaled counts')
    combo_plot2(msd_reg, arr_all_reg, dataset, plots_dir, 'region', 73, 50, 'Median binned counts')
    combo_plot2(msd_c1, arr_all_c1, dataset, plots_dir, 'scaled', 69, 1, 'Median scaled counts', classI=True)
    combo_plot2(msd_reg_c1, arr_all_reg_c1, dataset, plots_dir, 'region', 73, 50, 'Median binned counts', classI=True)
    
    return


def visualize_embeddings(dataset, points, method, key):
    fob.compute_visualization(dataset.match(~F(points).is_null()), embeddings=points, method=method, brain_key=key)
    # fob.compute_visualization(dataset.values(points), embeddings=points, method=method, brain_key=key)


def bin_residuals(dataset, n_divisions):
    
    resid_arr = np.array(dataset.values("median_sub_counts"))
    bins = np.linspace(0, len(resid_arr[1]), n_divisions)
    digitized = np.digitize(np.arange(len(resid_arr[1])), bins)
    grouped_medians = []
    grouped_std = []
    class_medians = []
    for counts in tqdm(resid_arr):
        medians = [np.median(counts[digitized == i]) for i in range(1, len(bins))]
        std = [statistics.stdev(counts[digitized == i]) for i in range(1, len(bins))]
        grouped_medians.append(medians)
        grouped_std.append(std)
        # create class list
        median_class = ''.join(map(lambda m: '1' if m else '0', np.array(medians) > 0))
        class_medians.append(median_class)
                
    return np.array(grouped_medians), np.array(grouped_std), np.array(class_medians)


def score_residuals(dataset):
    
    resid_arr = np.array(dataset.values("median_sub_counts"))
    # calculate a score based on the binary classification of each bin (+1 if bin > 0)
    resid_score = []
    for counts in tqdm(resid_arr):
        # score regions
        score = sum(1 for m in counts if m > 0 )
        resid_score.append(score)
    
    return np.array(resid_score)


def score_residuals_net(dataset):
    
    resid_arr = np.array(dataset.values("median_sub_counts"))
    # get statistics for distribution of all bin values
    stdev = statistics.stdev(resid_arr.flatten())
    # mean = resid_arr.flatten().mean()
    median = statistics.median(resid_arr.flatten())
    # calculate a score based on distribution (0 if within 1std of median AKA 0, +1 if greater, -1 if less)
    resid_score = []
    for counts in tqdm(resid_arr):
        # score regions
        score = sum(1 for m in counts if m > (median + stdev)) + sum((-1) for m in counts if m < (median - stdev))
        resid_score.append(score)
    
    return np.array(resid_score)


def score_residuals_verbose(dataset, n_divisions):
    
    resid_arr = np.array(dataset.values("median_sub_counts"))
    bins = np.linspace(0, len(resid_arr[1]), n_divisions)
    digitized = np.digitize(np.arange(len(resid_arr[1])), bins)
    # get statistics for distribution of all bin values
    stdev = statistics.stdev(resid_arr.flatten())
    # mean = resid_arr.flatten().mean()
    median = statistics.median(resid_arr.flatten())
    # calculate a score based on distribution (0 if within 1std of median AKA 0, +1 if greater, -1 if less)
    grouped_score = []
    for counts in tqdm(resid_arr):
        # score regions
        resid_score = []
        for i in range(1, len(bins)):
            score = sum(1 for m in counts[digitized == i] if m > (median + stdev)) + sum(-1 for m in counts[digitized == i] if m < (median - stdev))
            resid_score.append(score)
        grouped_score.append(resid_score)
    
    return np.array(grouped_score)


def add_dataset_values_to_df(dataset, metadata_dict, value_list, suffix):
    
    updated_metadata = {}
    for cellline, df in metadata_dict.items():
        print(cellline)
        ds_cl = dataset.match(F('cell_line') == cellline)
        merged_df = df.copy()
        for value in value_list:
            rtids = ds_cl.values('RT_ID')
            col = ds_cl.values(value)
            col_df = pd.DataFrame({'RT_ID': rtids, value: col})
            merged_df = pd.merge(merged_df, col_df, how='left', on='RT_ID')
        updated_metadata[cellline] = merged_df
        # save new metadata file
        m_filepath = 'C:/Users/abmcs/Desktop/shared_biosamples/read-through/fused_dec/final_output/binning_test/' + cellline + '.' + suffix.rsplit('.', maxsplit=3)[1] + '.final_metadata.txt'
        merged_df.to_csv(m_filepath, sep='\t', header=True, index=False)
        
    return updated_metadata


def main():
    
    ### build datasets
    
    # cell type specific datasets
    filepath = <path_to_working_directory>
    listpath = <path_to_gene_list>
    nspath = <path_to_downstream_genes_ns>
    sspath = <path_to_downstream_genes_ss>
    rpkmpath = <path_to_cellline_rpkms_bed>
    file_suffix = '.RTsegments.merged.bin_250.fixed_region_17377.sum_counts.bed'
    
    cell_list = ['A673', 'Caco2', 'Calu3', 'GM12878', 'HCT116', 'HepG2', 'HMEC', 'HUVEC', 'IMR90', 'K562', 'MCF10A', 'MCF7', 'OCILY7', 'panc1', 'PC3', 'PC9']
    
    genes = load_gene_list(listpath, nspath, sspath)
    rpkm = load_rpkm(rpkmpath, cell_list)
    
    ds1, ds2, ds3, ds4, ds5, metadata = create_datasets(cell_list, file_suffix, filepath, genes, rpkm)
    
    # plot aggregate signals
    plots_dir = 'C:/Users/abmcs/Desktop/shared_biosamples/read-through/fused_dec/final_output/binning_test/plots/mean_scaled_signal/'
    for dataset in [ds1, ds2, ds3, ds4, ds5]:
        plot_aggregate_signals(dataset, cell_list, plots_dir)
    
        
    # clustering
    for dataset in [ds2, ds3, ds4, ds5]:
        print('Clustering...')
        print(dataset.name)
        if not dataset.has_brain_run('median_sub_signal_umap'):
            visualize_embeddings(dataset, 'median_sub_counts', 'umap', 'median_sub_signal_umap')
            
        if not dataset.has_brain_run('median_sub_signal_pca'):
            visualize_embeddings(dataset, 'median_sub_counts', 'pca', 'median_sub_signal_pca')
        
        if not dataset.has_brain_run('mean_sub_signal_umap'):
            visualize_embeddings(dataset, 'mean_sub_counts', 'umap', 'mean_sub_signal_umap')
            
        if not dataset.has_brain_run('mean_sub_signal_pca'):
            visualize_embeddings(dataset, 'mean_sub_counts', 'pca', 'mean_sub_signal_pca')
        
        ### ERROR: 'ValueError: Found array with 0 sample(s) (shape=(0, 0)) while a minimum of 1 is required by PCA.'
        # if not dataset.has_brain_run('mean_sub_signal_tsne'):
        #     visualize_embeddings(dataset, 'mean_sub_count', 'tsne', 'mean_sub_signal_tsne')
   
    # save median and mean sub counts per cell line
    resid_arr = np.array(ds4.values("median_sub_counts"))
    # resid_arr2 = np.array(ds4.values("mean_sub_counts"))
    ds4_rt_ids = np.array(ds4.values("RT_ID"))
    ds4_cellline = np.array(ds4.values("cell_line"))
    ds4_gene = np.array(ds4.values("gene"))
    
    resid_df = pd.DataFrame(resid_arr)
    ds4_meta_df = pd.DataFrame({'gene': ds4_gene, 'RT_ID': ds4_rt_ids, 'cellline': ds4_cellline})
    
    median_sub_df = pd.concat([ds4_meta_df, resid_df], axis=1)
    median_sub_df.to_csv(filepath+'RT_segments.merged.bin_250.fixed_region_17377.median_sub_scaled_counts.txt', sep='\t', na_rep='NA', header=True, index=False)
    
    # perform PCA and clustering of residuals
    pca = PCA()
    pca.fit(resid_arr)
    
    singular_values = pca.singular_values_
    print(singular_values/singular_values.sum())
     
    # plot PCs/eigenvectors
    plt.plot(pca.components_[0])
    plt.plot(pca.components_[1])
    plt.plot(pca.components_[2])
    plt.plot(pca.components_[3])
    plt.show()
    
    # Histogram of weights for each sample for prinical component 0
    pca_results = pca.transform(resid_arr)
                
    # plot weights
    my_bins = np.arange(-5,5.1,0.1).tolist()
    
    plt.hist(pca_results[:,0], bins=my_bins)
    plt.hist(pca_results[:,1], bins=my_bins)
    plt.hist(pca_results[:,2], bins=my_bins)
    plt.hist(pca_results[:,3], bins=my_bins)
    # plt.show()
    
    # classify signals based on mean signal in 3 quantiles
    median, stdev, median_class = bin_residuals(ds4, 4) # n_divisions = 3 gives 2 groups
    # score = score_residuals(ds4)
    # net_score = score_residuals_net(ds4)
    grouped_score = score_residuals_verbose(ds4, 4)
    net_score = grouped_score.sum(axis=1)
    
    # plt.hist(score, bins = 69)
    
    plt.hist(net_score, bins = 80)
    plt.hist(grouped_score[:,0], bins = 30)
    plt.hist(grouped_score[:,1], bins = 30)
    plt.hist(grouped_score[:,2], bins = 30)
    
       
    
    median.shape
    median.max(axis=0)
    median.min(axis=0)
    
    group, count = np.unique(median_class, return_counts=True)
    class_counts = dict(zip(group, count))
    plt.bar(range(len(class_counts)), count, tick_label=group)
    
    # add classes based on mean and median signal cutoffs to dataset
    ids = ds4.values("id")
    
    median1 = dict(zip(ids, median[:,0]))
    median2 = dict(zip(ids, median[:,1]))
    median3 = dict(zip(ids, median[:,2]))
    ds4.set_values("first_23_median", median1, key_field="id")
    ds4.set_values("middle_23_median", median2, key_field="id")
    ds4.set_values("last_23_median", median3, key_field="id")
    
    median_class1 = dict(zip(ids, median_class))
    ds4.set_values("median_class", median_class1, key_field="id")
    
    # score1 = dict(zip(ids, score))
    # ds4.set_values("bin_score", score1, key_field="id")
    
    net_score1 = dict(zip(ids, net_score))
    ds4.set_values("net_bin_score", net_score1, key_field="id")
    
    grouped_score1 = dict(zip(ids, grouped_score[:,0]))
    grouped_score2 = dict(zip(ids, grouped_score[:,1]))
    grouped_score3 = dict(zip(ids, grouped_score[:,2]))
    ds4.set_values("first_23_net_score", grouped_score1, key_field="id")
    ds4.set_values("middle_23_net_score", grouped_score2, key_field="id")
    ds4.set_values("last_23_net_score", grouped_score3, key_field="id")
    
    
    # add verbose classification
    s = statistics.stdev(resid_arr.flatten())
    m = resid_arr.flatten().mean()
    d = statistics.median(resid_arr.flatten())
    plt.hist(resid_arr.flatten(), bins = 100)
    plt.axvline(x = m, color = 'green')
    plt.axvline(x = d, color = 'black')
    plt.axvline(x = d - (s/2), color = 'orange')
    plt.axvline(x = d + (s/2), color = 'orange')
    plt.axvline(x = d - s, color = 'red')
    plt.axvline(x = d + s, color = 'red')
    
    plt.hist(resid_arr[:,30], bins = 100)
    
    plt.plot(resid_arr.mean(axis = 0))
    
    
    
    # add mean class info to metadata dfs and save dfs
    updated_metadata = add_dataset_values_to_df(ds4, metadata, ['first_23_median', 'middle_23_median', 'last_23_median', 'median_class', 'first_23_net_score', 'middle_23_net_score', 'last_23_net_score', 'net_bin_score'], file_suffix)
    
    
