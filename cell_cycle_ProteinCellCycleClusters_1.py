import matplotlib.pyplot as plt
import seaborn as sbn
# Make PDF text readable
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["savefig.dpi"] = 600

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn

def values_comp(values_cell, values_nuc, values_cyto, wp_iscell, wp_isnuc, wp_iscyto):
    '''Get the values for the annotated compartment'''
    values_comp = np.empty_like(values_cell)
    values_comp[wp_iscell] = np.array(values_cell, dtype=object)[wp_iscell]
    values_comp[wp_isnuc] = np.array(values_nuc, dtype=object)[wp_isnuc]
    values_comp[wp_iscyto] = np.array(values_cyto, dtype=object)[wp_iscyto]
    return np.array(values_comp) 

def np_save_overwriting(fn, arr):
    '''Helper function to always overwrite numpy pickles'''
    with open(fn,"wb") as f:    
        np.save(f, arr, allow_pickle=True)

## STATISTICS HELPERS

def gini(array):
    """Calculate the Gini coefficient of a numpy array."""
    # based on bottom eq: http://www.statsdirect.com/help/generatedimages/equations/equation154.svg
    # from: http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    # All values are treated equally, arrays must be 1d:
    # Written by: Olivia Guest github.com/oliviaguest/gini/blob/master/gini.py
    array = array.flatten()
    if np.amin(array) < 0: 
        array -= np.amin(array) # Values cannot be negative
    array = np.sort(array + 0.0000001) # Values must be sorted and nonzero
    index = np.arange(1, array.shape[0] + 1) # Index per array element
    n = array.shape[0] # Number of array elements
    return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array))) # Gini coefficient

def _ecdf(x):
    '''no frills empirical cdf used in fdrcorrection'''
    nobs = len(x)
    return np.arange(1, nobs + 1)/float(nobs)

def benji_hoch(alpha, pvals):
    '''benjimini-hochberg multiple testing correction
    source: https://www.statsmodels.org/dev/_modules/statsmodels/stats/multitest.html'''
    pvals_array = np.array(pvals)
    pvals_array[np.isnan(pvals_array)] = 1 # fail the ones with not enough data
    pvals_sortind = np.argsort(pvals_array)
    pvals_sorted = np.take(pvals_array, pvals_sortind)
    ecdffactor = _ecdf(pvals_sorted)
    reject = pvals_sorted <= ecdffactor*alpha
    reject = pvals_sorted <= ecdffactor*alpha
    if reject.any():
        rejectmax = max(np.nonzero(reject)[0])
        reject[:rejectmax] = True
    pvals_corrected_raw = pvals_sorted / ecdffactor
    pvals_corrected = np.minimum.accumulate(pvals_corrected_raw[::-1])[::-1]
    del pvals_corrected_raw
    pvals_corrected[pvals_corrected>1] = 1
    pvals_corrected_BH = np.empty_like(pvals_corrected)

    # deal with sorting
    pvals_corrected_BH[pvals_sortind] = pvals_corrected
    del pvals_corrected
    reject_BH = np.empty_like(reject)
    reject_BH[pvals_sortind] = reject
    return pvals_corrected_BH, reject_BH

def bonf(alpha, pvals):
    '''Bonferroni multiple testing correction'''
    pvalsarr = np.array(pvals)
    pvalsarr[np.isnan(pvalsarr)] = 1 # fail the ones with not enough data    
    pvals_sortind = np.argsort(pvals)
    pvals_sorted = np.take(pvalsarr, pvals_sortind)
    alphaBonf = alpha / float(len(pvalsarr))
    rejectBonf = pvals_sorted <= alphaBonf
    pvals_correctedBonf = pvals_sorted * float(len(pvalsarr))
    pvals_correctedBonf_unsorted = np.empty_like(pvals_correctedBonf) 
    pvals_correctedBonf_unsorted[pvals_sortind] = pvals_correctedBonf
    rejectBonf_unsorted = np.empty_like(rejectBonf)
    rejectBonf_unsorted[pvals_sortind] = rejectBonf
    return pvals_correctedBonf_unsorted, rejectBonf_unsorted

## PLOTTING HELPERS

def weights(vals):
    '''normalizes all histogram bins to sum to 1'''
    return np.ones_like(vals)/float(len(vals))

def general_boxplot_setup(group_values, group_labels, xlabel, ylabel, title, showfliers, ylim=()):
    '''Set up a boxplot given equal length group_values and group_labels'''
    if len(group_values) != len(group_labels): 
        print("Error: general_boxplot() requires equal length group_values and group_labels.")
        exit(1)
    mmmm = np.concatenate(group_values)
    cccc = np.concatenate([[label] * len(group_values[iii]) for iii, label in enumerate(group_labels)])
    boxplot = sbn.boxplot(x=cccc, y=mmmm, showfliers=showfliers, color="grey")
    boxplot.set_xlabel(xlabel, size=36)
    boxplot.set_ylabel(ylabel, size=18)
    boxplot.tick_params(axis="both", which="major", labelsize=14)
    if len(ylim) > 0: boxplot.set(ylim=ylim)
    plt.title(title)
    return cccc, mmmm, boxplot

def general_boxplot(group_values, group_labels, xlabel, ylabel, title, showfliers, outfile, ylim=()):
    '''Make a boxplot given equal length group_values and group_labels'''
    general_boxplot_setup(group_values, group_labels, xlabel, ylabel, title, showfliers, ylim)
    plt.savefig(outfile)
    plt.close()

def boxplot_with_stripplot(group_values, group_labels, xlabel, ylabel, title, showfliers, outfile, alpha=0.3, size=5, jitter=0.25, ylim=()):
    plt.figure(figsize=(10,10))
    cccc, mmmm, ax = general_boxplot_setup(group_values, group_labels, xlabel, ylabel, title, showfliers, ylim)
    boxplot = sbn.stripplot(x=cccc, y=mmmm, alpha=alpha, color=".3", size=size, jitter=jitter)
    plt.savefig(outfile)
    plt.close()

def general_scatter(x, y, xlabel, ylabel, outfile, showLegend=True):
    '''Make a general scatterplot with matplotlib'''
    plt.figure(figsize=(10,10))
    plt.scatter(x, y, label="all")
    plt.ylabel(xlabel)
    plt.xlabel(ylabel)
    if showLegend: 
        plt.legend()
    plt.savefig(outfile)
    plt.close()
    
def general_scatter_color(x, y, xlabel, ylabel, c, clabel, show_color_bar, title, outfile, cmap="viridis", alpha=1):
    '''Make a general scatterplot with color using matplotlib'''
    plt.figure(figsize=(10,10))
    plt.scatter(x, y, c=c, cmap=cmap, alpha=alpha)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if show_color_bar:
        cb = plt.colorbar()
        cb.set_label(clabel)
    plt.title(title)
    plt.savefig(outfile)
    plt.close()

def general_histogram(x, xlabel, ylabel, alpha, outfile):
    '''Make a general histogram'''
    plt.hist(x, alpha=alpha)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()

def format_p(p):
    '''3 decimal places, scientific notation'''
    return '{:0.3e}'.format(p)

def barplot_annotate_brackets(num1, num2, data, center, height, yerr=None, dh=.05, barh=.05, fs=None, maxasterix=None):
    """ 
    Annotate barplot with p-values. 
    https://stackoverflow.com/questions/11517986/indicating-the-statistically-significant-difference-in-bar-graph

    :param num1: number of left bar to put bracket over
    :param num2: number of right bar to put bracket over
    :param data: string to write or number for generating asterixes
    :param center: centers of all bars (like plt.bar() input)
    :param height: heights of all bars (like plt.bar() input)
    :param yerr: yerrs of all bars (like plt.bar() input)
    :param dh: height offset over bar / bar + yerr in axes coordinates (0 to 1)
    :param barh: bar height in axes coordinates (0 to 1)
    :param fs: font size
    :param maxasterix: maximum number of asterixes to write (for very small p-values)
    """

    if type(data) is str:
        text = data
    else:
        # * is p < 0.05
        # ** is p < 0.005
        # *** is p < 0.0005
        # etc.
        text = ''
        p = .05

        while data < p:
            text += '*'
            p /= 10.

            if maxasterix and len(text) == maxasterix:
                break

        if len(text) == 0:
            text = 'n. s.'

    lx, ly = center[num1], height[num1]
    rx, ry = center[num2], height[num2]

    if yerr:
        ly += yerr[num1]
        ry += yerr[num2]

    ax_y0, ax_y1 = plt.gca().get_ylim()
    dh *= (ax_y1 - ax_y0)
    barh *= (ax_y1 - ax_y0)

    y = max(ly, ry) + dh

    barx = [lx, lx, rx, rx]
    bary = [y, y+barh, y+barh, y]
    mid = ((lx+rx)/2, y+barh)

    plt.plot(barx, bary, c='black')

    kwargs = dict(ha='center', va='bottom')
    if fs is not None:
        kwargs['fontsize'] = fs

    plt.text(*mid, text, **kwargs)

## GENE NAME - ENSG CONVERSIONS

def getGeneNameDict():
    '''Make dictionary of IDs to names'''
    gene_info = pd.read_csv("input/RNAData/IdsToNames.csv.gz", index_col=False, header=None, names=["gene_id", "name", "biotype", "description"])
    geneIdNameDict = dict([(ggg[0], ggg[1]) for idx, ggg in gene_info.iterrows()])
    return geneIdNameDict
    
def ccd_gene_names(id_list_like, geneIdNameDict):
    '''Convert gene ID list to gene name list'''
    return np.unique([geneIdNameDict[ggg] for ggg in id_list_like if ggg in geneIdNameDict])

def ccd_gene_names_gapped(id_list_like, geneIdNameDict):
    '''Convert gene ID list to gene name list'''
    return [geneIdNameDict[idd] if idd in geneIdNameDict else "" for idd in id_list_like]

def getHgncDict():
    '''Make dictionary of IDs to HGNC symbols'''
    geneIdToHgncTable = pd.read_csv("input/ProteinProperties/ENSGToHGNC.csv", index_col=False, header=0)
    geneIdToHgncDict = dict([(ggg[1], ggg[0]) for idx, ggg in geneIdToHgncTable.iterrows()])
    return geneIdToHgncDict

def geneIdToHngc(id_list_like, geneDict):
    '''Convert gene ID list to HNGC symbol if it exists'''
    return np.unique([geneDict[ggg] for ggg in id_list_like if ggg])

def geneIdToHngc_withgaps(id_list_like, geneDict):
    '''Convert gene ID list to HNGC symbol if it exists'''
    return [geneDict[ggg] for ggg in id_list_like]

def ccd_gene_lists(adata):
    '''Read in the published CCD genes / Diana's CCD / Non-CCD genes'''
    gene_info = pd.read_csv("input/RNAData/IdsToNames.csv.gz", index_col=False, header=None, names=["gene_id", "name", "biotype", "description"])
    ccd_regev=pd.read_csv("input/RNAData/ccd_regev.txt")   
    wp_ensg = np.load("output/pickles/wp_ensg.npy", allow_pickle=True)
    ccd_comp = np.load("output/pickles/ccd_comp.npy", allow_pickle=True)
    nonccd_comp = np.load("output/pickles/nonccd_comp.npy", allow_pickle=True)
    ccd=wp_ensg[ccd_comp]
    nonccd=wp_ensg[nonccd_comp]
    ccd_regev_filtered = list(gene_info[(gene_info["name"].isin(ccd_regev["gene"])) & (gene_info["gene_id"].isin(adata.var_names))]["gene_id"])
    ccd_filtered = list(ccd[np.isin(ccd, adata.var_names)])
    nonccd_filtered = list(nonccd[np.isin(nonccd, adata.var_names)])
    return ccd_regev_filtered, ccd_filtered, nonccd_filtered

def save_category(genelist, filename):
    pd.DataFrame({"gene" : genelist}).to_csv(filename, index=False, header=False)

def save_gene_names_by_category(adata, wp_ensg, ccd_comp, nonccd_comp, ccdtranscript):
    '''Save files containing the gene names for each category of CCD proteins/transcripts'''
    ccd_regev_filtered, ccd_filtered, nonccd_filtered = ccd_gene_lists(adata)
    genes_analyzed = np.array(pd.read_csv("output/gene_names.csv")["gene"])
    bioccd = np.genfromtxt("input/ProteinData/BiologicallyDefinedCCD.txt", dtype='str') # from mitotic structures
    knownccd1 = np.genfromtxt("input/ProteinData/knownccd.txt", dtype='str') # from gene ontology, reactome, cyclebase 3.0, NCBI gene from mcm3
    knownccd2 = np.genfromtxt("input/ProteinData/known_go_ccd.txt", dtype='str') # from GO cell cycle
    knownccd3 = np.genfromtxt("input/ProteinData/known_go_proliferation.txt", dtype='str') # from GO proliferation
    knownccd = np.concatenate((knownccd1, knownccd2, knownccd3))

    # Get the ENSG symbols for lists for GO analysis
    ensg_ccdtranscript = np.unique(adata.var_names[ccdtranscript])
    ensg_nonccdtranscript = np.unique(adata.var_names[~ccdtranscript])
    ensg_ccdprotein = np.unique(np.concatenate((wp_ensg[ccd_comp], bioccd)))
    ensg_nonccdprotein = np.unique(wp_ensg[nonccd_comp & ~np.isin(wp_ensg, bioccd)])
    ensg_ccdprotein_treg = np.unique(ensg_ccdprotein[np.isin(ensg_ccdprotein, ensg_ccdtranscript)])
    ensg_ccdprotein_nontreg = np.unique(ensg_ccdprotein[~np.isin(ensg_ccdprotein, ensg_ccdtranscript)])
    ensg_knownccdprotein = ensg_ccdprotein[np.isin(ensg_ccdprotein, knownccd)]
    ensg_novelccdprotein = ensg_ccdprotein[~np.isin(ensg_ccdprotein, knownccd)]
    
    # Get the HGNC symbols for lists for GO analysis
    geneIdToHgncDict = getHgncDict()
    hgnc_ccdtranscript = geneIdToHngc(ensg_ccdtranscript, geneIdToHgncDict)
    hgnc_ccdprotein_transcript_regulated = geneIdToHngc(ensg_ccdprotein_treg, geneIdToHgncDict)
    hgnc_ccdprotein_nontranscript_regulated = geneIdToHngc(ensg_ccdprotein_nontreg, geneIdToHgncDict)
    hgnc_nonccdprotein = geneIdToHngc(ensg_nonccdprotein, geneIdToHgncDict)
    hgnc_ccdprotein = geneIdToHngc(ensg_ccdprotein, geneIdToHgncDict)
    
    # Convert to gene names and store them as such
    geneIdNameDict = getGeneNameDict()
    names_ccdtranscript = ccd_gene_names(ensg_ccdtranscript, geneIdNameDict)
    names_nonccdtranscript = ccd_gene_names(ensg_nonccdtranscript, geneIdNameDict)
    names_ccdprotein = ccd_gene_names(ensg_ccdprotein, geneIdNameDict)
    names_nonccdprotein = ccd_gene_names(ensg_nonccdprotein, geneIdNameDict)
    names_ccdprotein_transcript_regulated = ccd_gene_names(ensg_ccdprotein_treg, geneIdNameDict)
    names_ccdprotein_nontranscript_regulated = ccd_gene_names(ensg_ccdprotein_nontreg, geneIdNameDict)
    names_genes_analyzed = ccd_gene_names(genes_analyzed, geneIdNameDict)
    names_ccd_regev_filtered = ccd_gene_names(ccd_regev_filtered, geneIdNameDict)
    names_ccd_filtered = ccd_gene_names(ccd_filtered, geneIdNameDict)
    
    # Save the HGNC gene names for each category
    save_category(hgnc_ccdtranscript, "output/hgnc_ccdtranscript.csv")
    save_category(hgnc_ccdprotein_transcript_regulated, "output/hgnc_ccdprotein_transcript_regulated.csv")
    save_category(hgnc_ccdprotein_nontranscript_regulated, "output/hgnc_ccdprotein_nontranscript_regulated.csv")
    save_category(hgnc_nonccdprotein, "output/hgnc_nonccdprotein.csv")
    save_category(hgnc_ccdprotein, "output/hgnc_ccdprotein.csv")

    # Save the geneIds for each category
    save_category(ensg_ccdtranscript, "output/ensg_ccdtranscript.csv")
    save_category(ensg_nonccdtranscript, "output/ensg_nonccdtranscript.csv")
    save_category(ensg_ccdprotein_treg, "output/ensg_ccdprotein_transcript_regulated.csv")
    save_category(ensg_ccdprotein_nontreg, "output/ensg_ccdprotein_nontranscript_regulated.csv")
    save_category(ensg_nonccdprotein, "output/ensg_nonccdprotein.csv")
    save_category(ensg_ccdprotein, "output/ensg_ccdprotein.csv")
    save_category(ensg_knownccdprotein, "output/ensg_knownccdprotein.csv")
    save_category(ensg_novelccdprotein, "output/ensg_novelccdprotein.csv")

    # Save the gene names for each category
    save_category(names_ccdtranscript, "output/names_ccdtranscript.csv")
    save_category(names_nonccdtranscript, "output/names_nonccdtranscript.csv")
    save_category(names_ccdprotein, "output/names_ccdprotein.csv")
    save_category(names_nonccdprotein, "output/names_nonccdprotein.csv")
    save_category(names_ccdprotein_transcript_regulated, "output/names_ccdprotein_transcript_regulated.csv")
    save_category(names_ccdprotein_nontranscript_regulated, "output/names_ccdprotein_nontranscript_regulated.csv")
    save_category(names_genes_analyzed, "output/names_genes_analyzed.csv")
    save_category(names_ccd_regev_filtered, "output/names_ccd_regev_filtered.csv")
    save_category(names_genes_analyzed, "output/names_genes_analyzed.csv")
    save_category(names_ccd_filtered, "output/names_ccd_filtered.csv")
    
    return ((ensg_ccdtranscript, ensg_nonccdtranscript, ensg_ccdprotein, ensg_nonccdprotein, 
            ensg_ccdprotein_treg, ensg_ccdprotein_nontreg, 
            genes_analyzed, ccd_regev_filtered, ccd_filtered),
        (names_ccdtranscript, names_nonccdtranscript, names_ccdprotein, names_nonccdprotein, 
            names_ccdprotein_transcript_regulated, names_ccdprotein_nontranscript_regulated, 
            names_genes_analyzed, names_ccd_regev_filtered, names_ccd_filtered))
    
	
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from SingleCellProteogenomics import utils
plt.rcParams['pdf.fonttype'], plt.rcParams['ps.fonttype'], plt.rcParams['savefig.dpi'] = 42, 42, 600 #Make PDF text readable

EMPTYWELLS = set(["B11_6745","C11_6745","D11_6745","E11_6745","F11_6745","G11_6745","H11_6745",
    "A12_6745","B12_6745","C12_6745","D12_6745","E12_6745","F12_6745","G12_6745"]) 
# EMPTYWELLS: These wells on the last plate didn't have cells; the segmentation algorithm still annotated some, so remove them
MIN_CELL_COUNT = 60 # Minimum number of cells per sample required for cell cycle analysis with pseudotime

# 0: use mean, (testing intensity that's already normalized for cell size)
# 1: use integrated, (testing integrated because it reflects that small cells are often brighter because they have rounded up and are thicker)
# 2: use integrated / nucleus area (testing a normalization by nucleus size, which may be more consistent in segmentation)
INTENSITY_SWITCH = 0 # cell size does increase into G2 and that has a substantial effect, so use the mean intensity, which is well behaved

def read_raw_data():
    '''Read in the raw protein IF data'''
    print("reading raw protein IF data")
    my_df1 = pd.read_csv("input/ProteinData/FucciDataFirstPlates.csv.gz")
    my_df2 = pd.read_csv("input/ProteinData/FucciDataSecondPlates.csv.gz")
    my_df = pd.concat((my_df1, my_df2), sort=True)
    print("loaded raw data")
    return my_df

def read_sample_info(df):
    '''Get the metadata for all the samples'''
    plate = np.asarray(df.plate)
    u_plate = np.unique(plate)
    well_plate = np.asarray(df.well_plate)
    imgnb = np.asarray(df.ImageNumber)
    well_plate_imgnb = np.asarray([f"{wp}_{imgnb[i]}" for i,wp in enumerate(well_plate)])
    u_well_plates = np.unique(well_plate)
    ab_objnum = np.asarray(df.ObjectNumber)
    well_plate_imgnb_objnb = np.asarray([f"{wp}_{imgnb[i]}_{ab_objnum[i]}" for i,wp in enumerate(well_plate)])
    area_cell = np.asarray(df.Area_cell)
    area_nuc = np.asarray(df.AreaShape_Area)
    area_cyto = np.asarray(df.Area_cyto)
    name_df = pd.read_csv("input/ProteinData/FucciStainingSummaryFirstPlates.csv")
    wppp1, ensggg1, abbb1, rrrr, cccc1 = list(name_df["well_plate"]), list(name_df["ENSG"]), list(name_df["Antibody"]), list(name_df["Results_final_update"]), list(name_df["Compartment"])
    name_df2 = pd.read_csv("input/ProteinData/FucciStainingSummarySecondPlates.csv")
    wppp2, ensggg2, abbb2, cccc2 = list(name_df2["well_plate"]), list(name_df2["ENSG"]), list(name_df2["Antibody"]), list(name_df2["Compartment"])
    wppp, ensggg, abbb, cccc = wppp1 + wppp2, ensggg1 + ensggg2, abbb1 +  abbb2, cccc1 + cccc2
    ensg_dict = dict([(wppp[i], ensggg[i]) for i in range(len(wppp))])
    ab_dict = dict([(wppp[i], abbb[i]) for i in range(len(wppp))])
    result_dict = dict([(wppp[i], rrrr[i]) for i in range(len(wppp1))])
    compartment_dict = dict([(wppp[i], cccc[i]) for i in range(len(wppp))])
    ENSG = np.asarray([ensg_dict[wp] if wp in ensg_dict else "" for wp in well_plate])
    antibody = np.asarray([ab_dict[wp] if wp in ab_dict else "" for wp in well_plate])
    result = np.asarray([result_dict[wp] if wp in result_dict else "" for wp in well_plate])
    compartment = np.asarray([compartment_dict[wp] if wp in compartment_dict else "" for wp in well_plate])
    
    # Pickle the results
    if not os.path.exists("output/"): os.mkdir("output/")
    if not os.path.exists("output/pickles/"): os.mkdir("output/pickles/")
    if not os.path.exists("figures/"): os.mkdir("figures/")
    np_save_overwriting("output/pickles/plate.npy", plate)
    np_save_overwriting("output/pickles/u_plate.npy", u_plate)
    np_save_overwriting("output/pickles/u_well_plates.npy", u_well_plates)
    np_save_overwriting("output/pickles/area_cell.npy", area_cell)
    np_save_overwriting("output/pickles/area_nuc.npy", area_nuc)
    np_save_overwriting("output/pickles/area_cyto.npy", area_cyto)
    np_save_overwriting("output/pickles/well_plate.npy", well_plate)
    np_save_overwriting("output/pickles/well_plate_imgnb.npy", well_plate_imgnb)
    np_save_overwriting("output/pickles/well_plate_imgnb_objnb.npy", well_plate_imgnb_objnb)
    
    return plate, u_plate, well_plate, well_plate_imgnb, u_well_plates, ab_objnum, area_cell, area_nuc, area_cyto, ensg_dict, ab_dict, result_dict, compartment_dict, ENSG, antibody, result, compartment

def previous_results(u_well_plates, result_dict, ensg_dict, ab_dict):
    '''Process the results metadata into lists of previously annotated CCD proteins'''
    wp_ensg = np.asarray([ensg_dict[wp] if wp in ensg_dict else "" for wp in u_well_plates])
    wp_ab = np.asarray([ab_dict[wp] if wp in ab_dict else "" for wp in u_well_plates])
    wp_prev_ccd = np.asarray([wp in result_dict and result_dict[wp].startswith("ccd") for wp in u_well_plates])
    wp_prev_notccd = np.asarray([wp in result_dict and result_dict[wp].startswith("notccd") for wp in u_well_plates])
    wp_prev_negative = np.asarray([wp in result_dict and result_dict[wp].startswith("negative") for wp in u_well_plates])
    prev_ccd_ensg = wp_ensg[wp_prev_ccd]
    prev_notccd_ensg = wp_ensg[wp_prev_notccd]
    prev_negative_ensg = wp_ensg[wp_prev_negative]
    
    # Pickle the results
    np_save_overwriting("output/pickles/wp_ensg.npy", wp_ensg)
    np_save_overwriting("output/pickles/wp_ab.npy", wp_ab)
    np_save_overwriting("output/pickles/wp_prev_ccd.npy", wp_prev_ccd)
    np_save_overwriting("output/pickles/wp_prev_notccd.npy", wp_prev_notccd)
    np_save_overwriting("output/pickles/wp_prev_negative.npy", wp_prev_negative)
    np_save_overwriting("output/pickles/prev_ccd_ensg.npy", prev_ccd_ensg)
    np_save_overwriting("output/pickles/prev_notccd_ensg.npy", prev_notccd_ensg)
    np_save_overwriting("output/pickles/prev_negative_ensg.npy", prev_negative_ensg)
    
    return wp_ensg, wp_ab, wp_prev_ccd, wp_prev_notccd, wp_prev_negative, prev_ccd_ensg, prev_notccd_ensg, prev_negative_ensg

def apply_manual_filtering(my_df, result_dict, ab_dict):
    '''Filter raw data based on manual annotations'''
    # filter some wells in the last plate didn't have anything.
    print(f"{len(my_df)}: number of cells before filtering empty wells")
    my_df = my_df[~my_df.well_plate.isin(EMPTYWELLS)]
    print(f"{len(my_df)}: number of cells after filtering empty wells")
    
    my_df_filtered = my_df
    print("filtering out of focus")
    oof = pd.read_csv("input/ProteinData/OutOfFocusImages.txt", header=None)[0]
    well_plate = np.asarray(my_df_filtered.well_plate)
    imgnb = np.asarray(my_df_filtered.ImageNumber)
    well_plate_imgnb = np.asarray([f"{wp}_{imgnb[i]}" for i,wp in enumerate(well_plate)])
    print(f"{len(my_df_filtered)}: number of cells before filtering out of focus images")
    my_df_filtered = my_df_filtered[~np.isin(well_plate_imgnb, oof)]
    print(f"{len(my_df_filtered)}: number of cells after filtering out of focus images")
    print("finished filtering")
    
    print("filtering negative staining")
    new_data_or_nonnegative_stain = [wp not in result_dict or (not result_dict[wp].lower().startswith("negative") and not wp.startswith("H12")) for wp in my_df_filtered.well_plate]
    print(f"{len(my_df_filtered)}: number of cells before filtering negative staining from first batch")
    my_df_filtered = my_df_filtered[new_data_or_nonnegative_stain]
    print(f"{len(my_df_filtered)}: number of cells after filtering negative staining from first batch")
    print("finished filtering")
     
    print("filtering bad fields of view (negative staining, unspecific, etc)")
    filterthese = pd.read_csv("input/ProteinData/FOV_ImgNum_Lookup.csv")
    badfov = filterthese["well_plate_imgnb"][(filterthese["UseImage"] == 0)]
    well_plate = np.asarray(my_df_filtered.well_plate)
    imgnb = np.asarray(my_df_filtered.ImageNumber)
    well_plate_imgnb = np.asarray([f"{wp}_{imgnb[i]}" for i,wp in enumerate(well_plate)])
    negative_controls = np.asarray([wp.startswith("H12") for wp in well_plate])
    print(f"{len(my_df_filtered)}: number of cells before filtering out of focus images")
    my_df_filtered = my_df_filtered[~np.isin(well_plate_imgnb, badfov) & ~negative_controls]
    print(f"{len(my_df_filtered)}: number of cells after filtering out of focus images")
    print("finished filtering")
    
    print("filtering failed antibodies")
    failedab = np.genfromtxt("input/ProteinData/RecentlyFailedAntibodies.txt", dtype='str')
    print(f"{len(my_df_filtered)}: number of cells before filtering antibodies failed in HPAv19")
    my_df_filtered = my_df_filtered[~np.isin([ab_dict[wp] for wp in my_df_filtered.well_plate], failedab)]
    print(f"{len(my_df_filtered)}: number of cells after filtering antibodies failed in HPAv19")
    print("finished filtering")
    
    print("filtering mitotic proteins")
    mitoticab = np.genfromtxt("input/ProteinData/RemoveMitoticAndMicrotubules.txt", dtype='str')
    print(f"{len(my_df_filtered)}: number of cells before filtering mitotic/microtubule proteins")
    my_df_filtered = my_df_filtered[~np.isin([ab_dict[wp] for wp in my_df_filtered.well_plate], mitoticab)]
    print(f"{len(my_df_filtered)}: number of cells after filtering mitotic/microtubule proteins")
    print("finished filtering")
    
    return my_df_filtered
    
def plot_areas(areas, title):
    '''Histogram for areas of cell/nuc/cytoplasm'''
    bins = plt.hist(areas, bins=100, alpha=0.5)
    plt.vlines(np.mean(areas), 0, np.max(bins[0]))
    plt.vlines(np.mean(areas) - 2 * np.std(areas), 0, np.max(bins[0]))
    plt.vlines(np.mean(areas) + 2 * np.std(areas), 0, np.max(bins[0]))
    plt.title(title)
    plt.ylabel("Count")
    plt.xlabel("Area")
    plt.savefig(f"figures/areas{title}.png")
    plt.close()

def apply_big_nucleus_filter(my_df):
    '''Filter the super big nuclei'''
    area_cell, area_nuc, area_cyto = my_df.Area_cell, my_df.AreaShape_Area, my_df.Area_cyto
    plot_areas(area_cell, "area_cell")
    plot_areas(area_nuc, "area_nuc")
    plot_areas(area_cyto, "area_cyto")
    
    upper_nucleus_cutoff = np.mean(area_nuc) + 2 * np.std(area_nuc)

    my_df_filtered = my_df
    print("filtering super big nuclei")
    cell_passes_nucleus_filter = my_df_filtered.AreaShape_Area < upper_nucleus_cutoff
    print(f"{len(my_df_filtered)}: number of cells before filtering out super big nuclei")
    my_df_filtered = my_df_filtered[cell_passes_nucleus_filter]
    print(f"{len(my_df_filtered)}: number of cells after filtering out super big nuclei")
    print("finished filtering on nuclei")
    
    area_cell_filtered, area_nuc_filtered, area_cyto_filtered = my_df_filtered.Area_cell, my_df_filtered.AreaShape_Area, my_df_filtered.Area_cyto
    plot_areas(area_cell_filtered, "area_cell_filtered")
    plot_areas(area_nuc_filtered, "area_nuc_filtered")
    plot_areas(area_cyto_filtered, "area_cyto_filtered")
    return my_df_filtered

def apply_cell_count_filter(my_df):
    '''Filter low cell counts per sample'''
    my_df_filtered = my_df
    well_plate = np.asarray(my_df_filtered.well_plate)
    u_well_plates = np.unique(my_df_filtered.well_plate)
    cell_count_dict = {}
    for wp in well_plate:
        if wp in cell_count_dict: cell_count_dict[wp] += 1
        else: cell_count_dict[wp] = 1
    cell_counts = np.array([cell_count_dict[wp] for wp in well_plate])
    print("filtering low cell counts")
    my_df_filtered = my_df_filtered[cell_counts >= MIN_CELL_COUNT]
    print(f"{len(my_df)}: number of cells before filtering out samples with < {MIN_CELL_COUNT} cells")
    print(f"{len(my_df_filtered)}: number of cells after filtering out samples with < {MIN_CELL_COUNT} cells")
    print("finished filtering on cell count")
    return my_df_filtered

def apply_variation_filter(my_df_filtered, result_dict, unfiltered_df):
    '''Separate the varying and nonvarying samples'''
    my_df_filtered_variation, my_df_filtered_novariation = my_df_filtered, my_df_filtered
    variable_firstbatch = np.asarray([wp in result_dict and not result_dict[wp].replace(" ","").startswith("novariation") for wp in my_df_filtered.well_plate])
    
    varann_secondbatch = pd.read_csv("input/ProteinData/SecondBatchVariableLookup.csv")
    variable_ann_secondbatch = np.asarray([str(vv).lower().startswith("yes") for vv in varann_secondbatch["IsVariable"]])
    variable_wp_secondbatch = np.asarray(varann_secondbatch["well_plate"][variable_ann_secondbatch])
    variable_secondbatch = np.isin(my_df_filtered.well_plate, variable_wp_secondbatch)
    
    my_df_filtered_variation = my_df_filtered[variable_firstbatch | variable_secondbatch]
    my_df_filtered_novariation = my_df_filtered[~(variable_firstbatch | variable_secondbatch)]
    print(f"{len(unfiltered_df)}: number of cells before filtering for variation")
    print(f"{len(my_df_filtered_variation)}: number of cells in samples with variation")
    print(f"{len(my_df_filtered_novariation)}: number of cells in samples without variation")
    return my_df_filtered_variation, my_df_filtered_novariation

def metacompartments(u_well_plates, compartment_dict, my_df_filtered_variation):
    '''Get the compartments for the unique wellplates'''
    wp_iscell = np.asarray([compartment_dict[wp].lower().startswith("cell") if wp in compartment_dict else False for wp in u_well_plates])
    wp_isnuc = np.asarray([compartment_dict[wp].lower().startswith("nuc") if wp in compartment_dict else False for wp in u_well_plates])
    wp_iscyto = np.asarray([compartment_dict[wp].lower().startswith("cyto") if wp in compartment_dict else False for wp in u_well_plates])
    
    # Pickle the results
    np_save_overwriting("output/pickles/wp_iscell.npy", wp_iscell)
    np_save_overwriting("output/pickles/wp_isnuc.npy", wp_isnuc)
    np_save_overwriting("output/pickles/wp_iscyto.npy", wp_iscyto)

    wp_nocompartmentinfo = ~wp_iscell & ~wp_isnuc & ~wp_iscyto
    print(f"{sum(wp_nocompartmentinfo)}: samples without compartment information; to be filtered since they're biologically defined as CCD and not included in the analysis")
    print(f"{len(my_df_filtered_variation)}: number of cells before filtering for compartment information")
    my_df_filtered_compartmentvariation = my_df_filtered_variation[~np.isin(my_df_filtered_variation.well_plate, u_well_plates[wp_nocompartmentinfo])]
    print(f"{len(my_df_filtered_compartmentvariation)}: number of cells before filtering for compartment information")
    return wp_iscell, wp_isnuc, wp_iscyto, my_df_filtered_compartmentvariation

def read_sample_data(df):
    '''Read antibody intensity data for each sample and save it to a file for later use.'''
    # Antibody data (mean intensity)
    ab_nuc = np.asarray([df.Intensity_MeanIntensity_ResizedAb, 
                         df.Intensity_IntegratedIntensity_ResizedAb, 
                         df.Intensity_IntegratedIntensity_ResizedAb / df.AreaShape_Area][INTENSITY_SWITCH])
    ab_cyto = np.asarray([df.Mean_ab_Cyto, 
                          df.Integrated_ab_cyto, 
                          df.Integrated_ab_cyto / df.AreaShape_Area][INTENSITY_SWITCH])
    ab_cell = np.asarray([df.Mean_ab_cell, 
                          df.Integrated_ab_cell, 
                          df.Integrated_ab_cell / df.AreaShape_Area][INTENSITY_SWITCH])
    mt_cell = np.asarray([df.Mean_mt_cell, 
                          df.Integrated_mt_cell, 
                          df.Integrated_mt_cell / df.AreaShape_Area][INTENSITY_SWITCH])

    # Fucci data (mean intensity)
    green_fucci = np.asarray(df.Intensity_MeanIntensity_CorrResizedGreenFUCCI)
    red_fucci = np.asarray(df.Intensity_MeanIntensity_CorrResizedRedFUCCI)
    
    # Pickle the results
    np_save_overwriting("output/pickles/ab_nuc.npy", ab_nuc)
    np_save_overwriting("output/pickles/ab_cyto.npy", ab_cyto)
    np_save_overwriting("output/pickles/ab_cell.npy", ab_cell)
    np_save_overwriting("output/pickles/mt_cell.npy", mt_cell)
    np_save_overwriting("output/pickles/green_fucci.npy", green_fucci)
    np_save_overwriting("output/pickles/red_fucci.npy", red_fucci)

    return ab_nuc, ab_cyto, ab_cell, mt_cell, green_fucci, red_fucci	
	
	
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sbn
import scipy
#from SingleCellProteogenomics import utils
import sklearn.mixture
plt.rcParams['pdf.fonttype'], plt.rcParams['ps.fonttype'], plt.rcParams['savefig.dpi'] = 42, 42, 600 #Make PDF text readable

def zero_center_fucci(green_fucci, red_fucci, u_plate, well_plate, plate):
    '''Zero center and rescale FUCCI data in the log space'''
    log_green_fucci, log_red_fucci = np.log10(green_fucci), np.log10(red_fucci)
    wp_p_dict = dict([(str(p), plate == p) for p in u_plate])
    logmed_green_fucci_p = dict([(str(p), np.log10(np.median(green_fucci[wp_p_dict[str(p)]]))) for p in u_plate])
    logmed_red_fucci_p = dict([(str(p), np.log10(np.median(red_fucci[wp_p_dict[str(p)]]))) for p in u_plate])
    logmed_green_fucci = np.array([logmed_green_fucci_p[wp.split("_")[1]] for wp in well_plate])
    logmed_red_fucci = np.array([logmed_red_fucci_p[wp.split("_")[1]] for wp in well_plate])
    log_green_fucci_zeroc = np.array(log_green_fucci) - logmed_green_fucci
    log_red_fucci_zeroc = np.array(log_red_fucci) - logmed_red_fucci
    log_green_fucci_zeroc_rescale = (log_green_fucci_zeroc - np.min(log_green_fucci_zeroc)) / np.max(log_green_fucci_zeroc)
    log_red_fucci_zeroc_rescale = (log_red_fucci_zeroc - np.min(log_red_fucci_zeroc)) / np.max(log_red_fucci_zeroc)
    fucci_data = np.column_stack([log_green_fucci_zeroc_rescale,log_red_fucci_zeroc_rescale])
    result = (log_green_fucci, log_red_fucci,
              log_green_fucci_zeroc, log_red_fucci_zeroc,
              log_green_fucci_zeroc_rescale, log_red_fucci_zeroc_rescale,
              fucci_data)
    
    # Pickle the results
    np_save_overwriting("output/pickles/log_green_fucci_zeroc.npy", log_green_fucci_zeroc)
    np_save_overwriting("output/pickles/log_red_fucci_zeroc.npy", log_red_fucci_zeroc)
    np_save_overwriting("output/pickles/log_green_fucci_zeroc_rescale.npy", log_green_fucci_zeroc_rescale)
    np_save_overwriting("output/pickles/log_red_fucci_zeroc_rescale.npy", log_red_fucci_zeroc_rescale)
    np_save_overwriting("output/pickles/fucci_data.npy", fucci_data)
    
    return result

def gaussian_boxplot_result(g1, s, g2, outfolder, ensg):
    '''Boxplot for intensities within each cell cycle phase'''
    if not os.path.exists(f"{outfolder}_png"): os.mkdir(f"{outfolder}_png")
    if not os.path.exists(f"{outfolder}_pdf"): os.mkdir(f"{outfolder}_pdf")
    mmmm = np.concatenate((g1, s, g2))
    cccc = (["G1"] * len(g1))
    cccc.extend(["G1/S"] * len(s))
    cccc.extend(["G2"] * len(g2))
    boxplot = sbn.boxplot(x=cccc, y=mmmm, showfliers=False, color="grey")
    boxplot.set_xlabel("", size=36)
    boxplot.set_ylabel("Normalized Mean Intensity", size=18)
    boxplot.tick_params(axis="both", which="major", labelsize=14)
    plt.ylim(0,1)
    plt.title("")
    plt.savefig(f"{outfolder}_png/GaussianClusteringProtein_{ensg}.png")
    plt.savefig(f"{outfolder}_pdf/GaussianClusteringProtein_{ensg}.pdf")
    plt.close()
    
def gaussian_clustering(log_green_fucci_zeroc_rescale, log_red_fucci_zeroc_rescale, clusternames):
    '''Perform gaussian clustering of FUCCI data into 3 phases: G1, S, G2'''
    gaussian = sklearn.mixture.GaussianMixture(n_components=3, random_state=1, max_iter=500)
    cluster_labels = gaussian.fit_predict(np.array([log_green_fucci_zeroc_rescale, log_red_fucci_zeroc_rescale]).T)
    for cluster in range(3):
        plt.hist2d(log_green_fucci_zeroc_rescale[cluster_labels == cluster],log_red_fucci_zeroc_rescale[cluster_labels == cluster],bins=200)
        plt.title(f"Gaussian clustered data, {clusternames[cluster]}")
        plt.xlabel("Log10 Green Fucci Intensity")
        plt.ylabel("Log10 Red Fucci Intensity")
        plt.savefig(f"figures/FucciPlotProteinIFData_unfiltered_Gauss{cluster}.png")
        # plt.show()
        plt.close()
    return cluster_labels

def get_phase_strings(is_g1, is_sph, is_g2):
    '''Make strings to represent the metacompartment'''
    phasestring = np.array(["G1"] * len(is_g1))
    phasestring[is_sph] = "S" 
    phasestring[is_g2] = "G2"
    return phasestring

def gaussian_clustering_analysis(alpha_gauss, doGeneratePlots, g1, sph, g2, 
             wp_ensg, well_plate, u_well_plates, ab_cell, ab_nuc, ab_cyto, mt_cell, wp_iscell, wp_isnuc, wp_iscyto):
    '''Analyze the results of Gaussian clustering of FUCCI data for each protein antibody staining'''
    wp_cell_kruskal, wp_nuc_kruskal, wp_cyto_kruskal, wp_mt_kruskal = [],[],[],[]
    curr_wp_phases = []
    mockbulk_phases = np.array(["  "] * len(ab_cell))
    fileprefixes = np.array([f"{ensg}_{sum(wp_ensg[:ei] == ensg)}" for ei, ensg in enumerate(wp_ensg)])
    for iii, wp in enumerate(u_well_plates):
        curr_well_inds = well_plate==wp
        curr_wp_g1 = curr_well_inds & g1
        curr_wp_sph = curr_well_inds & sph
        curr_wp_g2 = curr_well_inds & g2
        curr_wp_phase_list = get_phase_strings(g1[curr_well_inds], sph[curr_well_inds], g2[curr_well_inds])
        mockbulk_phases[curr_well_inds] = np.asarray(curr_wp_phase_list)
        curr_wp_phases.append(curr_wp_phase_list)
        wp_cell_kruskal.append(scipy.stats.kruskal(ab_cell[curr_wp_g1], ab_cell[curr_wp_sph], ab_cell[curr_wp_g2])[1])
        wp_nuc_kruskal.append(scipy.stats.kruskal(ab_nuc[curr_wp_g1], ab_nuc[curr_wp_sph], ab_nuc[curr_wp_g2])[1])
        wp_cyto_kruskal.append(scipy.stats.kruskal(ab_cyto[curr_wp_g1], ab_cyto[curr_wp_sph], ab_cyto[curr_wp_g2])[1])
        wp_mt_kruskal.append(scipy.stats.kruskal(mt_cell[curr_wp_g1], mt_cell[curr_wp_sph], mt_cell[curr_wp_g2])[1])
        max_val_for_norm = np.max(ab_cell[curr_well_inds] if wp_iscell[iii] else ab_nuc[curr_well_inds] if wp_isnuc[iii] else ab_cyto[curr_well_inds])
        max_mt_for_norm = np.max(mt_cell[curr_well_inds])
        if doGeneratePlots:
            gaussian_boxplot_result(
                    (ab_cell[curr_wp_g1] if wp_iscell[iii] else ab_nuc[curr_wp_g1] if wp_isnuc[iii] else ab_cyto[curr_wp_g1]) / max_val_for_norm,
                    (ab_cell[curr_wp_sph] if wp_iscell[iii] else ab_nuc[curr_wp_sph] if wp_isnuc[iii] else ab_cyto[curr_wp_sph]) / max_val_for_norm,
                    (ab_cell[curr_wp_g2] if wp_iscell[iii] else ab_nuc[curr_wp_g2] if wp_isnuc[iii] else ab_cyto[curr_wp_g2]) / max_val_for_norm,
                    "figures/GaussianBoxplots", fileprefixes[iii])
            gaussian_boxplot_result(
                mt_cell[curr_wp_g1] / max_mt_for_norm,
                mt_cell[curr_wp_sph] / max_mt_for_norm,
                mt_cell[curr_wp_g2] / max_mt_for_norm,
                "figures/GaussianBoxplots_mt", f"{fileprefixes[iii]}_mt")
        
    # multiple testing correction for protein of interest
    wp_comp_kruskal_gaussccd_p = values_comp(wp_cell_kruskal, wp_nuc_kruskal, wp_cyto_kruskal, wp_iscell, wp_isnuc, wp_iscyto)
    wp_comp_kruskal_gaussccd_adj, wp_pass_kruskal_gaussccd_bh_comp = benji_hoch(alpha_gauss, wp_comp_kruskal_gaussccd_p)
    np_save_overwriting("output/pickles/wp_comp_kruskal_gaussccd_adj.npy", wp_comp_kruskal_gaussccd_adj)
    np_save_overwriting("output/pickles/wp_pass_kruskal_gaussccd_bh_comp.npy", wp_pass_kruskal_gaussccd_bh_comp)

    # multiple testing correction for microtubules
    wp_mt_kruskal_gaussccd_adj, wp_pass_gaussccd_bh_mt = benji_hoch(alpha_gauss, wp_mt_kruskal) 
    np_save_overwriting("output/pickles/wp_mt_kruskal_gaussccd_adj.npy", wp_mt_kruskal_gaussccd_adj)
    np_save_overwriting("output/pickles/wp_pass_gaussccd_bh_mt.npy", wp_pass_gaussccd_bh_mt)
    
    # save the phase information
    np_save_overwriting("output/pickles/curr_wp_phases.npy", np.array(curr_wp_phases, dtype=object))
    np_save_overwriting("output/pickles/mockbulk_phases.npy", np.array(mockbulk_phases))

    print(f"{len(wp_pass_kruskal_gaussccd_bh_comp)}: number of genes tested")
    print(f"{sum(wp_pass_kruskal_gaussccd_bh_comp)}: number of passing genes at {alpha_gauss*100}% FDR in compartment")

    return wp_comp_kruskal_gaussccd_adj, wp_pass_kruskal_gaussccd_bh_comp, wp_mt_kruskal_gaussccd_adj, wp_pass_gaussccd_bh_mt

def address_replicates(alpha_gauss, wp_pass_kruskal_gaussccd_bh_comp, wp_ensg, wp_ab, u_well_plates):
    '''Look for replicated protein samples and antibody stainings'''
    # address gene redundancy
    wp_ensg_counts = np.array([sum([1 for eeee in wp_ensg if eeee == ensg]) for ensg in wp_ensg])
    ensg_is_duplicated = wp_ensg_counts > 1
    duplicated_ensg = np.unique(wp_ensg[ensg_is_duplicated])
    duplicated_ensg_pairs = [u_well_plates[wp_ensg == ensg] for ensg in duplicated_ensg]
    print(f"{sum(wp_pass_kruskal_gaussccd_bh_comp[~ensg_is_duplicated])}: number of passing genes at {alpha_gauss*100}% FDR in compartment (no replicate)")
    duplicated_ensg_ccd = np.array([sum(wp_pass_kruskal_gaussccd_bh_comp[wp_ensg == ensg]) for ensg in duplicated_ensg])
    print(f"{sum(duplicated_ensg_ccd == 2)}: number of CCD genes shown to be CCD in both replicates")
    print(f"{sum(duplicated_ensg_ccd == 1)}: number of CCD genes shown to be CCD in just one replicate")
    print(f"{sum(duplicated_ensg_ccd == 0)}: number of CCD genes shown to be non-CCD in both replicate")
    
    # any antibody redundancy?
    wp_ab_counts = np.array([sum([1 for aaaa in wp_ab if aaaa == ab]) for ab in wp_ab])
    ab_is_duplicated = wp_ab_counts > 1
    duplicated_ab = np.unique(wp_ab[ab_is_duplicated])
    print(f"{sum(wp_pass_kruskal_gaussccd_bh_comp[~ab_is_duplicated])}: number of passing antibodies at {alpha_gauss*100}% FDR in compartment (no replicate)")
    duplicated_ab_ccd = np.array([sum(wp_pass_kruskal_gaussccd_bh_comp[wp_ab == ab]) for ab in duplicated_ab])
    print(f"{sum(duplicated_ab_ccd == 2)}: number of duplicated antibodies shown to be CCD in both replicates")
    print(f"{sum(duplicated_ab_ccd == 1)}: number of duplicated antibodies shown to be CCD in just one replicate")
    print(f"{sum(duplicated_ab_ccd == 0)}: number of duplicated antibodies shown to be non-CCD in both replicate")



my_df = read_raw_data()
plate, u_plate, well_plate, well_plate_imgnb, u_well_plates, ab_objnum, area_cell, area_nuc, area_cyto, ensg_dict, ab_dict, result_dict, compartment_dict, ENSG, antibody, result, compartment = read_sample_info(
    my_df
)
wp_ensg, wp_ab, wp_prev_ccd, wp_prev_notccd, wp_prev_negative, prev_ccd_ensg, prev_notccd_ensg, prev_negative_ensg = previous_results(
    u_well_plates, result_dict, ensg_dict, ab_dict
)


my_df_filtered = apply_manual_filtering(
    my_df, result_dict, ab_dict
)
my_df_filtered = apply_big_nucleus_filter(my_df_filtered)
my_df_filtered = apply_cell_count_filter(my_df_filtered)


plate, u_plate, well_plate, well_plate_imgnb, u_well_plates, ab_objnum, area_cell, area_nuc, area_cyto, ensg_dict, ab_dict, result_dict, compartment_dict, ENSG, antibody, result, compartment = read_sample_info(
    my_df_filtered
)
wp_ensg, wp_ab, wp_prev_ccd, wp_prev_notccd, wp_prev_negative, prev_ccd_ensg, prev_notccd_ensg, prev_negative_ensg = previous_results(
    u_well_plates, result_dict, ensg_dict, ab_dict
)

my_df_filtered_variation, my_df_filtered_novariation = apply_variation_filter(
    my_df_filtered, result_dict, my_df
)


plate, u_plate, well_plate, well_plate_imgnb, u_well_plates, ab_objnum, area_cell, area_nuc, area_cyto, ensg_dict, ab_dict, result_dict, compartment_dict, ENSG, antibody, result, compartment = read_sample_info(
    my_df_filtered_variation
)
wp_iscell, wp_isnuc, wp_iscyto, my_df_filtered_compartmentvariation = metacompartments(
    u_well_plates, compartment_dict, my_df_filtered_variation
)


plate, u_plate, well_plate, well_plate_imgnb, u_well_plates, ab_objnum, area_cell, area_nuc, area_cyto, ensg_dict, ab_dict, result_dict, compartment_dict, ENSG, antibody, result, compartment = read_sample_info(
    my_df_filtered_compartmentvariation
)
wp_iscell, wp_isnuc, wp_iscyto, my_df_filtered_compartmentvariation = metacompartments(
    u_well_plates, compartment_dict, my_df_filtered_compartmentvariation
)
wp_ensg, wp_ab, wp_prev_ccd, wp_prev_notccd, wp_prev_negative, prev_ccd_ensg, prev_notccd_ensg, prev_negative_ensg = previous_results(
    u_well_plates, result_dict, ensg_dict, ab_dict
)



ab_nuc, ab_cyto, ab_cell, mt_cell, green_fucci, red_fucci = read_sample_data(
    my_df_filtered_compartmentvariation
)
log_green_fucci, log_red_fucci, log_green_fucci_zeroc, log_red_fucci_zeroc, log_green_fucci_zeroc_rescale, log_red_fucci_zeroc_rescale, fucci_data = zero_center_fucci(
    green_fucci, red_fucci, u_plate, well_plate, plate
)

plt.hist2d(log_green_fucci_zeroc_rescale, log_red_fucci_zeroc_rescale, bins=200)
plt.xlabel("Log10 Green Fucci Intensity")
plt.ylabel("Log10 Red Fucci Intensity")
plt.savefig("figures/FucciPlotProteinIFData_unfiltered.png")
# plt.show()
plt.close()


sbn.displot(ab_cell, kind="hist")
plt.xlabel("Mean Intensity")
plt.ylabel("Density")
plt.savefig("figures/antibody_cell_intensity.pdf")
# plt.show()
plt.close()


g1_idx, sph_idx, g2_idx = 1, 2, 0
clusternames = [
    "G2" if g2_idx == 0 else "G1" if g1_idx == 0 else "S-ph",
    "G2" if g2_idx == 1 else "G1" if g1_idx == 1 else "S-ph",
    "G2" if g2_idx == 2 else "G1" if g1_idx == 2 else "S-ph",
]

cluster_labels = gaussian_clustering(
    log_green_fucci_zeroc_rescale, log_red_fucci_zeroc_rescale, clusternames
)



g1 = cluster_labels == g1_idx
sph = cluster_labels == sph_idx
g2 = cluster_labels == g2_idx
alpha_gauss, doGenerateBoxplotsPerGene = 0.05, False
wp_comp_kruskal_gaussccd_adj, wp_pass_kruskal_gaussccd_bh_comp, wp_mt_kruskal_gaussccd_adj, wp_pass_gaussccd_bh_mt = gaussian_clustering_analysis(
    alpha_gauss,
    doGenerateBoxplotsPerGene,
    g1,
    sph,
    g2,
    wp_ensg,
    well_plate,
    u_well_plates,
    ab_cell,
    ab_nuc,
    ab_cyto,
    mt_cell,
    wp_iscell,
    wp_isnuc,
    wp_iscyto,
)

address_replicates(
    alpha_gauss, wp_pass_kruskal_gaussccd_bh_comp, wp_ensg, wp_ab, u_well_plates
)