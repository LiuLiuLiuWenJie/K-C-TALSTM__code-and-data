### Data Availability

【Data.xlsx】contains 109 climate phenomena indices and raw monthly runoff data of four hydrological stations for model training and testing.

【Results_for_kmeans.xlsx】It contains the top four most important  climate phenomena indices and corresponding measured runoff data from four hydrological stations screened by MIC method.

### Software Availability

【MIC.py】It is the MIC algorithm, which is used to calculate the MIC coefficient between 109 climate phenomenon indexes and the measured runoff. According to this algorithm, the most important 4 climate phenomenon indexes are screened, and combined with the measured runoff as the most important 5 influencing factors. The calculated results are shown in 【Results_for_kmeans.xlsx】.

【CSA-Kmeans.py】It is an improved kmeans algorithm using CSA to optimize the initial clustering centers, which is used to cluster the measured runoff and obtain cluster labels under different cluster numbers [3,4,5].

【K-means_CSA_LSTM_Attention.py】This is the core model K-C-TALSTM in this paper. First, model input-output sets under different clustering labels are constructed, and then the coupled cooperative search algorithm and the long and short memory nerve of the triple attention mechanism are used for model training and model test, and the prediction results under different clustering labels are obtained.

【Statistical_index_calculation.py】The purpose of this program is to summarize the forecast results under the specified number of cluster K and calculate statistical indicators.