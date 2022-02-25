
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from kneed import KneeLocator
from sklearn.metrics import silhouette_score, calinski_harabasz_score,davies_bouldin_score
import logging

class Clustering:
    """Generic Clustering class.
    
    Parameters
    ----------
    data_obj : DataDf class object
        The DataDf class object to hold the dataframe.
        
    cluster_by : string
        The attribute/colm on the basis of which we want to cluster the data.
        
    k : int
        Number of clusters you want to create. In default it is set to None.
           
    """

    def __init__(self, data_obj, cluster_by ,k=None):
        """Attributes are defined here."""
        self.data_obj = data_obj
        self.processing_df = None
        self.no_of_clusters = k
        self.cluster_by = cluster_by
        self.total_cases = data_obj.df.shape[0]
        self.valid_cases = None
        self.outlier_cases = None
        self.duplicate_cases = None
        self.temp = None

    def remove_outliers(self, metrics_colms):
        """Remove outliers from numeric data colms."""
        logging.debug('In remove_outliers')
        for colm in metrics_colms:
            Q1 = self.data_obj.df[colm].quantile(0.25)
            Q3 = self.data_obj.df[colm].quantile(0.75)
            IQR = Q3 - Q1
            self.data_obj.df = self.data_obj.df[(self.data_obj.df[colm] >= Q1 - 1.5 * IQR) & 
                                      (self.data_obj.df[colm] <= Q3 + 1.5 * IQR)]
        
        self.valid_cases = self.data_obj.df.shape[0]
        self.outlier_cases = self.total_cases-self.duplicate_cases-self.data_obj.df.shape[0]
        
    def metric_variables_scaling(self, metrics_colms, scaling_type=None):
        """Scale numeric colms based on scaling type provided."""
        logging.debug('In metric_variables_scaling')
        scaler = None
        if scaling_type:
            if scaling_type == 'min_max':
                print('Scaling:','\n MIN_MAX')
                scaler = MinMaxScaler()
            elif scaling_type == 'absolute':
                print('Scaling:','\n ABSOLUTE')
                scaler = MaxAbsScaler()      
            else:
                print('Scaling:','\n Z-scaling')
                #0 mean and unit variance
                scaler = StandardScaler()
                
            self.processing_df = scaler.fit_transform(self.data_obj.df[metrics_colms])
            self.processing_df = pd.DataFrame(self.processing_df)
            self.processing_df.columns = metrics_colms
            self.valid_cases = self.processing_df.shape[0] 
        else:
            self.processing_df = self.data_obj.df[metrics_colms]
            self.processing_df.columns = metrics_colms
            self.valid_cases = self.processing_df.shape[0] 
               
        return self.processing_df
    
    def find_the_best_k(self):
        """Find the best number of cluster that ensure optimal similarity between same cluster and 
        high difference between differetn cluster instances.
        
        Return
        ------
        k : int
            Best value for num of cluster for the provided data.
            
        distortions : list [float]
            Score calculated for each cluster. The higher the better. Required for elbow chart.
        
        ranges : list [int]
            Range in which we search for best cluster. Required for elbow chart.
        """
        logging.debug('In find_the_best_k')
        ranges = list(range(2,11))
        distortions = []
        cali_score = []
        devis=[]
        inertias = []
        print('In find_the_best_k() \n')

        for n_clusters in ranges:
            kmeanModel = KMeans(n_clusters=n_clusters, init='k-means++').fit(self.processing_df)
            preds = kmeanModel.predict(self.processing_df)
            label_unique_count = np.unique(preds)

            if self.processing_df.shape[0]-1 < len(label_unique_count):
                break
            
            cali_score.append(calinski_harabasz_score(self.processing_df, preds))
            distortions.append(sum(np.min(cdist(self.processing_df, kmeanModel.cluster_centers_,
                                                'euclidean'), axis=1)) / self.processing_df.shape[0])


        kl = KneeLocator(ranges, distortions, curve="convex", direction="decreasing")
        k = kl.knee
        self.no_of_clusters = k
        
        logging.debug(f' best k: {k}')

        return k, distortions, ranges
