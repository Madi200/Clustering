
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from clustering import Clustering
from sklearn.cluster import KMeans
import logging
import traceback
from sklearn.preprocessing import MinMaxScaler

FORMAT = '%(asctime)s:%(name)s:%(levelname)s - %(message)s'
# "%d %b %Y %H:%M:%S"
logging.basicConfig(filename='sample.log',format=FORMAT, level=logging.DEBUG)

class Kmeans(Clustering):
    """Kmeans Clustering class.
    
    Parameters
    ----------
    df : dataframe
        The dataframe contains the data with any type of attributes.
        
    cluster_by : string
        The attribute/colm on the basis of which we want to cluster the data.
        
    k : int
        Number of clusters you want to create. In default it is set to None.
        
    metric_colms : list
        List of Numeric colms. These colms will be used for k-means.
           
    """
    def __init__(self, df, cluster_by, metric_colms, k=None):
        """Class attributes are defined here."""
        super().__init__(df, cluster_by, k=None)
        self.metric_colms = metric_colms
        
    def __remove_outliers(self):
        """Remove outliers from numeric colms."""
        logging.debug('Removing Outliers')
        super().remove_outliers(self.metric_colms)
        
    def __variables_scaling(self, scaling_type=None):
        """Scale the variables baesd on the provided scaling type."""
        super().metric_variables_scaling(self.metric_colms, scaling_type)
        
    def __find_the_best_k(self):
        """Search the best number for k-means."""
        k, distortions, ranges = super().find_the_best_k()
        return k, distortions, ranges

    def perform_clustering(self, scaling_type=None, max_iterations=None, k=None,user=None,file_name=None):
        """Main method that handles the whole clustering process.
        
        Steps
        -----
        * Choose the attribute on which you want to cluster the data
        * Remove Outliers
        * Normalize/Scale the metric colms
        * Search for best k for k-means
        * Apply sklearn's KMeans() method using the provided k and max_iterations
        * Assign Clusters to the data points
        * Shape the output for Front-End
        """
        result ={}
        s = f'scaling_type: {scaling_type} | max_iteratoins: {max_iterations} | k: {k}'
        print(s)
        try:
            self.data_obj.df.drop_duplicates(subset=self.cluster_by, keep='last', inplace=True)
            self.duplicate_cases = self.total_cases - self.data_obj.df.shape[0]

            self.data_obj.df[self.cluster_by] = self.data_obj.df[self.cluster_by].astype(str)
            self.data_obj.df[self.metric_colms] = self.data_obj.df[self.metric_colms].astype(float)

            TOTAL_STEP = 3
            CURRENT_STEP = 1
            if scaling_type:
                TOTAL_STEP = TOTAL_STEP + 1 

            self.__remove_outliers()

            CURRENT_STEP = CURRENT_STEP + 1

            if scaling_type:
                self.__variables_scaling(scaling_type)

                CURRENT_STEP = CURRENT_STEP + 1

            else:
                self.processing_df = self.data_obj.df[self.metric_colms]
                self.processing_df.columns = self.metric_colms
            
            distortions = ranges = None
            
            if k is not None and k > self.processing_df.shape[0]:
                result['data'] = None
                result['msg'] =  f'{k} is greater than total valid cases {self.processing_df.shape[0]}'
                result['success'] = False
                logging.debug(f'{k} is greater than total valid cases {self.processing_df.shape[0]}')

                return result

            best_k = None
            if k is not None:
                best_k, distortions, ranges = self.__find_the_best_k()

            if k is None:
                k, distortions, ranges = self.__find_the_best_k()
                best_k = k

            CURRENT_STEP = CURRENT_STEP + 1

            if max_iterations is None:
                max_iterations = 51

            logging.debug(f'Going to run Kmeans with k:{k} and iteration:{max_iterations}')
            kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=max_iterations)

            logging.debug('Done with Clustering')

            
            if hasattr(kmeans, 'labels_'):
                y_pred = kmeans.labels_.astype(int)
            else:
                y_pred =  kmeans.fit_predict(self.processing_df)
                

            self.data_obj.df.loc[:,'Cluster_Id'] = y_pred
            #self.processing_df['Cluster_Id'] = y_pred
            # 
            feat_means_df = self.data_obj.df.groupby(['Cluster_Id'])[self.metric_colms].agg(['mean'])
        
            scaling = MinMaxScaler(feature_range=(0, 5))      ## limit defined
            feat_means = scaling.fit_transform(feat_means_df)
            spider_data = feat_means.tolist()

            logging.debug('Setting Up the Output Structure')

            result ={            
                    'data' : {
                    'per_cluster_count':self.data_obj.df.groupby(['Cluster_Id']).size().to_dict(),
                    'cluster_statistics': {'total_cases':self.total_cases, 'valid_cases':self.valid_cases,
                                        'missing_cases': self.outlier_cases,
                                        'duplicate_cases': self.duplicate_cases},
                    'cluster_list': list(self.data_obj.df['Cluster_Id'].unique()),
                    'elbow_chart': {'x':ranges, 'y':distortions, 'best_k':best_k},
                    'spider': spider_data,
                    'spider_columns' : list(feat_means_df.columns.levels[0])
                    },
                
                'success':True,
                'msg':'CLUSTERS CREATED'
            }

        except Exception as e:
            result['data'] = None
            result['msg'] =  f'{e.__class__} occurred! \n {traceback.print_exc()}'
            result['success'] = False
            logging.debug(f'Error: {traceback.print_exc()}')
            return result


        return result

