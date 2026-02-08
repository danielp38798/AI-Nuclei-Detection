import numpy as np
import pandas as pd
import csv
import os
from scipy import stats

 
class AutoStatistics:
    """
    Class to automatically calculate significance between two samples.
    The required statistical test is automatically chosen according to:
    https://statistik-und-beratung.de/2013/07/statistischer-vergleich-von-zwei-gruppen/
    
    """
    def __init__(self, verbose:bool=False):
        self.verbose = verbose
        self.saving_path = r"R:\10_Labs\BFM\20_Projekte\40_Promotionen\30_ECMO_Clotting_Kranz\33_Dissertation\TeX\data\chap3"
        
    def test_significance(self, data1:pd.DataFrame=None, 
                          data2:pd.DataFrame=None, paired=False)-> dict:
        """
        Test the significance between two samples from two DataFrames.
        
        Parameters:
        df1 (DataFrame): The first DataFrame containing the first sample data
        column1 (str): The name of the first sample column in df1
        df2 (DataFrame): The second DataFrame containing the second sample data
        column2 (str): The name of the second sample column in df2
        paired (bool): Whether the samples are paired or not
        
        Returns:
        dict: A dictionary containing the test results
        """
        
        # Extract samples from DataFrames
        sample1 = data1.dropna().values
        sample2 = data2.dropna().values
        
        
        # Test for normality
        self.pvalue_gauss1 = stats.shapiro(sample1).pvalue
        self.pvalue_gauss2 = stats.shapiro(sample2).pvalue
        
        self.normal1 = self.pvalue_gauss1 > 0.05
        self.normal2 = self.pvalue_gauss2 > 0.05
        
        self.median1 = np.median(sample1)
        self.median2 = np.median(sample2)
        
        self.mean1 = np.mean(sample1)
        self.mean2 = np.mean(sample2)
        
        self.std1 = np.std(sample1)
        self.std2 = np.std(sample2)
        
        self.ci_low1, self.ci_high1 = stats.norm.interval(confidence=0.95, 
                loc=np.mean(sample1), scale=stats.sem(sample1))
        self.ci_low2, self.ci_high2 = stats.norm.interval(confidence=0.95, 
                loc=np.mean(sample2), scale=stats.sem(sample1))
        
        # Initialize result dictionary
        self.result = {
            "pvalue_gauss1": f'{self.pvalue_gauss1:.3f}',
            "pvalue_gauss2": f'{self.pvalue_gauss2:.3f}',
            "normality_sample1": self.normal1,
            "normality_sample2": self.normal2,
            "mean1": f'{self.mean1:.3f}',
            "mean2": f'{self.mean2:.3f}',
            "std1": f'{self.std1:.3f}',
            "std2": f'{self.std2:.3f}',
            "median1": f'{self.median1:.3f}',
            "median2": f'{self.median2:.3f}',
            #"ci1": f'{self.ci_low1[0]:.2f}-{self.ci_high1[0]:.2f}',
            #"ci2": f'{self.ci_low2[0]:.2f}-{self.ci_high2[0]:.2f}',
            "test_used": None,
            "pvalue": None,
            "significant": None
        }
        
        # Choose the appropriate test based on normality and whether samples are paired
        if paired:
            if self.normal1 and self.normal2:
                # Paired t-test
                test_result = stats.ttest_rel(sample1, sample2)
                self.result["test_used"] = "Paired t"
            else:
                # Wilcoxon signed-rank test
                test_result = stats.wilcoxon(sample1, sample2)
                self.result["test_used"] = "Wilcoxon"
        else:
            if self.normal1 and self.normal2:
                # Levene test to check for equal variances
                pvalue_levene = stats.levene(sample1, sample2).pvalue
                equal_variances = pvalue_levene > 0.05
                if not equal_variances:
                    # Welch t-test
                    self.result["test_used"] = "Welch t"
                else: # Independent t-test 
                    self.result["test_used"] = "Ind t"
                # ttest_ind function automatically uses Welch or Independent t-test,
                # depending on value of equal_var
                test_result = stats.ttest_ind(sample1, sample2, 
                                              equal_var=equal_variances)
            else:
                # Mann-Whitney U test
                test_result = stats.mannwhitneyu(sample1, sample2)
                self.result["test_used"] = "MWU"
        
        # Populate result dictionary with test results
        self.result["pvalue"] = test_result.pvalue
        self.result["significant"] = test_result.pvalue < 0.05
        return self.result
    
    def getPvalue(self) -> float:
        return self.result['pvalue']
    
    def getSig(self) -> bool:
        return self.result['significant']
    
    def getTest(self) -> str:
        return self.result['test_used']
    
    def getGauss(self, dataset:int=0) -> [bool, float]:
        if dataset == 0:
            return [self.result['normality_sample1'], self.result['pvalue_gauss1']]
        elif dataset == 1:
            return [self.result['normality_sample2'], self.result['pvalue_gauss2']]
        else:
            raise Exception("No matching data set found")
            
    def getCi(self, dataset:int= 0, element:int=2):
        if dataset == 0:
            if element == 0:
                return self.ci_low1[0]
            elif element == 1:
                return self.ci_high1[0]
            elif element == 2:
                return (self.ci_low1[0], self.ci_high1[0])
            else:                 
                raise Exception(f"No matching element found with element={element}")
        if dataset == 1:
            if element == 0:
                return self.ci_low1[0]
            elif element == 1:
                return self.ci_high1[0]
            elif element == 2:
                return (self.ci_low1[0], self.ci_high1[0])
            else:
                raise Exception(f"No matching element found with element={element}")
        else:        
            raise Exception(f"No matching dataset found with dataset={dataset}")
            
        
    def save_statistics(self, filename:str='results.csv') -> None:
        saving_path = self.saving_path + os.sep + filename 
        with open(saving_path, 'w', newline='') as csvfile:
            # Erstelle ein csv-Schreiber-Objekt
            writer = csv.writer(csvfile)            
            # Schreibe die Kopfzeile
            writer.writerow(self.result.keys())
            # Schreibe die Daten
            writer.writerow(self.result.values())
            csvfile.close()
            if self.verbose:
                print(f'Csv-file {saving_path} saved.')
                
    def get_result(self) -> dict:
        return self.result




def main()-> None:
    df1 = pd.DataFrame([1, 2, 3, 4, 5, 6, 7])
    df2 = pd.DataFrame([23, 434, 67, 1, 12, 34, 56])
    ast = AutoStatistics()
    res = ast.test_significance(df1, df2, paired=True)
    pvalue = ast.getPvalue()
    test = ast.getTest()
    sig = ast.getSig()
    gauss1, pvalue_gauss1 = ast.getGauss(dataset=0)
    gauss2, pvalue_gauss2 = ast.getGauss(dataset=1)
    low, high =  ast.getCi(dataset=1, element=2)
 
    print(res)
    print(pvalue)
    print(test)
    print(sig)
    print(gauss1)
    print(pvalue_gauss1)
    print(gauss2)
    print(pvalue_gauss2)
    print(low)

    ast.save_statistics(filename='results.csv')

if __name__ == '__main__':
    main()