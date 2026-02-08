from gui.core.reporting.reporting import Reporting, generate_report_for_trial
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pprint import pprint as pp
import seaborn as sns

class StatisticalTest:

    def __init__(self, test_data_1, test_data_2):
        self.test_data_1 = test_data_1
        self.test_data_2 = test_data_2

    def analyze_data(self, data):

        print("\n------------Analyzing the data ---------------\n")

        print("\n------------Calculating the 95 % confidence interval for the data ---------------\n")
        # calculate the 95 % confidence interval for the data
        conf_interval = stats.norm.interval(confidence=0.95, loc=np.mean(data), scale=np.std(data))
        print(f"95 % confidence interval for the data: {conf_interval}")

        # calculate the descriptive statistics for the data
        print("\n------------Calculating the descriptive statistics for the data ---------------\n")
        mean = np.mean(data)
        std = np.std(data)
        median = np.median(data)
        variance = np.var(data)
        print(f"Mean of the data: {mean}")
        print(f"Standard deviation of the data: {std}")
        print(f"Median of the data: {median}")
        print(f"Variance of the data: {variance}")

        # detect outliers in the data
        print("\n------------Detecting outliers in the data ---------------\n")
        outliers = []
        for i in data:
            z = (i - mean) / std
            if z.any() > 3 or z.any() < -3:
                outliers.append(i)
        print(f"Outliers in the data: {outliers}")

        # test if the data is normally distributed
        print("\n------------Testing if the data is normally distributed ---------------\n")
        p_value = stats.shapiro(data)[1]
        print(f"p-value for the data: {p_value}")
        if p_value > 0.05:
            normal_dist = True
            print("The data is normally distributed")
        else:
            normal_dist = False
            print("The data is not normally distributed")

        # test if the data is paired
        print("\n------------Testing if the data is paired ---------------\n")
        p_value = stats.ttest_rel(data[0], data[1])[1]
        print(f"p-value for the data: {p_value}")
        if p_value > 0.05:
            paired = True
            print("The data is paired")
        else:
            paired = False
            print("The data is not paired")

        return mean, std, median, variance, outliers, normal_dist, paired


    def perform_statistical_test(self, data_expert, data_ai, connected, normally_distributed_expert, normally_distributed_ai):
        if connected:  # connected; there is a dependency between the data -> Paired t-test or Wilcoxon test
            if normally_distributed_expert and normally_distributed_ai:
                # Paired t-test
                t_stat, p_val = stats.ttest_rel(data_expert, data_ai)  # connected and normally distributed
                print("Paired t-test was performed.")
                print(f"T-statistic: {t_stat}, p-value: {p_val}")
                print("\n")
                test = "t-test_paired"
            else:
                # Wilcoxon test
                _, p_val = stats.wilcoxon(data_expert, data_ai)  # connected and not normally distributed
                print("The data is connected but not normally distributed. Wilcoxon test was performed.")
                print(f"p-value: {p_val}")
                print("\n")
                test = "Wilcoxon"
        else:  # unconnected; there is no dependency between the data -> t-test, Welch's test, or Mann-Whitney U test
            if not (normally_distributed_expert and normally_distributed_ai):
                # Mann-Whitney U test
                _, p_val = stats.mannwhitneyu(data_expert, data_ai)  # unconnected and not normally distributed
                print("The data is unconnected and not normally distributed. Mann-Whitney U test was performed.")
                print(f"p-value: {p_val}")
                print("\n")
                test = "Mann-Whitney"
            else:
                # Checking equality of variances using Levene's test
                w_stat, p_val_levene = stats.levene(data_expert, data_ai)  # H0: Variances are equal
                if p_val_levene > 0.05:  # p-value > 0.05: Variances are equal
                    # Variances are equal, t-test
                    t_stat, p_val = stats.ttest_ind(data_expert, data_ai, equal_var=True)  # t-test
                    print("Variances are equal. A t-test was performed.")
                    print(f"T-statistic: {t_stat}, p-value: {p_val}")
                    test = "t-test_ind"
                else:
                    # Variances are not equal, Welch's test
                    t_stat, p_val = stats.ttest_ind(data_expert, data_ai, equal_var=False)  # Welch's test
                    print("Variances are not equal. Welch's test was performed.")
                    print(f"T-statistic: {t_stat}, p-value: {p_val}")
                    test = "t-test_welch"

        # pass test string to statnot 
        return test
        


trial_dir = r"C:\Users\pod44433\Documents\Mask_R_CNN\AI_nuclei_detection\GUI\statistics_test"
reporting = Reporting(unit="mm", trial_dir=trial_dir)
ai_predicted_nuclei_counts_list = reporting.get_nuclei_counts()
ai_predicted_decond_nuclei_counts_list = reporting.get_decond_nuclei_counts()

# load the data
ai_data = [ai_predicted_decond_nuclei_counts_list, ai_predicted_nuclei_counts_list ]
# create dummy data for the expert

#expert_data = [np.random.randint(1, 100, 100), np.random.randint(1, 100, 100)]
excel_file = r"AI_validation.xlsx"
expert_df = pd.read_excel(excel_file)
decond_data_expert = expert_df['Anzahl decond.'].values.tolist()
nuclei_data_expert = expert_df['Anzahl cond.'].values.tolist()

expert_data = [decond_data_expert, nuclei_data_expert, ]
print(f"expert_data: {expert_data}")#




# check if arrays have the same length
if len(ai_data[0]) != len(expert_data[0]) or len(ai_data[1]) != len(expert_data[1]):
    print("The arrays do not have the same length.")
    print("The statistical test cannot be performed.")
    print(f"Length of AI data: {len(ai_data[0])}, {len(ai_data[1])}")
    print(f"Length of expert data: {len(expert_data[0])}, {len(expert_data[1])}")
    exit()

stat_test = StatisticalTest(ai_data, expert_data)    

# analyze the data
ai_mean, ai_std, ai_median, ai_variance, ai_outliers, ai_normal_dist, ai_paired = stat_test.analyze_data(ai_data)
expert_mean, expert_std, expert_median, expert_variance, expert_outliers, expert_normal_dist, expert_paired = stat_test.analyze_data(expert_data)

# perform statistical test
if expert_paired is True and expert_paired is True:
    connected = True
else:
    connected = False
test_to_perform = stat_test.perform_statistical_test(expert_data, ai_data, 
                         connected=connected, 
                         normally_distributed_expert=expert_normal_dist, 
                         normally_distributed_ai=ai_normal_dist)




# plot the data show two plots each for decond and cond for expert and AI data

from statannotations.Annotator import Annotator



import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



# create a dataframe
df = pd.DataFrame({'expert_decond': expert_data[0], 'expert_cond': expert_data[1],
             'ai_decond': ai_data[0], 'ai_cond': ai_data[1]})

# melt the dataframe as seaborn prefers long format; sort depending on the type decond or cond
df_melted = pd.melt(df, value_vars=['expert_decond', 'expert_cond', 'ai_decond', 'ai_cond'],
              var_name='cells', value_name='count')
df_melted['x'] = df_melted['cells'].apply(lambda x: 'expert' if 'expert' in x else 'ai')
df_melted['hue'] = df_melted['cells'].apply(lambda x: 'decond' if 'decond' in x else 'cond')

# define the order of hue and x
hue_order = ['expert', 'ai']
x_order = ['decond', 'cond']

# create a boxplot
plt.figure(figsize=(10, 6))

ax = sns.boxplot(x='hue', y='count', hue='x', data=df_melted, hue_order=hue_order, order=x_order)

# add statistical cells if necessary
pairs=[(("expert", "decond"), ("expert", "cond")),
           (("ai", "decond"), ("ai", "cond"))]
# add statistical cells
annot = Annotator(ax, x='x', y='count', hue='hue', data=df_melted,  # x is the annotation type, y is the cell count, hue is the cell type
       pairs=pairs,
       perform_stat_test=True, text_format='star', loc='inside', verbose=2)
annot.new_plot(ax, pairs, x='x', y='count', hue='hue', data=df_melted, order=hue_order, hue_order=x_order)
annot.configure(test='Mann-Whitney', verbose=2)
annot.apply_test()
annot.annotate()
plt.legend(title='Type')
plt.xlabel('Annotation Type')
plt.ylabel('Cell Count')
plt.title('Boxplot of expert and AI data')

plt.show()