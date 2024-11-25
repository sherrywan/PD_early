'''
Author: sherrywaan sherrywaan@outlook.com
Date: 2024-11-06 15:43:38
LastEditors: sherrywaan sherrywaan@outlook.com
LastEditTime: 2024-11-20 18:44:34
FilePath: /wxy/PD/PD_early/code/CI_0.95_of_mean.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import numpy as np

def CI95(mean, SD):
    mean_score = mean
    std_score = SD / np.sqrt(10)  # 因为是10折交叉  
    # t分布的自由度为n_splits - 1  
    t_value = 1.96  # 95%置信区间对应的双尾t分布的临界值，可以使用scipy.stats.t.ppf(0.975, df=len(scores)-1)来计算  
    confidence_interval = (mean_score - (t_value * std_score), mean_score + (t_value * std_score))  
    print('95%置信区间:', confidence_interval)

if __name__ == "__main__":
    mean = 0.8385225885225885
    SD = 0.087646181599007
    CI95(mean, SD)