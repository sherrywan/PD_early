
###
 # @Author: sherrywaan sherrywaan@outlook.com
 # @Date: 2024-09-25 14:24:02
 # @LastEditors: sherrywaan sherrywaan@outlook.com
 # @LastEditTime: 2024-10-15 20:14:30
 # @FilePath: /wxy/PD/PD_early/pipeline.sh
 # @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
### 
# full modality
# train and test models for full modality
python code/full_modality_main.py

# analysis results (five metrics for four models and auc for ten folds)
python code/full_modality_results_analysis.py

# analysis |shap|-ratio of each modality for each model
python code/modality_shapley_analysis.py

# vis |shap|-ratio of each modality for each model
python code/modality_shapley_analysis.py

# sole modality
# train and test models for sole modality
python code/sole_modality_main.py

# analysis results (five metrics for four models and auc for ten folds)
python code/sole_modality_results_analysis.py

# analysis stability (cv in 10-fold auc for four models)
python code/modality_classification_performance.py

# combination modality
# train and test models for combination modality
python code/com_modality_main.py

# analysis results (five metrics for four models and four combinations)
python code/com_modality_results_analysis.py

# analysis |shap|-ratio of each modality for each model in com primary modality model
python code/com_modality_shapley_analysis.py

# vis |shap|-ratio of each modality for each model in com primary modality model
python code/com_modality_shapley_analysis.py
