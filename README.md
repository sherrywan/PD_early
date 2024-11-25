<!--
 * @Author: sherrywaan sherrywaan@outlook.com
 * @Date: 2024-11-25 15:43:58
 * @LastEditors: sherrywaan sherrywaan@outlook.com
 * @LastEditTime: 2024-11-25 16:01:24
 * @FilePath: /wxy/PD/PD_early/README.md
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
-->
# [Development and Validation of the Multi-modality AI Diagnostic Model for Early Parkinson’s Disease]

### Project Description

Goal: To distiguish early-stage PDs from HCs with motor features extracted from our multi-modality AI system.

### Environment

You can create the environment via:

```bash
pip install -r requirements.txt
```

### Datas

The data includes:
1. motor features (extracted from motor recordings captured by our multi-modality AI system)
2. label (the state of the sample: PD or HC)
3. parameters (the PDBs which is more sensitive among PD related features, identified by statistical analysis)

Note: To maintain the privacy of participants, we currently do not publish all data. The datas would be available from the corresponding author on reasonable request. Data may only be used for research purposes.

### Pipeline

-   To train, analysize, and test models, run the bash in sequence:

```bash
bash pipeline.sh
```

