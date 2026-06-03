# DOVE — Results Report

_Auto-generated. 16 experiments._

## 1. Experiment Results

| # | Experiment | Test Acc | Top-3 Acc |
|---|-----------|----------|----------|

## 2. Top-10 Configurations

| name                               | backbone        | fusion          | head          |   test_accuracy |   top3_accuracy |
|:-----------------------------------|:----------------|:----------------|:--------------|----------------:|----------------:|
| efficientnet_b3_none_random_forest | efficientnet_b3 | none            | random_forest |        0.147864 |               0 |
| swin_t_none_random_forest          | swin_t          | none            | random_forest |        0.146769 |               0 |
| vgg19_none_random_forest           | vgg19           | none            | random_forest |        0.142388 |               0 |
| swin_t_none_naive_bayes            | swin_t          | none            | naive_bayes   |        0.139102 |               0 |
| efficientnet_b3_none_naive_bayes   | efficientnet_b3 | none            | naive_bayes   |        0.138007 |               0 |
| mobilenet_none_naive_bayes         | mobilenet       | none            | naive_bayes   |        0.13034  |               0 |
| mobilenet_none_random_forest       | mobilenet       | none            | random_forest |        0.128149 |               0 |
| vgg19_none_naive_bayes             | vgg19           | none            | naive_bayes   |        0.128149 |               0 |
| swin_t_cross_attention_mlp         | swin_t          | cross_attention | mlp           |      nan        |             nan |
| swin_t_cross_attention_linear      | swin_t          | cross_attention | linear        |      nan        |             nan |

## 3. Conclusions

Best configuration: `efficientnet_b3_none_random_forest` with **14.79%** test accuracy (top-3: 0.00%).
