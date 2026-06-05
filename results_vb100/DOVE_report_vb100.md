# DOVE — Results Report

_Auto-generated. 24 experiments._

## 1. Experiment Results

| # | Experiment | Test Acc | Top-3 Acc |
|---|-----------|----------|----------|
| 1 | `swin_t_cross_attention_mlp` | 0.9552 | 0.9919 |
| 2 | `swin_t_cross_attention_linear` | 0.9558 | 0.9882 |
| 3 | `swin_t_concat_mlp` | 0.9577 | 0.9894 |
| 4 | `swin_t_concat_linear` | 0.9565 | 0.9894 |
| 5 | `efficientnet_b3_cross_attention_mlp` | 0.9677 | 0.9944 |
| 6 | `efficientnet_b3_cross_attention_linear` | 0.9652 | 0.9913 |
| 7 | `efficientnet_b3_concat_mlp` | 0.9590 | 0.9919 |
| 8 | `efficientnet_b3_concat_linear` | 0.9565 | 0.9938 |
| 9 | `mobilenet_cross_attention_mlp` | 0.9515 | 0.9913 |
| 10 | `mobilenet_cross_attention_linear` | 0.9558 | 0.9869 |
| 11 | `mobilenet_concat_mlp` | 0.9552 | 0.9900 |
| 12 | `mobilenet_concat_linear` | 0.9565 | 0.9919 |
| 13 | `vgg19_cross_attention_mlp` | 0.9521 | 0.9932 |
| 14 | `vgg19_cross_attention_linear` | 0.9583 | 0.9894 |
| 15 | `vgg19_concat_mlp` | 0.9527 | 0.9888 |
| 16 | `vgg19_concat_linear` | 0.9565 | 0.9894 |
| 17 | `swin_t_none_random_forest` | 0.8551 | 0.0000 |
| 18 | `swin_t_none_naive_bayes` | 0.7345 | 0.0000 |
| 19 | `efficientnet_b3_none_random_forest` | 0.8420 | 0.0000 |
| 20 | `efficientnet_b3_none_naive_bayes` | 0.7220 | 0.0000 |
| 21 | `mobilenet_none_random_forest` | 0.8520 | 0.0000 |
| 22 | `mobilenet_none_naive_bayes` | 0.7195 | 0.0000 |
| 23 | `vgg19_none_random_forest` | 0.8495 | 0.0000 |
| 24 | `vgg19_none_naive_bayes` | 0.7307 | 0.0000 |

## 2. Top-10 Configurations

| name                                   | backbone        | fusion          | head   |   test_accuracy |   top3_accuracy |
|:---------------------------------------|:----------------|:----------------|:-------|----------------:|----------------:|
| efficientnet_b3_cross_attention_mlp    | efficientnet_b3 | cross_attention | mlp    |        0.967662 |        0.994403 |
| efficientnet_b3_cross_attention_linear | efficientnet_b3 | cross_attention | linear |        0.965174 |        0.991294 |
| efficientnet_b3_concat_mlp             | efficientnet_b3 | concat          | mlp    |        0.958955 |        0.991915 |
| vgg19_cross_attention_linear           | vgg19           | cross_attention | linear |        0.958333 |        0.989428 |
| swin_t_concat_mlp                      | swin_t          | concat          | mlp    |        0.957711 |        0.989428 |
| swin_t_concat_linear                   | swin_t          | concat          | linear |        0.956468 |        0.989428 |
| efficientnet_b3_concat_linear          | efficientnet_b3 | concat          | linear |        0.956468 |        0.993781 |
| mobilenet_concat_linear                | mobilenet       | concat          | linear |        0.956468 |        0.991915 |
| vgg19_concat_linear                    | vgg19           | concat          | linear |        0.956468 |        0.989428 |
| swin_t_cross_attention_linear          | swin_t          | cross_attention | linear |        0.955846 |        0.988184 |

## 3. Conclusions

Best configuration: `efficientnet_b3_cross_attention_mlp` with **96.77%** test accuracy (top-3: 99.44%).
