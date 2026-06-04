# DOVE — Results Report

_Auto-generated. 16 experiments._

## 1. Experiment Results

| # | Experiment | Test Acc | Top-3 Acc |
|---|-----------|----------|----------|
| 1 | `swin_t_cross_attention_mlp` | 0.9321 | 0.9847 |
| 2 | `swin_t_cross_attention_linear` | 0.9343 | 0.9858 |
| 3 | `swin_t_concat_mlp` | 0.9288 | 0.9869 |
| 4 | `swin_t_concat_linear` | 0.9299 | 0.9836 |
| 5 | `efficientnet_b3_cross_attention_mlp` | 0.9244 | 0.9792 |
| 6 | `efficientnet_b3_cross_attention_linear` | 0.9409 | 0.9825 |
| 7 | `efficientnet_b3_concat_mlp` | 0.9255 | 0.9814 |
| 8 | `efficientnet_b3_concat_linear` | 0.9376 | 0.9880 |
| 9 | `mobilenet_cross_attention_mlp` | 0.9376 | 0.9836 |
| 10 | `mobilenet_cross_attention_linear` | 0.9409 | 0.9858 |
| 11 | `mobilenet_concat_mlp` | 0.9376 | 0.9814 |
| 12 | `mobilenet_concat_linear` | 0.9343 | 0.9901 |
| 13 | `vgg19_cross_attention_mlp` | 0.9310 | 0.9869 |
| 14 | `vgg19_cross_attention_linear` | 0.9387 | 0.9858 |
| 15 | `vgg19_concat_mlp` | 0.9299 | 0.9825 |
| 16 | `vgg19_concat_linear` | 0.9299 | 0.9880 |

## 2. Top-10 Configurations

| name                                   | backbone        | fusion          | head   |   test_accuracy |   top3_accuracy |
|:---------------------------------------|:----------------|:----------------|:-------|----------------:|----------------:|
| efficientnet_b3_cross_attention_linear | efficientnet_b3 | cross_attention | linear |        0.940854 |        0.982475 |
| mobilenet_cross_attention_linear       | mobilenet       | cross_attention | linear |        0.940854 |        0.985761 |
| vgg19_cross_attention_linear           | vgg19           | cross_attention | linear |        0.938664 |        0.985761 |
| efficientnet_b3_concat_linear          | efficientnet_b3 | concat          | linear |        0.937568 |        0.987952 |
| mobilenet_cross_attention_mlp          | mobilenet       | cross_attention | mlp    |        0.937568 |        0.983571 |
| mobilenet_concat_mlp                   | mobilenet       | concat          | mlp    |        0.937568 |        0.98138  |
| swin_t_cross_attention_linear          | swin_t          | cross_attention | linear |        0.934283 |        0.985761 |
| mobilenet_concat_linear                | mobilenet       | concat          | linear |        0.934283 |        0.990142 |
| swin_t_cross_attention_mlp             | swin_t          | cross_attention | mlp    |        0.932092 |        0.984666 |
| vgg19_cross_attention_mlp              | vgg19           | cross_attention | mlp    |        0.930997 |        0.986857 |

## 3. Conclusions

Best configuration: `efficientnet_b3_cross_attention_linear` with **94.09%** test accuracy (top-3: 98.25%).
