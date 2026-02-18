# InfoBridge: Mutual Information estimation via Bridge Matching

This is the official PyTorch implementation of our ICLR 2026 paper
**“InfoBridge: Mutual Information estimation via Bridge Matching”**  
by **[Sergei Kholkin](https://scholar.google.com/citations?user=KwhztSMAAAAJ&hl=en), [Ivan Butakov](https://scholar.google.com/citations?user=AYr57xsAAAAJ&hl=en), [Evgeny Burnaev](https://scholar.google.com/citations?user=pCRdcOwAAAAJ&hl=ru), [Nikita Gushchin](https://scholar.google.com/citations?user=UaRTbNoAAAAJ&hl=ru), [Alexander Korotin](https://scholar.google.com/citations?user=1rIIvjAAAAAJ&hl=ru)**.  

Paper: https://openreview.net/forum?id=y8Kzu9SKpv

**Abstract**: Diffusion bridge models have recently become a powerful tool in the field of generative modeling. In this work, we leverage their power to address another important problem in machine learning and information theory, the estimation of the mutual information (MI) between two random variables. Neatly framing MI estimation as a domain transfer problem, we construct an unbiased estimator for data posing difficulties for conventional MI estimators. We showcase the performance of our estimator on three standard MI estimation benchmarks, i.e., low-dimensional, image-based and high MI, and on real-world data, i.e., protein language model embeddings.

---

## Installation

python=3.10

Install project requirements:

`pip install -r requirements.txt`

---

## Usage

### Initialize InfoBridge Pytorch Lightning Module

Create InfoBridge module:

```python
from info_bridge import InfoBridge

infobridge_model = InfoBridge(
    backbone=neural_network,
    eps=eps,
    lr=lr,
    base_mc_iters=1,
    default_batch_size=batch_size,
    ema_decay=ema_decay
).to(device)
```

Where 

- `backbone` is a suitable neural network (take a look at example notebooks) 
- `eps` is the volatility coefficient (eps=1 for image data and eps=0.01 for high MI bench) 
- `base_mc_iters` is the number of inner time samples in the MI estimation (high "base_mc_iters" = more stable MI estimate)
- `ema_decay` is the Exponential Moving Average decay coefficient

### Fit the InfoBridge

Then one can train the InfoBridge model given data:

```python
infobridge_model.fit(x0_train, x1_train, x0_eval, x1_eval, max_epochs=max_epochs, trainer_kwargs={"check_val_every_n_epoch": val_epoch_freq})
```

where `x0_train, x1_train, x0_eval, x1_eval`  the `torch.tensor` instances containing corresponding train or val datasamples from `x0` and `x1` random variables.

### Estimate the MI

Given validation dataset `x0_eval, x1_eval` one can estimate the MI:

```python
mi_est = infobridge_model.estimate_mi(x0_eval, x1_eval, mc_iters=mc_iters)
```

where `base_mc_iters` is the number of inner time samples in the MI estimation (high "mc_iters" = more stable MI estimate)

---

## Examples

To run both examples, it is necessary to install the [mutinfo library](https://github.com/VanessB/mutinfo)

### Image benchmark [Section 5.2](https://arxiv.org/abs/2502.01383)

Run `InfoBridge_images_example.ipynb` jupyter notebook.

### High MI benchmark [Section 5.4](https://arxiv.org/abs/2502.01383)

Run `InfoBridge_high_dim_example.ipynb` jupyter notebook.

---

## Citation 

```
@inproceedings{
kholkin2026infobridge,
title={InfoBridge: Mutual Information estimation via Bridge Matching},
author={Sergei Kholkin and Ivan Butakov and Evgeny Burnaev and Nikita Gushchin and Alexander Korotin},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=y8Kzu9SKpv}
}
```
