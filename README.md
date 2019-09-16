Self-AVAE (Self-adversarial Variational Autoencoder)
=====

Implementation of 'Self-adversarial Variational Autoencoder with Gaussian Anomaly Prior Distribution for Anomaly Detection' (Self-AVAE) [<a href="https://github.com/YeongHyeon/CVAE-AnomalyDetection">Related repository</a>].

## Architecture
<div align="center">
  <img src="./figures/self-avae.png" width="500">  
  <p>Simplified Self-AVAE architecture.</p>
</div>

## Graph in TensorBoard
<div align="center">
  <img src="./figures/graph.png" width="800">  
  <p>Graph of Self-AVAE.</p>
</div>

## Results
<div align="center">
  <img src="./figures/restoring.png" width="800">  
  <p>Restoration result by Self-AVAE.</p>
</div>

<div align="center">
  <img src="./figures/test-box.png" width="350"><img src="./figures/histogram-test.png" width="390">
  <p>Box plot and histogram of restoration loss in test procedure.</p>
</div>

<div align="center">
  <img src="./figures/test-latent_z.png" width="350"><img src="./figures/test-latent_z_T.png" width="350">
  <p>Normal latent space, and transformed latent space (regared as abnormal).</p>
</div>

## Environment
* Python 3.7.4  
* Tensorflow 1.14.0  
* Numpy 1.17.1  
* Matplotlib 3.1.1  
* Scikit Learn (sklearn) 0.21.3  

## Reference
[1] Xuhong Wang, et al. (2019). <a href="https://arxiv.org/abs/1903.00904">Self-adversarial Variational Autoencoder with Gaussian Anomaly Prior Distribution for Anomaly Detection.</a>. arXiv preprint arXiv:1903.00904.
