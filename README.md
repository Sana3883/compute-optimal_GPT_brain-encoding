# Computing Optimal Generative Models for Brain Encoding
For the brain encoding task, subjects played the Shinobi video game while corresponding video data was recorded. We utilized a self-supervised generative model (GPT-2) to extract efficient spatiotemporal features from these videos. During the training process, the model aimed to predict the next sequence of frames based on the previous sequence of frames. Subsequently, we trained a Ridge regression model to predict brain activity based on these extracted features. 



## Brain Encoding Maps for Different Scenarios of Data Size and Model Parameters



### Data size
<table>
<tr>
<td><img src="R_predicted frames/data_size.png" alt="Data Size" width="400"/></td>
<td><img src="R_predicted frames/data_size_1.png" alt="Data Size 1" width="400"/></td>
</tr>
</table>



### Hidden Layers Dimension in GPT Architecture
<table>
<tr>
<td><img src="R_predicted frames/hidd_dim%20(1).png" alt="Hidden Dimension" width="400"/></td>
<td><img src="R_predicted frames/1_hidd.png" alt="1 Hidden Dimension" width="400"/></td>
</tr>
</table>

### Number of Layers  in GPT Architecture
<table>
<tr>
<td><img src="R_predicted frames/num_layers%20(1).png" alt="Number of Layers" width="400"/></td>
<td><img src="R_predicted frames/num_layers_1.png" alt="Number of Layers 1" width="400"/></td>
</tr>
</table>


### Number of Heads in the Multi-Head Attention Mechanism 
<table>
<tr>
<td><img src="R_predicted frames/head_num%20(2).png" alt="Head Number" width="400"/></td>
<td><img src="R_predicted frames/num_heads.png" alt="Number of Heads" width="400"/></td>
</tr>
</table>


###  High performance computing
<table>
<tr>
<td><img src="R_predicted frames/Floating_Point_Precision%20(1).png" alt="Floating Point Precision" width="400"/></td>
<td><img src="R_predicted frames/1FP.png" alt="1FP" width="400"/></td>
</tr>
</table>


---

# Model Training Details

## Model Sizes

### (a) GPT-2 Model Size

| Name             | Type              | Parameters |
|------------------|-------------------|------------|
| vqvae            | VQ-VAE            | 35.8 M     |
| resnet           | ResidualBlock     | 2.4 M      |
| fc_in            | Linear            | 73.7 K     |
| attn_stack       | AttentionStack    | 37.2 M     |
| fc_out           | Linear            | 589 K      |
| **Trainable params**  |                   | **40.3 M** |
| **Non-trainable params** |               | **35.8 M** |
| **Total params**      |                   | **76.2 M** |
| **Total model params size (MB)** |        | **304.606** |


### (b) VQVAE Model Size

| Name             | Type              | Parameters |
|------------------|-------------------|------------|
| encoder           | encoder           | 18.7 M     |
| decoder           | decoder           | 17.1 M     |
| pre_vq_conv       | SamePadConv3d     | 30.8 K     |
| post_vq_conv      | SamePadConv3d     | 31.0 K     |
| codebook          | Codebook          | 0          |
| **Trainable params**  |                   | **35.8 M** |
| **Non-trainable params** |               | **0**      |
| **Total params**      |                   | **35.8 M** |
| **Total model params size (MB)** |        | **143.306** |

### (c) Hyperparameters for VQVAE and GPT-2 components

| Parameter       | VQVAE            | GPT-2     |
|-----------------|-------------------|----------|
| embedding_dim   | 256               | -        |
| n_codes         | 2048              | -        |
| n_hiddens       | 240               | -        |
| n_res_layers    | 4                 | -        |
| downsample      | (4, 4, 4)         | -        |
| heads           | 2                 | 4        |
| hidden_dim      | -                 | 576      |
| layers          | -                 | 8        |
| dropout         | -                 | 0.2      |
| attn_dropout    | -                 | 0.3      |



The architecture of the VideoGPT model is an adaptation of VQ-VAE  and GPT-2 architectures. In the first phase of VideoGPT, we train VQ-VAE to reconstruct 16 sequences of frames. In the second phase, we train GPT to predict the next 16 sequence of frames based on the previous 16 sequences of frames. In the second phased, we use VQ-VAE as a pretrained network to represent sequences of frames with the codebook as the input for GPT. 

In the first phased of Video GPT, a set of discrete latent codes will be trained for the Shinobi dataset through the VQ-VAE, in effect downsampling windows of video frames (sequence length of 16) into a discrete space-time codebook. The encoder of the VQ-VAE will include a series of 3D convolutions followed by attention residual blocks (a replacement for standard residual blocks) to better capture complex spatiotemporal dependencies within video data. In this architecture,  each attention residual block includes Convolution, LayerNorm, position embedding and axial attention layers. The position embedding is shared between all axial attention layers in the encoder and decoder.  The architecture of the decoder starts with attention residual blocks which are then followed by a series of 3D transposed convolution (reverse of encoder) to upsample the video frames across space-time. 

**Multi-Head Attention**: The multi-head attention mechanism uses the following formula to calculate the attention for each head (i):
```latex
\text{Attention}_i(X) = \text{softmax}\left(\frac{Q_i K_i^T}{\sqrt{d_k}}\right) V_i
```

The final multi-head attention result is:
```latex
\text{MultiHeadAttention}(X) = \text{Concat}(\text{Attention}_1(X), \ldots, \text{Attention}_H(X))W^O
```

**Axial Attention**: To process high-dimensional video data, attention is computed along each dimension (width, height, and temporal). The equations for each attention are as follows:

- Width attention:
```latex
\text{Attention}_{\text{width}}(X) = \text{softmax}\left(\frac{Q_{\text{width}} K_{\text{width}}^T}{\sqrt{d_k}}\right) V_{\text{width}}
```

- Height attention:
```latex
\text{Attention}_{\text{height}}(X) = \text{softmax}\left(\frac{Q_{\text{height}} K_{\text{height}}^T}{\sqrt{d_k}}\right) V_{\text{height}}
```

- Temporal attention:
```latex
\text{Attention}_{\text{temporal}}(X) = \text{softmax}\left(\frac{Q_{\text{temporal}} K_{\text{temporal}}^T}{\sqrt{d_k}}\right) V_{\text{temporal}}
```

These attentions are combined into:
```latex
\text{Attention}_{\text{combined}}(X) = \text{Attention}_{\text{width}}(X) + \text{Attention}_{\text{height}}(X) + \text{Attention}_{\text{temporal}}(X)
```

**Loss Function**: The VQ-VAE model uses the following loss function:
```latex
L = |x - D(e)|_2^2 + |sg[E(x)] - e|_2^2 + \beta |sg[e] - E(x)|_2^2
```

This loss consists of three terms: reconstruction loss, codebook loss, and commitment loss. The stop gradient operation (sg) ensures that certain terms are excluded from backpropagation.

**Optimizer**: The Adam optimizer is used with the following learning rate schedule:
```latex
\eta_t = \eta_{\text{min}} + \frac{1}{2} (\eta_{\text{max}} - \eta_{\text{min}}) \left(1 + \cos\left(\frac{t}{T_{\text{max}}}\pi\right)\right)
```

The codebook update mechanism follows an exponential moving average (EMA):
```latex
\text{N} \leftarrow 0.99 \times \text{N} + 0.01 \times \text{n_total}
```

<div style="display: flex; gap: 10px;">
  <img src="R_predicted frames/1.gif" width="450" height="450" alt="GIF 1">
  <img src="R_predicted frames/2.gif" width="450" height="450" alt="GIF 2">
  <img src="R_predicted frames/3.gif" width="450" height="450" alt="GIF 3">
<img src="R_predicted frames/4.gif" width="450" height="450" alt="GIF 1">
  <img src="R_predicted frames/5.gif" width="450" height="450" alt="GIF 2">
  <img src="R_predicted frames/6.gif" width="450" height="450" alt="GIF 3">
<img src="R_predicted frames/7.gif" width="450" height="450" alt="GIF 1">
  <img src="R_predicted frames/8.gif" width="450" height="450" alt="GIF 2">
 
</div>



