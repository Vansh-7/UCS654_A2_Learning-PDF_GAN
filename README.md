# Learning Probability Density Functions using GANs

**Assignment-2: Generative Adversarial Networks**

* **Student Roll Number:** 102483084
* **Dataset:** Air Quality Data (NO2 concentration)

---

## 1. Objective

The goal of this assignment is to learn the unknown Probability Density Function (PDF) of a transformed random variable  using a Generative Adversarial Network (GAN). The model is trained purely on data samples without assuming any parametric form (like Gaussian or Exponential).

---

## 2. Methodology

### A. Data Transformation

The input feature  (NO2 concentration) was transformed into a new variable  using a specific transformation function derived from the university roll number.

**Parameter Calculation:**

**Roll Number (r):** 102483084

*$a_r$** Calculation:**
$$r \pmod 7 = 4 \implies a_r = 0.5 \times 4 = \mathbf{2.0}$$

*$b_r$** Calculation:**
$$r \pmod 5 = 4 \implies b_r = 0.3 \times (4 + 1) = \mathbf{1.5}$$


**Transformation Equation:**
$$z = x + 2.0 \cdot \sin(1.5 \cdot x)$$

### B. Preprocessing

1. **Cleaning:** Missing values in the `no2` column were dropped.
2. **Transformation:** The equation above was applied to the raw data.
3. **Scaling:** The transformed data  was normalized using `StandardScaler` to have zero mean and unit variance. This is crucial for stabilizing GAN training.

### C. GAN Architecture

A standard vanilla GAN architecture was implemented using PyTorch:

* **Generator ():**
* Input: Random Gaussian Noise vector (Latent Dimension = 5).
* Hidden Layers: Two dense layers (32, 64 units) with `LeakyReLU` activation.
* Output: Single continuous value ().


* **Discriminator ():**
* Input: Single value ().
* Hidden Layers: Two dense layers (64, 32 units) with `LeakyReLU` activation.
* Output: Single probability score (Sigmoid activation) indicating if the sample is Real or Fake.



### D. Training Configuration

* **Loss Function:** Binary Cross Entropy (BCELoss).
* **Optimizer:** Adam (, ).
* **Epochs:** 100 (sufficient for 1D convergence).
* **Batch Size:** 64.

---

## 3. Results

### Summary Table

| Parameter | Value / Description |
| --- | --- |
| **Transformation ** | 2.0 |
| **Transformation ** | 1.5 |
| **Latent Dimension** | 5 |
| **Final Generator Loss** | ~0.69 (Oscillating) |
| **Final Discriminator Loss** | ~1.38 (Oscillating) |
| **Observation** | Successful convergence (Nash Equilibrium) |

### Visualizations

#### Figure 1: Learned PDF vs. Real Distribution

The plot below compares the Kernel Density Estimation (KDE) and Histogram of the real transformed data (Blue) versus the GAN-generated data (Red).

*![Learned PDF Graph](gan_pdf_result.png)*

> **Visual Analysis:** The overlapping histograms demonstrate that the GAN has successfully approximated the underlying distribution of the transformed variable .

---

## 4. Observations

Based on the generated samples and training logs, the following observations were made:

1. **Mode Coverage:**
The Generator successfully captured the primary modes (peaks) of the distribution. The high-density regions in the real data (visible in the histogram) are well-represented by the generated samples, indicating that the model did not suffer from major mode collapse.
2. **Training Stability:**
The loss graph shows that neither the Generator nor the Discriminator loss collapsed to zero or diverged to infinity. Instead, they oscillated within a stable range. This indicates a healthy "minimax" game where both networks improved iteratively.
3. **Quality of Generated Distribution:**
The Kernel Density Estimation (KDE) of the generated samples () closely aligns with the real data (). While minor noise exists in the low-density tail regions, the overall shape and spread of the probability density function were learned effectively.
