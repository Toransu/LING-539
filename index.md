## Technical Tutorial for GAN-BERT 
<a> Yijie Chen </a>

### Introduction of GAN-BERT

As we know, one limitation of recent NLP tasks is the way to obtain **high-quality annotated** data. Therefore, researchers focus on different data augmentation functions trying to increase the size of labeled training sets by applying **class-preserving transformations** to create copies of labeled data points. In this tutorial, I want to cover 2 data augmnetation algothrims, the **GAN-BERT** which mainly focuses on GAN, and then compare it with **Snorkel**, which directly **leverages and exploits SME domain knowledge** of transformation operations.

>GAN-BERT is an extension of the BERT model within the Generative Adversarial Network (GAN) framework (Goodfellow et al, 2014). In particular, the Semi-Supervised GAN (Salimans et al, 2016) is used to make the BERT fine-tuning robust in such training scenarios where obtaining annotated material is problematic.
<p align = "center">
<img src = "https://raw.githubusercontent.com/crux82/ganbert/master/ganbert.jpg">
</p>






































