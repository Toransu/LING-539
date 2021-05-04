## Technical Tutorial for Using Multiple Labels in Snorkel
<a> Yijie Chen </a>

### Introduction to Snorkel

As we know, one limitation of recent NLP tasks is the way to obtain **high-quality annotated** data. Therefore, researchers focus on different data augmentation functions trying to increase the size of labeled training sets by applying **class-preserving transformations** to create copies of labeled data points. In this tutorial, I want to show how to create multi-annotation with **Snorkel**, which directly **leverages and exploits SME domain knowledge** of transformation operations.

### Install the Snorkel
<summary> Follow the steps if you did not install the snorkel </summary>
<pre><code=python>
#[OPTIONAL] Activate a virtual environment
```python
conda create --yes -n spam python=3.6
conda activate spam
```
#Install requirements (both shared and tutorial-specific)
pip install environment_kernels
#We specify PyTorch here to ensure compatibility, but it may not be necessary.
conda install pytorch==1.1.0 -c pytorch
conda install snorkel==0.9.5 -c conda-forge
pip install -r spam/requirements.txt
#Launch the Jupyter notebook interface
jupyter notebook spam
</code></pre>





































