## Technical Tutorial for Using Multiple Labels in Snorkel
<a> Yijie Chen </a>

### Introduction to Snorkel

As we know, one limitation of recent NLP tasks is the way to obtain **high-quality annotated** data. Therefore, researchers focus on different data augmentation functions trying to increase the size of labeled training sets by applying **class-preserving transformations** to create copies of labeled data points. In this tutorial, I want to show how to create multi-annotation with **Snorkel**, which directly **leverages and exploits SME domain knowledge** of transformation operations.

### Install the Snorkel
<details>
<summary> Follow the steps if you did not install the snorkel </summary>
<pre><code>
### [OPTIONAL] Activate a virtual environment
conda create --yes -n spam python=3.6
conda activate spam
### Install requirements (both shared and tutorial-specific)
pip install environment_kernels
### We specify PyTorch here to ensure compatibility, but it may not be necessary.
conda install pytorch==1.1.0 -c pytorch
conda install snorkel==0.9.5 -c conda-forge
pip install -r spam/requirements.txt
### Launch the Jupyter notebook interface
jupyter notebook spam
</code></pre>
</details>

### Create the dataset
I collected 120 twitters with the keyword "Vaccine", and manually label it as "Annoncement", "Positive", "Negative"

### Build the Multi-labeled function
```python
from snorkel.labeling import labeling_function

@labeling_function()
def lf_keyword_good(x):
    positive_w = ["good","optimistc","great"]
    return POSITIVE if item in positive_w in x.text.lower() else ABSTAIN
@labeling_function()
def lf_keyword_bad(x):
    return NEGATIVE if "bad" in x.text.lower() else ABSTAIN
@labeling_function()
def lf_keyword_fair(x):
    return NEUTRAL if "fair" in x.text.lower() else ABSTAIN
```



































