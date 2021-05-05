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
I collected 120 twitters with the keyword "Vaccine", and manually label 20 as "Annoncement", "Positive", "Negative".
![train_set](https://github.com/Toransu/LING-539/blob/d6a3a7fd41f716cb0a7c2e8818dc8fc1232fbdc3/Screenshot%20from%202021-05-04%2017-49-44.png)

### Build the Multi-labeled function
```python
from snorkel.labeling import labeling_function

def lf_keyword_good(context):
    Positive_word = r"(good|great|well|(be vaccinated)|better)"
    return POSITIVE if re.search(Positive_word, context) else ABSTAIN
def lf_keyword_bad(context):
    Negative_word = r"(cancer|bad|(won't take)|traced|(won't work)|refuse|convinced)"
    return NEGATIVE if re.search(Negative_word, context) else ABSTAIN
def lf_keyword_annoncement(context):
    Anno_word = r"(clinic|volunteer|offically|(find a clinic))"
    return ANNO if re.search(Anno_word, context) else ABSTAIN
```



































