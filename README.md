Here is the source code for the article entitled "*Informed sampling and recommendation of cycling routes: leveraging crowd-sourced trajectories with weighted-latent Dirichlet allocation*".

## Hardware Requirement

This code was executed on Lenovo Legion R9000P2021H. The processor is an AMD Ryzen 7 5800H with Radeon Graphics, 16 GB memory, and NVIDIA GeForce RTX 3060 Laptop GPU 6 GB. Given that the dataset used in this project is relatively small, we believe it can be run on a PC without a GPU as well.

## Python Version

For compatibility reasons, the Python version used for this project is Python 3.7.6, which you can download via this link: https://www.python.org/downloads/release/python-376/

## Environment Setup

1. **Create a virtual environment**:

   - you can use the existing ".venv" which we already set up, or delete it to create a new ".venv" using the command:
     ```
     python -m venv .venv or python3 -m venv .venv
     ```

2. **Activate the virtual environment**:

   - For Windows, activate using:
     ```
     .venv\Scripts\activate
     ```
   - For macOS or Linux, activate using:
     ```
     source .venv/bin/activate
     ```

3. **Install the required dependencies**:

   - To install dependencies listed in `requirements.txt`, use:
     ```
     pip install -r requirements.txt
     ```

   *Note: You can also use the shortcut "Ctrl + Shift + P" (or "Cmd + Shift + P" on macOS) in VSCode to set up the virtual environment and install the requirements.txt.*

## Running the Code

Run the `Weight_lda.py` script

## Data Input

- **Textual Corpus**: `data/Train_3features` contains the spatial context words.
- **Weights**: `data/Weight_3features` stores associated weights.
- **Parameters**: `setting.conf` stores various parameters.

## Data Processing

- **`dataprocess.py`**:
  - Reads `Train_3features` and `Weight_3features`.
  - Main function: Splits words and builds word indexes.

## Training the Model

- **`Weight_lda.py`**: Main file for training.

## Output

- **Model Files**:
  - `data/tmp/model_theta.dat`: Topic distributions of trajectories.
  - `data/tmp/model_phi.dat`: Word distributions of topics.
- **Excel Output**: `data/Topic_output/Topic_output.xlsx` saves topics and track-topic probabilities.

## Route Recommendation

- **`Recommend_Route_Sampling.py`**: Script for route recommendation.

## PCoA of Topics

- **`Topic_PCoA.py`**: Script for the Topics' distance visualization

## Visualize Topics

```
pic = pyLDAvis.prepare(lda.phi, lda.theta, dpre.doc_lengths, dpre.vacab,  dpre.term_freqs)
pyLDAvis.save_html(pic, 'lda_pass' + str(len(lda.phi)) + '.html')
```




