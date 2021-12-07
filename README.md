# TCRBert
## Predicting SARS-CoV-2 epitope-specific TCR recognition using pre-trained protein embeddings
We developed a predictive model based on the pre-trained <a href='https://arxiv.org/abs/1906.08230'>Tasks Assessing Protein Embeddings (TAPE) model</a>[1], employing the self-supervised transfer learning, to predict SARS-CoV-2 T-cell epitope-specific TCR recognition. The predictive model was generated by fine-tuning the pre-trained TAPE model using epitope-specific TCR CDR3β sequence datasets in a progressively specialized manner. The fine-tuned model showed a markedly high predictive performance for two evaluation datasets containing the SARS-CoV-2 S-protein<sub>269-277</sub> epitope (<b>YLQPRTFLL</b>)-specific CDR3β sequences, and outperformed the recent Gaussian process-based model, TCRGP[2], for the ImmuneCODE dataset. In particular, the output attention weights of our model suggest that the proline at the position 4 (<b>P4</b>) in the epitope may contribute critically to TCR recognition of the epitope. The <b>P272L</b> mutation in SARS-Cov-2 S-protein, which is corresponding to the <b>P4</b>, is known to be highly relevant to the viral escape associated with the second pandemic wave in Europe.
A recent experimental study of SARS-CoV-2 variants has demonstrated that CD8+ T-cells failed to respond to the <b>P272L</b> variant[3]. Our attention-based approach, which can capture all motifs in both the epitope and CDR3β sequences in epitope-specific TCR recognition, can be more useful for predicting immunogenic changes in T-cell epitopes derived from SARS-CoV-2 mutations than MSA-based approaches which depend entirely on TCR sequences. We anticipate that our findings will provide new directions for constructing a reliable data-driven model for predicting the immunogenic T-cell epitopes using limited training data and help accelerate the development of an effective vaccine for the respond to SARS-CoV-2 variants, by identifying critical amino acid positions that are important in epitope-specific TCR recognition.

<hr>

### Publication
For full description of analysis and approach, refer to the following manuscript:

Han, Y. & Aeri, L. (2021). Predicting SARS-CoV-2 epitope-specific TCR recognition using pre-trained protein embeddings, _bioRxiv_ 2021.11.17.468929; doi: https://doi.org/10.1101/2021.11.17.468929

https://www.biorxiv.org/content/10.1101/2021.11.17.468929v1

<hr>

### Run the notebook for our works
#### Install requiremnets
In python version >= 3.7,
```bash
pip install torch
pip install tape_proteins

```
Run the jupyter notebook, <a href='notebook/exp1.ipynb'>notebook/exp1.ipynb</a>

<hr>

### Web server

<img src="webserver.png" width="50%">
<p>
The main web interface consists of the input form panel (left) and the result list panel (right). 
Users can submit multiple TCR CDR3β sequences and a specific epitope in the input form panel. 
Once the prediction process is completed, the user can see a list of the prediction results for the input CDR3β sequences grouped by sequence lengths in the result list panel. 
For each prediction results by CD3β sequence length, the user can also see the marginalized position-wise attention weights captured by our model for the epitope-specific CDR3β sequence pairs predicted as a binder via a pop-up panel
</p>

#### Run the web server
Just run www/main.py as follows:

```bash
cd www
python main.py
```
<hr>

### Reference
1. Rao, R. et al. Evaluating Protein Transfer Learning with TAPE. Advances in neural information processing systems 32, 9689–9701 (2019).
2. Jokinen, E., Huuhtanen, J., Mustjoki, S., Heinonen, M. & Lähdesmäki, H. Predicting recognition between T cell receptors and epitopes with TCRGP. PLoS Computational Biology 17, e1008814 (2021). 
3. Dolton, G. et al. Emergence of immune escape at dominant SARS-CoV-2 killer T-cell epitope. medRxiv doi:https://doi.org/10.1101/2021.06.21.21259010 (2021).
