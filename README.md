# Non-Generative Textual Style Transfer
This is the implementation of my dissertation, written in part fulfilment for my bachelor's degree at the University of York in the UK. See [_report.pdf_](report.pdf) for the write-up. In short, I come up with a simple non-generative novel mechanism based on doc2vec that performs textual style transfer. The code published here comprises the content illustrated solely under section 2.4 within [_report.pdf_](report.pdf). Other content is not included.

See the table below, extracted from the report, for a demonstration of the capabilities of the developed method:
<p align="center">
    <img src="https://user-images.githubusercontent.com/17494044/58577561-a4719080-823e-11e9-9149-81df9e80563d.png">
</p>

Dependencies
---
All project dependencies are listed below:
<p align="center">
    <img src="https://user-images.githubusercontent.com/17494044/58575931-1051fa00-823b-11e9-8264-4c28daf6e3aa.PNG">
</p>
The code may run correctly if packages with more recent versions are installed, but no testing has been conducted to guarantee such.

Running
---
Firstly, it is necessary to download the political corpus from RtGender and place it in the _corpora/original_ folder. At the time of writing, this data can be downloaded from the following [link](http://tts.speech.cs.cmu.edu/style_models/political_data.tar). If the data has been appropriately placed in the right folder, the file structure of the _corpora_ folder should be the same as below:
<p align="center">
    <img src="https://user-images.githubusercontent.com/17494044/58576471-5491ca00-823c-11e9-89eb-d91749ab04bb.PNG">
</p>
Then, scripts may be run in the following order:

1. corpora_resplit.py
2. corpora_sanitise.py
3. classifier_baseline.py
4. doc2vec.py
5. style_transfer.py
6. style_transfer_appendix.py (optional)

This should correctly reproduce the main style transfer experiment, saving outputs to the _out_ folder. Raise an issue if some problem is encountered while trying to run the scripts.

Bibliography
---
S. Prabhumoye et al., “Style Transfer Through Back-Translation,” in Association for Computational Linguistics, 2018, pp. 866–876.

T. Shen et al., “Style Transfer from Non-Parallel Text by Cross-Alignment,” in Neural Information Processing Systems, 2017.

Q. Le and T. Mikolov, “Distributed Representations of Sentences and Documents,” in International Conference on Machine Learning, 2014.

R. Voigt et al., “RtGender: A Corpus for Studying Differential Responses to Gender,” in Language Resources and Evaluation Conference, 2018, pp. 2814–2820.

Y. Kim, “Convolutional Neural Networks for Sentence Classification,” in Empirical Methods in Natural Language Processing, 2014, pp. 1746–1751.

M. Rikters, “Impact of Corpora Quality on Neural Machine Translation,” in Baltic Human Language Technologies, 2018.

Citation
---
P. H. M. Wigderowitz, “Towards a More Accessible Style Transfer Mechanism,” BSci Thesis, Dept. of Comp. Science, Univ. of York, York, UK, 2019.

License
---
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
