# About Keyword Extraction

In the era of information overload it is essential to efficiently extract concise, precise and quality information from large texts. One aspect of information extraction is keyword extraction
where large texts are represented as sets of keywords. This prospect of keyword extraction is
paramount to researchers as they deal with huge numbers of scientific papers, and having a
good and concise representation of those papers is essential for them. This thesis addresses that
problem in the realm of Natural Language Processing (NLP).
Using core NLP concepts and modeling texts as graphs, we are going to build a model for the
automatic extraction of keywords. This is done in an unsupervised manner as the importance of
a word is calculated through the position and weights associated with respective words in the
graph. The first metric used to calculate the graph weights are co-occurrence matrices and the
other metric are word embeddings. Word embeddings became a crucial way of representing the
semantic information of words as dense vectors.
The results of this paper were compared with keywords that were provided by authors of scientific papers in the area of computer science which act as the ground truth, but crucially are not
a component in the model construction, but just serve as a verifier of the model’s accuracy.

# About the repository

To see some examples of the repository usage, checkout the `main.ipynb` file. 

The required python libraries can be found in the `requirements.txt` file.

The $5000$ paper abstracts dataset can be found at `data/CS_unique.csv`.

You can read more about the implementation and the thesis background at `paper/thesis.pdf`.