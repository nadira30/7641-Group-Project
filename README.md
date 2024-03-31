 

## I. Introduction 

As technology becomes more widespread and data grows in size and complexity, methods for interpreting advanced ML algorithms become paramount. Monti et al [7] showcase how Graph Neural Network can be applied to detect fake news by leveraging the structural information in social media graphs. The usage of graphs for real-life data is increasing given the flexibility and efficiency for this data structure to represent diverse datasets. GNN uses graph analysis by identifying a subset of nodes/features to explain the AI decision making process.  

This project will utilize the publicly available Fake News dataset from the [Deep Graph Library Python package](https://docs.dgl.ai/en/0.8.x/generated/dgl.data.FakeNewsDataset.html). This dataset comprises labeled news articles with associated Twitter information.    

## II. Problem Definition: 

As existing ML models often lack transparency in their decision-making process, we are seeking to develop a robust and interpretable system for identifying fake news articles.   

### Motivation: 

The widespread dissemination of fake news poses a significant threat to various aspects of society, including: 

* Undermining trust in credible information sources. 

* Misleadingly swaying individuals' opinions on critical issues. 

* Fueling existing societal divides. 

By leveraging GNNs and explainability techniques, this project aims to produce a fake news detection system that contributes to the tackling of the challenges posed by fake news. 

 
## III. Methods : 

GNNExplainer: Utilizes perturbation-based techniques, employing mean field variational approximation to minimize conditional entropy for extracting GNN explanations. 

PGMExplainer: Employs a probabilistic graphical model to calculate conditional probabilities of explained features, potentially effective via analyzing a surrogate graph structure. 

### Data Preprocessing Methods: 

* Text Cleaning: Apply techniques like tokenization, stop word removal, stemming/lemmatization, and named entity recognition.  

* Feature Engineering: Extract relevant features such as word frequencies, sentiment scores, and named entity counts.  

* Graph Construction: Utilize social network information to construct a graph representing user interactions between news articles. 

### Machine Learning Algorithms: 

* Graph Neural Network (GNN): Implement a GNN model to learn from the graph representation and binary classify news articles. The GNN model will leverage the inherent capability to capture relationships and information diffusion within the social network. 

* GNN explainer: Integrate a GNN explainer like GNNExplainer or AttnExplain to provide insights into the factors influencing the GNN model's predictions for specific news articles, which enables us to understand the rationale behind its classifications. 

### Unsupervised and Supervised Learning Methods Identified: 

* Use Principal Component Analysis to identify the most informative features from the data to feed into the GNN model. 

* Use SVMs as a baseline supervised learning model for comparison with the GNN model's performance. 

## IV. (Potential) Results and Discussion:  

For fake news detection, we propose the following evaluation metrics: 

* Accuracy 

* Precision 

* F1 Score 

For explainability, we propose the following evaluation metrics: 

* Ease of understanding to a user 

* Completeness of explanation  

* Conciseness of explanation  


### Project Goals: 

* Develop ML models for fake news detection using proposed methods and algorithms. 

* Evaluate model performance using quantitative metrics. 

* Explore feature engineering techniques and model hyperparameters' impact on performance. 

### Expected Results: 

* Expect Model to deliver competitive accuracy, precision, and recall. 

* Expect final deliverable to capture linguistic and stylistic disparities between real and fake news. 
 

## V. References: 

1. Agarwal, C., Queen, O., Lakkaraju, H. et al. Evaluating explainability for graph neural networks. Sci Data 10, 144 (2023). https://doi.org/10.1038/s41597-023-01974-x 

2. Castillo, C., Mendoza, M., & Poblete, B. (2011). Information credibility on Twitter. In Proceedings of the 20th international conference on World wide web (pp. 675-684). 

3. Shu, K., Sliva, A., Wang, S., Tang, J., & Liu, H. (2017). Fake news detection on social media: A data mining perspective. ACM SIGKDD Explorations Newsletter, 19(1), 22-36.  

4. Sundararajan, M., Taly, A., & Yan, Q. (2017). Axiomatic Attribution for Deep Networks. International Conference on Machine Learning. 

5. Vu, M.N., & Thai, M.T. (2020). PGM-Explainer: Probabilistic Graphical Model Explanations for Graph Neural Networks. ArXiv, abs/2010.05788. 

6. Ying, R., Bourgeois, D., You, J., Zitnik, M., & Leskovec, J. (2019). GNNExplainer: Generating Explanations for Graph Neural Networks. Advances in neural information processing systems, 32, 9240-9251 . 

7. Monti, Federico, et al. "Fake news detection on social media using geometric deep learning." arXiv preprint arXiv:1902.06673 (2019). 

8. GraphXAI repository: https://github.com/mims-harvard/GraphXAI 

9. GNNExplainer : https://github.com/RexYing/gnn-model-explainer 
## Gantt Chart 
![Gantt chart](https://ucarecdn.com/2745ad43-212c-4a21-b475-61dc32fb6213/GanttChart.png)
## Contribution Table 
![Contribution table](https://ucarecdn.com/1490c528-a050-489a-9960-6fd7a64ff217/contribution_table.png)

 
