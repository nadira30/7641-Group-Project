 

## 1. Introduction/Background:  

Literature Review: As technology becomes more widespread and datasets grow in size and complexity, methods for interpreting advanced machine learning algorithms become more paramount. "Fake News Detection on Social Media using Geometric Deep Learning" by Monti et al. (2019) showcases how Graph Neural Network can be effectively applied to the task of fake news detection by leveraging the structural information present in social media graphs. The use of graphs for real-life datasets is increasingly common given the flexibility and efficiency for this data structure to represent diverse datasets. GNN uses graph analysis by identifying a subset of nodes/features to explain the AI decision making process.  

Dataset: This project will utilize the publicly available Fake News dataset from the Deep Graph Library Python package: [https://docs.dgl.ai/en/0.8.x/generated/dgl.data.FakeNewsDataset.html]. This dataset comprises labeled news articles with associated social network (Twitter) information.  

 

## 2. Problem Definition: 

As existing ML models often lack transparency in their decision-making process, we are seeking to develop a robust and interpretable system for identifying fake news articles.  

 

### Motivation: 

The widespread dissemination of fake news poses a significant threat to various aspects of society, including: 

Undermining trust in credible information sources. 

Misleadingly sway individuals' opinions on critical issues, potentially leading to harmful consequences. 

Fueling existing societal divides and hinder constructive dialogue. 

By leveraging the power of GNNs and explainability techniques, this project aims to produce a fake news detection system that contributes to the creation of more trustworthy and transparent AI solutions for tackling the challenge of fake news. 

 

## 3. Methods : 

The GraphXAI repository  provides an overview of perturbation-based approaches (GNNExplainer), gradient-based approaches (Integrated Gradients), and surrogate network-based approaches (PGMExplainer). 

 

* GNNExplainer  is a method for extracting GNN explanations, employing a perturbation-based approach using mean field variational approximation of subgraph distributions to minimize conditional entropy.  

 

* Integrated Gradients is a technique which uses network gradients to extract explanations in an axiomatic manner based on the difference between a baseline (zero signal input) and an input.  

 

* PGMExplainer uses a probabilistic graphical model to calculate the conditional probabilities of explained features. This approach may be effective to disentangle node / graph features with a surrogate graph structure. 

 

### Data Preprocessing Methods Identified 

 

* Text Cleaning: Apply techniques like tokenization, stop word removal, stemming/lemmatization, and named entity recognition to clean the news article text data.  

* Feature Engineering: Extract relevant features such as word frequencies, sentiment scores, and named entity counts.  

* Graph Construction: Utilize the social network information to construct a graph representing user interactions and relationships between news articles. 

 

 

### Machine Learning Algorithms/Models Identified: 

 

* Graph Neural Network (GNN): Implement a GNN model to learn from the graph representation and binary classify news articles. The GNN model will leverage the inherent capability to capture relationships and information diffusion within the social network. 

* GNN explainer: Integrate a GNN explainer like GNNExplainer or AttnExplain to provide insights into the factors influencing the GNN model's predictions for specific news articles, which enables us to understand the rationale behind its classifications. 

 

### Unsupervised and Supervised Learning Methods Identified: 

* Principal Component Analysis (PCA): Use PCA to identify the most informative features from the text data before feeding them into the GNN model. 

* Support Vector Machines (SVMs): Use SVMs as a baseline supervised learning model for comparison with the GNN model's performance. 


4. (Potential) Results and Discussion:  

An essential criterion for explanations is that they must be interpretable by providing a qualitative understanding of the relationship between the input nodes and the prediction. Also,  a GNN explainer should consider the structure of the underlying graph and the associated features when available. 


For fake news detection, we propose the following evaluation metrics: 

* Accuracy: to measure the overall proportion of correctly classified news articles.  

* Precision: to reflect the proportion of true positives among the predicted positive cases. 

* F1 Score: Harmonic mean of precision and recall, providing a balanced evaluation metric for binary classification tasks. 

 

For explainability, we propose the following evaluation metrics: 

* Ease of understanding to a particular user 

* Completeness of explanation – How complete the explanation is wrt the goal 

* Conciseness of explanation  


### Project Goals: 

Develop and optimize ML models for fake news detection using proposed preprocessing methods and algorithms. 

Evaluate model performance using quantitative metrics, comparing effectiveness in discerning real from fake news. 

Explore feature engineering techniques and model hyperparameters' impact on predictive performance and generalization. 

### Expected Results: 

ML models trained on fake news detection dataset anticipated to deliver competitive accuracy, precision, and recall. 

Final deliverable expected to accurately capture linguistic and stylistic disparities between real and fake news, facilitating precise misinformation detection. 

 

## 5. References: 

* Agarwal, C., Queen, O., Lakkaraju, H. et al. Evaluating explainability for graph neural networks. Sci Data 10, 144 (2023). https://doi.org/10.1038/s41597-023-01974-x 

* Castillo, C., Mendoza, M., & Poblete, B. (2011). Information credibility on Twitter. In Proceedings of the 20th international conference on World wide web (pp. 675-684). 

* Shu, K., Sliva, A., Wang, S., Tang, J., & Liu, H. (2017). Fake news detection on social media: A data mining perspective. ACM SIGKDD Explorations Newsletter, 19(1), 22-36.  

* Sundararajan, M., Taly, A., & Yan, Q. (2017). Axiomatic Attribution for Deep Networks. International Conference on Machine Learning. 

* Vu, M.N., & Thai, M.T. (2020). PGM-Explainer: Probabilistic Graphical Model Explanations for Graph Neural Networks. ArXiv, abs/2010.05788. 

* Ying, R., Bourgeois, D., You, J., Zitnik, M., & Leskovec, J. (2019). GNNExplainer: Generating Explanations for Graph Neural Networks. Advances in neural information processing systems, 32, 9240-9251 . 

* Monti, Federico, et al. "Fake news detection on social media using geometric deep learning." arXiv preprint arXiv:1902.06673 (2019). 

* GraphXAI repository: https://github.com/mims-harvard/GraphXAI 

* GNNExplainer : https://github.com/RexYing/gnn-model-explainer 

* Integrated Gradients: https://github.com/ankurtaly/Integrated-Gradients 

 

 

 
