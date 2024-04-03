## I. Introduction 

Our project focuses on the development of a machine learning model for Fake News vs Real News Classification. The project builds upon previous research in the field of natural language processing (NLP) and machine learning, drawing inspiration from state-of-the-art techniques such as BERT (Bidirectional Encoder Representations from Transformers) and deep learning architectures. Through a combination of feature engineering, model training, and evaluation, we seek to develop a classification model that achieves high accuracy and generalization performance across diverse news datasets.
In addition to technical challenges, our project also addresses broader ethical and societal considerations surrounding the detection and classification of fake news. We recognize the importance of transparency, accountability, and responsible use of technology in combating misinformation and promoting media literacy.
Ultimately, our project aims to contribute to the ongoing efforts to create a more trustworthy and reliable media landscape.

## II. Litterature Review 

Researchers have explored various approaches to address this challenge, employing both traditional machine learning algorithms and advanced deep learning models to develop fake news detection systems.

Traditional Machine Learning Approaches:
* Castillo et al.,[2] in 2011 and Conroy et al.,[3] in 2015 utilized algorithms like SVMs, Decision Trees, and Logistic Regression with handcrafted features to tackle fake news.
* They focused on text characteristics and metadata, requiring extensive feature engineering.
* Jain et al.,[13] utilize Naive Bayes classification capabilities to predict whether Facebook posts will be labelled as “real news” or “fake news”.
* Reis et al.,[14] use several supervised learning algorithms to measure the effectiveness of their classification abilities while considering features such as lexical features, domain location (for urls), engagement, temporal patterns, etc. 

Advancements with Deep Learning:
* Zhou et al[4] adopted RNNs and CNNs for processing sequential data and capturing local text patterns to differentiate between real and fake news.
* Monti et al[11] use geometric deep learning in order to detect fake news titles on social media.
* Deep learning models excel in learning features directly from text without manual extraction.

BERT and Transformer Models:
* Devlin et al. [5] introduced BERT as a significant advance in fake new classification with its bidirectional training and attention mechanisms.
* Demonstrated superior performance in understanding context and semantic relationships in text[6]. 

## III. Dataset Description 

Our data are Json files publicly available and stored on web; therefore, we download it via GCS storage APIs and convert it into pandas dataframe. We download the data in 4 files: 

* Training – Training dataset of real news. [Dataset link](https://storage.googleapis.com/public-resources/dataset/real_train.json)
* Testing – Testing dataset of real news. [Dataset link](https://storage.googleapis.com/public-resources/dataset/real_test.json)
* Training – Training dataset of fake news. [Dataset link](https://storage.googleapis.com/public-resources/dataset/fake_train.json)
* Testing – Testing dataset of fake news. [Dataset link](https://storage.googleapis.com/public-resources/dataset/fake_test.json)
There are 800 rows and 3 columns (url, title, text) for real and fake news news in their respective training sets, we will only use the 'text' column for modeling (for simplicity's sake). 

Here is a quick snapshot of the training dataset for real news (Please see the Colab notebook in GitHub repository for more details). 

![image](https://raw.githubusercontent.com/nadira30/7641-Group-Project/main/_includes/2.png)

Here is a quick snapshot of the training dataset for fake news (Please see the Colab notebook in GitHub repository for more details).
!(image)[https://github.com/nadira30/7641-Group-Project/blob/main/_includes/2.png?raw=true]

To train, validate, and evaluate the performance of our Fake News or Real News Classification model, we divided our dataset into three distinct subsets: training, validation, and testing. 

* Training Dataset: The training dataset comprised the largest portion of our data and was used to train the model's parameters. It consisted of a diverse collection of labeled news articles, including both real and fake news samples. This dataset served as the foundation for the model to learn the underlying patterns and features associated with each class.
* Validation Dataset: The validation dataset was used to fine-tune the model's hyperparameters and monitor its performance during training. It provided an independent set of data samples that the model did not see during training, allowing us to assess its generalization ability and prevent overfitting. The validation dataset played a crucial role in optimizing the model's architecture and training process. 
* Testing Dataset: The testing dataset served as the final benchmark to evaluate the model's performance after training and validation. It consisted of a separate set of news articles, again with labels indicating their authenticity. The model's predictions on the testing dataset were compared against the ground truth labels to assess its accuracy, precision, recall, and other performance metrics. This dataset provided valuable insights into the model's real-world effectiveness and its ability to accurately classify unseen news articles. 

 
## IV. Problem Definition
Our goal is to develop a machine learning model capable of distinguishing between fake and real news articles. Given a dataset of news articles, the objective is to first identify real and fake news articles through data pre-processing, labelling them and then training a classification model to accurately identify the authenticity of unseen news articles as either fake or real. 

#### Scope: 
What we aim to do is fake news detection based on textual features. What we do not cover in our project is checking the truthfulness of content of an article/ fact. Using the help of external sources, we built our code to solve the issue stated above (see appendix for code references). 

#### Change in Research Direction:
Our research initially focused on detecting fake news using structural and language features without verifying content truthfulness and focusing more on user modelling because we realize that was a more challenging problem to solve and leveraged GNNs. Additionally, due to Twitter’s privacy laws, we could not view the dataset features necessary to utilize GNNs. Hence, we have refined our approach to concentrate on Fake News or Real News Classification using BERT. 

Shifting our focus allows us to provide a more specific solution to the broader issue of misinformation. By classifying news articles as either fake or real, we aim to empower users to discern the authenticity of news sources and combat the spread of misinformation. 

## V. Motivation: 

The widespread dissemination of fake news poses a significant threat to various aspects of society, including: 

* Undermining trust in credible information sources.
* Misleadingly swaying individuals' opinions on critical issues.
* Fueling existing societal divides.
By leveraging BERT and supervised and unsupervised ML techniques, this project aims to produce a fake news detection system that contributes to the tackling of the challenges posed by fake news. 

 
## VI. Methods: (BERT and Logistic regression) 

#### BERT for Feature extraction, classification, training and evaluation: 

We leverage Bidirectional Encoder Representations from Transformers (BERT), a powerful pre-trained language model, to encode textual data from news articles. Since BERT excels at capturing semantic relationships between words and contextual meaning within sentences, we chose it as a primary tool for detecting fake news. 
During training, the provided code utilizes a pre-trained BERT model (‘bert-base-uncased') to transform preprocessed news text into numerical representations. These representations act as features that capture the essence of the news content. 

#### Logistic Regression (To be implemented): 
We also plan to use logistic regression to compare the results with BERT. Logistic regression is a well-suited algorithm for binary classification tasks like real vs. fake news detection. Our hypothesis is that it will learn a separating hyperplane between the two classes (real and fake) like the extracted BERT features. New unseen news articles can then classify as real or fake based on their position relative to the learned hyperplane in the feature space. 

#### Data Preprocessing Method Implemented: 
We preprocessed our original text into input features BERT can read. The process is basically tokenizing and converting our original text into token IDs that can be read by the algorithm. The words are tokenized based on the vocabulary dictionary it was pretrained on (vocabulary size of 30,522 words), and unknown words are broken down into smaller words contained in the dictionary. Maximum sequence length is also specified so we can pad all sequences into the same length. However, please note that the final sequence length would be larger than specified since BERT tokenizer breaks unknown words into multiple small known words. 
Since BERT algorithm can only accept sentence length up to 512 words, we need to preprocess our data (long news) to feed into the algorithm. To do so, we follow the idea from this paper and segment each of the text into multiple subtexts of no longer than 150 words. The subtexts will have some overlapping, specifically, the last 30 words for first subtext will be the first 30 words of the second subtext. 
We incorporated several data preprocessing steps to prepare the news text data for BERT and logistic regression: 
* Text Cleaning: We removed punctuation, converted text to lowercase, and handled extra spaces to standardize the text format. This ensures consistent processing by BERT and avoids introducing noise due to formatting variations. 
* Text Splitting: We split each news article into subtexts with a maximum length of 150 words. This step caters to the input requirements of BERT, which has limitations on the sequence length it can process effectively. 
* BERT Tokenization: We use a pre-trained BERT tokenizer (‘bert-base-uncased') to convert text into numerical tokens (ids) that BERT understands. This allows BERT to process the textual data and extract meaningful features. 
* Padding and Truncation: We padded shorter sequences with zeros and truncated longer sequences to a fixed length (‘max_seq_len’) to ensure consistent input size for BERT. This step ensures all inputs are compatible with the BERT model's architecture. 
Thus, by carefully preparing and padding our input data, we ensured that our Fake News or Real News Classification model received standardized inputs, allowing it to effectively learn and make accurate predictions on news articles of varying lengths and complexities. 

#### Feature Reduction (Not Implemented - Potential Future Exploration) 
Our initial implementation will not include feature reduction techniques like Principal Component Analysis (PCA) or feature selection. However, these techniques can be explored in future iterations to: 
* Potentially improve model performance by reducing dimensionality. 
* Decrease training time. 
* Identify and remove irrelevant features that might hinder classification. 

Note: Since BERT already performs dimensionality reduction, additional feature reduction might not be necessary in this specific case. However, it remains a potential area for future exploration. 

## VII. Machine Learning Algorithms Implemented: 
This section outlines the planned implementation of machine learning algorithms for the final report (Section 3b). 

#### 1. BERT (Pre-trained - Implemented): 

BERT, a pre-trained deep learning model, will be used for feature extraction. BERT excels at learning valuable representations of words and their relationships within text data. 
Below is the high-level process that we followed: 
* Clean and tokenize the text data. 
* Feed the processed data into the pre-trained BERT model. 
* Obtain informative features from BERT's output, capturing the semantic meaning of the text. 

To construct our BERT model, we utilized the Hugging Face transformers library, leveraging the powerful BERT architecture pre-trained on vast amounts of text data. We initialized a BERT base model with uncased tokens (‘bert-base-uncased') and fine-tuned it for our specific task of Fake News or Real News Classification. The model consists of: 
1. An input layer for tokenized sequences, followed by the pre-trained BERT layers.  
2. We added a dropout layer for regularization and a dense layer with a sigmoid activation function for binary classification.  
3. Then we loaded the pretrained weight of BERT and finetune it. The source of pretrained weights is called bert_news.h5. 
4. The model was compiled using the Adam optimizer with a specified learning rate, and binary cross-entropy loss was used as the evaluation metric. 

During training, we employed early stopping and model checkpointing techniques to prevent overfitting and save the best-performing model based on validation accuracy. This approach enabled us to develop a robust BERT model capable of accurately distinguishing between fake and real news articles. 

#### 2. Logistic Regression (Planned): 

Logistic regression is the planned classification model for distinguishing real from fake news. It will leverage the BERT-extracted features: 
* The features will serve as input to the logistic regression model. 
* The model will learn a decision boundary in the feature space, enabling it to classify new unseen text data as real or fake news. 

#### 3. Support Vector Machines (SVM) (Planned): 

SVMs are powerful supervised learning algorithms that can effectively classify data points into different categories. They excel at finding hyperplanes (decision boundaries) in high-dimensional feature spaces, which makes them well-suited for text classification tasks like fake news detection. 
We think SVMs are a good fit for this use case because of the following: 
* High-dimensional data: Fake news detection often involves analyzing textual data, which translates into high-dimensional feature spaces. SVMs can handle these complex spaces efficiently.
* Non-linear decision boundaries: Fake news often employ manipulative language or obfuscation techniques. SVMs can learn non-linear decision boundaries to separate real and fake news data points even if the patterns aren't perfectly linear. 
* Interpretability: SVMs offer a level of interpretability compared to complex deep learning models like BERT tying to our original goal of incorporating explainability into a black box neural network model. This can clarify the features the model uses to make predictions. 

## VII. Results and Discussion:  
#### Visualizations 

We used the training set to perform exploratory analysis. First, we wanted to look at the word count for each news and see if there is difference between real and fake news. We can see in the below graph that most real news is within 1000 words, and the distribution of word count is skewed to the right. 
![img3](https://raw.githubusercontent.com/nadira30/7641-Group-Project/main/_includes/3.png)

As for the fake news, we see some outliers from the below graph making it hard to interpret. So, we plot it again below with outlier (news that has more than 20,000 words) removed. 
![img4](https://raw.githubusercontent.com/nadira30/7641-Group-Project/main/_includes/4.png)

The below graph shows fake news training dataset with outliers (>20,000 words removed) 

![img5](https://raw.githubusercontent.com/nadira30/7641-Group-Project/main/_includes/5.png)

The histogram below depicts frequencies of titles based on title length (number of characters in title) for both real and fake news. 

![histogram_title_length.png](https://raw.githubusercontent.com/nadira30/7641-Group-Project/main/_includes/histogram_title_length.png)

The word cloud for real vs fake news based on news text is depicted below. We can see most of the real news is about COVID19 virus, and the common words are names of countries and other neutral words.

![fake_real](https://raw.githubusercontent.com/nadira30/7641-Group-Project/main/_includes/fake_real_cloud.png)

A similar word cloud is obtained for real and fake news, but only based on the title and inserted below. The topics seem to be the same for fake news. However, it contains words that are distinguishable. For example, Bill Gates is spelled wrongly in some instances. Additionally, there are a few standalone letters such as “u”. 

![fake_real2](https://raw.githubusercontent.com/nadira30/7641-Group-Project/main/_includes/fake_real2.png)

To gain better understanding of the sentiments across real vs fake news across titles as well as texts, their sentiments were analyzed, and their polarities are represented in the plots below. 

![sentiment](https://raw.githubusercontent.com/nadira30/7641-Group-Project/main/_includes/sentiment_distribution.png)

The plots below show the N-gram analysis conducted to obtain the top 10 n-gram phrases and / or words in real and fake news titles and texts respectively. 
![n_gram2](https://raw.githubusercontent.com/nadira30/7641-Group-Project/main/_includes/n_gram2.png)

The plot below shows clustering of word embeddings for real and fake news. Word2Vec is used to represent words in titles and texts as dense vectors. Clustering algorithm is used to cluster similar titles and texts together and visualize the clusters. 
![real_clustering](https://raw.githubusercontent.com/nadira30/7641-Group-Project/main/_includes/real_cluster.png)
![fake_clustering](https://raw.githubusercontent.com/nadira30/7641-Group-Project/main/_includes/fake_cluster.png)

Next we wanted to see the Top “k” word proportion of the real/fake news. In other words, we wanted to see how many of the words used in the news are from the top 10 common words, top 100, and so on. Our hypothesis was that since fake news are machine generated and it will likely use many high frequency words comparing to real news. 
![top_k](https://raw.githubusercontent.com/nadira30/7641-Group-Project/main/_includes/top_k_proportion.png)

Our analysis confirmed our hypothesis, and we see from the above bar chart that the difference is significant only for the Top 1000 most frequently used words. 

#### Quantitative Metrics

After we trained the model, we evaluated the training set, the validation set, and the test set. We used the following metrics: 
* Accuracy 
* Precision,  
* Recall  
* F1 Score 
* AUC (Area under the Curve) - to get a more thorough evaluation of our model 

#### Evaluation Observations:

We can see that we have about ~approx. 95.0 % accuracy for the training set and about 95.40 % accuracy for our validation set.  

The prediction is on the augmented test set, but this needs to be translated to the prediction of the original text. Therefore, for each original text, we should average the probability of each subtext and obtain final prediction for that piece of news. 

#### Analysis of 1+ Algorithm/Model 

In this project, BERT is employed as a feature extractor. It takes raw text data as input and produces high-dimensional vectors as output, capturing the semantic meaning and context of the text. As expected, pre-trained on massive text corpora, BERT captured semantic relationships and nuances in language that might be difficult for simpler feature extraction techniques. Our metrics showed great results. Even AUC was shown hugging the top-left corner indicating that our model was performing well. If our planned implementation of the Logistic Regression model achieves high accuracy, precision, recall, and F1-score, it suggests BERT is capturing relevant information for fake news classification. We think that  

Additionally, by learning a lower-dimensional representation of text, BERT can potentially improve the efficiency and performance of the final classification model (Logistic Regression).  

We think that one of our key advantages of using BERT was using its pre-training on large corpora of text data. During pre-training, BERT learns to predict missing words in sentences, masked tokens, and sentence relationships, which imbues it with a robust understanding of language. This pre-trained knowledge is leveraged in the project by fine-tuning BERT on the specific task of fake news detection. Hence, we were able to get excellent metrics.  

#### Potential Limitations (to Investigate with Logistic Regression): 

BERT might not be specifically optimized for the task of fake news detection. While pre-trained on a large corpus, it might not capture the specific linguistic cues or patterns indicative of fake news. The complexity of BERT can lead to overfitting, especially with smaller datasets. 

By implementing Logistic Regression and performing the suggested analyses, we can gain a more comprehensive understanding of BERT's effectiveness in this specific use case and identify potential areas for improvement
 
#### Next Steps 

1. Implement Logistic Regression:
    a. Train and evaluate a Logistic Regression model using the features extracted by BERT.
    b. Analyze its performance using metrics like accuracy, precision, recall, F1-score, and confusion matrix.
    c. This will provide a quantitative assessment of how well BERT, combined with Logistic Regression, classifies real vs. fake news.
2. Hyperparameter Tuning: Experiment with different hyperparameters for both BERT and Logistic Regression to potentially improve performance. 
3. Comparison Baseline: Train a Logistic Regression model directly on the raw text data (without BERT) as a baseline.
    a. Compare its performance with the model using BERT features to quantify the effectiveness of feature extraction by BERT. 

#### Gantt Chart:
!(Gantt Chart)[https://raw.githubusercontent.com/nadira30/7641-Group-Project/main/_includes/GanttChart.png]

#### Contribution table:
!(Contribution Table)[https://raw.githubusercontent.com/nadira30/7641-Group-Project/main/_includes/contribution_table.png]

## References: 

1. Agarwal, C., Queen, O., Lakkaraju, H. et al. Evaluating explainability for graph neural networks. Sci Data 10, 144 (2023). https://doi.org/10.1038/s41597-023-01974-x
2. Castillo, C., Mendoza, M., & Poblete, B. (2011). Information credibility on Twitter. In Proceedings of the 20th international conference on World wide web (pp. 675-684). 
3. Conroy, N. K., Rubin, V. L., & Chen, Y. (2015). Automatic deception detection: Methods for finding fake news. Proceedings of the association for information science and technology, 52(1), 1-4. 
4. Zhou, X., & Zafarani, R. (2019). Network-based fake news detection: A pattern-driven approach. ACM SIGKDD explorations newsletter, 21(2), 48-60. 
5. Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805. 
6. Yang, J., Wang, M., Zhou, H., Zhao, C., Zhang, W., Yu, Y., & Li, L. (2020, April). Towards making the most of bert in neural machine translation. In Proceedings of the AAAI conference on artificial intelligence (Vol. 34, No. 05, pp. 9378-9385). 
7. Shu, K., Sliva, A., Wang, S., Tang, J., & Liu, H. (2017). Fake news detection on social media: A data mining perspective. ACM SIGKDD Explorations Newsletter, 19(1), 22-36. 
8. Sundararajan, M., Taly, A., & Yan, Q. (2017). Axiomatic Attribution for Deep Networks. International Conference on Machine Learning. 
9. Vu, M.N., & Thai, M.T. (2020). PGM-Explainer: Probabilistic Graphical Model Explanations for Graph Neural Networks. ArXiv, abs/2010.05788. 
10. Ying, R., Bourgeois, D., You, J., Zitnik, M., & Leskovec, J. (2019). GNNExplainer: Generating Explanations for Graph Neural Networks. Advances in neural information processing systems, 32, 9240-9251 . 
11. Monti, Federico, et al. "Fake news detection on social media using geometric deep learning." arXiv preprint arXiv:1902.06673 (2019). 
12. Zhang, X., & Ghorbani, A. A. (2020). An overview of online fake news: Characterization, detection, and discussion. Information Processing & Management, 57(2), 102025. 
13. Jain, A., & Kasbe, A. (2018, February). Fake news detection. In 2018 IEEE International Students' Conference on Electrical, Electronics and Computer Science (SCEECS) (pp. 1-5). IEEE. 
14. Reis, J. C., Correia, A., Murai, F., Veloso, A., & Benevenuto, F. (2019). Supervised learning for fake news detection. IEEE Intelligent Systems, 34(2), 76-81. 


## Appendix A- Code references: 

* [Code reference 1](https://stackoverflow.com/questions/65085991/bert-model-show-up-invalidargumenterror-condition-x-y-did-not-hold-element-wi)
* [Code reference 2](https://colab.research.google.com/github/singularity014/BERT_FakeNews_Detection_Challenge/blob/master/Detect_fake_news.ipynb#scrollTo=oPz41H-zs-XN)
* [GraphXAI repository]( https://github.com/mims-harvard/GraphXAI)
* [GNNExplainer](https://github.com/RexYing/gnn-model-explainer)
* [Integrated Gradients](https://github.com/ankurtaly/Integrated-Gradients)

 