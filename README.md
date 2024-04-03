 

## I. Introduction 

Our project focuses on the development of a machine learning model for Fake News vs Real News Classification. The project builds upon previous research in the field of natural language processing (NLP) and machine learning, drawing inspiration from state-of-the-art techniques such as BERT (Bidirectional Encoder Representations from Transformers) and deep learning architectures. Through a combination of feature engineering, model training, and evaluation, we seek to develop a classification model that achieves high accuracy and generalization performance across diverse news datasets. 

 

In addition to technical challenges, our project also addresses broader ethical and societal considerations surrounding the detection and classification of fake news. We recognize the importance of transparency, accountability, and responsible use of technology in combating misinformation and promoting media literacy. 

 

Ultimately, our project aims to contribute to the ongoing efforts to create a more trustworthy and reliable media landscape. 

[Deep Graph Library Python package](https://docs.dgl.ai/en/0.8.x/generated/dgl.data.FakeNewsDataset.html)   

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

############Image of the dataset####################

Here is a quick snapshot of the training dataset for fake news (Please see the Colab notebook in GitHub repository for more details).
############Image of the dataset####################

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

VII. Machine Learning Algorithms Implemented: 
This section outlines the planned implementation of machine learning algorithms for the final report (Section 3b). 



 
