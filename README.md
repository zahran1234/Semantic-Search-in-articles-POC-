# Semantic Search in Articles: A Classification-Based Approach

## Overview
This project focuses on implementing a semantic search system for articles by framing the task as a classification problem. Each article is classified into predefined categories based on its semantic meaning. A user query is transformed into a vector, and articles are ranked based on their semantic similarity to the query.

## Dataset
- **Source**: Data collected using ChatGPT.
- **Size**: Over 400 samples.
- **Categories**: Two categories - AI and Sports.
- **Splits**: 
  - Training: 80% of the dataset.
  - Testing: 20% of the dataset.
- **Features**: Text and corresponding class labels.

## Project Pipeline
1. **Problem Formulation**: Transform the semantic search problem into a classification task.
2. **Dataset Preparation**: Preprocess the dataset and split it into training and testing sets.
3. **Model Selection**: Compare Word2Vec and Sentence Transformers for semantic representation.
4. **Implementation**: Implement Word2Vec and Sentence Transformers as separate classes to handle embeddings and classification.
5. **Evaluation**: Measure model performance using accuracy and F1 score, and analyze runtime efficiency.

## Techniques and Models Used
### 1. Word2Vec Model (Skip-Gram)
**Overview**: The Word2Vec model is used to generate word embeddings based on the skip-gram neural network architecture. This method captures semantic relationships between words.

**Class Implementation**: `word2vec`
- **Purpose**: Classifies articles using Word2Vec embeddings and a K-Nearest Neighbors (KNN) classifier.

**Key Methods**:
1. `__init__(self, word2vec_model, df)`:
   - Initializes the class with a pre-trained Word2Vec model and a DataFrame.
   - Computes average word vectors for each article and splits the data.
2. `get_average_word_vector(self, concept)`:
   - Calculates the average vector representation for an input text.
3. `test_word2vec_model(self)`:
   - Trains a KNN classifier and evaluates accuracy and F1 score.

**Tools Used**:
- **Python**: Core programming language.
- **Gensim**: For loading pre-trained Word2Vec models.
- **Pandas and NumPy**: For data handling and vector manipulation.
- **Scikit-learn**: For implementing KNN and calculating evaluation metrics.
- **NLTK**: For tokenizing text.

**Results**:
- **Accuracy**: 82.76%
- **F1 Score**: 86.00%

### 2. Sentence Transformers
**Overview**: Sentence Transformers are built on the Transformer architecture and use self-attention mechanisms to capture semantic relationships between sentences. This approach is more effective for long-range dependencies.

**Class Implementation**: `sentence_transformers`
- **Purpose**: Uses embeddings from Sentence Transformers and a vector database for similarity-based classification.

**Key Methods**:
1. `__init__(self, embeddings, df)`:
   - Initializes the class with the embedding model and the dataset, splitting it into training and testing sets.
2. `vector_db(self)`:
   - Sets up a Chroma vector database and adds articles for similarity search.
3. `follow_steps(self)`:
   - Retrieves top-3 similar documents for each test sample, predicts the majority class, and prints similarities.
4. `test(self)`:
   - Evaluates the model by comparing predicted and true labels, calculating accuracy and F1 score.
5. `pipeline(self)`:
   - Runs the complete pipeline: vector database setup, similarity search, and evaluation.

**Tools Used**:
- **Python**
- **Sentence-Transformers**: For embedding generation.
- **Chroma**: Vector database for similarity search.
- **Pandas and Scikit-learn**: For data processing and evaluation.
- **Counter**: For majority voting during classification.

**Results**:
- Sentence Transformers showed better semantic understanding compared to Word2Vec but required more runtime. The use of a vector database significantly improved runtime efficiency.

## Challenges Faced
1. **Runtime Efficiency**:
   - Sentence Transformers were computationally expensive.
   - A vector database (Chroma) was used to enhance query response times.
2. **Vocabulary Limitations**:
   - Word2Vec embeddings depend heavily on the training data, resulting in limited vocabulary coverage.

## Key Learnings
- Improved understanding of the differences between Word2Vec (skip-gram) and Sentence Transformers (Transformer architecture).
- Gained insights into optimizing semantic search systems for runtime efficiency using vector databases.

## Results Summary
- Word2Vec:
  - Accuracy: 82.76%
  - F1 Score: 86.00%
- Sentence Transformers:
  - Demonstrated better semantic understanding but with higher computational cost.

## Tools and Technologies
- **Programming Language**: Python
- **Libraries**: Gensim, NLTK, Scikit-learn, Sentence-Transformers, Chroma, Pandas, NumPy
- **External Resources**: Pre-trained Word2Vec and Sentence Transformer models.

## Conclusion
Sentence Transformers capture semantic meaning more effectively than Word2Vec but require more runtime. The addition of a vector database improved response time, making the solution more scalable for real-world applications. 

This project highlights the trade-offs between semantic understanding and computational efficiency when implementing a semantic search system.

## Future Work
- Explore other embedding models, such as FastText or BERT.
- Optimize the runtime of Sentence Transformers further using quantization or model distillation.
- Expand the dataset to include more categories and samples.
