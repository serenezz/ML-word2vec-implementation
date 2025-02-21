# Word2Vec Embedding from Rotten Tomato Movie Review

This project implements a Word2Vec model using the Gensim library to learn word embeddings from a dataset of movie reviews. The model is trained on Rotten Tomatoes critic reviews and is designed to capture semantic relationships between movie-related terms by finding similar words based on context.

## Dataset

The model is trained using the `rotten_tomatoes_critic_reviews.csv` dataset obtained from [Kaggle](https://www.kaggle.com/datasets/stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset). The CSV file contains a column named `review_content` with the full text of the movie reviews. These reviews are used as input to the Word2Vec model, which learns to generate vector representations of words based on their context.

## Preprocessing

The preprocessing steps are as follows:

1. **Missing Data**: Any rows with missing `review_content` are removed.
2. **Text Tokenization**: Each review is tokenized using Gensim's `simple_preprocess` method. This method:
   - Converts text to lowercase.
   - Removes punctuation and non-alphanumeric characters.
   - Splits the text into individual words.

## Training the Word2Vec Model

The Word2Vec model is trained using the **skip-gram** approach. In this approach, the model learns word vectors by predicting context words (surrounding words) given a target word. The model parameters are as follows:

- **Window size (`window`)**: Defines the maximum distance between the current and predicted word within a sentence. A window size of 10 is used.
- **Minimum word frequency (`min_count`)**: Ignores words that appear less than 5 times in the corpus, helping to filter out infrequent words.
- **Number of CPU workers (`workers`)**: Uses 5 CPU cores for parallel training to speed up the process.

Once trained, the model is capable of generating vector representations for words that can be used for finding similar words or measuring word similarity.
