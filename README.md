# Quora Question Pair Similarity
   ![Quora](https://github.com/akashbangalkar/quora_question_pair_simi/blob/main/Quora.png)
   
Quora is a place to gain and share knowledge - about anything. It’s a platform to ask questions and connect with people who contribute unique insights and quality answers. This empowers people to learn from each other and to better understand the world.

Over 100 million people visit Quora every month, so it's no surprise that many people ask similarly worded questions. Multiple questions with the same intent can cause seekers to spend more time finding the best answer to their question, and make writers feel they need to answer multiple versions of the same question. Quora values canonical questions because they provide a better experience to active seekers and writers, and offer more value to both of these groups in the long term.

**Credits:** Kaggle

### Problem Statement :
Identify which questions asked on Quora are duplicates of questions that have already been asked.

### Real World/Business Objectives and Constraints:
   - The cost of a mis-classification can be very high.
   - You would want a probability of a pair of questions to be duplicates so that you can choose any threshold of choice.
   - No strict latency concerns.
   - Interpretability is partially important.

### Performance Metric:
   - log-loss 
   - Binary Confusion Matrix

### Data Overview:
Train.csv contains 5 columns : qid1, qid2, question1, question2, is_duplicate. 
Total we have 404290 entries. I worked on 50k entries. Splitted data into train and test with 70% and 30%.

I derived some features from questions like number_of_common_words, word_share and some distances_between_questions with the help of word vectors. Will discuss those below. You can check my total work [here](https://github.com/akashbangalkar/quora_question_pair_simi).
### Some Analysis:
- ##### Distribution of Data Points Among Output Classes  
   ![No of Datapoints Per Class](https://github.com/akashbangalkar/quora_question_pair_simi/blob/main/Images/class.png "No of Datapoints Per Class") 
- ##### Number of Unique Questions
   ![Number of Unique Questions](https://github.com/akashbangalkar/quora_question_pair_simi/blob/main/Images/output_1.png "Number of Unique Questions") 
- ##### Number of Occurrences of Each Question
   ![Number of Occurrences of Each Question](https://github.com/akashbangalkar/quora_question_pair_simi/blob/main/Images/occurance.png "Number of Occurrences of Each Question")

### Feature Extraction:
- ##### Basic Features - Extract Some Features Before Cleaning of Data -
  - <b>freq_qid1</b> = Frequency of qid1's
  - <b>freq_qid2</b> = Frequency of qid2's
  - <b>q1len</b> = Length of q1
  - <b>q2len</b> = Length of q2
  - <b>q1_n_words</b> = Number of words in Question 1
  - <b>q2_n_words</b> = Number of words in Question 2
  - <b>word_Common</b> = (Number of common unique words in Question 1 and Question 2)
  - <b>word_Total</b> =(Total num of words in Question 1 + Total num of words in Question 2)
  - <b>word_share</b> = (word_common)/(word_Total)
  - <b>freq_q1+freq_q2</b> = sum total of frequency of qid1 and qid2
  - <b>freq_q1-freq_q2</b> = absolute difference of frequency of qid1 and qid2
   
- ##### Advance Features - Preprocessing of texts and extract some other features.
  - <b>cwc_min</b> = common_word_count / (min(len(q1_words), len(q2_words)) 
  - <b>cwc_max</b> = common_word_count / (max(len(q1_words), len(q2_words)) 
  - <b>csc_min</b> = common_stop_count / (min(len(q1_stops), len(q2_stops)) 
  - <b>csc_max</b> = common_stop_count / (max(len(q1_stops), len(q2_stops)) 
  - <b>ctc_min</b> = common_token_count / (min(len(q1_tokens), len(q2_tokens)) 
  - <b>ctc_max</b> = common_token_count / (max(len(q1_tokens), len(q2_tokens)) 
  - <b>last_word_eq</b> = Check if Last word of both questions is equal or not (int(q1_tokens[-1] == q2_tokens[-1]))
  - <b>first_word_eq</b> = Check if First word of both questions is equal or not (int(q1_tokens[0] == q2_tokens[0]) )
  - <b>abs_len_diff</b> = abs(len(q1_tokens) - len(q2_tokens))
  - <b>mean_len</b> = (len(q1_tokens) + len(q2_tokens))/2
  - <b>fuzz_ratio</b> = How much percentage these two strings are similar, measured with edit distance.
  - <b>fuzz_partial_ratio</b> = if two strings are of noticeably different lengths, we are getting the score of the best matching lowest length substring.
  - <b>token_sort_ratio</b> = sorting the tokens in string and then scoring fuzz_ratio.
  - <b>longest_substr_ratio</b> = len(longest common substring) / (min(len(q1_tokens), len(q2_tokens))
  
### Some Features analysis and visualizations:
- ##### Word Share - We can check from below that it is overlaping a bit, but it is giving some classifiable score for disimilar questions.
   ![Wordshare](https://github.com/akashbangalkar/quora_question_pair_simi/blob/main/Images/word%20share.png "Wordshare")
   
- ##### Word Total -
   ![Word Total](https://github.com/akashbangalkar/quora_question_pair_simi/blob/main/Images/word%20total.png "Word Total")
   
- ##### Bivariate analysis of features 'ctc_min', 'cwc_min', 'csc_min', 'token_sort_ratio'. We can observe that we can divide duplicate and non duplicate with some of these features with some patterns. 
   ![Pair plot](https://github.com/akashbangalkar/quora_question_pair_simi/blob/main/Images/pair.png "Pair plot")
   
- ##### Fuzz Ratio -
   ![Fuzz Ratio](https://github.com/akashbangalkar/quora_question_pair_simi/blob/main/Images/fuzz.png "Fuzz Ratio")
   
### Machine Learning Models:
   - Trained a random model to check worst case log loss and got log loss as 0.8826
   - Trained some models and also tuned hyperparameters.

| Model         | Log Loss |
| ------------- | -------- |
| Logistic Regression  | 0.4829  |
| Linear SVM           | 0.5071  |
| XG Boost        | 0.4050  |

##### References:
- Source: https://www.kaggle.com/c/quora-question-pairs
- Discussions : https://www.kaggle.com/anokas/data-analysis-xgboost-starter-0-35460-lb/comments
- Kaggle Winning Solution and other approaches: https://www.dropbox.com/sh/93968nfnrzh8bp5/AACZdtsApc1QSTQc7X0H3QZ5a?dl=0
- Blog 1 : https://engineering.quora.com/Semantic-Question-Matching-with-Deep-Learning
- Blog 2 : https://towardsdatascience.com/identifying-duplicate-questions-on-quora-top-12-on-kaggle-4c1cf93f1c30

