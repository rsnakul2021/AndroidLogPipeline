# AndroidLogPipeline
A PySpark model used to classify Android Logs and detect anomalies in them.
The "Logcat" tool in Android Studio is commonly used to view Android event logs, which are a record of system events and actions that take place on an Android device. In essence, they are a detailed log of everything that occurs within the operating system and applications, including user interactions, system processes, network activity, and potential errors. These logs can be accessed and analyzed for debugging and troubleshooting purposes. 
# Types of Andoid Log Levels
1. Verbose (Log.v)
Logs detailed and highly granular information, typically used for debugging purposes. Includes everything, even the least significant logs. This type of log is very noisy and should be used sparingly in production environments. Used for debugging specific scenarios during development to capture the complete flow of an operation.
2. Debug (Log.d)
Logs information useful for debugging during development. Less verbose than Verbose logs but provides sufficient context about the app's state and operations. Used for tracking application flow, values of variables, and non-critical operations.
3. Info (Log.i)
Logs general information about application events. Provides messages that indicate the normal functioning of the application. These logs are more concise and less detailed than Debug logs. Used for reporting key lifecycle events or application status (e.g., "User logged in").
4. Error (Log.e)
Logs error messages when something goes wrong in the application. Used for logging issues that need immediate attention, such as exceptions or failures in operations. Used for tracking critical failures, such as failed network requests, crashes, or unhandled exceptions.
5. Warning (Log.w)
Captures potential problems that are not yet errors but could lead to issues. Used for identifying areas that might need improvement or fixing in future releases.
6. Fatal/Assert (Log.wtf)
Logs critical problems that should "never happen" and often represent serious bugs. It is typically used for situations where recovery is impossible or undesirable. Corrupted data or application state inconsistency. Used for debugging extreme cases that indicate a breach of fundamental assumptions in your code.
# Dataset 
https://github.com/logpai/loghub/tree/master/Android
# Feature Engineering
The content column contains a bundle of related text. Instead of using the words as it is, the data is converted into a key-value pair using the following pattern.
 
For example, transferRemoteInputFocus: 190 is converted into transferRemoteInputFocus -> 190.
The component column contains 1 or 2 word brief description of the log. A tokenizer is used to pair up the words. After which stop words such as in, there etc., are removed for. The StopWordsRemover is a custom UDF where unnecessary log words are also removed for.
 
Finally, an N-Gram UDF is used to created a paid of 2-words grams. An N-gram is a contiguous sequence of N items (words, characters, or tokens) from a given text or speech. It is commonly used in natural language processing (NLP) and text analysis.
 
The dataset that we now have contain the following columns: date, time, pid, tid, component, content, content_tokens, content_cleaned, content_ngram, key_value_map. These will be used feature engineering pipeline that is based on a TF-IDF algorithm to convert the words into meaningful numbers. A numerical metric called TF-IDF (Term Frequency-Inverse Document Frequency) is used in natural language processing to assess a word's significance to a document within a corpus, or group of documents. It strikes a balance between a word's rarity throughout the corpus (IDF) and its frequency in a single document (TF). This approach is useful for finding pertinent phrases while weeding out common, less instructive ones since words that are widespread in a document but uncommon throughout the corpus have higher TF-IDF scores.
	Then a String Indexer is used to convert the categories into numbers. StringIndexer is a feature transformer in PySpark that converts categorical text labels into numerical indices, assigning unique integers to each distinct label based on their frequency. This is commonly used for preparing categorical features for machine learning algorithms.
# Classification
Different classification algorithms were tested out with regularization and Logistic Regression came out to be the best in terms of accuracy and F1 score.
# How to run the files?
1) Import the 6 datasets along with the 3 code files onto your GCP bucket.
2) Submit job-1 with Level_classification.py as main file and the 6 android log files as arguments.
3) The outputs from feature engineering, classification pipeline and accuracy metrics will be shown in Dataproc.
4) The script also saves the pipeline model in your GCP bucket.
5) Run the Load_model.py with your new unlabeled dataset as job-2 to label them.
