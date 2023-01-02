#!/usr/bin/env python
# coding: utf-8

# ### Importing Libraries

# In[2]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import json
import seaborn as sns
import nltk
import spacy
import re
import string


# ### Load Data

# In[3]:


# file path
my_file = open("C:/Users/vishakha_deore/Documents/TRAINING/NLP/Client_data.json")
# returns JSON object as a dictionary
data = json.load(my_file)
# print(data)


# In[4]:


# data

# load data using Python JSON module
# with open('C:/Users/vishakha_deore/Documents/TRAINING/NLP/Client_data.json','r') as my_file:
#     input_json = json.loads(my_file.read())
#     print(input_json)
# df = pd.read_json('C:/Users/vishakha_deore/Documents/TRAINING/NLP/Client_data.json')
# print(df)


# In[5]:


# input_json = json.loads(my_file.read())


# In[6]:


# Flatten data
input_df = pd.json_normalize(data)
input_df.head()


# In[7]:


# checking columns
input_df.columns


# In[8]:


input_new_df = input_df.rename(columns = {'_source.product': 'Banking_Services','_source.sub_product':'Debts','_source.complaint_what_happened':'Complaints'})
input_new_df = input_new_df[['Banking_Services','Debts','Complaints']]
input_new_df


# In[9]:


input_new_df.info()


# In[10]:


input_new_df.describe()


# In[11]:


#check  NAN values
input_new_df.Complaints.isnull().sum()


# In[12]:


#check  NAN values
input_new_df.Banking_Services.isnull().sum()


# In[13]:


#check  NAN values
input_new_df.Debts.isnull().sum()


# In[14]:


#check for blank cells any
a = input_new_df[input_new_df['Complaints']==''].count()
# b=input_new_df[input_new_df['Banking_Services']==''].count()
# c=input_new_df[input_new_df['Debts']==''].count()
# a,b,c
a


# In[15]:


input_new_df[input_new_df['Complaints']=='']=np.nan


# In[16]:


input_new_df


# In[17]:


input_new_df.Complaints.isnull().sum()


# In[18]:


# drop the blanks
input_new_df = input_new_df[~input_new_df['Complaints'].isnull()]
input_new_df


# In[19]:


#checking if any null values
input_new_df.Complaints.isnull().sum()


# In[20]:


input_new_df.info()


# In[21]:


input_new_df.describe()


# In[22]:


#cleaning
def cleaning(context):
    context = context.lower()
    punctuation_list = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    print(punctuation_list)
        # Removing punctuations in string
    for values in context:
        if values in punctuation_list:
            test_str = test_str.replace(values, "")
            print("The string after removing punctuation: " + context)
        else: 
                print("The string has no punctuation" + context)
    # Removing square brackets    
    context = re.sub('\[.*\]','',context).strip()
    # Removing numbers from string
    context = re.sub('\S*\d\S*\s*','',context).strip()
    return context.strip()


# In[23]:


input_new_df.Complaints = input_new_df.Complaints.apply(lambda x : cleaning(x))
input_new_df.Complaints.head()


# In[24]:


import en_core_web_sm
nlp = en_core_web_sm.load()


# In[25]:


from nltk.corpus import stopwords


# In[26]:


# Lemmatization
nltk_stopwords = nlp.Defaults.stop_words#list
# words = set(nltk_stopwords)
def lemmatizer(text):
    doc = nlp(text)
    sent = [token.lemma_ for token in doc if not token.text in set(nltk_stopwords)]
    return ' '.join(sent)


# In[27]:


input_new_df


# In[28]:


# Apply lemmatization on complaints
input_new_df['lemma'] = input_new_df.Complaints.apply(lambda x: lemmatizer(x))


# In[30]:


input_new_df.head()


# In[41]:


clean_input_df = input_new_df[['Banking_Services','Complaints','lemma']]
clean_input_df.head()


# In[44]:


def pos_tags(text):
    txt = nlp(text)
    sentence = [token.text for token in txt if token.tag_ =='NN']
    return ' '.join(sentence)


# In[45]:


#modified dataframe
clean_input_df['pos_removed'] = clean_input_df.lemma.apply(lambda x: pos_tags(x))
clean_input_df.head()


# In[47]:


#data visualisation
plt.figure(figsize = (15, 8))
text_length= [len(d) for d in clean_input_df ]
plt.hist(text_length, bins = 50)


# In[49]:


clean_input_df['clean_complaints'] = clean_input_df['pos_removed'].str.replace('-PRON-', '')
clean_input_df = clean_input_df.drop(['pos_removed'], axis = 1)


# In[55]:


print(clean_input_df)


# In[50]:


get_ipython().system('pip install wordcloud')


# In[58]:


# from wordcloud import WordCloud

# w_cloud = WordCloud(stopwords = stopwords, max_words = 40).generate(str(clean_input_df.pos_removed))

# print(w_cloud)
# plt.figure(figsize = (15, 8))
# plt.imshow(w_cloud)
# plt.axis('off')
# plt.show()


# In[ ]:





# In[ ]:


#removing masked text
clean_input_df['clean_complaints'] = clean_input_df['clean_complaints'].str.replace('XXXX','')


# In[59]:


clean_input_df.head()


# In[ ]:


## Grouping data into five categories namely 
# Banking services- _source.product
# loans - _source.sub_product
# Fraudalent reporting - _source.complaint_what_happened
# Card
# others

# df = input_df[['_source.product','_source.sub_product','_source.complaint_what_happened']]
# df = input_df[['Banking_Service','Debts','Complaints']]


# In[62]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[65]:


# Feature Extraction
tf_dif = TfidfVectorizer(min_df = 2, max_df = 0.95, stop_words = 'english')


# In[71]:


doc_tm = tf_dif.fit_transform(clean_input_df.clean_complaints)


# In[73]:


print(doc_tm)


# In[74]:


tf_dif.get_feature_names()[:10]


# In[75]:


len(tf_dif.get_feature_names())


# In[68]:


from sklearn.decomposition import NMF


# In[77]:


ini_topics = 5

#nmf model
model_nmf = NMF(n_components = ini_topics, random_state = 40)
w1 = model_nmf.fit_transform(doc_tm)
h1 = model_nmf.components_


# In[83]:


topic_top_words = 15

vocabulary = np.array(tf_dif.get_feature_names())

t_words = lambda t:[vocabulary[i] for i in np.argsort(t)[:-topic_top_words-1:-1]]
topics = ([t_words(t) for t in h1])

top = [' '.join(t) for t in topics]
print(vocabulary)


# In[84]:


top


# In[87]:


colnames = ['Topic' + str(i) for i in range(model_nmf.n_components)]
docnames = ['Doc' + str(i) for i in range(len(clean_input_df.clean_complaints))]
topic_doc_df = pd.DataFrame(np.round(w1, 2), columns = colnames, index = docnames)
signi_topic = np.argmax(topic_doc_df.values, axis =1)
topic_doc_df['dominant_topic'] = signi_topic
topic_doc_df.head()


# In[88]:


clean_input_df['Topic'] = signi_topic


# In[89]:


pd.set_option('display.max_colwidth', -1)


# In[97]:


clean_input_df[['Complaints', 'clean_complaints', 'Banking_Services', 'Topic']][clean_input_df.Topic == 4].head(30)


# In[98]:


#first 10 complaints

tmp = clean_input_df[['Complaints', 'clean_complaints', 'Banking_Services', 'Topic']].groupby('Topic').head(10)
tmp.sort_values('Topic')


# In[102]:


top_map = {
    0:'Bank Account Services',
    1:'Cards',
    2:'Others',
    3:'Issues',
    4:'Loan'
}

clean_input_df['Topic'] = clean_input_df['Topic'].map(top_map)
clean_input_df.head()


# In[106]:


#plot
plt.figure(figsize = (8,4))
sns.countplot(x = 'Topic', data = clean_input_df)


# In[107]:


train_dt = clean_input_df[['Complaints', 'Topic']]


# In[108]:


train_dt.head()


# In[130]:


# Reverse topic names
topic_mapping_reverse = {
    'Bank Account Services':0,
    'Cards':1,
    'Others':2,
    'Issues':3,
    'Loan':4
}


# In[131]:


train_dt[['Complaints', 'Topic']][train_dt.Topic == 2].head(30)


# In[132]:


#Split

X = train_dt.Complaints
y = train_dt.Topic


# In[133]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split


# In[134]:


# Get vector count

vector_count = CountVectorizer()
X_vect = vector_count.fit_transform(X)


# In[135]:


tfdif_transform = TfidfTransformer()
X_tfdif = tfdif_transform.fit_transform(X_vect)


# In[136]:


#Train-Test Split

x_train, x_test, y_train, y_test = train_test_split(X_tfdif, y, test_size = 0.25, random_state = 40, stratify = y)


# In[137]:


# Trying Models

from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_recall_fscore_support

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay


# In[138]:


#function to display the model report
def display_model_report(metric,model):
    
    pred_prob_y_train = model.predict_proba(X_train)
    pred_prob_y_test = model.predict_proba(X_test)
    train_roc_auc_score = round(roc_auc_score(y_train, pred_prob_y_train,average='weighted',multi_class='ovr'),2)
    test_roc_auc_score = round(roc_auc_score(y_test, pred_prob_y_test,average='weighted',multi_class='ovr'),2)
    print("ROC AUC Score Train:", train_roc_auc_score)
    print("ROC AUC Score Test:", test_roc_auc_score)
    metric.append(train_roc_auc_score)
    metric.append(test_roc_auc_score)
    #prediction
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    #metrics
    precision_train,recall_train,fscore_train,support_train=precision_recall_fscore_support(y_train,y_train_pred,average='weighted')
    precision_test,recall_test,fscore_test,support_test=precision_recall_fscore_support(y_test,y_test_pred,average='weighted')
    #accuracy score
    acc_score_train = round(accuracy_score(y_train,y_train_pred),2)
    acc_score_test = round(accuracy_score(y_test,y_test_pred),2)
    
    metric.append(acc_score_train)
    metric.append(acc_score_test)
    metric.append(round(precision_train,2))
    metric.append(round(precision_test,2))
    metric.append(round(recall_train,2))
    metric.append(round(recall_test,2))
    metric.append(round(fscore_train,2))
    metric.append(round(fscore_test,2))
    
    print('The training accuracy is :',acc_score_train)
    print('The testing accuracy is :',acc_score_test)
        
     # Plot the confusion matrix
    fig, ax = plt.subplots(figsize=(12, 8))
    # Create the confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    cmp = ConfusionMatrixDisplay(cm, display_labels=model.classes_)
    cmp.plot(ax=ax)
    plt.xticks(rotation=80)

    plt.show();


# In[158]:


# create 5 folds
folds = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 40)


# In[159]:


#method to get the GridSearchCV object
def grid_search(folds,model, parameters,scores):
    
    grid_search = GridSearchCV(model,
                                cv=folds, 
                                param_grid=parameters, 
                                scoring=scores, 
                                n_jobs=-1, verbose=1)
    return grid_search


# In[160]:


#function to display the best score and parameters of the GridSearchCV model
def best_score_params(model):
    print("Score: ", model.best_score_)
    print("Hyperparameters: ", model.best_params_)

# create MNB model object
multi_nb = MultinomialNB()
# fit the model
multi_nb.fit(x_train, y_train)

#### Decision Tree Classification
d_tree = DecisionTreeClassifier(random_state=40)
# fit model
d_tree.fit(x_train,y_train)

#### Random Forest Classification
random_f = RandomForestClassifier(n_estimators = 500,random_state=40, n_jobs = -1,oob_score=True)
# fit model
random_f.fit(x_train,y_train)


#### Logistic Regression Classification
logistic_reg = LogisticRegression(random_state=40,solver='liblinear')
# fit model
logistic_reg.fit(x_train,y_train)


# oob
print('OOB SCORE :',random_f.oob_score_)


# Multinomial Naive Bayes with GridSearchCV
mnb = MultinomialNB()
mnb_params = {  
'alpha': (1, 0.1, 0.01, 0.001, 0.0001)  
}

# create gridsearch object
grid_search_mnb = grid_search(mnb, folds, mnb_params, scores=None)

# fit model
grid_search_mnb.fit(x_train, y_train)

# print best hyperparameters
best_score_params(grid_search_mnb)

# logistic regression
logistic_reg = LogisticRegression()

# hyperparameter for Logistic Regression
logistic_params = {'C': [0.01, 1, 10], 
          'penalty': ['l1', 'l2'],
          'solver': ['liblinear','newton-cg','saga']
         }

# create gridsearch object
grid_search_log = grid_search(logistic_reg, folds,logistic_params, scores=None)

# fit model
grid_search_log.fit(x_train, y_train)

# print best hyperparameters
best_score_params(grid_search_log)


# Decision Tree Classification with GridSearchCV
dtc = DecisionTreeClassifier(random_state=40)
dtc_params = {
    'max_depth': [5,10,20,30],
    'min_samples_leaf': [5,10,20,30]
}

# create gridsearch object
d_tree_grid_search = grid_search(dtc, folds, dtc_params, scores='roc_auc_ovr')

# fit model
d_tree_grid_search.fit(x_train, y_train)

# best hyperparameters
best_score_params(d_tree_grid_search)


# In[ ]:


rf = RandomForestClassifier(random_state=40, n_jobs = -1,oob_score=True)

# hyperparameters for Random Forest
rf_params = {'max_depth': [10,20,30,40],
          'min_samples_leaf': [5,10,15,20,30],
          'n_estimators': [100,200,500,700]
        }

# create gridsearch object
grid_search_rf = grid_search(rf, folds, rf_params, scores='roc_auc_ovr')

# fit model
grid_search_rf.fit(x_train, y_train)

# oob score
print('OOB SCORE :',grid_search_rf.best_estimator_.oob_score_)

# print best hyperparameters
print_best_score_params(grid_search_rf)


# In[ ]:


test_complaint= 'Hi, I tried to withdraw money from ATM machine and also used correct pin, but still it is rejecting my card, please resolve this issue'
test = count_vect.transform([test_complaint])
test_tfidf = tfidf_transformer.transform(test)


# In[ ]:


# predict
prediction=grid_search_log.predict(test_tfidf)
prediction


# In[ ]:


topic_mapping[prediction[0]]


# In[ ]:




