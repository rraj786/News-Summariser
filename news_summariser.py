"""
    news_summariser: This script reads in various news articles and performs abstractive 
    or extractive summarisation based on user inputs.

    Author: Rohit Rajagopal
"""

from newspaper import Article
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from collections import Counter
from heapq import nlargest
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
from scipy.special import softmax
import re 
from keybert import KeyBERT
from autocorrect import Speller

def extractive_summarisation(spacy_model, text):

    '''
        Performs extractive summarisation on pre-defined text by applying
        the standard small English spaCy model.

        Inputs:
            - spacy_model: NLP model used to create doc object
            - text: Pre-defined text to summarise

        Outputs:
            - e_sum: Text summary of article
    '''

    doc = spacy_model(text)

    # Add keywords to an array, ensuring that punctuation and stop words are ignored
    keyword = []
    stopwords = list(STOP_WORDS)
    pos_tag = ['PROPN', 'ADJ', 'NOUN', 'VERB']
    for token in doc:
        if token.text in stopwords or token.text in punctuation:
            continue
        elif token.pos_ in pos_tag:
            keyword.append(token.text)

    # Find frequency of keywords in text
    freq = Counter(keyword)

    # Normalise frequencies 
    max_freq = freq.most_common(1)[0][1]
    for word in freq.keys():
        freq[word] = freq[word]/max_freq

    # Find weightings of each sentence
    sent_sum = {}
    for sent in doc.sents:
        for word in sent:
            if word.text in freq.keys():
                if sent in sent_sum.keys():
                    sent_sum[sent] += freq[word.text]
                else:
                    sent_sum[sent] = freq[word.text]

    top_sent = nlargest(5, sent_sum, key = sent_sum.get)

    # Format and return summarised output
    summary = [w.text for w in top_sent]
    final_summary = ' '.join(summary)
    
    return final_summary


def abstractive_summarisation(spacy_model, text):

    '''
        Performs abstractive summarisation on pre-defined text by applying 
        a pre-trained Transformers base model.

        Inputs:
            - spacy_model: NLP model used to create doc object
            - text: Pre-defined text to summarise

        Outputs:
            - a_sum: Text summary of article
    '''

    # Initialize pretrained model
    model = T5ForConditionalGeneration.from_pretrained('t5-base')
    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    device = torch.device('cpu')

    # Remove stop words
    stopwords = list(STOP_WORDS)
    words = text.split()
    target = []
    for word in words:
        if words not in stopwords:
            target.append(word)

    # Pre-process text
    t5_input = 'summarize: ' + ' '.join(target)

    # Create tokenized text
    tokenized_text = tokenizer.encode(t5_input, return_tensors = 'pt', max_length = 8192, truncation = True).to(device)

    # Run model
    summary_ids = model.generate(tokenized_text, min_length = 100, max_length = 200)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens = True)

    # Format and return summarised output
    format_summary = spacy_model(summary)
    sents = [sent.text[0].capitalize() + sent.text[1:] for sent in format_summary.sents]
    final_summary = ' '.join(sents)

    return final_summary


def sentiment_analysis(text):

    '''
        Detects the sentiment of a piece of text - postive, negative
        and neutral components.

        Inputs:
            - text: Text used to predict sentiment

        Outputs:
            - sentiment: A list of sentiment scores
    '''

    # Instantiate BERT model
    tokenizer = AutoTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment-latest')
    model = AutoModelForSequenceClassification.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment-latest')

    # Encode tokens and calculate sentiment
    tokens = tokenizer.encode(text, return_tensors = 'pt')
    result = model(tokens)

    # Convert result into probability distribution using softmax function
    output = softmax(result.logits[0].detach().tolist())

    sentiment = [("negative", round(output[0], 2)), ("neutral", round(output[1], 2)), ("positive", round(output[2], 2))]
    sentiment = sorted(sentiment, key = lambda x: x[1], reverse = True)

    return sentiment


def emotion_analysis(text):

    '''
        Detects the emotions exuded by a piece of text.

        Inputs:
            - text: Text used to predict emotions

        Outputs:
            - emotions: A list of scores for various emotions
    '''

    if len(text) == 0:
        return [], 'No quotations found'
    
    # Instantiate BERT model
    classifier = pipeline(task = "text-classification", model = "SamLowe/roberta-base-go_emotions", top_k = 3)

    # Extract model outputs
    emotions = {} 
    top_label = []
    for quote in text:
        emotions[quote] = classifier(quote)[0] 
        top_label.append(emotions[quote][0]['label'])

    top_emotions = Counter(top_label).most_common(3)    
    
    return emotions, top_emotions


def keywords(text):
    
    '''
        Extract keywords from a given piece of text.

        Inputs:
            - text: Text used to predict emotions

        Outputs:
            - keywords: A list of keywords and their importance.
    '''

    # Instantiate model
    model = KeyBERT()

    # Extract keywords
    keywords = model.extract_keywords(text)

    return keywords


def question_answer(questions, text):
    
    '''
        Find the best answers to questions related to a piece of text.

        Inputs:
            - questions: List of questions related to text
            - text: Text used to generate answers

        Outputs:
            - answer: A list of answers from the given text.
    '''

    # Instantiate model
    model = pipeline("question-answering", model = "deepset/roberta-base-squad2", tokenizer = "deepset/roberta-base-squad2")

    # Clean input (spell-check)
    spell = Speller(lang = 'en')
    question_clean = spell(question)

    # Find answer to each question
    prompt = {'question': question_clean, 'context': text}
    result = model(prompt)
    answer = result['answer']

    return answer


if __name__ == '__main__':

    # Return contents for a particular article
    url = 'https://www.nytimes.com/2024/04/25/us/politics/trump-supreme-court-immunity-case.html'
    article = Article(url)
    article.download()
    article.parse()
   
    # Load small English model
    nlp = spacy.load("en_core_web_sm") 
    doc = nlp(article.text)

    # Pre-process text (add full stops to create complete sentences)  
    pattern = r'([^.\n]+)\n(?!\.”$)'                  
    processed1 = re.sub(pattern, r'\1.\n', article.text.strip())    
    processed2 = processed1.replace('.\n', '.')
    processed3 = processed2.replace('\n', ' ')
    proc_final = processed3.replace('“', '"').replace('”', '"').replace('.".', '".')

    # Perform abstractive summarisation
    summary = abstractive_summarisation(nlp, proc_final)

    # Perform sentiment analysis on summary
    sentiment = sentiment_analysis(summary)
    
    # Detect type of emotions from text (only consider quotes in text)
    emotion_text = re.findall(r'"([^"]*)"', proc_final)
    _, top_emotions = emotion_analysis(emotion_text)

    # Return keywords and people from text (for finance articles, see what keywords drive up and down price, 
    # and also look at financial BERT model)
    words = keywords(proc_final)
    people = [ent.text for ent in doc.ents if ent.label_ == 'PERSON']

    # Could potentially train a model on news articles/key words and whether stock went up or down

    # Print results
    print('\n\n')
    print('----------------------------------------------------------------------------------------------------------------------------------------------------------------------')
    print(article.title.upper() + ': \n' + article.meta_description + '\n' + summary)
    print('\n' + 'Sentiment:', sentiment)
    print('\n' + 'Top Emotion Evoked:', top_emotions)
    print('\n' + 'Key Words:', words)
    print('\n' + 'Key People:', people)
    print('----------------------------------------------------------------------------------------------------------------------------------------------------------------------')

    # Questions related to text
    cont = True
    while cont:
        question = input("Do you have any questions related to this article? (Press enter to skip): ")
        if len(question) != 0:
            answers = question_answer(question, proc_final)
            print(answers)
        else:
            cont = False
