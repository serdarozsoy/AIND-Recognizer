import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []

    """
        1- Calculating the logL scores for each word in model 
        2- Appending these scores to "probabilities" list.
        3- Finding words with maximum scores, append these are "guesses" lists.
    """


    try:
        for word_id in range(0, len(test_set.get_all_sequences())):
            logL_words = {}
            best_score = float('-inf')
            best_word = None
            X, lengths = test_set.get_item_Xlengths(word_id)

            for word, model in models.items():
                try:
                    score = model.score(X, lengths)
                    logL_words[word] = score
                    if score > best_score:
                        best_word = word
                        best_score = score
                except:
                    logL_words[word] = float("-inf")
                    
            probabilities.append(logL_words)
            guesses.append(best_word)

    except:
        pass

    return probabilities, guesses



