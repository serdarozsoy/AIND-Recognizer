import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''
    
    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        """ 
        BIC penalizes complex models. Minimizing score to find best model.
        """
        best_score = float("inf")
        best_model = None


        for num_states in range(self.min_n_components, self.max_n_components+1):
            try:

                model = self.base_model(num_states)
                logL = model.score(self.X, self.lengths)
                p = num_states * num_states + 2 * num_states * len(self.X[0]) - 1
                N = len(self.sequences) #len(self.X)
                score = (-2 * logL + p * math.log(N))

                if score < best_score:
                    best_score = score
                    best_model = model
            except:
                pass
        return best_model





class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''
    # best description ->       max - DIC = log(P(word)) - average(log(P(others)))
    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        """ 
        DIC penalizes similarity between other words while trying to find best likelihood for that word.
        Maximizing score to find best model.
        DIC ->  log(P(word)) - average(log(P(others)))
        """
        best_score = float("-inf")
        best_model = None
        M = len(self.words)

        for num_states in range(self.min_n_components, self.max_n_components+1):
            try:
                model =self.base_model(num_states)
                logP_word = model.score(self.X, self.lengths)
                
                logP_others = []
                for word in self.words:
                    if word != self.this_word:
                        X, lengths = self.hwords[word]
                        logP_others.append(model.score(X, lengths))
                score = logP_word - np.mean(logP_others)            #alternatively score = logP_word-(1/(M-1))*sum(logP_others)
                                
                
                if score > best_score:
                    best_score = score
                    best_model = model
            except:
                pass


        return best_model

class SelectorCV(ModelSelector):
       
        
    ''' select best model based on average log Likelihood of cross-validation folds
    '''
 
    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV

        best_score = float('-inf')
        best_model = None
        
        if len(self.sequences) < 2:
            return self.base_model(self.n_constant)
        
        for num_states in range(self.min_n_components, self.max_n_components + 1):
            try:
                scores = []
                model = self.base_model(num_states)
                split_method = KFold(n_splits=min(3,len(self.lengths)))
        
                for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                    
                    X_train, length_train = combine_sequences(cv_train_idx, self.sequences)
                    X_test, length_test = combine_sequences(cv_test_idx, self.sequences)
                    model.fit(X_train, length_train)
                    one_score = model.score(X_test, length_test)
                    scores.append(one_score)
                    
       
                    
                score = np.mean(scores)

                if score > best_score:
                    best_score = score
                    best_model = model
            except:
                pass
        return best_model