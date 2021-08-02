import pandas as pd
from dataclasses import dataclass
from typing import List, Any, NamedTuple
import math

@dataclass
class Branch:
    best_feature: str
    subtrees: Any
    default_value: Any = None

@dataclass
class PandasDecisionTree():
    '''INPUT VARIABLES'''
    df: pd.DataFrame
    feature_columns: List[str]
    target_feature: Any

    '''OUTPUT VARABLES'''
    branch = Branch('None', {})
    leaf: Any = None
    
    




    '''ENTROPY CALCULATIONS'''
    def entropy(self, dataframe = None) -> float:
        '''return the entropy of the dataframe according to the target feature'''

        if dataframe is None:
            dataframe = self.df 

        return sum(-p * math.log(p, 2)
                    for p in self.class_probabilities(dataframe)
                    if p > 0)

    def class_probabilities(self, dataframe = None) -> List[float]:
        '''return the probabilites each output of target feature column in the dataframe'''

        if dataframe is None:
            dataframe = self.df 

        total_count = dataframe.shape[0]

        counts = dataframe[self.target_feature].value_counts()

        return [count/total_count for index, count in counts.items()]




    '''FORMATTING FUNCTIONS'''
    def partition_dataframe(self, feature: str = None ) -> dict:
        '''returns dict of dataframes indexed by unique falues in the feature column'''

        if feature is None:
            feature = self.feature_columns[0]

        dictionary = {}
        #get unique values in column
        series = self.df[feature].unique() #returns numpy array

        #split dataframe according to unique values
        for key in series:
            dictionary[key] = self.df.loc[self.df[feature] == key]

        return dictionary

    def partition_entropy(self, feature: str = None ) -> float:
        '''returns total entropy of a partitioned feature column.
            used to determine the best feature column to branch on - 
            total entropy is a weighted sum of entropies, where the weight is the proportion of
            data in partition to the data in the full dataframe '''

        if feature is None:
            feature = self.feature_columns[0]

        partitions = self.partition_dataframe(feature)
        
        
        return sum(self.entropy(dataframe) * dataframe.shape[0] / self.df.shape[0]
                    for dataframe in partitions.values() )

    
    def train(self):
        '''build decision tree'''
         #get most common target_attribute for given dataframe
        target_feature_counts = self.df[self.target_feature].value_counts()

        most_common_output = target_feature_counts.keys()[0]

        # HANDLE BASE CASES

            # if all survival counts are same for given dataframe
        if len(target_feature_counts) == 1:
            self.leaf = most_common_output
            return self.leaf

            # no more features left
        if not self.feature_columns:
            self.leaf = most_common_output
            return self.leaf

        #HANDLE BRANCHING

            # find best feature (lowest entropy)
        best_feature = min(self.feature_columns, key=self.partition_entropy)

            # partition about best feature
        partitions = self.partition_dataframe()

            # get next feature_columns
        next_feature_columns = [feat for feat in self.feature_columns if feat != best_feature]

            # RECURSIVELY BUILD SUBTREE
        subtrees = {feature : PandasDecisionTree(partitions[feature], next_feature_columns, self.target_feature)
                    for feature in partitions }

        [subtrees[branch].train() for branch in subtrees]

        self.branch.best_feature = best_feature
        self.branch.subtrees = subtrees
        self.branch.default_value = most_common_output

        return self.branch


    def classify(self, input: pd.DataFrame,  tree = None):
        '''classify the input using the given decision tree'''

        if tree is None:
            tree = self

        # convert input to Dataframe
        if isinstance(input, dict):
            input = pd.DataFrame.from_dict(input)

        #return leaf value if the tree is a leaf node
        if tree.leaf is not None:
            return tree.leaf
        
        #in best_feature column, get value to for branch 
        subtree_key = input.get(tree.branch.best_feature, '')

        #if the the tree doesn't contain the key return default
        if subtree_key not in tree.branch.subtrees:
            return tree.branch.default_value

        # get subtree to follow
        subtree = tree.branch.subtrees[subtree_key]

        # classify input at the next branch
        return self.classify(subtree, input)

    '''MANIPULATION FUNCTIONS FOR TESTING/SAMPLING'''
    def split_data(self, ratio: float) -> tuple:
        '''split dataframe into (train, test) data. Parameter is the ratio of train data'''
        split_row = math.ceil(self.df.shape[0]*ratio)
        shuffle = self.df.sample(frac=1)
        return (shuffle.iloc[:split_row], shuffle.iloc[split_row:])

    def test(self, test_set: pd.DataFrame) -> float:
        '''return accuracy of model'''
        correct = 0
        total = test_set.shape[0]
        for row in range(0, total):
            guess =  self.classify(test_set.iloc[row])
            answer = test_set.iloc[row][self.target_feature]
            if guess == answer:
                correct += 1

        return correct/total

            



    





df = pd.read_csv('./phpMYEkMl.csv')


tree = PandasDecisionTree(df, ['pclass', 'sex', 'age', 'sibsp', 'parch', 'embarked', 'cabin'], 'survived')

train, test = tree.split_data(.75)

#intialize Decision Tree with train data
train_frame = PandasDecisionTree(train, ['pclass', 'sex', 'age', 'sibsp', 'parch', 'embarked', 'cabin'], 'survived')
#train the model
train_frame.train()

# Test the model
print(train_frame.test(test))






