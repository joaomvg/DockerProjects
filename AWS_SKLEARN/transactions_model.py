from os import stat
from scipy.sparse import csr_matrix,hstack,vstack
import pickle
import pandas as pd
import numpy as np
import re
from collections import Counter


class KEYWORDS:
        
    """
    KEYWORDS class
    
    """
    
    def __init__(self,file='credit_keywords.pkl'):
        self.keywords=pickle.load(open(file,'rb'))
        self.all_words=self.__allwords()
        self.words_contained=None
        self.__contained()
    
    def __allwords(self):
        all_keys=[]
        for _,lst in self.keywords.items():
            all_keys+=lst
        all_keys=list(set(all_keys))
        
        return sorted(all_keys,key=len)
        
    def __contained(self):
        self.words_contained={}
        
        for i,word1 in enumerate(self.all_words):
            self.words_contained[word1]=[]
            for word2 in self.all_words[i+1:]:
                if word2.lower().find(word1.lower())!=-1:
                    self.words_contained[word1].append(word2.lower())
                    
                    
class TransactionsModel:
    
    """
    Credits and Debits model based on SKlearn Random Forests
    
    """
    
    def __init__(self,credit_model_file,debit_model_file,vocabulary_file,person_file,credit_keywords_file,catg_dic):
        
        self.credit_model=pickle.load(open(credit_model_file,'rb'))
       
        self.debit_model=pickle.load(open(debit_model_file,'rb'))

        self.vocabulary=pickle.load(open(vocabulary_file,'rb'))
        
        self.credit_person=pd.read_pickle(person_file)
        
        self.vocab_size=len(self.vocabulary)
        self.credit_keyword_object=KEYWORDS(credit_keywords_file)
        self.credit_keywords=self.credit_keyword_object.keywords
        
        self.credit_keywords_dic={word:i for i, word in enumerate(sorted(self.credit_keyword_object.all_words))}
        
        self.credit_catg_dic=catg_dic['credits']
        self.debit_catg_dic=catg_dic['debits']
        
        
    def __data_transform(self,data, account_holder):
        
        '''
        DEBITS:
        '''
        
        if 'Debit' in data['type'].values:
            debit=data[data.type=='Debit'].fillna('')
            
            #check columns
            cols=['description','additional','payee_information','payee']
            for col in cols:
                if col not in debit.columns:
                    debit.loc[:,col]=''
            
            debit_row_sparse=[]
            debit_internal=[]
            for text in zip(debit.description,debit.additional,debit.payee_information):
                debit_internal.append([self.__internal(text[0],account_holder)])
                text_concat=self.__concat_text(text)
                text_counter=self.__vocab_clean(text_concat)
                text_sparse=self.__sparse_text(text_counter,self.vocab_size)
                debit_row_sparse.append(text_sparse)

            debit_foreign=debit.payee.str.contains('^(?![NL])[A-Z][A-Z]\d\d',regex=True).values
            debit_foreign=debit_foreign.astype('int8').reshape(-1,1)
            
            debit_csr=vstack(debit_row_sparse)
            debit_csr=hstack([debit_csr,debit.amount.values.reshape(-1,1),debit_internal,debit_foreign])
        else:
            debit_csr=None
        
        '''
        CREDITS:
        '''
        
        def search_counter(text,keywords,contained):
            match=Counter()
            for key in keywords:
                re_string='[^\s]*'+key+'[^\s]*'
                ids=[(i.start(),i.end()) for i in re.finditer(re_string,text.lower())]
                match_substrings=[text.lower()[i:j] for i,j in ids]

                for substring in match_substrings:
                    if substring not in contained[key]:
                        match.update([key])

            return match
        
        if 'Credit' in data['type'].values:
            credit=data[data.type=='Credit'].fillna('')
            credit=credit.join(self.credit_person.set_index('payer_information'),on='payer_information')
            
            #check columns
            cols=['description','additional','payer_information','payer']
            for col in cols:
                if col not in credit.columns:
                    credit.loc[:,col]=''
                    
            contained=self.credit_keyword_object.words_contained
            col=[]
            row=[]
            data_csr=[]
            credit_internal=[]
            i=0
            for text in zip(credit.description,credit.additional,credit.payer_information):
                credit_internal.append([self.__internal(text[0],account_holder)])
                text_concat=self.__concat_text(text)
                text_match=search_counter(text_concat,self.credit_keywords_dic,contained)
                for w,cnt in text_match.items():
                    col.append(self.credit_keywords_dic[w])
                    row.append(i)
                    data_csr.append(cnt)
                i+=1
            
            credit_csr=csr_matrix((data_csr,(row,col)),shape=(credit.shape[0],len(self.credit_keywords_dic)))
            p2p=(credit.p2p=='person').values.reshape(-1,1)
            p2p=p2p.astype('int8')
            credit_foreign=credit.payer.str.contains('^(?![NL])[A-Z][A-Z]\d\d',regex=True).values
            credit_foreign=credit_foreign.astype('int8').reshape(-1,1)

            credit_csr=hstack([credit_csr,credit.amount.values.reshape(-1,1),credit_internal,p2p,credit_foreign],format='csr')
        else:
            credit_csr=None
        
        return credit_csr, debit_csr
    
    def predict(self,data_trns, account_holder):
        
        credit_csr, debit_csr= self.__data_transform(data_trns,account_holder)
        
        if credit_csr is not None:
            credit_predict=self.credit_model.predict(credit_csr)
        else:
            credit_predict=None
        
        if debit_csr is not None:
            debit_predict=self.debit_model.predict(debit_csr)
        else:
            debit_predict=None
        
        return credit_predict, debit_predict
    
    
    def __sparse_text(self,text_counter,vocab_size):
        """ x: counter
            col: column index
            data: count of word
        """
        col=[]
        row=[]
        data=[]
        for word,i in text_counter.items():
            if word in self.vocabulary:
                col.append(self.vocabulary[word])
                data.append(i)
                row.append(0)
        
        row_sparse=csr_matrix((data,(row,col)),shape=(1,vocab_size))

        return row_sparse
    
    def __internal(self,description,account_holder):
        
        irrelevant=['hr','mf','van','den','der', 'en/of', 'dtg', 'c.w.', 'm.g.']
        if type(account_holder)==str:
            holder=re.sub('\W',' ',account_holder)
        else:
            holder=''
        
        holder_counter=self.vocab(holder)
        relevant_holder=self.relevant(holder_counter,irrelevant)
        internal_label=self.check_internal(description,relevant_holder) 
        
        return internal_label
    
    @staticmethod
    def __vocab_clean(text):
        """
            Splits text and removes punctuation.
            returns: counter([list])
        """
        lst=text.split()
        lst=[re.sub('[\W]','',n) for n in lst if len(n)>2]

        return Counter(lst)
    
    @staticmethod
    def __concat_text(text_list):
        concat_string=''
        for text in text_list:
            if type(text)==str:
                concat_string+=text
            concat_string+=' '
        return concat_string.strip()

    @staticmethod
    def vocab(text):
            if text is not None and type(text)==str:
                lst=text.split()
                lst=[re.sub(',','',n) for n in lst]
                return Counter(lst)
            else:
                return Counter()
            
    @staticmethod
    def relevant(counter,irrelevant):
        lst=[word.lower() for word in counter if word.lower() not in irrelevant and len(word)>3]

        return lst

    @staticmethod
    def check_internal(description,relevant_holder):
        for name in relevant_holder:
            if description.lower().find(name)!=-1:
                return 1
        return 0
    

credit_catg_dic={'INTERNAL':'INTERNAL_PAYMENT',
                 'UNCATEGORIZED':'UNCATEGORYZED',
                 'Person2Person':'UNCATEGORYZED',
                 'INCOME_SALARY':'INCOME_SALARY',
                 'GAMBLING':'GAMBLING',
                 'OTHER_INCOME':'OTHER_INCOME', 
                 'INCOME_FOREIGN':'INCOME_FOREIGN',
                 'SUSPECT_PAYMENT':'CRIME',
                 'ALIMONY':'ALIMONY',
                 'CASH':'CASH',
                 'TAX_PAYBACK':'UNCATEGORYZED',
                 'OTHER_LOAN':'OTHER_LOAN',
                 'REFUND':'UNCATEGORYZED',
                 'STORNO':'STORNO'}

debit_catg_dic={'GROCERIES':'GROCERIES',
                'GAMBLING':'GAMBLING',
                'ENTERTAINMENT':'ENTERTAINMENT',
                'TRANSPORTATION':'TRANSPORTATION',
                'CLOTHING&APPEARAL':'CLOTHING & APPEARAL',
                'ONLINE_PAYMENT':'UNCATEGORYZED',
                'UTILITIES':'UTILITIES',
                'INTERNAL':'INTERNAL_PAYMENT',
                'INSURANCE':'INSURANCE',
                'DRUGSTORE':'UNCATEGORYZED',
                'BANKS':'UNCATEGORYZED',
                'TAX_PAY':'UNCATEGORYZED',
                'PERSON2PERSON':'UNCATEGORYZED',
                'CASH':'CASH',
                'OTHER_LOAN':'OTHER_LOAN',
                'OUTGOING_FOREIGN':'OUTGOING_FOREIGN',
                'CREDIT_CARD':'UNCATEGORYZED',
                'RENT':'RENT',
                'SERVICES':'UNCATEGORYZED',
                'HEALTH_COSTS':'UNCATEGORYZED',
                'SPORTS':'SPORTS',
                'MORTGAGE':'MORTGAGE',
                'CHILD_CARE':'CHILD_CARE',
                'STUDY_LOAN':'STUDY_LOAN',
                'LAW/JUSTICE':'UNCATEGORYZED',
                'EDUCATION':'UNCATEGORYZED',
                'HOUSING_COSTS':'UNCATEGORYZED',
                'ALIMONY':'ALIMONY',
                'NOTARY':'UNCATEGORYZED'}

catg_dic={'credits': credit_catg_dic,
          'debits': debit_catg_dic
         }