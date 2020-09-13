from flask import Flask, request
from flask_restful import Resource, Api
import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
#cleaning the texts
import re
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer




app = Flask(__name__)
api = Api(app)

class GaneshTesting(Resource):
    def get(self):
        try:
            ds=pd.read_csv('Train.csv')
            emotes=['😜','\U0001F539','😍','😂','😉','🎄','📷','🔥','😘','❤','😁','us','☀','✨','💙','💕','😎','😊','💜','💯']

            print(request.json)
            args = request.args
            print (args) # For debugging
            g=args['input']
            c=[]
            for i in range(0,7000):
                review=re.sub('[^a-zA-Z]',' ',ds['TEXT'][i])
                review=review.lower()
                review=review.split()
                ps=PorterStemmer()
                review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
                review=' '.join(review)
                c.append(review)
               
            #creating athe bag of words  
            from sklearn.feature_extraction.text import CountVectorizer
            cv=CountVectorizer(max_features=1500)
            x=cv.fit_transform(c).toarray()
            y=ds.iloc[:7000,2].values

            #cleaning my text
            d=[]
            txt=re.sub('[^a-zA-Z]',' ',g)
            txt=txt.lower()
            txt=txt.split()
            ps=PorterStemmer()
            txt=[ps.stem(word) for word in txt if not word in set(stopwords.words('english'))]
            txt=' '.join(txt)
            d.append(txt)

            #baging my text
            txt=txt.split()
            z=np.zeros((7000,len(txt)))
            for e in c:
                for q in txt:
                    if(q in e):z[c.index(e)][txt.index(q)]=1

            sg=0
            nsg=0
            pr=0
            nr=0
            s1=0
            s2=0
            p1=1
            lst=[]
            lst1=[]
            lst2=[]
            lst3=[]

            #to claculate probablitu of good and day in review (denominator)
            for j in range(0,len(txt)):
                s1=0
                for i in range(0,7000):
                    if(z[i][j]==1):s1+=1
                ps1=s1/7000
                if(ps1==0):ps1=0.0001
                lst1.append(ps1)
               
            for m in lst1:
                p1=p1*m


            count=np.zeros((20,len(txt)))
            count1=np.zeros(20)

            #counting emoji
            for i in range(0,7000):
                count1[y[i]]+=1

            #to calculate numerator  
            for j in range(0,len(txt)):
                for i in range(0,7000):
                    if(z[i][j]==1):count[y[i],j]+=1
            for i in range(0,20):
                for j in range(0,len(txt)):
                    if(count[i,j]==0):count[i,j]=100
                   
            for i in range(0,20):
                pg=1
                for j in range(0,len(txt)):
                    pg=pg*(count[i][j]/count1[i])
                pg=pg*(count1[i]/7000)
                lst.append(pg)
            for i in range(0,20):    
                pp0=lst[i]/(p1)
                lst3.append(pp0)

            #regularization
            for i in range(0,20):
                lst2.append(lst3[i]/sum(lst3))  


            #printing emojies and probablity
            for i in range(0,20):
                print(emotes[i],lst2[i],sep=' ')

            #finding second max
            sm=float(0)
            for yr in lst2:
                if(yr!=max(lst2) and yr>sm):sm=yr

            #finding third max
            tm=0
            for t in lst2:
                if(t!=max(lst2) and t!=sm and  t>tm):tm=t

            print(" ")
            print("recommended emojies")
            result = [emotes[lst2.index(max(lst2))],emotes[lst2.index(sm)],emotes[lst2.index(tm)]
            ]
            return result
        except Exception as e:
                return {
                    "exception" : str(e),
                }
    
api.add_resource(GaneshTesting,'/ganesh')

if __name__ == '__main__':
    app.run()
