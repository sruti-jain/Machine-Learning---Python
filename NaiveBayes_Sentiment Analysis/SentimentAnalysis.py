import re, random

PRINT_ERRORS=0

def readFiles(sentimentDictionary,sentencesTrain,sentencesTest,sentences):

    posSentences = open('posSentences', 'r', encoding="ISO-8859-1")
    posSentences = re.split(r'\n', posSentences.read())

    negSentences = open('negSentences', 'r', encoding="ISO-8859-1")
    negSentences = re.split(r'\n', negSentences.read())

    positive = open('positive.txt', 'r')
    negative = re.split(r'\n', positive.read())

    negative = open('negative.txt', 'r', encoding="ISO-8859-1")
    negative = re.split(r'\n', negative.read())
 
    posDictionary = open('positive-words.txt', 'r', encoding="ISO-8859-1")
    posWordList = re.findall(r"[a-z\-]+", posDictionary.read())

    negDictionary = open('negative-words.txt', 'r', encoding="ISO-8859-1")
    negWordList = re.findall(r"[a-z\-]+", negDictionary.read())

    for i in posWordList:
        sentimentDictionary[i] = 1
    for i in negWordList:
        sentimentDictionary[i] = -1

    #create Training and Test Datsets:
    for i in posSentences:
        if random.randint(1,10)<2:
            sentencesTest[i]="positive"
        else:
            sentencesTrain[i]="positive"

    for i in negSentences:
        if random.randint(1,10)<2:
            sentencesTest[i]="negative"
        else:
            sentencesTrain[i]="negative"

    for i in positive:
            sentences[i]="positive"
    for i in negative:
            sentences[i]="negative"

def trainBayes(sentencesTrain, pWordPos, pWordNeg, pWord):
    freqPositive = {}
    freqNegative = {}
    dictionary = {}
    posWordsTot = 0
    negWordsTot = 0
    allWordsTot = 0

    for sentence, sentiment in sentencesTrain.items():
        wordList = re.findall(r"[\w']+", sentence)
        bigramList=wordList
        for x in range(len(wordList)-1):
           bigramList.append(wordList[x]+"_" + wordList[x+1])
        
        for word in bigramList: #calculate over bigrams
            allWordsTot += 1 # keeps count of total words in dataset
            if not (word in dictionary):
                dictionary[word] = 1
            if sentiment=="positive" :
                posWordsTot += 1 # keeps count of total words in positive class

                #keep count of each word in positive context
                if not (word in freqPositive):
                    freqPositive[word] = 1
                else:
                    freqPositive[word] += 1    
            else:
                negWordsTot+=1# keeps count of total words in negative class
                
                #keep count of each word in positive context
                if not (word in freqNegative):
                    freqNegative[word] = 1
                else:
                    freqNegative[word] += 1

    for word in dictionary:
        if not (word in freqNegative):
            freqNegative[word] = 1
        if not (word in freqPositive):
            freqPositive[word] = 1

        # Calculate p(word|positive)
        pWordPos[word] = freqPositive[word] / float(posWordsTot)

        # Calculate p(word|negative) 
        pWordNeg[word] = freqNegative[word] / float(negWordsTot)

        # Calculate p(word)
        pWord[word] = (freqPositive[word] + freqNegative[word]) / float(allWordsTot) 

def testBayes(sentencesTest, dataName, pWordPos, pWordNeg, pWord,pPos):
    pNeg=1-pPos

    total=0
    correct=0
    totalpos=0
    totalpospred=0
    totalneg=0
    totalnegpred=0
    correctpos=0
    correctneg=0

    #for each sentence, sentiment pair in the dataset
    for sentence, sentiment in sentencesTest.items():
        wordList = re.findall(r"[\w']+", sentence)


        bigramList=wordList #initialise bigramList
        for x in range(len(wordList)-1):
           bigramList.append(wordList[x]+"_" + wordList[x+1])

        pPosW=pPos
        pNegW=pNeg

        for word in bigramList: #calculate over bigrams
            if word in pWord:
                if pWord[word]>0.00000001:
                    pPosW *=pWordPos[word]
                    pNegW *=pWordNeg[word]

        prob=0;            
        if pPosW+pNegW >0:
            prob=pPosW/float(pPosW+pNegW)


        total+=1
        if sentiment=="positive":
            totalpos+=1
            if prob>0.5:
                correct+=1
                correctpos+=1
                totalpospred+=1
            else:
                correct+=0
                totalnegpred+=1
                if PRINT_ERRORS:
                    print ("ERROR (pos classed as neg %0.2f):" %prob + sentence)
        else:
            totalneg+=1
            if prob<=0.5:
                correct+=1
                correctneg+=1
                totalnegpred+=1
            else:
                correct+=0
                totalpospred+=1
                if PRINT_ERRORS:
                    print ("ERROR (neg classed as pos %0.2f):" %prob + sentence)
 
    acc=correct/float(total)
    print (dataName + " Accuracy (All)=%0.2f" % acc + " (%d" % correct + "/%d" % total + ")\n")

    precision_pos=correctpos/float(totalpospred)
    recall_pos=correctpos/float(totalpos)
    precision_neg=correctneg/float(totalnegpred)
    recall_neg=correctneg/float(totalneg)
    f_pos=2*precision_pos*recall_pos/(precision_pos+recall_pos);
    f_neg=2*precision_neg*recall_neg/(precision_neg+recall_neg);

    print (dataName + " Precision (Pos)=%0.2f" % precision_pos + " (%d" % correctpos + "/%d" % totalpospred + ")")
    print (dataName + " Recall (Pos)=%0.2f" % recall_pos + " (%d" % correctpos + "/%d" % totalpos + ")")
    print (dataName + " F-measure (Pos)=%0.2f" % f_pos)

    print (dataName + " Precision (Neg)=%0.2f" % precision_neg + " (%d" % correctneg + "/%d" % totalnegpred + ")")
    print (dataName + " Recall (Neg)=%0.2f" % recall_neg + " (%d" % correctneg + "/%d" % totalneg + ")")
    print (dataName + " F-measure (Neg)=%0.2f" % f_neg + "\n")


def testDictionary(sentencesTest, dataName, sentimentDictionary, threshold):
    total=0
    correct=0
    totalpos=0
    totalneg=0
    totalpospred=0
    totalnegpred=0
    correctpos=0
    correctneg=0
    for sentence, sentiment in sentencesTest.items():
        Words = re.findall(r"[\w']+", sentence)
        score=0
        for word in Words:
            if word in sentimentDictionary:
               score+=sentimentDictionary[word]
 
        total+=1
        if sentiment=="positive":
            totalpos+=1
            if score>=threshold:
                correct+=1
                correctpos+=1
                totalpospred+=1
            else:
                correct+=0
                totalnegpred+=1
        else:
            totalneg+=1
            if score<threshold:
                correct+=1
                correctneg+=1
                totalnegpred+=1
            else:
                correct+=0
                totalpospred+=1
 
    acc=correct/float(total)
    print (dataName + " Accuracy (All)=%0.2f" % acc + " (%d" % correct + "/%d" % total + ")\n")
    precision_pos=correctpos/float(totalpospred)
    recall_pos=correctpos/float(totalpos)
    precision_neg=correctneg/float(totalnegpred)
    recall_neg=correctneg/float(totalneg)
    f_pos=2*precision_pos*recall_pos/(precision_pos+recall_pos);
    f_neg=2*precision_neg*recall_neg/(precision_neg+recall_neg);


    print (dataName + " Precision (Pos)=%0.2f" % precision_pos + " (%d" % correctpos + "/%d" % totalpospred + ")")
    print (dataName + " Recall (Pos)=%0.2f" % recall_pos + " (%d" % correctpos + "/%d" % totalpos + ")")
    print (dataName + " F-measure (Pos)=%0.2f" % f_pos)

    print (dataName + " Precision (Neg)=%0.2f" % precision_neg + " (%d" % correctneg + "/%d" % totalnegpred + ")")
    print (dataName + " Recall (Neg)=%0.2f" % recall_neg + " (%d" % correctneg + "/%d" % totalneg + ")")
    print (dataName + " F-measure (Neg)=%0.2f" % f_neg + "\n")


def mostUseful(pWordPos, pWordNeg, pWord, n):
    predictPower={}
    for word in pWord:
        if pWordNeg[word]<0.0000001:
            predictPower=1000000000
        else:
            predictPower[word]=pWordPos[word] / (pWordPos[word] + pWordNeg[word])

            
    sortedPower = sorted(predictPower, key=predictPower.get)
    head, tail = sortedPower[:n], sortedPower[len(predictPower)-n:]
    print("NEGATIVE:")
    print(head)
    print("\nPOSITIVE:")
    print(tail)


# Main Script ======================================================================================


sentimentDictionary={} # {} initialises a dictionary [hash function]
sentencesTrain={}
sentencesTest={}
sentences={}

#initialise datasets and dictionaries
readFiles(sentimentDictionary,sentencesTrain,sentencesTest,sentences)

pWordPos={} # p(W|Positive)
pWordNeg={} # p(W|Negative)
pWord={}    # p(W) 

trainBayes(sentencesTrain, pWordPos, pWordNeg, pWord)

print ("Naive Bayes")

testBayes(sentencesTest,  "Films  (Test Data, Naive Bayes)\t", pWordPos, pWordNeg, pWord,0.5)

testDictionary(sentencesTest,  "Films  (Test Data, Rule-Based)\t",  sentimentDictionary, -4)

mostUseful(pWordPos, pWordNeg, pWord, 100) # print most useful words