# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 22:42:13 2020

@author: jdfab
"""


import xml.etree.ElementTree as et
import pandas as pd
from sklearn.cluster import KMeans
#import matplotlib.pyplot as plt
import numpy as np
import sys
import os



def get_data_frame_from_xml(xml_fn):
    ''' Takes an XML in the format we are using for input into the OCR model,
        and outputs a dataframe with the same information
    '''
    
    cards_list = list()

    tree = et.parse(xml_fn)
    root = tree.getroot()

    for obj in root.iter('object'):
        card = obj.find('name').text
        xmlbox = obj.find('bndbox')
        xmin = xmlbox.find('xmin').text
        xmax = xmlbox.find('xmax').text
        ymin = xmlbox.find('ymin').text
        ymax = xmlbox.find('ymax').text
        # all the data formats seem to use 'T' rather than '10', so we'll do that as we impor the XML
        cards_list.append([card.replace('10','T'),xmin,xmax,ymin,ymax])
    return pd.DataFrame(cards_list,columns = ['card','xmin','xmax','ymin','ymax'])


def get_hands_from_filename(filename):
    ''' Takes an XML in the format we are using for input into the OCR model,
        and returns a dataframe with hand labels (using K-means clustering)
    '''
    
    cards = get_data_frame_from_xml(filename)
    
    #check the data types are ok
    cards.xmin = pd.to_numeric(cards.xmin)
    cards.xmax = pd.to_numeric(cards.xmax)
    cards.ymin = pd.to_numeric(cards.ymin)
    cards.ymax = pd.to_numeric(cards.ymax)
    
    #cards only have 4 corners. If there are any more, then something is wrong. 
        
    if cards.groupby('card').count().max().max() > 4:
        counts = cards.groupby('card').count().max(axis = 1)
        too_many = counts[counts > 4].index
        print('There is a duplicated card in the xml: %s' %(' '.join(too_many)))
    
    
    #we'll leave the card name as the index here, so we can just pass the whole data frame to the clustering algorithm
    all_cards = [value + suit for value in list('23456789TJQK') for suit in list('CDHS')]
    print(all_cards)
    cards = cards.groupby('card').mean()
    if len(cards) < 52:

        missing_cards = set(all_cards).difference(set(cards.index))
        print('These cards are missing: %s' %(' '.join(missing_cards)))
    
    kmeans = KMeans(n_clusters=4).fit(cards)
    centroids = kmeans.cluster_centers_
    hands = dict()
    
    #Assume the hand with the largest y-coordinate is north, and so on.
    #Note that there are two x and two y coordinates in 'centroids',
    #which is why we use indices 0 and 2 below.
    
    hands[np.argmax(centroids[:,2])] = 'n'
    hands[np.argmin(centroids[:,2])] = 's'
    hands[np.argmin(centroids[:,0])] = 'e'
    hands[np.argmax(centroids[:,0])] = 'w'
    
    
    #This code was used in testing, but haven't included matplotlib in requirements, so I'm removing it for now
    #plt.scatter(cards['xmin'], cards['ymin'], c= kmeans.labels_.astype(float), s=50, alpha=0.5)
    #plt.scatter(centroids[:, 0], centroids[:, 2], c='red', s=50)
    #plt.show()
    
    cards['player'] = kmeans.predict(cards)
    cards['player'] = cards['player'].replace(hands)
    
    if cards.player.value_counts().min() < 13: 
        print('one of the hands doesnt have enough cards')
    
    print(cards.player.value_counts())
    if cards.player.value_counts().max() > 13: 
        print('one of the hands has too many cards')
        
    # we need to distinguish suit and value, so let's do it now

    
    cards = cards.reset_index()
    cards['suit'] = cards.card.str[-1]
    cards['value'] = cards.card.str[:-1]
    
    return cards[['player','suit','value']].sort_values(['player','suit'],ascending =[True,False])



def dataframe_to_pbn(cards):
    ''' takes a dataframe with columns 'labels', 'suit' and 'value' 
    and returns a pbn of the hand
    '''
    
    #group the cards for each player by suit, and then sort them in descending order
    cards = cards.groupby(['player','suit']).sum().reset_index()
    cards.value = cards.value.apply(lambda x: ''.join(sorted(x, key = lambda y: -'23456789TJQKA'.index(y))))
    
    #The correct order for the suits is reverse alphabetical order
    cards = cards.sort_values('suit',ascending = False)
    pbn = '[Deal "N:'

    
    for player in 'nesw':
        pbn += '.'.join(cards[cards.player == player].value)
        pbn += ' ' 
    pbn = pbn[:-1] + '"]'
    
    return pbn
    
if __name__ == '__main__':
    #for testing
    # 'test_errors.xml' has 8 references to the JD, and no 8S.

    if sys.argv[1] == 'use_default':
        filename = 'C:/Users/jdfab/Dropbox/Bridge/bridge_hackathon/OCR/hand_detection/IMG_20200621_114258.xml'
    else:
        filename = os.path.join(os.getcwd(),'hand_detection',sys.argv[1])

    cards = get_hands_from_filename(filename)
    pbn = dataframe_to_pbn(cards)

    print(pbn)


    