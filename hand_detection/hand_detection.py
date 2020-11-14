# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 16:49:58 2020

@author: jdfab
"""
import os
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np



cards = pd.read_csv('C:/Users/jdfab/Dropbox/Bridge/OCR/test_files/cards.names',header=None)
cards.columns = ['card']

file_path = 'C:/Users/jdfab/Dropbox/Bridge/OCR/test_files/test10_out/6/'
file_path = 'C:/Users/jdfab/Dropbox/Bridge/OCR/test_files/'

def get_data_frame_from_files(file_path,cards,confidence_threshold=95):
    '''
    This function should return a dataframe with the best guess
    of the position of each card, given the output of the predictive model
    
    The output is a dataframe with index 'card' which contains card names 
    in the format 'AC', '6H', etc., and two numerical columns.
    
    Current logic is *very* simple- just take the mean of all predictions 
    above a certain confidence level to get the x,y coordinates. 
    This seems to give passable reuslts on some outputs, but needs more work.
      
    '''

    boxes = {}
    
    for file_name in os.listdir(file_path):
        if file_name[-4:] == '.txt':
            if os.path.getsize(file_path+file_name) > 0:
                print(file_name)
                file_data = pd.read_csv(file_path + file_name,header=None, sep = ' ')
                file_data.columns = ['label','x_ratio','y_ratio','width','height','confidence']
                file_data['Name'] = file_name[file_name.find('_')+1:-4]
                boxes[file_name] = file_data
    
    
    boxes = pd.concat(boxes[file_name] for file_name in boxes)

    boxes[['top_x','top_y']] = boxes['Name'].str.split('_',expand=True)
    
    
    boxes['top_x'] = pd.to_numeric(boxes['top_x'])
    boxes['top_y'] = pd.to_numeric(boxes['top_y'])
    boxes = boxes.merge(cards,left_on = 'label',right_index=True)
                
    boxes['x'] = boxes.x_ratio*720 + boxes.top_x
    
    boxes['y'] = boxes.y_ratio*720 + boxes.top_y     
    
    boxes = boxes[['confidence','x','y','card']]

    if boxes.confidence.mean() < 1: 
        confidence_threshold = confidence_threshold/100
    confident = boxes[boxes.confidence > confidence_threshold]

    
    confident = confident[['card','x','y']].groupby('card').mean()
    return confident


def add_players_to_card_positions(card_positions):
    '''
    take a dataframe with cards and positions, and label each card according to which hand it's in
    NB - assumes 'north' is at the top. 
    keyword argument: 
        card_positions: a dataframe, with columns ['card','x','y']
    
    adds a column 'player' with the label (from 'nesw') of the hand that card is in
    '''
    kmeans = KMeans(n_clusters=4).fit(card_positions)
    centroids = kmeans.cluster_centers_
    hands = dict()
    hands[np.argmin(centroids[:,1])] = 'e'
    hands[np.argmax(centroids[:,1])] = 's'
    hands[np.argmax(centroids[:,0])] = 'w'
    hands[np.argmin(centroids[:,0])] = 'n'
    
    #This is a cool visualisation, and useful for testing, but not really required, just leaving it here for now
    #plt.scatter(card_positions['x'], card_positions['y'], c= kmeans.labels_.astype(float), s=50, alpha=0.5)
    #plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
    #plt.show()
        
    card_positions['player'] = kmeans.predict(card_positions)
    card_positions['player'] = card_positions['player'].replace(hands)
    return card_positions
    
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

    #TODO - doesn't handle voids properly
    for player in 'nesw':
        pbn += '.'.join(cards[cards.player == player].value)
        pbn += ' ' 
    pbn = pbn[:-1] + '"]'
    
    return pbn
    
if __name__ == '__main__':
    confident = get_data_frame_from_files(file_path,cards)
    card_positions = add_players_to_card_positions(confident)
    card_positions = card_positions.reset_index()
    card_positions.card = card_positions.card.str.replace('10','T')
    card_positions['value'] = card_positions.card.str.slice(0,1)
    card_positions['suit'] = card_positions.card.str.slice(1)
    print(dataframe_to_pbn(card_positions[['player','value','suit']]))
