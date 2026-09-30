import pickle5 as pickle
import numpy as np
from numpy import genfromtxt
from collections import defaultdict
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import csv
import re
import tabulate


def load_data():
    item_train = genfromtxt('./数据/content_item_train.csv', delimiter=',')
    user_train = genfromtxt('./数据/content_user_train.csv', delimiter=',')
    y_train    = genfromtxt('./数据/content_y_train.csv', delimiter=',')
    with open('./数据/content_item_train_header.txt', newline='') as f:    #csv 读取器对带引号的字符串处理得更好
        item_features = list(csv.reader(f))[0]
    with open('./数据/content_user_train_header.txt', newline='') as f:
        user_features = list(csv.reader(f))[0]
    item_vecs = genfromtxt('./数据/content_item_vecs.csv', delimiter=',')
       
    movie_dict = defaultdict(dict)
    count = 0
#    with open('./数据/movies.csv', newline='') as csvfile:
    with open('./数据/content_movie_list.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for line in reader:
            if count == 0: 
                count +=1  #跳过表头
                #print(line)
            else:
                count +=1
                movie_id = int(line[0])  
                movie_dict[movie_id]["title"] = line[1]  
                movie_dict[movie_id]["genres"] =line[2]  

    with open('./数据/content_user_to_genre.pickle', 'rb') as f:
        user_to_genre = pickle.load(f)

    return(item_train, user_train, y_train, item_features, user_features, item_vecs, movie_dict, user_to_genre)


def pprint_train(x_train, features,  vs, u_s, maxcount = 5, user=True):
    """ Prints user_train or item_train nicely """
    if user:
        flist = [".0f",".0f",".1f", 
                 ".1f", ".1f", ".1f", ".1f",".1f",".1f", ".1f",".1f",".1f", ".1f",".1f",".1f",".1f",".1f"]
    else:
        flist = [".0f",".0f",".1f", 
                 ".0f",".0f",".0f", ".0f",".0f",".0f", ".0f",".0f",".0f", ".0f",".0f",".0f",".0f",".0f"]

    head = features[:vs]
    if vs < u_s: print("error, vector start {vs} should be greater then user start {u_s}")
    for i in range(u_s):
        head[i] = "[" + head[i] + "]"
    genres = features[vs:]
    hdr = head + genres
    disp = [split_str(hdr, 5)]
    count = 0
    for i in range(0,x_train.shape[0]):
        if count == maxcount: break
        count += 1
        disp.append( [ 
                      x_train[i,0].astype(int),  
                      x_train[i,1].astype(int),   
                      x_train[i,2].astype(float), 
                      *x_train[i,3:].astype(float)
                    ])
    table = tabulate.tabulate(disp, tablefmt='html',headers="firstrow", floatfmt=flist, numalign='center')
    return(table)


def pprint_data(y_p, user_train, item_train, printfull=False):
    np.set_printoptions(precision=1)

    for i in range(0,1000):
        #print(f"{y_p[i,0]: 0.2f}, {ynorm_train.numpy()[i].item(): 0.2f}")
        print(f"{y_pu[i,0]: 0.2f}, {y_train[i]: 0.2f}, ", end='') 
        print(f"{user_train[i,0].astype(int):d}, ",  end='')   # 用户 id
        print(f"{user_train[i,1].astype(int):d}, ", end=''),  #  评分数量
        print(f"{user_train[i,2].astype(float): 0.2f}, ",  end='')       # 评分平均值
        print(": ", end = '')
        print(f"{item_train[i,0].astype(int):d}, ",  end='')   # 电影 id
        print(f"{item_train[i,2].astype(float):0.1f}, ", end='')   # 电影平均评分
        if printfull:
          for j in range(8, user_train.shape[1]):
            print(f"{user_train[i,j].astype(float):0.1f}, ", end='')   # 评分
          print(":", end='')
          for j in range(3, item_train.shape[1]):
            print(f"{item_train[i,j].astype(int):d}, ", end='')   # 评分
          print()
        else:
          a = user_train[i, uvs:user_train.shape[1]]
          b = item_train[i, ivs:item_train.shape[1]]
          c = np.multiply(a,b)
          print(c)

def split_str(ifeatures, smax):
    ofeatures = []
    for s in ifeatures:
        if ' ' not in s:  # 跳过已经有空格的字符串
            if len(s) > smax:
                mid = int(len(s)/2)
                s = s[:mid] + " " + s[mid:]
        ofeatures.append(s)
    return(ofeatures)
    
def pprint_data_tab(y_p, user_train, item_train, uvs, ivs, user_features, item_features, maxcount = 20, printfull=False):
    flist = [".1f", ".1f", ".0f", ".1f", ".0f", ".0f", ".0f",
             ".1f",".1f",".1f",".1f",".1f",".1f",".1f",".1f",".1f",".1f",".1f",".1f",".1f",".1f"]
    user_head = user_features[:uvs]
    genres = user_features[uvs:]
    item_head = item_features[:ivs]
    hdr = ["y_p", "y"] + user_head + item_head + genres
    disp = [split_str(hdr, 5)]
    count = 0
    for i in range(0,y_p.shape[0]):
        if count == maxcount: break
        count += 1
        a = user_train[i, uvs:user_train.shape[1]]
        b = item_train[i, ivs:item_train.shape[1]]
        c = np.multiply(a,b)

        disp.append( [ y_p[i,0], y_train[i], 
                      user_train[i,0].astype(int),   # 用户 id
                      user_train[i,1].astype(int),   # 评分数量
                      user_train[i,2].astype(float), # 用户评分平均值
                      item_train[i,0].astype(int),   # 电影 id
                      item_train[i,1].astype(int),   # 年份
                      item_train[i,2].astype(float),  # 电影平均评分
                      *c
                     ])
    table = tabulate.tabulate(disp, tablefmt='html',headers="firstrow", floatfmt=flist, numalign='center')
    return(table)




def print_pred_movies(y_p, user, item, movie_dict, maxcount=10):
    """打印对新用户的预测结果。输入的预期顺序为
        已排序、未缩放。"""
    count = 0
    movies_listed = defaultdict(int)
    disp = [["y_p", "movie id", "rating ave", "title", "genres"]]

    for i in range(0, y_p.shape[0]):
        if count == maxcount:
            break
        count += 1
        movie_id = item[i, 0].astype(int)
        if movie_id in movies_listed:
            continue
        movies_listed[movie_id] = 1
        disp.append([y_p[i, 0], item[i, 0].astype(int), item[i, 2].astype(float),
                    movie_dict[movie_id]['title'], movie_dict[movie_id]['genres']])

    table = tabulate.tabulate(disp, tablefmt='html',headers="firstrow")
    return(table)

def gen_user_vecs(user_vec, num_items):
    """给定一个用户向量，返回：
        与 item_vecs 尺寸一致的用户预测矩阵"""
    user_vecs = np.tile(user_vec, (num_items, 1))
    return(user_vecs)

# 对所有内容做预测，打印/使用时再筛选
def predict_uservec(user_vecs, item_vecs, model, u_s, i_s, scaler, ScalerUser, ScalerItem, scaledata=False):
    """给定一个用户向量，对 item_vecs 中的所有电影做预测，返回
        按预测评分排序的预测数组，
        以及按预测评分排序索引排列的用户数组和物品数组"""
    if scaledata:
        scaled_user_vecs = ScalerUser.transform(user_vecs)
        scaled_item_vecs = ScalerItem.transform(item_vecs)
        y_p = model.predict([scaled_user_vecs[:, u_s:], scaled_item_vecs[:, i_s:]])
    else:
        y_p = model.predict([user_vecs[:, u_s:], item_vecs[:, i_s:]])
    y_pu = scaler.inverse_transform(y_p)

    if np.any(y_pu < 0) : 
        print("Error, expected all positive predictions")
    sorted_index = np.argsort(-y_pu,axis=0).reshape(-1).tolist()  #取负号以便把评分最高的排在前面
    sorted_ypu   = y_pu[sorted_index]
    sorted_items = item_vecs[sorted_index]
    sorted_user  = user_vecs[sorted_index]
    return(sorted_index, sorted_ypu, sorted_items, sorted_user)


def print_pred_debug(y_p, y, user, item, maxcount=10, onlyrating=False,  printfull=False):
    """希望能复用的打印函数。保留用于调试"""
    count = 0
    for i in range(0, y_p.shape[0]):
        if onlyrating == False or (onlyrating == True and y[i,0] != 0):
            if count == maxcount: break
            count += 1
            print(f"{y_p[i, 0]: 0.2f}, {y[i,0]: 0.2f}, ", end='') 
            print(f"{user[i, 0].astype(int):d}, ",  end='')       # 用户 id
            print(f"{user[i, 1].astype(int):d}, ", end=''),       #  评分数量
            print(f"{user[i, 2].astype(float):0.1f}, ", end=''),       #  评分平均值
            print(": ", end = '')
            print(f"{item[i, 0].astype(int):d}, ",  end='')       # 电影 id
            print(f"{item[i, 2].astype(float):0.1f}, ", end='')   # 电影平均评分
            print(": ", end = '')
            if printfull:
                for j in range(uvs, user.shape[1]):
                    print(f"{user[i, j].astype(float):0.1f}, ", end='') # 评分
                print(":", end='')
                for j in range(ivs, item.shape[1]):
                    print(f"{item[i, j].astype(int):d}, ", end='')    # 评分
                print()
            else:
                a = user[i, uvs:user.shape[1]]
                b = item[i, ivs:item.shape[1]]
                c = np.multiply(a,b)
                print(c)    
                
                
def get_user_vecs(user_id, user_train, item_vecs, user_to_genre):
    """给定 user_id，返回：
        与 item_vecs 尺寸一致的用户训练/预测矩阵
        含所有已评分电影评分、其余为 0 的 y 向量，长度与 item_vecs 一致"""

    if user_id not in user_to_genre:
        print("error: unknown user id")
        return(None)
    else:
        user_vec_found = False
        for i in range(len(user_train)):
            if user_train[i, 0] == user_id:
                user_vec = user_train[i]
                user_vec_found = True
                break
        if not user_vec_found:
            print("error in get_user_vecs, did not find uid in user_train")
        num_items = len(item_vecs)
        user_vecs = np.tile(user_vec, (num_items, 1))

        y = np.zeros(num_items)
        for i in range(num_items):  # 遍历 item_vecs 中的电影，看看用户是否给它们打过分
            movie_id = item_vecs[i, 0]
            if movie_id in user_to_genre[user_id]['movies']:
                rating = user_to_genre[user_id]['movies'][movie_id]
            else:
                rating = 0
            y[i] = rating
    return(user_vecs, y)


def get_item_genre(item, ivs, item_features):
    offset = np.where(item[ivs:] == 1)[0][0]
    genre = item_features[ivs + offset]
    return(genre, offset)


def print_existing_user(y_p, y, user, items, item_features, ivs, uvs, movie_dict, maxcount=10):
    """打印对数据库中已有用户的预测结果。输入的预期顺序为已排序、未缩放。"""
    count = 0
    movies_listed = defaultdict(int)
    disp = [["y_p", "y", "user", "user genre ave", "movie rating ave", "title", "genres"]]
    listed = []
    count = 0
    for i in range(0, y.shape[0]):
        if y[i, 0] != 0:
            if count == maxcount:
                break
            count += 1
            movie_id = items[i, 0].astype(int)

            offset = np.where(items[i, ivs:] == 1)[0][0]
            genre_rating = user[i, uvs + offset]
            genre = item_features[ivs + offset]
            disp.append([y_p[i, 0], y[i, 0],
                        user[i, 0].astype(int),      # 用户 id
                        genre_rating.astype(float),
                        items[i, 2].astype(float),    # 电影平均评分
                        movie_dict[movie_id]['title'], genre])

    table = tabulate.tabulate(disp, tablefmt='html', headers="firstrow", floatfmt=[".1f", ".1f", ".0f", ".2f", ".2f"])
    return(table)