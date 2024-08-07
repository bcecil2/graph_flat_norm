import numpy as np
import pickle
import sys
from new_node import newnode

sys.setrecursionlimit(100000)
    
def parse_table(filename):
    table = np.loadtxt(filename,delimiter=",")
    angles = table[:,0]
    values = table[:,1]
    return angles,values
        
def insert_node(root,angle,value):
    if root == None:
        return newnode(angle,value)
    if angle < root.key:
        root.left = insert_node(root.left,angle,value)
    if angle > root.key:
        root.right = insert_node(root.right,angle,value)
    return root

def maxDiffUtil(ptr, k, min_diff, min_diff_key):
    if ptr == None:
        return
         
    if ptr.key == k:
        min_diff_key[0] = k
        min_diff_key[1] = ptr.integral_value
        return
 
    if min_diff > abs(ptr.key - k):
        min_diff = abs(ptr.key - k)
        min_diff_key[0] = ptr.key
        min_diff_key[1] = ptr.integral_value
 
    if k < ptr.key:
        maxDiffUtil(ptr.left, k, min_diff,
                                 min_diff_key)
    else:
        maxDiffUtil(ptr.right, k, min_diff,
                                  min_diff_key)

def closest_angle(root, k):
    min_diff, min_diff_key = 999999999999, [-1,-1]
    maxDiffUtil(root, k, min_diff, min_diff_key)
    return min_diff_key

def build_tree(angles,values):
    """assumes input is sorted"""
    midpt = len(angles)//2
    root = newnode(angles[midpt],values[midpt])
    for i,entry in enumerate(angles):
        insert_node(root,entry,values[i])
    return root

def save_tree(root,filename):
    file = open(filename,"wb")
    pickle.dump(root,file)
    
def load_tree(filename):
    file = open(filename,"rb")
    return pickle.load(file)
 
def build_binary_search_table(filename):
    angles,values = parse_table(filename)
    return build_tree(angles,values)
    
if __name__ == '__main__':
    table_filename = "2d_lookup_table100000.txt"
    root = build_binary_search_table(table_filename)
    save_tree(root,"2d_lookup_tree100k.txt")
    # imported_tree = load_tree("3d_lookup_tree2k.txt")
    # k = 0
    # print(closest_angle(imported_tree,k)[1]) #integral associated with closest angle to k
