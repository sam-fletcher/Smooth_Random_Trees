''' A class that builds Differentially Private Random Decision Trees, using Smooth Sensitivity '''

from collections import Counter, defaultdict
import random
import numpy as np
import math
from scipy import stats # for Exponential Mechanism
import os
import multiprocessing as multi

NO_NOISE = False # for making random trees without any privacy parameters
SMOOTH_SENSITIVITY = True # not used by JAG
DISJOINT_DATA = False # FALSE = Kellaris13 samples. Each tree gets a subset of the dataset, and the full epsilon budget.
JAG = False # Jagannathan's DP RDT. assumes NOT smooth_sensitivity and NOT disjoint data
MULTI_THREAD = False

class DP_RDT_2016:    
    ''' Make a forest of Random Trees, then filter the training data through each tree to fill the leafs. '''
    def __init__(self, attribute_indexes, attribute_domains, categs, num_trees, max_depth, epsilon, class_values, train, test):
        self._attribute_domains = attribute_domains
        self._categs = categs
        self._num_trees = num_trees
        self._max_depth = max_depth
        self._missed_records = []
        self._flipped_majorities = []
        self._av_sensitivity = []
        self._empty_leafs = []

        random.shuffle(train)
        class_labels = [int(x) for x in class_values]
        actual_labels = [int(x[0]) for x in test]
        voted_labels = [defaultdict(int) for x in test]
        
        ''' SELECT THE NUMBER OF CORES TO USE '''
        if MULTI_THREAD:        
            num_threads = multi.cpu_count() - 2
            pool = multi.Pool(processes = num_threads)
            processes = []

        subset_size = int(len(train)/self._num_trees) # if using subsets, we don't need to divide the epsilon budget
        curr_tree = 0
        for i in range(self._num_trees):
            if MULTI_THREAD:
                if DISJOINT_DATA: 
                    processes.append(pool.apply_async(self.build_tree, (attribute_indexes, train[i*subset_size:(i+1)*subset_size], epsilon, class_values, test)))
                else:
                    #processes.append(pool.apply_async(self.build_tree, (attribute_indexes, train, epsilon/float(self._num_trees), class_values, test)))
                    ''' random sub-samples using Kellaris13 '''
                    new_train = random.sample(train, subset_size)
                    p = 1./float(self._num_trees) # P CAN BE ANY USER-DEFINED NUMBER BETWEEN 0 AND 1
                    new_epsilon = math.log( 1 + (math.exp(epsilon/float(self._num_trees))-1.) / p)
                    processes.append(pool.apply_async(self.build_tree, (attribute_indexes, new_train, new_epsilon, class_values, test)))
            else: # not multi-threaded
                if DISJOINT_DATA: 
                    results = self.build_tree(attribute_indexes, train[i*subset_size:(i+1)*subset_size], epsilon, class_values, test)
                else:
                    results = self.build_tree(attribute_indexes, train, epsilon/float(self._num_trees), class_values, test)

                curr_votes = results['voted_labels']
                for rec_index in range(len(test)):
                    for lab in class_labels:
                        voted_labels[rec_index][lab] += curr_votes[rec_index][lab]
                self._missed_records.append(results['missed_records'])
                self._flipped_majorities.append(results['flipped_majorities'])
                self._av_sensitivity.append(results['av_sensitivity'])
                self._empty_leafs.append(results['empty_leafs'])
                curr_tree += 1
                if curr_tree >= self._num_trees: break

        if MULTI_THREAD:        
            for i in range(self._num_trees):
                print(str(i), end=' ')
                curr_votes = processes[i].get()['voted_labels']
                for rec_index in range(len(test)):
                    for lab in class_labels:
                        voted_labels[rec_index][lab] += curr_votes[rec_index][lab]
                self._missed_records.append(processes[i].get()['missed_records'])
                self._flipped_majorities.append(processes[i].get()['flipped_majorities'])
                self._av_sensitivity.append(processes[i].get()['av_sensitivity'])
                self._empty_leafs.append(processes[i].get()['empty_leafs'])
            pool.close()

        final_predictions = []
        for i,rec in enumerate(test):
            final_predictions.append( Counter(voted_labels[i]).most_common(1)[0][0] )
        counts = Counter([x == y for x, y in zip(final_predictions, actual_labels)])
        self._predicted_labels = final_predictions
        self._accuracy = float(counts[True]) / len(test)
        
    
    def build_tree(self, attribute_indexes, train, epsilon, class_values, test):
        root = random.choice(attribute_indexes)
        tree = Tree(attribute_indexes, root, self)
        tree.filter_training_data_and_count(train, epsilon, class_values)
        missed_records = tree._missed_records
        flipped_majorities = tree._flip_fraction
        av_sensitivity = tree._av_sensitivity
        empty_leafs = tree._empty_leafs
        voted_labels = [defaultdict(int) for x in test]
        for i,rec in enumerate(test):
            label = tree.classify(tree._root_node, rec)
            voted_labels[i][label] += 1
        del tree
        return {'voted_labels':voted_labels, 'missed_records':missed_records, 'flipped_majorities':flipped_majorities, 
                'av_sensitivity':av_sensitivity, 'empty_leafs':empty_leafs}


class Tree(DP_RDT_2016):
    ''' Set the root for this tree and then start the random-tree-building process. '''
    def __init__(self, attribute_indexes, root_attribute, pc):
        self._id = 0
        self._categs = pc._categs
        self._max_depth = pc._max_depth
        self._num_leafs = 0

        root = node(None, None, root_attribute, 1, 0, []) # the root node is level 1
        attribute_domains = pc._attribute_domains
        
        if root_attribute not in self._categs: # numerical attribute
            split_val = random.uniform(attribute_domains[str(root_attribute)][0], attribute_domains[str(root_attribute)][1])
            left_domain = {k : v if k!=str(root_attribute) else [v[0], split_val] for k,v in attribute_domains.items() }
            right_domain = {k : v if k!=str(root_attribute) else [split_val, v[1]] for k,v in attribute_domains.items() }
            root.add_child( self.make_children([x for x in attribute_indexes], root, 2, '<'+str(split_val), split_val, left_domain) ) # left child
            root.add_child( self.make_children([x for x in attribute_indexes], root, 2, '>='+str(split_val), split_val, right_domain) ) # right child
        else: # categorical attribute
            for value in attribute_domains[str(root_attribute)]: 
                root.add_child( self.make_children([x for x in attribute_indexes if x!=root_attribute], root, 2, value, None, attribute_domains) ) # categorical attributes can't be tested again
        self._root_node = root
                
    ''' Recursively make all the child nodes for the current node, until a termination condition is met. '''
    def make_children(self, candidate_atts, parent_node, current_depth, splitting_value_from_parent, svfp_numer, attribute_domains):
        self._id += 1
        if not candidate_atts or current_depth >= self._max_depth+1: # termination conditions. leaf nodes don't count to the depth.
            self._num_leafs += 1
            return node(parent_node, splitting_value_from_parent, None, current_depth, self._id, None, svfp_numer=svfp_numer) 
        else:
            new_splitting_attr = random.choice(candidate_atts) # pick the attribute that this node will split on
            current_node = node(parent_node, splitting_value_from_parent, new_splitting_attr, current_depth, self._id, [], svfp_numer=svfp_numer) # make a new node

            if new_splitting_attr not in self._categs: # numerical attribute
                split_val = random.uniform(attribute_domains[str(new_splitting_attr)][0], attribute_domains[str(new_splitting_attr)][1])
                left_domain = {k : v if k!=str(new_splitting_attr) else [v[0], split_val] for k,v in attribute_domains.items() }
                right_domain = {k : v if k!=str(new_splitting_attr) else [split_val, v[1]] for k,v in attribute_domains.items() }
                current_node.add_child( self.make_children([x for x in candidate_atts], current_node, current_depth+1, '<', split_val, left_domain) ) # left child
                current_node.add_child( self.make_children([x for x in candidate_atts], current_node, current_depth+1, '>=', split_val, right_domain) ) # right child
            else: # categorical attribute
                for value in attribute_domains[str(new_splitting_attr)]: # for every value in the splitting attribute
                    child_node = self.make_children([x for x in candidate_atts if x!=new_splitting_attr], current_node, current_depth+1, value, None, attribute_domains)
                    current_node.add_child( child_node ) # add children to the new node
            return current_node


    ''' Record which leaf each training record belongs to, and then set the (noisy) majority label. '''
    def filter_training_data_and_count(self, records, epsilon, class_values):
        ''' epsilon = the epsilon budget for this tree (each leaf is disjoint, so the budget can be re-used). '''
        num_unclassified = 0.
        for rec in records:
            num_unclassified += self.filter_record(rec, self._root_node, class_index=0)
        self._missed_records = num_unclassified
        flipped_majorities, empty_leafs, sensitivities = self.set_all_noisy_majorities(epsilon, self._root_node, class_values, 0, 0, [])
        self._av_sensitivity = np.mean(sensitivities) # excludes empty leafs

        if self._num_leafs == 0:
            print("\n\n~~~ WARNING: NO LEAFS. num_unclassified = "+str(num_unclassified)+" ~~~\n\n")
            self._empty_leafs = -1.0
        else:
            self._empty_leafs = empty_leafs / float(self._num_leafs)

        if empty_leafs == self._num_leafs:
            print("\n\n~~~ WARNING: all leafs are empty. num_unclassified = "+str(num_unclassified)+" ~~~\n\n")
            self._flip_fraction = -1.0
        else:
            self._flip_fraction = flipped_majorities / float(self._num_leafs-empty_leafs)
    
    def filter_record(self, record, node, class_index=0):
        if not node:
            return 0.00001 # doesn't happen in my experience
        if not node._children: # if leaf
            node.increment_class_count(record[class_index])
            return 0.
        else:
            child = None
            if node._splitting_attribute not in self._categs: # numerical attribute
                rec_val = record[node._splitting_attribute]
                for i in node._children:
                    if i._split_value_from_parent.startswith('<') and rec_val < i._svfp_numer:
                        child = i
                        break
                    if i._split_value_from_parent.startswith('>=') and rec_val >= i._svfp_numer:
                        child = i
                        break
            else: # categorical attribute
                rec_val = str(record[node._splitting_attribute])
                for i in node._children:
                    if i._split_value_from_parent == rec_val:
                        child = i
                        break
            if child is None and node._splitting_attribute in self._categs: # if the record's value couldn't be found:
                #print(str([i._split_value_from_parent,])+" vs "+str([record[node._splitting_attribute],])+" out of "+str(len(node._children)))
                return 1.
            elif child is None: # if the record's value couldn't be found:
                return 0.001
            return self.filter_record(record, child, class_index)

    def set_all_noisy_majorities(self, epsilon, node, class_values, flipped_majorities, empty_leafs, sensitivities):
        if node._children:
            for child in node._children:
                flipped_majorities, empty_leafs, sensitivities = self.set_all_noisy_majorities(
                                                        epsilon, child, class_values, flipped_majorities, empty_leafs, sensitivities)
        else:
            flipped_majorities += node.set_noisy_majority(epsilon, class_values)
            empty_leafs += node._empty
            if node._sensitivity>=0.0: sensitivities.append(node._sensitivity)
        return flipped_majorities, empty_leafs, sensitivities


    def classify(self, node, record):
        if not node:
            return None
        elif not node._children: # if leaf
            return node._noisy_majority
        else: # if parent
            attr = node._splitting_attribute
            child = None
            if node._splitting_attribute not in self._categs: # numerical attribute
                rec_val = record[attr]
                for i in node._children:
                    if i._split_value_from_parent.startswith('<') and rec_val < i._svfp_numer:
                        child = i
                        break
                    if i._split_value_from_parent.startswith('>=') and rec_val >= i._svfp_numer:
                        child = i
                        break
            else: # categorical attribute
                rec_val = str(record[attr])
                for i in node._children:
                    if i._split_value_from_parent == rec_val:
                        child = i
                        break
            if child is None: # if the record's value couldn't be found, just return the latest majority value
                return node._noisy_majority #majority_value, majority_fraction

            return self.classify(child, record)