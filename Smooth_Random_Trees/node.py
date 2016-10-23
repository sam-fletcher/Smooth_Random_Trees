''' A class defining the nodes in our Differentially Private Random Decision Forest '''

class node:
    def __init__(self, parent_node, split_value_from_parent, splitting_attribute, tree_level, id, children, svfp_numer=None):
        self._parent_node = parent_node
        self._split_value_from_parent = split_value_from_parent
        self._svfp_numer = svfp_numer
        self._splitting_attribute = splitting_attribute
        #self._level = tree_level # comment out unless needed. saves memory.
        #self._id = id # comment out unless needed. saves memory.
        self._children = children
        self._class_counts = defaultdict(int)
        self._noisy_majority = None
        self._empty = 0 # 1 if leaf and has no records
        self._sensitivity = -1.0

    def add_child(self, child_node):
        self._children.append(child_node)

    def increment_class_count(self, class_value):
        self._class_counts[class_value] += 1

    def set_noisy_majority(self, epsilon, class_values):
        if not self._noisy_majority and not self._children: # to make sure this code is only run once per leaf
            for val in class_values:
                if val not in self._class_counts: self._class_counts[val] = 0

            if max([v for k,v in self._class_counts.items()]) < 1:
                self._empty = 1
                self._noisy_majority = random.choice([k for k,v in self._class_counts.items()])
                return 0 # we dont want to count purely random flips
            else:
                ''' SMOOTH SENSITIVITY OR GLOBAL SENSITIVITY '''
                if SMOOTH_SENSITIVITY and not JAG:
                    all_counts = sorted([v for k,v in self._class_counts.items()], reverse=True)
                    count_difference = all_counts[0] - all_counts[1]
                    self._sensitivity = math.exp(-1 * count_difference * epsilon)
                    self._sens_of_sens = 1.
                    self._noisy_sensitivity = 1.
                else:
                    self._sensitivity = 1.
                    self._sens_of_sens = 1. # this line is only here to mirror the code in the Heuristics version
                    self._noisy_sensitivity = 1. # this line is only here to mirror the code in the Heuristics version

                if not JAG:
                    self._noisy_majority = self.expo_mech(epsilon, self._sensitivity, self._class_counts)
                else: # if Jagannathan's RDT
                    self._noisy_majority = self.laplace(epsilon, self._class_counts)

                if self._noisy_majority != int(max(self._class_counts.keys(), key=(lambda key: self._class_counts[key]))):
                    #print('majority: '+str(self._noisy_majority)+' vs. max_count: '+str( max(self._class_counts.keys(), key=(lambda key: self._class_counts[key]))))
                    return 1 # we're summing the flipped majorities
                else:
                    return 0
        else: return 0


    def laplace(self, e, counts):
        noisy_counts = {}
        for label,count in counts.items():
            noisy_counts[label] = max( 0, int(count + np.random.laplace(scale=float(1./e))) )
        return int(max(noisy_counts.keys(), key=(lambda key: noisy_counts[key])))


    def expo_mech(self, e, s, counts):
        ''' For this implementation of the Exponetial Mechanism, we use a piecewise linear scoring function,
        where the element with the maximum count has a score of 1, and all other elements have a score of 0. '''
        weighted = []
        if NO_NOISE:
            max_label = max(counts.keys(), key=(lambda key: counts[key]))
            return int(max_label)
        else:
            max_count = max([v for k,v in counts.items()])
        
            for label,count in counts.items():
                ''' if the score is non-monotonic, s needs to be multiplied by 2 '''
                if count == max_count:
                    if s<1.0e-10: power = 50 # e^50 is already astronomical. sizes beyond that dont matter
                    else: power = min( 50, (e*1)/(2*s) ) # score = 1
                else:
                    power = 0 # score = 0
                weighted.append( [label, math.exp(power)] ) 
            sum = 0.
            for label,count in weighted:
                sum += count
            for i in range(len(weighted)):
                weighted[i][1] /= sum   
            customDist = stats.rv_discrete(name='customDist', values=([lab for lab,cou in weighted], [cou for lab,cou in weighted]))
            best = customDist.rvs()
            #print("best_att examples = "+str(customDist.rvs(size=20)))
            return int(best)
