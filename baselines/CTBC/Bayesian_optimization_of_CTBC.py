# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as ss
import itertools
import bintrees
import csv
import os

#black box optimization function

##input strength of combination coverage
##input factors'values and corresponding importance index
N=2
efficiency_index=[]
impr_list=[]
cost_list=[]
parameter_details=[]
indexes_with_max_weight=[]
constrains=[]
sub_constrains=[]
lines=open('parameters_scenario2.txt').readlines()
length=len(lines)
candidate=list(range(length))
combination=list(itertools.combinations(candidate,N))
valid=[]
res=[0]*N
p=[0]*length
#next_input_i = 0

def generate_valid_combination(i,elements,constrains):
    if i==len(elements):
        for constrain in constrains:
            condition=eval(constrain[1])
            if condition:
                assertion=eval(constrain[2])
                if not assertion:
                    return
        weight=0
        for j in range(len(res)):
            weight+=parameter_details[elements[j]][res[j]][1]
        combo=((-weight,tuple(res),tuple(elements)),0)
        valid.append(combo)
        return
    current=elements[i]
    for j in range(len(parameter_details[current])):
        res[i]=j
        p[current]=parameter_details[current][j][0]
        generate_valid_combination(i+1,elements,sub_constrains)
    return 
def validation(current,coverage,constrains):
    p=[0]*length
    elements=set()
    for parameter in coverage:
        p[parameter]=parameter_details[parameter][coverage[parameter]][0]
    for i in range(N):
        parameter=current[2][i]
        p[parameter]=parameter_details[parameter][current[1][i]][0]
    for constrain in constrains:
        flag=True
        for element in constrain[0]:
            if element not in elements:
                flag=False
                break
        if flag:
            condition=eval(constrain[1])
            if condition:
                assertion=eval(constrain[2])
                if not assertion:
                    return False
    return True
#########################################################################################################################################################
def blackboxfunction(x, return_scenarios=False):
#########################################################################################################################################################        
    for line in lines:
        details=[]
        max_weight=0
        max_index=0
        i=0
        for pair in line.split():
            detail=pair.split(',')
            value=str(detail[0])
            weight=float(detail[1])
            if weight>max_weight:
                max_index=i
                max_weight=weight
            details.append((value,weight))
            i+=1
        parameter_details.append(details)
        indexes_with_max_weight.append(max_index)
    
    scenarios_list = []  # List to store test scenarios when return_scenarios=True

    for com in combination:
        elements=set(com)
        sub_constrains=[]
        for constrain in constrains:
            flag=True
            for element in constrain[0]:
                if element not in elements:
                    flag=False
                    break
            if flag:
                sub_constrains.append(constrain)
        #res=[0]*N
        #p=[0]*length
        generate_valid_combination(0,com,sub_constrains)
    
    uncovered=bintrees.RBTree(valid)
    weights=[]
    max_weight=0
    for i in range(length):
        max_weight+=parameter_details[i][indexes_with_max_weight[i]][1]
        
    ##input the complexity improvement coefficient
    threshold=max_weight*x/10

    while uncovered:
        target=uncovered.min_item()[0]
        coverage={}
        covered=set()
        for i in range(N):
            coverage[target[2][i]]=target[1][i]
        covered.add(target)
        current=target

        #if (threshold+target[0])<0:------->threshold[0]
        if (threshold+target[0])<0:
            while len(coverage)<length:
                try:
                    current=uncovered.succ_item(current)[0]
                    flag=True
                    for i in range(N):
                        if current[2][i] in coverage:
                            if coverage[current[2][i]]!=current[1][i]:
                                flag=False
                                break
                    if not flag:
                        continue
                    if not validation(current,coverage,constrains):
                        continue
                    for i in range(N):
                        if current[2][i] not in coverage:
                            coverage[current[2][i]]=current[1][i]
                    covered.add(current)
                except:
                    break
        for combo in covered:
            uncovered.remove(combo)
    
        ##output the test scenarios set
        test_set={}
        weight_sum=0
        for i in range(length):
            if i in coverage:
                test_set[i]=parameter_details[i][coverage[i]][0]
                weight_sum+=parameter_details[i][coverage[i]][1]
            else:
                test_set[i]=parameter_details[i][indexes_with_max_weight[i]][0]
                weight_sum+=parameter_details[i][indexes_with_max_weight[i]][1]
        #print(test_set)
        scenarios_list.append(test_set)  # Collect the scenario
        weights.append(weight_sum)  # Original behavior for optimization
        # weights.append(weight_sum)
    
    ##output the set of test scenarios' complexity index
    #print(weights) 
    
    ##output the number of test scenarios
    cost_index = len(weights)
    #print(len(weights))
    
    improvement_index = sum(weights)/len(weights)
    #print(improvement_index)
    
    Z = 0.6*(improvement_index-0.295917284)/(0.5071-0.295917284)-0.4*(cost_index-81)/(1723-81)

    #Z is the output of CTBC
    #print(Z)
    #save the improvement_index,cost_index and Z in a list
    impr_list.append(improvement_index)
    cost_list.append(cost_index)
    efficiency_index.append(Z)
    
    #print(impr_list)
    #print(cost_list)
    #print(efficiency_index)
    
    ######plot Z and beta################
    #from matplotlib import pyplot as plt      
    #fig=plt.figure(figsize=(6,6))
    #plt.plot(search_step, efficiency_index)
    #print(Z)
    if return_scenarios:
        return scenarios_list  # Return list of test scenarios
    else:
        # ... [original code to compute Z] ...
        return Z

    # return Z
       

'''
gaussian base functions
'''
def gaussian_distribution(x, mean, stddev):
    #Gets input x, mean, and variance
    #Returns vector or scalar from input
    return (np.exp(-((x-mean)**2)/(2*stddev))) / (np.sqrt(2*stddev*np.pi))

def cdf(x, mean, variance):
    #Get values to compute cdf over
    dist_values = gaussian_distribution(np.arange(x-100, x, .1), mean, variance)
    
    #Equivalent to the last element of cumulative sum
    return sum(dist_values)

def get_cov_matrix(f, cov):
    #Given a vector f, generate the covariance matrix 
    #f because known inputs
    f_n = len(f)
    f_m = np.zeros(shape=(f_n, f_n))
    for row_i, f_i in enumerate(f):
        for col_i, f_j in enumerate(f):
            f_m[row_i, col_i] = cov.evaluate(f_i, f_j)

    return f_m

def get_cov_vector(f, test_f, cov):
    #Given a vector f and scalar f* (test_f)
    #Generate a covariance vector for each value in f
    f_n = len(f)
    f_v = np.zeros(shape=(f_n, 1))
    for row_i, f_i in enumerate(f):
        f_v[row_i] = cov.evaluate(test_f, f_i)

    return f_v

def cartesian_product(vectors):
    return [i for i in itertools.product(*vectors)]


'''
covariance functions
'''

class covariance_function(object):
    #Superclass

    def __init__(self, lengthscale, v):
        self.lengthscale = lengthscale
        self.v = v


class squared_exponential(covariance_function):

    def __init__(self, lengthscale, v):
        covariance_function.__init__(self, lengthscale, v)

    def evaluate(self, x_i, x_j):
        return np.exp((-1/(2.0*self.lengthscale))*np.linalg.norm(x_i - x_j)**2) 

        
class matern(covariance_function):

    def __init__(self, lengthscale, v):
        covariance_function.__init__(self, lengthscale, v)

    def evaluate(self, x_i, x_j):
        dist = np.linalg.norm(x_i-x_j)
        return np.nan_to_num(((2**(1-self.v))/(ss.gamma(self.v))) * ((np.sqrt(2*self.v) * (dist/self.lengthscale))**self.v) * ss.kv(self.v, (np.sqrt(2*self.v) * (dist/self.lengthscale))))



'''
acquistion function
'''

class acquisition_function(object):

    def __init__(self, confidence_interval):
        self.confidence_interval = confidence_interval

class probability_improvement(acquisition_function):

    def __init__(self, confidence_interval):
        acquisition_function.__init__(self, confidence_interval)
    
    def evaluate(self, means, variances, values):
        improvement_probs = np.array([np.nan_to_num(cdf(val, mean, np.sqrt(variance))) for val, mean, variance in zip(values, means, variances)])
        return np.argmax(improvement_probs)


class upper_confidence_bound(acquisition_function):

    def __init__(self, confidence_interval):
        acquisition_function.__init__(self, confidence_interval)
    
    def evaluate(self, means, variances, values):
        return np.argmax(means + self.confidence_interval * np.sqrt(variances))


'''
HyperParameter
'''
class HyperParameter(object):
    def __init__(self, min, max):
        self.min = min
        self.max = max


'''
Bayesian Blackbox Optimization
'''


detail_n = 100 #Number of evaluated input points
maximizing = True
bbf_evaluation_n = 15 #7
confidence_interval = 1.5 #1.5 
#Choice of acquisition function
acquisition_function = upper_confidence_bound(confidence_interval)
#Choice of covariance function
lengthscale = 2.50#2.50
v = [5.00/2]#[5/2.0]#For matern1(not currently functional)
covariance_function = squared_exponential(lengthscale, v)
#Initialize ranges for each parameter into a resulting matrix
hps = [HyperParameter(0, 10)]

independent_domains = np.array([np.arange(hp.min, hp.max, ((hp.max-hp.min)/float(detail_n))) for hp in hps])
n = detail_n**len(hps)
domain = [np.array(i) for i in cartesian_product(independent_domains)]

domain_x = np.copy(independent_domains[0])

#shuffle the original to get two(three) random input vectors
for independent_domain in independent_domains:
    np.random.shuffle(independent_domain)

#Get our different values easily by transposing
#x1, x2 = independent_domains.transpose()[:2]
#x1, x2, x3 = independent_domains.transpose()[:3]
x1 = np.array([0.40])    #0.40
x2 = np.array([2.50])    #2.50
x3 = np.array([6.20])    #4.55 for iterarion 7; 6.20 for iteration 15; 

#Now that we have our two(three) random input vectors, 
#evaluate them and store them in our blackboxfunction inputs and outputs vector
#bbf_inputs = [x1, x2]
bbf_inputs = [x1, x2, x3]

#bbf_evaluations = np.array([blackboxfunction(x1), blackboxfunction(x2)])
bbf_evaluations = np.array([blackboxfunction(x1), blackboxfunction(x2), blackboxfunction(x3)])

#for bbf_evaluation_i in range(2, bbf_evaluation_n):
for bbf_evaluation_i in range(3, bbf_evaluation_n):

#    print("Determining Point #%i" % (bbf_evaluation_i+1))
#    print("Determining Point #%i" % (bbf_evaluation_i+1))
    print("Point #%i" % (bbf_evaluation_i))
    #Since we reset this every time we generate through the domain
    test_means = np.zeros(shape=(n))
    test_variances = np.zeros(shape=(n))
    test_values = np.zeros(shape=(n))

    for test_input_i, test_input in enumerate(domain):
        #Generate our covariance matrices and vectors
        training_cov_m = get_cov_matrix(bbf_inputs, covariance_function)#K
        training_cov_m = training_cov_m + (np.eye(training_cov_m.shape[0])*1e-7)
        training_cov_m_inv = np.linalg.inv(training_cov_m)#K^-1
        test_cov = get_cov_vector(bbf_inputs, test_input, covariance_function)#K*
        test_cov_T = test_cov.transpose()#K*T
        test_cov_diag = covariance_function.evaluate(test_input, test_input)#K**

        #Compute test mean using our Multivariate Gaussian Theorems
        #print test_cov_T.shape, training_cov_m_inv.shape, bbf_evaluations.shape
        test_mean = np.dot(np.dot(test_cov_T, training_cov_m_inv), bbf_evaluations)

        #Compute test variance using our Multivariate Gaussian Theorems
        test_variance = test_cov_diag - np.dot(np.dot(test_cov_T, training_cov_m_inv), test_cov)

        #Store them for use with our acquisition function
        test_means[test_input_i] = test_mean
        test_variances[test_input_i] = test_variance + 0.01

    if maximizing:
        test_values = test_means + test_variances
    else:
        test_values = test_means - test_variances

    #Set the filename
    fname = "results/%s" % str(bbf_evaluation_i)

    #Plot
    plt.figure(figsize=(8.6,6.5)) #inch?3.3858:2.5590
    plt.grid(linestyle = "--")
    plt.plot(domain_x, test_means, color='steelblue',linewidth=2.0,label='Means')#
    plt.plot(domain_x, test_means+test_variances, 'r--',linewidth=2.0,label="Variances")#'firebrick',linewidth=1.5
    plt.plot(domain_x, test_means-test_variances, 'r--',linewidth=2.0)#'firebrick',linewidth=1.5
    plt.plot(bbf_inputs, bbf_evaluations, 'bo', label='CTBC Algorithm Evaluations')
#   plt.plot(bbf_inputs, bbf_evaluations, marker='o', c='b', s=100.0, label="CTBC Function Evaluations")
    
    if bbf_evaluation_i == bbf_evaluation_n-1:
        list_bbf = []
        for x_i in domain_x:
            list_bbf.append(blackboxfunction(x_i))    
        plt.plot(domain_x, list_bbf, 'k:',linewidth=3.0, label='Actual CTBC Algorithm')
    #plot axis
    ax=plt.gca()
    ax.set_xticks(np.linspace(0,10,6)) #ax.set_xticks(np.linspace(0,18,10))
    ax.set_xticklabels(('0', '0.2', '0.4', '0.6', '0.8',  '1.0'),fontname="Times New Roman",fontsize=20,fontweight='normal')#
    ax.set_yticks(np.linspace(0,0.40,9))
    ax.set_yticklabels(('0', '0.05', '0.10', '0.15', '0.20',  '0.25', '0.30', '0.35', '0.40'),fontname="Times New Roman",fontsize=20,fontweight='normal')#,fontsize=13
    plt.xlabel('Complexity improvement index $' u'\u03B2' '$',fontname="Times New Roman",fontsize=20)#fontweight='normal' u'\u00A0' ,fontsize=13
    plt.ylabel('Test effect $Z$($' u'\u03B2' '$)',fontname="Times New Roman",fontsize=20)
    plt.xlim(0,10)
    plt.ylim(0,0.4)        
            
    #show legend
    #plt.legend()          
    plt.legend(loc=0, numpoints=1)
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontname="Times New Roman",fontsize=20) #,fontweight='normal'
    
    #save figure
#        plt.savefig("%s.jpg" % fname, dpi=None, facecolor='w', edgecolor='w',
#            orientation='portrait', papertype=None, format=None,
#            transparent=False, bbox_inches='tight', pad_inches=0.1,
#            frameon=None)
    #remove margin 
#        plt.gca().xaxis.set_major_locator(plt.NullLocator())
#        plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.margins(0,0)

    # 1. Get the directory where this script is located
    # (.../baselines/CTBC)
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 2. Define the output directory
    # (.../baselines/CTBC/Figures)
    figures_dir = os.path.join(script_dir, "Figures")

    # 3. Create the directory if it doesn't exist
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)

    # 4. Construct the full file paths
    png_path = os.path.join(figures_dir, 'black_box_optimization3.png')
    svg_path = os.path.join(figures_dir, 'black_box_optimization3.svg')

    # 5. Save the figures
    plt.savefig(png_path, dpi=600.0)
    
    plt.margins(0,0)
    plt.savefig(svg_path, format='svg')
    plt.show()
    
    #Get the index of the next input to evaluate in black box function
    next_input_i = acquisition_function.evaluate(test_means, test_variances, test_values)

    #Add new input
    next_input = domain[next_input_i]
    #print "\tNew point: {}".format(next_input)
    bbf_inputs.append(np.array(next_input))

    #Evaluate new input
    bbf_evaluations = list(bbf_evaluations)
    bbf_evaluations.append(blackboxfunction(next_input))
    bbf_evaluations = np.array(bbf_evaluations)

best_input = bbf_inputs[np.argmax(bbf_evaluations)]
print(bbf_inputs, bbf_evaluations)
print("Best input found after {} iterations: {}".format(bbf_evaluation_n-3, best_input/10))
print("Max evaluation is %f" % blackboxfunction(best_input))

best_beta = best_input # 0.04  # Optimal β value from optimization results
test_scenarios = blackboxfunction(best_beta, return_scenarios=True)
# Write test scenarios to CSV
with open('test_scenarios_scenario2.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    # Write header: Factor0, Factor1, ..., Factor16
    header = [f'Factor{i}' for i in range(length)]
    writer.writerow(header)
    
    for scenario in test_scenarios:
        row = [scenario.get(i, '') for i in range(length)]
        writer.writerow(row)
