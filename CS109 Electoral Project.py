
# coding: utf-8

# In[1]:


#clear all variables
#import os
#cwd = os.getcwd()
#sys.modules[__name__].__dict__.clear()


# In[2]:


### Note to self: I cheated a little bit and edited party cell for 2000, 2004 and 2012 Minnesota
### (John Kerry and Al Gore) to be democrat instead of democratic-farmer-labor 

import numpy as np
import scipy
import matplotlib.pyplot as plt
import pandas as pd 
import math
import sys
#open up electoral data 
#file  = open("/Users/robertthompson/Desktop/CS 109/CS109 project electoral/1976-2016-president.csv")
#dtypes = [int, str, str, int, int,int, str, str, str, bool, int, int, int, str]
#prestxt = np.loadtxt(file, dtype = str, delimiter=",")


# In[3]:


#Load MIT data
presdf = pd.read_csv("/Users/robertthompson/Desktop/CS 109/CS109 project electoral/1976-2016-president.csv")
presdf = presdf[["year", "state", "state_po", "candidate", "party", "candidatevotes", "totalvotes"]]


# In[4]:


#get rid of asterisks in the Dave Leip data for the sake of consistency later
#change D.C. to District of Columbia to match the other dataset 
#get rid of faithess electors, assuming they voted as directed, if fixFaithless = true
def removeFaithlessAndAsterisks(electoralinfo, fixFaithless = False):
    stateList = electoralinfo.index.values.copy()
    newstatelist = []
    for state in stateList:
        if '*' in state:
            if (fixFaithless):
                EV = electoralinfo.loc[state, "EV"]
                D = electoralinfo.loc[state, "D"]
                if (D == "-"):
                    electoralinfo.loc[state, "R"] = EV
                else:
                    electoralinfo.loc[state, "D"] = EV
            new = state[0:len(state)-1]
            state = new
        if state == "D. C.":
            state = "District of Columbia"
        newstatelist.append(state)
    #print(newstatelist)
    electoralinfo["state"] = newstatelist
    electoralinfo.set_index("state", inplace = True)                             
    return electoralinfo


# In[5]:


#upload Dave Leip data
#clean, prepare in order to merge with voter-level data
def cleanElectoralInfo(year):
    #upload file
    yearstr = str(year)
    filepath = "/Users/robertthompson/Desktop/CS 109/CS109 project electoral/yearly electorals/" + yearstr + ".xlsx"
    electoralinfo = pd.read_excel(filepath)
    electoralinfo = electoralinfo[["EV", "President", "Vice President"]] #drop useless columns
    oldHead = electoralinfo.iloc[2,]
    #fix headers to be Democrat or republican
    newHeaderNames = []
    newHeaderNames.append("EV")
    if oldHead[1] == "Democratic":
        newHeaderNames.append("D")
        newHeaderNames.append("R")
    else:
        newHeaderNames.append("R")
        newHeaderNames.append("D")
    electoralinfo.columns = newHeaderNames
    electoralinfo = electoralinfo.drop(['Candidate', 'Home State', 'Party']) #drop useless rows
    electoralinfo = electoralinfo[["EV", "D", "R"]] #reorder
    electoralinfo = removeFaithlessAndAsterisks(electoralinfo, fixFaithless = False)
    return electoralinfo
a = cleanElectoralInfo(2004)
print(a)


# In[6]:


#merge data Dave Leip data with MIT voter-level data for a single year
def create_election_year(year):
    electoralinfo = cleanElectoralInfo(year)
    electoralinfo["Dvotes"] = electoralinfo["D"]
    electoralinfo["Rvotes"] = electoralinfo["D"]
    electoralinfo["Tvotes"] = electoralinfo["D"]
    voteinfo = presdf[presdf.year == year]
    stateList = electoralinfo.index.values[0:51]
    for state in stateList:
        row = electoralinfo.loc[state].copy()
        dem = voteinfo[(voteinfo["state"] == state) & (voteinfo["party"] == "democrat")]["candidatevotes"].iloc[0]
        rep = voteinfo[(voteinfo["state"] == state) & (voteinfo["party"] == "republican")]["candidatevotes"].iloc[0]
        total = voteinfo[(voteinfo["state"] == state) & (voteinfo["party"] == "republican")]["totalvotes"].iloc[0]
        electoralinfo.loc[state, "Dvotes"] = dem
        electoralinfo.loc[state, "Rvotes"] = rep
        electoralinfo.loc[state, "Tvotes"] = total
        #print(dem)
    electoralinfo.loc["Total", "Dvotes"] = electoralinfo[["Dvotes"]].sum().iloc[0]
    electoralinfo.loc["Total", "Rvotes"] = electoralinfo[["Rvotes"]].sum().iloc[0]
    electoralinfo.loc["Total", "Tvotes"] = electoralinfo[["Tvotes"]].sum().iloc[0]
    electoralinfo["TPvotes"] = electoralinfo['Tvotes'] - electoralinfo['Rvotes'] - electoralinfo['Dvotes']
    return electoralinfo
a = create_election_year(2012)
print(a)


# In[7]:


#add rows which implement

#Add IPS vote counts to a dataframe that has already merged Dave Leip and MIT data
def addIPSScheme(electoralinfo):
    stateList = electoralinfo.index.values
    electoralinfo["D IPS Alloc"] = electoralinfo["Dvotes"] #initializing
    electoralinfo["R IPS Alloc"] = electoralinfo["Dvotes"] #initializing

    for state in stateList[0:51]:
        demvotes = electoralinfo.loc[state, "Dvotes"]
        repvotes = electoralinfo.loc[state, "Rvotes"]
        totvotes = electoralinfo.loc[state, "Tvotes"]
        electors = electoralinfo.loc[state, "EV"]
        maxvotes = max(demvotes, repvotes)
        nextvotes = min(demvotes, repvotes)
        maxelectors = math.ceil(maxvotes / totvotes * electors)
        residualvotes = totvotes - maxvotes
        residualelectors = electors - maxelectors
        nextelectors = math.ceil(nextvotes / residualvotes * residualelectors)
        if demvotes == maxvotes:
            electoralinfo.loc[state, 'D IPS Alloc'] = maxelectors
            electoralinfo.loc[state, 'R IPS Alloc'] = nextelectors
        else :
            electoralinfo.loc[state, 'R IPS Alloc'] = maxelectors
            electoralinfo.loc[state, 'D IPS Alloc'] = nextelectors
    electoralinfo["leftover IPS Alloc"] = electoralinfo["EV"] - electoralinfo['D IPS Alloc'] - electoralinfo['R IPS Alloc']        
    electoralinfo.loc["Total", "D IPS Alloc"] = electoralinfo[["D IPS Alloc"]].iloc[0:51].sum().iloc[0]
    electoralinfo.loc["Total", "R IPS Alloc"] = electoralinfo[["R IPS Alloc"]].iloc[0:51].sum().iloc[0]
    electoralinfo.loc["Total", "leftover IPS Alloc"] = electoralinfo[["leftover IPS Alloc"]].iloc[0:51].sum().iloc[0]
    #print(electoralinfo[["D IPS Alloc"]].sum())
    #print(electoralinfo[["D IPS Alloc"]].iloc[0:51].sum())
    
    return electoralinfo

#add columns to an already merge dataframe which represent
#proportion of votes for democrats and repulicans, respectively
def addProportionColumns(electoralinfo):
    electoralinfo['Dprop'] = electoralinfo['Dvotes'] / electoralinfo["Tvotes"]
    electoralinfo['Rprop'] = electoralinfo['Rvotes'] / electoralinfo["Tvotes"]
    electoralinfo['propdiff'] = electoralinfo['Dprop'] - electoralinfo['Rprop']
    return electoralinfo

a = addProportionColumns(addIPSScheme(create_election_year(1996)))
print(a)


# In[8]:


#generate yearly reports for all election years available so that they can
#be accessed more quickly, without having to be regenerate each time
def generateAllYearlyReports():
    yearsavailable = range(1976, 2020, 4)
    datalist = []
    for i in yearsavailable:
        datalist.append(addProportionColumns(addIPSScheme(create_election_year(i))))
    return datalist
ALL_YEARLY_REPORTS = generateAllYearlyReports()


# In[9]:


#access a year of election from the array we have already generated
def getpreloadedyear(year):
    index = int((year - 1976) / 4)
    return ALL_YEARLY_REPORTS[index]
a = getpreloadedyear(2004)
print(a)


# In[10]:


#gather the election data from all years available
#and create a new data frame with all observations for a given state
def statewideBiasReport(state):
    yearsavailable = range(1976, 2020, 4)
    yearlist = []
    report = pd.DataFrame(columns = addIPSScheme(create_election_year(1976)).columns)
    report["year"] = report["R"]
    for i in yearsavailable:
        #print(i)
        year = getpreloadedyear(i)
        report = report.append(year.loc[state])
        yearlist.append(i)
    for i in range(0,11):
        report.iloc[i, 10] = yearlist[i]
    report.set_index("year", inplace = True)  
    return report
a = statewideBiasReport("District of Columbia")
print(a)


# In[11]:


#generate a list of all states and the District of Columbia
def genStateList():
    temp = getpreloadedyear(1976).index.values[0:51]
    return(temp)

#generate all statewide reports and store in an array so that 
#we do not have to recreate a report each time we want to access it
def generateAllStatewideReports():
    statesdict = {}
    statelist = genStateList()
    for state in statelist:
        statesdict[state] = statewideBiasReport(state)
    return statesdict
ALL_STATEWIDE_REPORTS = generateAllStatewideReports()

#access a statewide report from the array we have already generated
def getPreloadedState(state):
    return ALL_STATEWIDE_REPORTS[state]

print(genStateList())


# In[41]:


#Graph the proportion of votes for Democrats and Republicans
#and Democrats in a given state
def graphState(state):
    data = statewideBiasReport(state)
    dlist = data[["Dvotes"]].values / data[['Tvotes']].values
    rlist = data[['Rvotes']].values / data[['Tvotes']].values
    plt.figure(1)
    plt.plot(rlist)
    plt.plot(dlist)
graphState("California")


# In[13]:


###
#Now we begin testing models to predict future proportions of votes
#We will predict vote proportions instead of electoral votes because 
#Electoral votes is a noisy statistic

#Because our assumption is that these proportionings will be redone every census period,
#this will dictate our testing periods.
#The three we will use is 1992 through 2000, 2004 through 2008, and 2012 through 2016.
#It is unfortunate that the last period is incomplete.
#period 1 (p1): average of 1992, 1996, 2000
#period 2 (p2): average of 2004, 2008
#period 3 (p3): average of 2012, 2016
###

#get the average proportion voting for Democrats and Republicans, respectively
#in each period for a given state
def getActualOutcomes(state):
    data = getPreloadedState(state)
    Dprops = data[["Dprop"]].values
    Rprops = data[["Rprop"]].values
    ret = pd.DataFrame()
    dp1 = Dprops[4:7].mean()
    rp1 = Rprops[4:7].mean()
    dp2 = Dprops[7:9].mean()
    rp2 = Rprops[7:9].mean()
    dp3 = Dprops[9:11].mean()
    rp3 = Rprops[9:11].mean()
    ret["Dprop"] = np.array([dp1,dp2,dp3])
    ret["Rprop"] = np.array([rp1,rp2,rp3])
    ret["Period"] = np.array(["p1", "p2", "p3"])
    ret.set_index("Period", inplace = True)
    return ret

#generate a dataframe with predictions for Republicans and Democrats
#in all three periods for a given state
#also returns an int representing total error of all predictions
#and an array with average error for each time period 
#Predict solely based on previous election
def predictBasedOnPreviousSingleYear(state):
    data = getPreloadedState(state)
    dguess1 = data.loc[1988, "Dprop"]
    dguess2 = data.loc[2000, "Dprop"]
    dguess3 = data.loc[2008, "Dprop"]
    rguess1 = data.loc[1988, "Rprop"]
    rguess2 = data.loc[2000, "Rprop"]
    rguess3 = data.loc[2008, "Rprop"]
    ret = pd.DataFrame()
    ret["Dprop"] = np.array([dguess1,dguess2,dguess3])
    ret["Rprop"] = np.array([rguess1,rguess2,rguess3])
    ret["Period"] = np.array(["p1", "p2", "p3"])
    ret.set_index("Period", inplace = True)
    error = abs(ret - getActualOutcomes(state))
    terror = error.sum().sum()
    return (terror, ret, error)
    
b = getActualOutcomes("Wyoming")
a = predictBasedOnPreviousSingleYear("Wyoming")
print(a)
print(b)


# In[14]:


#find the total error across all 50 states for a given method
def totalErrorPreviousUsingMethod(method):
    statelist = genStateList()
    terror = 0.0
    for state in statelist:
        terror += method(state)[0]
    return terror
errormethod1 = totalErrorPreviousUsingMethod(predictBasedOnPreviousSingleYear)

#print error summary
def printErrorSummary(error, method):
    print("Total error " + method + ": " + str(error))
    print("Average error " + method + ": " + str(error/300))
printErrorSummary(errormethod1, "method 1 (last observation)")


# In[15]:


#generate a dataframe with predictions for Republicans and Democrats
#in all three periods for a given state
#also returns an int representing total error of all predictions
#and an array with average error for each time period 
#Predict solely based on average of previous three election years
def predictBasedOnAverageOfPreviousThreeYears(state):
    data = getPreloadedState(state)
    dguess1 = data.loc[1980:1988, "Dprop"].mean()
    dguess2 = data.loc[1992:2000, "Dprop"].mean()
    dguess3 = data.loc[2000:2008, "Dprop"].mean()
    rguess1 = data.loc[1980:1988, "Rprop"].mean()
    rguess2 = data.loc[1992:2000, "Rprop"].mean()
    rguess3 = data.loc[2000:2008, "Rprop"].mean()
    ret = pd.DataFrame()
    ret["Dprop"] = np.array([dguess1,dguess2,dguess3])
    ret["Rprop"] = np.array([rguess1,rguess2,rguess3])
    ret["Period"] = np.array(["p1", "p2", "p3"])
    ret.set_index("Period", inplace = True)
    error = abs(ret - getActualOutcomes(state))
    terror = error.sum().sum()
    return (terror, ret, error)

errormethod2 = totalErrorPreviousUsingMethod(predictBasedOnAverageOfPreviousThreeYears)
printErrorSummary(errormethod2, "method 2 (average of last 3 observations)")


# In[16]:


from sklearn.linear_model import LinearRegression

#generate a dataframe with predictions for Republicans and Democrats
#in all three periods for a given state
#also returns an int representing total error of all predictions
#and an array with average error for each time period 
#Predict using linear regression of previous three years
def linearRegressState(state):
    data = getPreloadedState(state)
    d1 = data.loc[1980:1988, "Dprop"].values
    d2 = data.loc[1992:2000, "Dprop"]
    d3 = data.loc[2000:2008, "Dprop"]
    r1 = data.loc[1980:1988, "Rprop"]
    r2 = data.loc[1992:2000, "Rprop"]
    r3 = data.loc[2000:2008, "Rprop"]
    d1l = LinearRegression()
    d2l = LinearRegression()
    d3l = LinearRegression()
    r1l = LinearRegression()
    r21 = LinearRegression()
    r31 = LinearRegression()
    simp = np.array([1,2,3]).reshape(-1,1)
    d1l.fit(simp,d1)
    d2l.fit(simp,d2)
    d3l.fit(simp,d3)
    r1l.fit(simp,r1)
    r21.fit(simp,r2)
    r31.fit(simp,r3)
    dguess1 = d1l.predict(4)[0]
    dguess2 = d2l.predict(4)[0]
    dguess3 = d3l.predict(4)[0]
    rguess1 = r1l.predict(4)[0]
    rguess2 = r21.predict(4)[0]
    rguess3 = r31.predict(4)[0]
    ret = pd.DataFrame()
    ret["Dprop"] = np.array([dguess1,dguess2,dguess3])
    ret["Rprop"] = np.array([rguess1,rguess2,rguess3])
    ret["Period"] = np.array(["p1", "p2", "p3"])
    ret.set_index("Period", inplace = True)
    error = abs(ret - getActualOutcomes(state))
    terror = error.sum().sum()
    return (terror, ret, error)
#linearRegressState("Wyoming")

error3 = totalErrorPreviousUsingMethod(linearRegressState)
printErrorSummary(error3, "method 3 (linear regression of previous three years)")

print(linearRegressState("Georgia"))
print(getActualOutcomes("Georgia"))
print(statewideBiasReport("Georgia"))


# In[17]:


###
#Now that we have predicted average proportion of voters, we attempt to predict 
#standard deviation using the same periods:
#period 1 (p1): average of 1992, 1996, 2000
#period 2 (p2): average of 2004, 2008
#period 3 (p3): average of 2012, 2016
#
#Note that we are calculating deviation from our estimate using method 1.
#Also note that we ignoring the possibility that my estimates are skewed in any one direction.
###

#Turn an array of 1d arrays into an array of ints
def fixDoubleArrayProblem(array,length):
    ret = np.empty(length)
    for i in range(length):
        ret[i] = array[i][0]
    return ret
        
#return a dataframe with errors for Republican and Democratic estimates for all years in all year periods
#all return absolute values of errors
def deviationOfSingleStateEstimate(state):
    guess = predictBasedOnPreviousSingleYear(state)
    data = getPreloadedState(state)
    Dprops = data[["Dprop"]].values
    Rprops = data[["Rprop"]].values
    Dprops = fixDoubleArrayProblem(Dprops, 11)
    Rprops = fixDoubleArrayProblem(Rprops, 11)
    ret = pd.DataFrame()
    dp1 = Dprops[4:7] - guess[1].iloc[0,0]
    rp1 = Rprops[4:7] - guess[1].iloc[0,1]
    dp2 = Dprops[7:9] - guess[1].iloc[1,0]
    rp2 = Rprops[7:9] - guess[1].iloc[1,1]
    dp3 = Dprops[9:11] - guess[1].iloc[2,0]
    rp3 = Rprops[9:11] - guess[1].iloc[2,1]
    Derror = np.concatenate((dp1,dp2,dp3))
    Rerror = np.concatenate((rp1,rp2,rp3))
    ret["Derror"] = np.array(Derror)
    ret["Rerror"] = np.array(Rerror)
    ret["Derror abs"] = abs(np.array(Derror))
    ret["Rerror abs"] = abs(np.array(Rerror))
    #ret["Period"] = np.array(["p1", "p2", "p3"])
    #ret.set_index("Period", inplace = True)
    return ret

#calculate statistics about the error in my estimates across all 50 states
#first value return is the error of all democrat estimates
#second value: error of all Republican estimates
#third value: array of std of error of Democratic estimate for all 50 states
#fourth value: array of std of error of Republican guess for all 50 states
#fifth value: array of covariance of errors for all 50 states
#sixth value: float, covariance of errors using observations from all 50 states
def overallDeviationStats():
    statelist = genStateList()
    DErrorArr = np.empty(51 * 7)
    RErrorArr = np.empty(51 * 7)
    DSTDerror = np.empty(51)
    RSTDerror = np.empty(51)
    covarErrorArr = np.empty(51)
    counter = 0
    counter7 = 0
    for state in statelist:
        data = deviationOfSingleStateEstimate (state)
        Derror = fixDoubleArrayProblem(data[["Derror"]].values, 7)
        Rerror = fixDoubleArrayProblem(data[["Rerror"]].values, 7)
        np.put(DErrorArr, list(range(counter7,counter7+7)), Derror)
        np.put(RErrorArr, list(range(counter7,counter7+7)), Rerror)
        DSTDerror[counter] =  Derror.std(ddof = 1)
        RSTDerror[counter] =  Rerror.std(ddof = 1)
        covarErrorArr[counter] = np.cov(np.array([Derror, Rerror]))[0,1]
        counter +=1
        counter7 += 7
    covarError = np.cov(np.array([DErrorArr, RErrorArr]))
    return (DErrorArr, RErrorArr, DSTDerror, RSTDerror, covarErrorArr, covarError)

a = deviationOfSingleStateEstimate("Wyoming")
print(a)
devStat = overallDeviationStats()
print(devStat)


# In[18]:


#Plot histogram of all error when guessing Democratic proportion
print(plt.hist(devStat[0], bins=25))
print("Looks \'normal\', no?")


# In[19]:


#plot histogram of all errors when estimating Republican proportion
print(plt.hist(devStat[1], bins=25))
print("Looks doesn't look very \'normal,\' sadly. Oh well.")


# In[20]:


###
#Let's just pretend that that second distribution looks normal
#If we have time we could go back at the end and try all years instead of 
#just the ones that actually occurred
###

###
#find mean and std of republican dist
#find mean and std of dem dist
###

DMEAN = devStat[0].mean()
RMEAN = devStat[1].mean()
DSTD = devStat[0].std()
RSTD = devStat[1].std()
MaxDemDev = max(devStat[2])
MaxRepDev = max(devStat[3])
print("Mean error when predicting Republican Dist: " + str(RMEAN))
print("STD of error when predicting Republican Dist: " + str(RSTD))
print("Mean error when predicting Democratic Dist: " + str(DMEAN))
print("STD of error when predicting Democratic Dist: " + str(DSTD))
print(MaxDemDev)
print(MaxRepDev)


# In[21]:


###
#Quick detour to determine via bootstrap if some states are volatile
###

#get a sample standard deviation of 6 errors
#intended to mimic the standard deviation of one state's errors in estimation
def getSampleSTD(dem = True):
    ret = np.empty(6)
    if (dem):
        ret = np.random.choice(devStat[0],6)
    else:
        ret = np.random.choice(devStat[1],6)
    return ret.std(ddof = 1)

#get the max of 51 samples chosen randomly
def getMaxOf51(sampleList):
    return (np.random.choice(sampleList, 51).max())

#get the minimum of 51 samples chossen randomly
def getMinOf51(sampleList):
    return (np.random.choice(sampleList, 51).min())
    
#Boot strap to determine if estimate of proportions for a given party are better
#or worse in some states than in others
#generate 100,000 samples of six errors and their standard error
#pull 51 of these sample standard errors 100,000 times 
#determine how often the maximum standard error observed among the 51 is greater
#than the maximum standard error observed across the fifty states and DC
def bootstrapVolatilities(dem = True):
    #generating 100,000 samples, then getting fifty of these samples 100,000 times
    #vastly more memory complexity than just sampling 50 100,000 times
    #Theoretically faster runtime? Maybe not b/c of cacheing but that is above my pay grade
    NUMSAMPLES = 100000
    NUMRESAMPLES = 100000
    samples = np.empty(NUMSAMPLES)
    maxOfResamples = np.empty(NUMRESAMPLES)
    for i in range(NUMSAMPLES):
        samples[i] = getSampleSTD(dem)
    print("Done Sampling. Starting Resampling")
    for i in range(NUMRESAMPLES):
        maxOfResamples[i] = getMaxOf51(samples)
    maxDev = MaxRepDev
    if (dem):
        maxDev = MaxDemDev
    pval = sum(maxDev > maxOfResamples) / NUMRESAMPLES
    return (maxOfResamples, pval)

a = bootstrapVolatilities()
print(a[1])
print("Conclusion : Democratic standard deviations are the same across states")


# In[22]:


b = bootstrapVolatilities(False)
print(b[1])

print("Conclusion: Republican standard deviations are the same across states")

###
#Note that we do not need to normalize for each state because we are assuming that
#error in terms of deviation from expected is centered around 0 and thus already 'normalized'
###


# In[23]:


#bootstrap to determine if my errors are just as bad for Democrats as Republicans
#randomly split all my errors in half and see how many times the difference in standard deviation
#is a large as the split we observe empirically between Republicans and Democrats
def bootStrapToDetermineIfRepAndDemAreDiff():
    completelist = np.concatenate((devStat[1], devStat[0]))
    NUMSAMPLES = 100000
    diffList = np.empty(NUMSAMPLES)
    stdList = np.empty(NUMSAMPLES)
    actualdiff = RSTD - DSTD
    for i in range(NUMSAMPLES):
        stdList[i] = np.random.choice(completelist, 357).std()
        diffList[i] = abs(np.random.choice(completelist, 357).std() - np.random.choice(completelist, 357).std())
    pval = sum(actualdiff < diffList) / NUMSAMPLES
    return (pval, diffList, stdList)

a = bootStrapToDetermineIfRepAndDemAreDiff()
print("P value of difference in deviations between Replican and Democratic Distributions: " + str(a[0]))
print("Conclusion: our standard deviations are different for estimates of Republican and Democratic Distrbutions")


# In[24]:


###
#It is unfortunate to conclude that our Republican estimates are worse than our democratic ones,
#as there is no reasonable real-world explanation for this that I can dream up
#Alas, we will asume that the distributions are in fact different
###


# In[25]:


###
#Quick detour to determine covariance of dem and rep props
###

###
#Resampling covariances with bootstraps can be problematic because
#The Rprops and Dprops that the distributions are drawn from are
#fundamentally different by my own assumptions.
#This is why we are resampling errors...?
###

#this method is mostly used as a sanity check
#to determine if the variances and covariances I calculate later on errors
#are relatively similar to variances and covariances on the proportions themselves
#no need to pay too much attention
def findCovarianceStats():
    statelist = genStateList()
    DpropArr = np.empty(51*11)
    RpropArr = np.empty(51*11)
    counter11 = 0
    for state in statelist:
        data = getPreloadedState(state)
        np.put(DpropArr, list(range(counter11,counter11+11)), data[["Dprop"]].values)
        np.put(RpropArr, list(range(counter11,counter11+11)), data[["Rprop"]].values)
        counter11 += 11
    return (np.cov(np.array([DpropArr,RpropArr])), DpropArr, RpropArr, DpropArr.std(), RpropArr.std())
COVAR_STATS = findCovarianceStats()
COVAR = COVAR_STATS[0][1][0]


###
#back to working with errors
###

DPROPSERROR = devStat[0]
RPROPSERROR = devStat[1]
COVAR_LIST = devStat[4]

#sample 6 pairs of Republican and Demcrat errors from the same estimate
#determine covariance of these six errors
def getSampleCVR():
    sample = np.random.randint(0,356,6)
    dsample = np.empty(6)
    rsample = np.empty(6)
    for i in range(0,6):
        dsample[i] = DPROPSERROR[sample[i]]
        rsample[i] = RPROPSERROR[sample[i]]
    return np.cov(np.array([dsample,rsample]))[1][0]
        
print(COVAR)
print(COVAR_LIST)
print(COVAR_LIST.mean())


# In[26]:


#Bootstrap:
#find the probability that any state displays abnormally high or low covariance
#sample 100,000 covariances of six observations 
#resample from this 100,000 another 100,000 samples of 51
#determine how many of these 100,000 exceed the maximum or are less than the 
#minimum covariance observed among the states, respectively
def bootStrapToDetermineCovarianceSignificance():
    covar_min = min(COVAR_LIST)
    covar_max = max(COVAR_LIST)
    #generating 100,000 samples, then getting fifty of these samples 100,000 times
    #vastly more memory complexity than just sampling 50 100,000 times
    #Theoretically faster runtime? Maybe not b/c of cacheing but this is above my pay grade
    NUMSAMPLES = 100000
    NUMRESAMPLES = 100000
    samples = np.empty(NUMSAMPLES)
    maxOfResamples = np.empty(NUMRESAMPLES)
    minOfResamples = np.empty(NUMRESAMPLES)
    for i in range(NUMSAMPLES):
        samples[i] = getSampleCVR()
    print("Done Sampling. Starting Resampling")
    for i in range(NUMRESAMPLES):
        maxOfResamples[i] = getMaxOf51(samples)
        minOfResamples[i] = getMinOf51(samples)
    maxCov = COVAR_LIST.max()
    minCov = COVAR_LIST.min()
    pval1 = sum(maxCov > maxOfResamples) / NUMRESAMPLES
    pval2 = sum(minCov < minOfResamples) / NUMRESAMPLES
    return (pval1, pval2)
    
print(bootStrapToDetermineCovarianceSignificance())
print("Conclusion: no state has a uniquely extreme covariance of errors. We will assume that they are all the same")


# In[27]:


###
#Now that we have an idea as to what our distribution looks like,
#we will start playing around with it
###

#creates a dictionary with estimated proportions for all 50 states and DC
#basically just makes access faster in later methods
def genDictWithAllGuessedProps():
    statelist = genStateList()
    dpropdict = {}
    rpropdict = {}
    propdict = {}
    for state in statelist:
        data = getPreloadedState(state)
        Dprop = data.loc[2016, 'Dprop']
        Rprop = data.loc[2016, 'Rprop']
        dpropdict[state] = Dprop
        rpropdict[state] = Rprop
        propdict[state] = np.array([Dprop, Rprop])
    return (propdict, dpropdict, rpropdict)
ASSUMED_PROPS_DICT = genDictWithAllGuessedProps()[0]

#create a dictionary which has the electoral value for 
#each state in 2016 stored. Basically makes 
#access easier and faster in later methods
def genDictWithEVs():
    statelist=genStateList()
    evdict = {}
    for state in statelist:
        data = getPreloadedState(state)
        evdict[state] = data.loc[2016, 'EV']
    return evdict
EVDICT = genDictWithEVs()


# In[28]:


###
#determine the distribution we will use to simulate elections
#our variance for republicans and democrats and our covariance between the two
#will remain constant across all 50 states
#after playing around with many distributions, however,
#I found it necessary to crank up the covariance to higher than we actually observed
#In order to generate more realistic elections
###

COVAR_MATRIX = np.cov(np.array([devStat[0],devStat[1]]))
COVAR_MATRIX[1][0] *= 2.53
COVAR_MATRIX[0][1] *= 2.53
print(COVAR_MATRIX)

#Ignore the rest of the code in this cell!

#print("COVAR: " + str(COVAR))
#print("RSTD: " + str(RSTD))
#print("DSTD: " + str(DSTD))

#COVAR_MATRIX = np.empty((2,2))
#COVAR_MATRIX[0][0] = DSTD ** 2
#COVAR_MATRIX[0][1] = COVAR
#COVAR_MATRIX[1][0] = COVAR 
#COVAR_MATRIX[1][1] = RSTD ** 2


#print("fudging ...")
#NDSTD = 0.05 ** 2
#NRSTD = 0.05 ** 2
#NCOVAR = -((NDSTD * NRSTD) ** (1/2))
#NCOVAR_MATRIX = np.empty((2,2))
#NCOVAR_MATRIX[0][0] = NDSTD 
#NCOVAR_MATRIX[0][1] = NCOVAR
#NCOVAR_MATRIX[1][0] = NCOVAR 
#NCOVAR_MATRIX[1][1] = NRSTD 
#print(NCOVAR_MATRIX)
#print(genDictWithAllGuessedProps())
#print(ASSUMED_PROPS_DICT)
#print(ASSUMED_PROPS_DICT["Idaho"])


# In[43]:


def genProportionGuess(state, numsamples):
    mean = ASSUMED_PROPS_DICT[state]
    return (np.random.multivariate_normal(mean, COVAR_MATRIX, numsamples))
a = genProportionGuess("California", 10000)

print(plt.scatter(a[:,0],a[:,1]))
b = sum((a[:,0] + a[:,1])>1)/100
print("percentage of distribution with too many votes cast: " + str(b) + "%")


# In[30]:


#find expected margin for a given state given that it uses winner-take-all allocation
#by simulating a bunch of elections
#takes electoral votes as input
#so that it can calculate expected margin if only part of electors are allocated WTA
def findNormalNetEEV(state, numtrials, EVs):
    ret = genProportionGuess(state, numtrials)
    propDemVict = sum((ret[:,0] - ret[:,1]) > 0) / numtrials
    return (propDemVict - (1 - propDemVict)) * EVs


# In[31]:


#returns electoral votes for republicans and democrats, respectively
#given a proportion of votes for each and the number of EVs to allocate
def getIPS(dprop, rprop, EVs):
    maxprop = max(dprop, rprop)
    nextprop = min(dprop, rprop)
    maxelectors = math.ceil(maxprop * EVs)
    residualprop = 1 - maxprop
    #print((nextprop, residualprop, maxprop))
    residualEV = EVs - maxelectors 
    nextelectors = math.ceil((nextprop / residualprop) * residualEV)
    if (dprop > rprop):
        return (maxelectors, nextelectors)
    else :
        return (nextelectors, maxelectors)
    
#Find expected margin for a given state with the given number of EVs
#by simulating a bunch of elections
#takes electoral votes as input
#so that it can calculate expected margin if only part of electors are allocated IPS
def findIPSNetEEV(state, numtrials, EVs):
    ret = genProportionGuess(state, numtrials)
    counter = 0
    for i in range(numtrials):
        temp = getIPS(ret[i][0], ret[i][1], EVs)
        counter += temp[0]
        counter -= temp[1]
    counter /= numtrials
    return counter
    
#Find the difference in expected net electoral votes for Democrats
#IE if originally democrats expect 12 votes, reps expect 1
#and with IPS dems expect 9, reps expect 4,
#net EEV for dems moves from 12-1 = 11 to 9-4 = 5
#and thus difference is 5 - 11 = -6
#
#In other words, democrats expect this many more electoral votes / 2
#And Republicans expecte this many fewer elector votes / 2
#For a net gain of this many votes for Democrats
def expectedDifferenceInNetEVs(state, numtrials):
    EVs = EVDICT[state]
    return findIPSNetEEV(state,numtrials, EVs) - findNormalNetEEV(state, numtrials, EVs)
print(findNormalNetEEV("Alabama", 100000, 9))
print(findIPSNetEEV("Alabama", 100000, 9))
print(expectedDifferenceInNetEVs("Alabama", 100000))


# In[32]:


#print the difference in NetEEV for all 50 states
#when adopting IPS
def printAllDifferenceInNetEEV():
    statelist = genStateList()
    total = 0
    for state in statelist:
        diff = expectedDifferenceInNetEVs(state, 10000)
        total += diff
        print(state +": Expected Diff: "+ str(diff))
    print(total)
printAllDifferenceInNetEEV()


# In[47]:


#Find expected electoral votes with a mixed system:
#In this system, x electors will be given to whoever wins a plurality of votes
#The other y electors will be apportioned using IPS
def findNetEEVMixedSystem(state, pluralityelectors, IPSelectors, numtrials):
    return (findIPSNetEEV(state, numtrials, IPSelectors)) + findNormalNetEEV(state, numtrials, pluralityelectors)

#finds the optimal amount of electors to allocate using IPS for a given state given that if wants to
#change its EEV by @param targetDiffEEV
#@param targetDiffEEV: we want dems to win this many more net votes or Republicans to win this many fewer
def findBestProportionToAllocateIPS(state, targetDiffEEV, numtrials):
    totalEV = EVDICT[state]
    startEEV = findNormalNetEEV(state, numtrials*5, totalEV)
    currDiff = 0
    numPluralityElectors = totalEV+1
    numIPSElectors = -1
    while(abs(currDiff) < abs(targetDiffEEV)):
        lastDiff = currDiff
        numPluralityElectors -= 1
        numIPSElectors += 1
        currDiff = findNetEEVMixedSystem(state, numPluralityElectors, numIPSElectors, numtrials) - startEEV
    diffUltimate = abs(targetDiffEEV - currDiff)
    diffPenUltimate = abs(targetDiffEEV - lastDiff)
    if (diffUltimate < diffPenUltimate):
        return(numIPSElectors, numPluralityElectors, currDiff)
    else :
        return(numIPSElectors - 1, numPluralityElectors + 1, lastDiff)

#finds the maximum fair number of electors for two states to allocate using IPS
#while maintaining parity between Republicans and Democrats
def makeTwoStatesFair(state1, state2, numtrials):
    state1diff = expectedDifferenceInNetEVs(state1, numtrials)
    state2diff = expectedDifferenceInNetEVs(state2, numtrials)
    state1electors = EVDICT[state1]
    state2electors = EVDICT[state2]
    state1dems = ASSUMED_PROPS_DICT[state1][0]
    state1reps = ASSUMED_PROPS_DICT[state1][1]
    state2dems = ASSUMED_PROPS_DICT[state2][0]
    state2reps = ASSUMED_PROPS_DICT[state2][1]
    print(str(state1) + " electors: " + str(state1electors))
    print(str(state2) + " electors: " + str(state2electors))
    print((str(state1) + " proportion democrats: " + str(state1dems) + '. proportion republicans ' + str(state1reps)))
    print((str(state2) + " proportion democrats: " + str(state2dems) + '. proportion republicans ' + str(state2reps)))
    if (state1diff > 0 and state2diff > 0):
        print('Both states will help Democrats by adopting an IPS scheme')
    if (state1diff <0 and state2diff <0):
        print('Both states will help Republicans by adopting an IPS scheme')
    if abs(state1diff) > abs(state2diff):
        ret = findBestProportionToAllocateIPS(state1, -state2diff, numtrials)
        print('Fair number of electors for ' + state1 + ' to allocate using IPS: ' + str(ret[0]))
        print('Fair number of electors for ' + state2 + ' to allocate using IPS: ' + str(EVDICT[state2]) + " (all electors)")
    else:
        ret = findBestProportionToAllocateIPS(state2, -state1diff, numtrials)
        print('Fair number of electors for ' + state1 + ' to allocate using IPS: ' + str(EVDICT[state1]) + " (all electors)")
        print('Fair number of electors for ' + state2 + ' to allocate using IPS: ' + str(ret[0]))
        

print(makeTwoStatesFair("California", "Texas", 100000))


# In[34]:


###
#begin setting up large simulation to determine
#how often each system results in popular vote winner losing EC for each system,
#how often each system results in no one winning, etc.
###

#set number of trials and retrials
NUM_TRIALS = 300000
NUM_RETRIALS = 1000000
#^ the reason this is taking so long, sorry!


# In[35]:


#create a dictionary with the population of each state that voted in the 2016 election
#for the sake of quick lookup later on
def generatePopDict():
    popdict = {}
    statelist = genStateList()
    for state in statelist:
        data=getPreloadedState(state)
        popdict[state] = data.loc[2016,"Tvotes"]
    return popdict
POP_DICT = generatePopDict()
    

#repeatedly create an election for a given state
#compile into an array
def getSampleOfStateElections(state, numtrials):
    sims = genProportionGuess(state, numtrials)
    EV = EVDICT[state]
    pop = POP_DICT[state]
    ret = np.empty((numtrials, 8))
    #retdf = pd.DataFrame(columns = ['dprop','rprop','dvotes','rvotes','dIPSelec','rIPSelec','dnormelec', 'rnormelec'])
    for i in range(numtrials):
        dprop = sims[i][0]
        rprop = sims[i][1]
        IPSelec = getIPS(dprop, rprop, EV)
        dvotes = int(pop * dprop)
        rvotes = int(pop*rprop)
        delec = 0
        relec = EV
        if(dprop > rprop):
            delec = EV
            relec = 0
        row = np.array([dprop, rprop, dvotes, rvotes, IPSelec[0], IPSelec[1], delec, relec])
        #rowdf = pd.DataFrame([dprop, rprop, dvotes, rvotes, IPSelec[0], IPSelec[1], delec, relec])
        #retdf += (rowdf)
        ret[i] = row
    #return (ret,retdf)
    return ret



#create a ditionary
#with a set of sample elections for each state
#computationally expensive
def genStatesDict(numtrials):
    statesSimDict = {}
    statelist = genStateList()
    for state in statelist:
        statesSimDict[state] = getSampleOfStateElections(state, numtrials)
    return statesSimDict
    
#print(getSampleOfStateElections("Alabama", 1000)[0][1])
STATE_SAMPLES_DICT = genStatesDict(NUM_TRIALS)
#print(STATE_SAMPLES_DICT["Iowa"][5])
print("finito")


# In[36]:


#takes an array representing an election in the form:
#'sum of dprops','sum of rprops','dvotes','rvotes','dIPSelec','rIPSelec','dnormelec', 'rnormelec'
#spits out an array in the form (with 0 representing democrats, 1 republicans, 2 other in all cases)
#popular vote winner, normal electoral college winner, IPS electoral college winner,... 
#winner of pop vote lost normal electoral college?, winner of pop vote lost IPS electoral college?
#last two will enter 0 if false, 1 if true, 2 if no one won the given electoral college
def processElection(election):
    popvotewinner = int(election[3] > election[2])
    if (max(election[5], election[4]) < 270):
        IPSWinner = 2
    else :
        IPSWinner = int(election[5] > election[4])
    if (max(election[7], election[6]) < 270):
        electoralCollegeWinner = 2
    else :
        electoralCollegeWinner = int(election[7] > election[6])
    if(electoralCollegeWinner == 2):
        entry4 = 2
    else :
        entry4 = int(electoralCollegeWinner != popvotewinner)
    if(IPSWinner == 2):
        entry5 = 2
    else :
        entry5 = int(IPSWinner != popvotewinner)
    return np.array([popvotewinner, electoralCollegeWinner, IPSWinner, entry4, entry5])

#uses the samples of state elections we have already sampled
#resamples these simulations numretrials times to create an election in all 50 states and DC
#process each election one by one 
#returns relevant details about each election in an array
def runElectionsBySamplingStateDict(numretrials, numtrials):
    statelist = genStateList()
    #numElectionsWherePopVoteWinnerLosesOGEC
    #numElecsWherePopVoteWinnerLosesIPSEC
    #elecArray = np.empty((numretrials, 8))
    ret = np.empty((numretrials, 5))
    for i in range(numretrials):
        election = np.empty((51,8))
        counter = 0
        for state in statelist:
            index = np.random.randint(0,NUM_TRIALS)
            #print(np.random.choice(STATE_SAMPLES_DICT[state]))
            election[counter] = STATE_SAMPLES_DICT[state][index]
            #print(election[i])
            counter += 1
        #print(election)
        total = np.sum(election, axis=0)
        #print(total)
        #print(processElection(total))
        ret[i] = processElection(total)
        #elecArray[i] = total
    return ret

SIMULATED_ELECTIONS_INFO = runElectionsBySamplingStateDict(NUM_RETRIALS, NUM_TRIALS)
print(SIMULATED_ELECTIONS_INFO)
print('finito')


# In[37]:


#finds the stats we are looking
#first value returned: proportion of elections in which democrats win the popular votes
#second value returned: proportion of elections in which republicans win the popular votes
#third value: propportion of elections in which Democrats win the normal (WTA) electoral college
#fourth value: proportion of elections in which Republicans win the normal (WTA) electoral college
#fifth value: propportion of elections in which Democrats win the IPS electoral college
#sixth value: proportion of elections in which Republicans win the IPS electoral college
#seventh value: proportion of elections in which popular vote winner loses normal (WTA) electoral college
#eigth value: proportion of elections in which popular vote winner loses IPS electoral college
#ninth value: proportion of elections in which no one wins normal (WTA) electoral college
#tenth value: proportion of elections in which no one wins IPS electoral college
def getOverallStats(simulationsinfo):
    propdempopvoteswins = sum(SIMULATED_ELECTIONS_INFO[:,0] == 0) / NUM_RETRIALS
    propreppopvoteswins = sum(SIMULATED_ELECTIONS_INFO[:,0] == 1) / NUM_RETRIALS
    propdemnormalecwins = sum(SIMULATED_ELECTIONS_INFO[:,1] == 0) / NUM_RETRIALS
    proprepnormalecwins = sum(SIMULATED_ELECTIONS_INFO[:,1] == 1) / NUM_RETRIALS
    propdemIPSwins = sum(SIMULATED_ELECTIONS_INFO[:,2] == 0) / NUM_RETRIALS
    proprepIPSwins = sum(SIMULATED_ELECTIONS_INFO[:,2] == 1) / NUM_RETRIALS
    propThrownElectionsNormal = sum(SIMULATED_ELECTIONS_INFO[:,3] == 1) / NUM_RETRIALS
    propThrownElectionsIPS = sum(SIMULATED_ELECTIONS_INFO[:,4] == 1) / NUM_RETRIALS
    propnowinnerNormal = sum(SIMULATED_ELECTIONS_INFO[:,3] == 2) / NUM_RETRIALS
    propnowinnerIPS = sum(SIMULATED_ELECTIONS_INFO[:,4] == 2) / NUM_RETRIALS
    ret = np.array([propdempopvoteswins, propreppopvoteswins, propdemnormalecwins, 
                    proprepnormalecwins, propdemIPSwins,proprepIPSwins,propThrownElectionsNormal, 
                    propThrownElectionsIPS, propnowinnerNormal, propnowinnerIPS])
    return ret
#print(SIMULATED_ELECTIONS_INFO)
print(getOverallStats(SIMULATED_ELECTIONS_INFO))
print('finito')


# In[38]:


from scipy.stats import kurtosis

###
#plot all errors in estimates
#find kurtosis of this distribution
###
allprops = np.concatenate([COVAR_STATS[1], COVAR_STATS[2]])
#print(allprops.size)
plt.hist(allprops, bins = 30)
kurtosis(allprops, fisher=False)


# In[39]:


#sanity check covariances
bettercovar = (np.cov(np.array([devStat[0],devStat[1]])))
print(COVAR)
print(bettercovar)

print(bettercovar[1][0] * bettercovar[0][1] * 10)
print(bettercovar[0][0] * bettercovar[1][1])

236/1000000


# In[40]:


39560000 / 55

