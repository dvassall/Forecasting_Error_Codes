# Get variables for 100m reading
def Vars100m(data, refData):
    import numpy as np
    import pandas as pd

    times = pd.concat([data['year'], data['month'], data['day'], data['hour'], data['minute']], axis=1)
    time = pd.to_datetime(times)
    h1 = np.cos(np.array(data['hour']+data['minute']/60)*np.pi/12)
    h2 = np.sin(np.array(data['hour']+data['minute']/60)*np.pi/12)
    ws = np.array(data['spd'])
    Dir = np.array(data['dir'])
    dirNS = np.cos(Dir*np.pi/180)
    dirEW = np.sin(Dir*np.pi/180)
    tc = data['tc']+273.15 # in Kelvin
    T = data['T']+273.15 # in Kelvin
    tke = .5*(np.array(data['u_u'])+np.array(data['v_v'])+np.array(data['w_w']))
    rmsu = np.sqrt(np.array(data['u_u']))
    ti = rmsu/ws
    turbHtFlux = np.array(data['w_tc'])
    u = -ws*np.sin(data['dir']*np.pi/180)
    v = -ws*np.cos(data['dir']*np.pi/180)
    w = np.array(data['w'])
    fricVel = (np.array(refData['u_w'])**2+np.array(refData['v_w'])**2)**.25 # measured at 20m
    normFricVel = fricVel/ws
    vpt = np.array(data['vpt'])
    arimaErr = data['arima error']
    arimaPred = data['arima preds']

    # Extra Vars
    Nsq10020 = np.array(data['Nsq10020'])
    Nsq10020[Nsq10020<0] = 0
    dtc10020 = (tc-np.array(refData['tc']+273.15))/80
    dT10020 = (T-np.array(refData['T']+273.15))/80

    # Gradient & Flux Rich between 100-20m
    refws = np.array(refData['spd'])
    refDir = np.array(refData['dir'])
    refU = -refws*np.sin(refDir*np.pi/180)
    refV = -refws*np.cos(refDir*np.pi/180)
    richFtop = (9.81/tc)*(np.array(data['w_tc']))
    dudz = (u-refU)/80
    dvdz = (v-refV)/80
    richFbottom = np.array(data['u_w'])*dudz+np.array(data['v_w'])*dvdz
    richF = richFtop/richFbottom
    richF[richF>5] = 5
    richF[richF<-5] = -5

    richGtop = Nsq10020
    richGbottom = dudz**2+dvdz**2
    richG = richGtop/richGbottom
    richG[richG>5] = 5

    cols = ['ws', 'u', 'v', 'w', 'dir', 'dirNS', 'dirEW', 'T',
              'tc', 'vpt', 'tke', 'rmsu', 'TI', 'turbHtFlux', 'fricVel', 'normFricVel',
              'Nsq10020', 'dT10020', 'dtc10020',
              'richF10020', 'richG10020', 't1', 't2', 'error', 'preds']

    Data = np.stack((ws, u, v, w, Dir, dirNS, dirEW, T, tc, vpt, tke,
                     rmsu, ti, turbHtFlux, fricVel, normFricVel,
                     Nsq10020, dT10020, dtc10020, richF,
                     richG, h1, h2, arimaErr, arimaPred), axis=1)
    pdData = pd.concat([pd.DataFrame(time, columns=['time']), pd.DataFrame(Data, columns=cols)], axis=1)
    
    return pdData

# Get variables for 20m reading
def Vars(data):
    import numpy as np
    import pandas as pd
    
    data = data.iloc[:-1, :]

    times = pd.concat([data['year'], data['month'], data['day'], data['hour'], data['minute']], axis=1)
    time = pd.to_datetime(times)
    ws = np.array(data['spd'])
    Dir = np.array(data['dir'])
    dirNS = np.cos(Dir*np.pi/180)
    dirEW = np.sin(Dir*np.pi/180)
    tc = data['tc']+273.15 # in Kelvin
    T = data['T']+273.15 # in Kelvin
    turbHtFlux = np.array(data['w_tc'])
    u = -ws*np.sin(data['dir']*np.pi/180)
    v = -ws*np.cos(data['dir']*np.pi/180)
    w = np.array(data['w'])
    
    cols = ['ws', 'u', 'v', 'w', 'dir', 'dirNS', 'dirEW', 'T',
              'tc']

    Data = np.stack((ws, u, v, w, Dir, dirNS, dirEW, T, tc), axis=1)
    pdData = pd.concat([pd.DataFrame(time, columns=['time']), pd.DataFrame(Data, columns=cols)], axis=1)
    
    return pdData


def data_Func(Data100, Data20, n_steps, n_skip, Vars):

    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split as tts

    # create separate vectors for error autocorrelation
    err = np.array(Data100['error'])
    ws100 = np.array(Data100['ws'])
    pred = np.array(Data100['preds'])
    if len(Vars)>0:
        dataset = np.zeros((len(err), len(Vars)))
        for i, var in enumerate(Vars):
            if var=='ws20':
                dataset[:, i] = np.array(Data20['ws'])
            elif var=='ws100':
                dataset[:, i] = np.array(Data100['ws'])
            else:
                dataset[:,i] = np.array(Data100[var])

    # loop through data and get input, target values
    x, errY = list(), list()
    tarWS, currPred = list(), list()
    inputIndices = (np.linspace(1, n_steps+1, n_steps+1)*n_skip-n_skip).astype('int')
    for i in range(len(err)):
        # find end of the pattern
        movingIndices = i+inputIndices
        end_ix = movingIndices[-1]
        # check if we are beyond the dataset
        if end_ix >= len(err):
            break
        # gather input and output parts of the pattern
        # autocorr. error goes one step farther back than the rest of the inputs
        erry = err[end_ix]
        # future wind speed
        fws = ws100[end_ix]
        # arima prediction for next time step
        aPred = pred[end_ix]
        if len(Vars)>0:
    #             X = dataset[movingIndices[1:], :]
            X = dataset[movingIndices[:-1], :]
            x.append(X)
        errY.append(erry)
        tarWS.append(fws)
        currPred.append(aPred)
    erry = np.array(errY)
    tarWS, currPred = np.array(tarWS), np.array(currPred)
    x = np.array(x)

    Vars_new = Vars.copy()
    totalVars = list()
    #     vars_Err = ['pred error']
    for i in range(n_steps):
        totalVars = totalVars+[In+'_'+str(n_steps-i) for In in Vars_new]
    X = np.reshape(x, (int(x.shape[0]), int(len(totalVars))))
    X = pd.DataFrame(X, columns=totalVars)
    y = pd.DataFrame(erry.flatten(), columns=['Error'])

    # # Split into training, testng, and validation data
    train_inputs, test_inputs, train_target, test_target = tts(
         X, y, test_size=0.25, random_state=1)
    trainTarWS, testTarWS, trainPred, testPred = tts(
        tarWS, currPred, test_size=0.25, random_state=1)


    return(train_inputs, train_target, test_inputs, test_target, totalVars, trainTarWS, trainPred, testTarWS, testPred)


def biasCorr(trainTar, testTar):
    import numpy as np
    import pandas as pd
    
    trainErr = np.array(trainTar).flatten()
    testErr = np.array(testTar).flatten()
    mu = trainErr.mean()
    trainTar = pd.DataFrame(trainErr-mu, columns=['Error'])
    testTar = pd.DataFrame(testErr-mu, columns=['Error'])
    
    return trainTar, testTar, mu

