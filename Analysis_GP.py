"""

    Copyright (C) 2022  Miguel Sabogal, Javier González

"""
import sys
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from gapp import dgp, covariance
import scipy.integrate as integrate

class Gp_Analisys_D:

    """This class contains the necessary methods to estimate sigma_{H0} from simulated
       data { Z , Ln(D) } associated with the LSST using the Gaussian process method,
       through the Marina Seikel code GaPP (Gaussian Processes in Python) """

    """
    Attributes and parameters of Gp_Analisys:

    cov_fun             Covariance function
    Z0                  Redshifft vector of the inputs
    SD                  Vector of Ln(D) = sigma/D of the inputs
    Sigma               Vector sigma of data
    Xstar(xmin,         Points where the function is to be reconstructed
    xmax,nstar)
    fmean               A priori mean function just in (Gp_Analisys_H)
    muargs              List of arguments to be passed to mu
    fgrad               'True' if the gradient of the covariance function is used for the GP, 'False' else
    Omega_m             Matter content
    H0                  Hubble parameter current value
    alpha/beta          Parameters associated with the HDE model of Granda-Oliveros
    ini_sigma           Initial seed for sigmaf
    ini_lpar'           Initial seed for lpar
    N_rec               Number of reconstructions
    N_simulated_by_rec  Number of simulated by reconstructions
    fiducial            Fiducial model

    """

    def __init__(self,z,sigma_lnD,parametros):
        self.Z0 = z ; self.SD = sigma_lnD;
        for name in sorted(parametros.keys()): setattr(self,name,parametros[name])

        ##### Select the Hubble fiducial model ####
        if self.fiducial == 'GO': self.H = self.HGO
        if self.fiducial == 'LCDM': self.H = self.HCDM
        if self.fiducial == 'WCDM': self.H = self.HWCDM

    def HCDM(self,z):   #### LAMBDACDM model
        return self.H0*(self.Omega_m*(1+z)**(3) + (1-self.Omega_m))**(1/2)
    
    def HWCDM(self,z):   #### LAMBDACDM model
        return self.H0*(self.Omega_m*(1+z)**(3) + (1-self.Omega_m)*(1+z)**(3*(1+self.w)) )**(1/2)

    def HGO(self,z):    #### Granda-Oliveros model
        H2 =self.H0**2 *((1 +(2*self.alpha-3*self.beta)/(2-2*self.alpha + 3*self.beta))*self.Omega_m*(1+z)**(3) +
                       (1-(2*self.Omega_m)/(2-2*self.alpha+3*self.beta))* (1+z)**(2*(self.alpha-1)/self.beta))
        return np.sqrt(H2)

    def D(self,z):      #### Diameter distance integral for fiducial
        if type(z) is not int :
            Dz = np.zeros(len(z));
            for i in range (0,len(z)):
                Dz[i] = integrate.quad(lambda x: 1/self.H(x), 0.0,z[i])[0]
        else:
            Dz = self.H0 *integrate.quad(lambda x: 1/self.H(x), 0.0,z)[0]
        return Dz

    def modelofiducial(self):
        """simulates the value of D from the fiducial and
           the {Z0,lnD} data in a Gaussian random way"""
        simulado = np.random.normal(self.D(self.Z0),self.D(self.Z0)*self.SD)
        return np.concatenate((np.array([0.0]),simulado));

    def N_models(self):
        """Generates a set of N_rec simulated models and
           set D(z=0) = 0 for all cases"""
        self.sigma_D = self.D(self.Z0)*self.SD
        self.Z = np.concatenate((np.array([0.0]),self.Z0));
        self.Sigma= np.concatenate((np.array([0.0]),self.sigma_D))

        self.Data = {}
        for i in range(0,self.N_rec):
            self.Data["model{0}".format(i)] = self.modelofiducial()

    def N_reconstructions(self):

        """Between xmin and xmax, nstar points of the function
           will be reconstructed for each of the N_rec cases"""

        xmin = self.xmin
        xmax = self.xmax
        nstar =self.nstar

        # initial values of the hyperparameters
        initheta = [self.ini_sigma, self.ini_lpar]

        self.rec_GP = {}; self.theta_GP = {};
        for i in tqdm (range(0,self.N_rec), desc="GP_reconstructions..."):
            g = dgp.DGaussianProcess(self.Z, self.Data["model{0}".format(i)],self.Sigma,
                                     cXstar=(xmin,xmax,self.nstar),covfunction=self.cov_fun,grad=self.fgrad)

            (rec,theta) = g.gp(theta=initheta)
            (drec, dtheta)= g.dgp(thetatrain='False')

            self.rec_GP["GP{0}".format(i)] = np.c_[drec[:,0],1.0/drec[:,1],(1.0/(drec[:, 1])**2)*drec[:,2]]
            self.theta_GP["GP{0}".format(i)] = theta


    def statistics(self,print_stad='False'):
        #Warning for convergence
        n_rand = np.random.randint(0,self.N_rec)
        if self.rec_GP["GP{0}".format(n_rand)][:,1][2] == self.rec_GP["GP{0}".format(n_rand)][:,1][-2]:
            sys.exit("Method 'N_reconstructions' FAIL, try again changing 'initheta' or --> fgrad = 'False'")

        #preliminary statistics for eliminate the outliers later
        preH = np.array([])
        for i in range (0,self.N_rec):
            rec = self.rec_GP["GP{0}".format(i)]
            preH = np.append(preH,rec[:,1][0])

        self.preH0 = preH.mean()
        self.preSigma = preH.std()

        # stat block
        HT={} ; self.outliers = 0;
        for i in tqdm (range(0,self.N_rec), desc="GP_statistics..."):
            rec = self.rec_GP["GP{0}".format(i)]

            if abs(rec[:,1][0]-self.preH0) >= 5.0*self.preSigma: #delete outliers
                self.outliers += 1
                continue

            # simulated N_simulated_by_rec cases for each reconstrucion
            # and concatenates each value with the others of it's respective z
            for j in range(0,self.nstar):
                try:
                    HT["H{0}".format(j)] = np.concatenate((HT["H{0}".format(j)],
                                                           np.random.normal(rec[:,1][j],rec[:,2][j],
                                                           size=self.N_simulated_by_rec) ))
                except:
                    HT["H{0}".format(j)] = np.array([])
                    HT["H{0}".format(j)] = np.concatenate((HT["H{0}".format(j)],
                                                           np.random.normal(rec[:,1][j],rec[:,2][j],
                                                           size=self.N_simulated_by_rec) ))

        HF=np.array([]); SF=np.array([]); # mean and std for each concatenated vector of z
        for j in range(0,self.nstar):
            HF= np.concatenate((HF,np.array([HT["H{0}".format(j)].mean()]) ))
            SF= np.concatenate((SF,np.array([HT["H{0}".format(j)].std() ]) ))

        self.HF = HF ; self.SF = SF; self.HT = HT;
        self.H0_final = HT['H0'].mean(); self.sigmaH0_final = HT['H0'].std();

        if print_stad != 'False':
            print("H0 =",self.H0_final,"  ","Sigma de H0 =",self.sigmaH0_final)
            print('')
            print('Limites =',self.H0_final+ self.sigmaH0_final,self.H0_final-self.sigmaH0_final)


    def plots(self,x_lims='None',y_lims='None',name_save='None'):

        plt.figure(figsize=(15,5))
        plt.subplot(121)

        if x_lims != 'None':
            plt.xlim(x_lims[0],x_lims[1])
        if y_lims != 'None':
            plt.ylim(y_lims[0],y_lims[1])

        for i in range(0,self.N_rec):
            rec = self.rec_GP["GP{0}".format(i)]
            if abs(rec[:,1][0]-self.preH0) >= 5.0*self.preSigma:
                continue
            plt.plot(rec[:, 0], rec[:, 1])

        plt.ylabel('H(z) [km/s/Mpc]')
        plt.xlabel('z')
        plt.tick_params(direction='in',length=4, width=2, colors='k', right=True, top=True,labelright=False,labeltop=False)

        plt.subplot(122)

        if x_lims != 'None':
            plt.xlim(x_lims[0],x_lims[1])
        if y_lims != 'None':
            plt.ylim(y_lims[0],y_lims[1])

        rec = self.rec_GP["GP0"]
        plt.fill_between(rec[:, 0], self.HF + self.SF, self.HF - self.SF,facecolor='lightblue')
        plt.plot(rec[:, 0], self.HF)

        plt.ylabel('H(z) [km/s/Mpc]')
        plt.xlabel('z')
        plt.tick_params(direction='in', length=4, width=2, colors='k', right=True, top=True,
                    labelright=False,labeltop=False)

        plt.title(r'$H_{0} = {1} \,\,\,\, \sigma = {2}$'.format(0,round(self.HT['H0'].mean(),ndigits=3),
                                                         round(self.HT['H0'].std(),ndigits=3)),size=14);
        if name_save != 'None':
            plt.savefig(name_save+'.pdf',format="pdf", dpi=2000,bbox_inches='tight')

        plt.show()

class Gp_Analisys_H(Gp_Analisys_D):

    """This class contains the necessary methods to estimate sigma_{H0} from simulated
       data { Z , Ln(H) } associated with the LSST using the Gaussian process method """

    def modelofiducial(self):
        """simulates the value of H from the fiducial and
           the {Z0,lnH} data in a Gaussian random way"""
        simulado = self.H(self.Z0)
        return np.c_[self.Z0,simulado,simulado*self.SD]

    def N_models(self):
        """Generates a set of N_rec simulated models"""

        reconstrucion = self.modelofiducial();

        self.Z = reconstrucion[:,0];
        self.model = reconstrucion[:,1];
        self.SigmaHZ= reconstrucion[:,2];

        self.Data = {}
        for i in range(0,self.N_rec): #poner mean y sigma avariar POR RECOSNTRUCCION
            self.Data["model{0}".format(i)] = np.random.normal(self.model,self.SigmaHZ)

    def N_reconstructions(self):

        """Between xmin and xmax, nstar points of the function
           will be reconstructed for each of the N_rec cases"""

        xmin = self.xmin
        xmax = self.xmax
        nstar =self.nstar

        # initial values of the hyperparameters
        initheta = [self.ini_sigma, self.ini_lpar]

        # select the mean prior function or None
        if self.f_mean == 'GO' and self.fiducial != 'GO':
            self.mean = self.HGO
        if self.f_mean == 'LCDM' and self.fiducial != 'LCDM':
            self.mean = self.HCDM
        if self.f_mean == 'None':
            self.mean = None;

        self.rec_GP = {}; self.theta_GP = {};
        for i in tqdm (range(0,self.N_rec), desc="GP_reconstructions..."):
            g = dgp.DGaussianProcess(self.Z,self.Data["model{0}".format(i)],self.SigmaHZ,mu=self.mean
                                      ,cXstar=(xmin,xmax,nstar),covfunction=self.cov_fun,grad=self.fgrad)

            (self.rec_GP["GP{0}".format(i)],self.theta_GP["GP{0}".format(i)]) = g.gp(theta=initheta);
