#################################################################
These are the EMRI populations used in Babak et al. 2017 (B+17). Please refer to that paper for a full description.

There are 15 models (see below) and for each of them there are 10 files EMRICAT* (*=1,...,10). Each file contains all EMRIs plunging in the Universe (up to z=4.5, or z=6.5 when m2=30msun) in 1 year. So for each model there are 10 years of EMRIs.

Columns in the file are as follow:

 column 1: base 10 log of MBH mass (in solar masses);
 column 2: redshift of the EMRI;
 column 3: spin of the MBH [0, 0.998];
 column 4: inclination angle between the spin of the MBH and the orbital plane of the EMRI [0, pi];
 column 5: distance in Gpc (derived from z, you can ignore this and derive the distance according to the cosmology you want to use.

Notes:

-the mass of the small BH is given in the file name (see below). It is usually 10msun but for one model we used 30msun;

-The eccentricity at plunge, not recorded in the files is roughly taken to be uniform between 0 and 0.2, so you can use random values between those two extremes;

-the spin of the small BH is generally not modeled in EMRI waveforms (at least not in the ones we are using);

-you can randomize on all other parameters (sky location, various initial phases and inclinations etc), see how-to in Barack & Cutler 2004. Then, again, use the sensitivity and orbit of TianQin to compute SNR, estimate parameters etc.


#############################################
below is how to read the file. You can match those keys with the models in table 1 in B+17. Note that in there there are 12 models, whereas there are 15 in this directory. This is because there are some extra models we used for cross-checkin and we did not included in the paper. You can identify those and ignore them.

KEY TO FILES:

-MBH: 10=10msun; 30=30msun
-SIGMA: 1=pess msigma; 2=def msigma; 3=opt msigma
-NPL: 0=no plunges; 10=10 pl per emri; 100=100 pl per emri
-CUSP: 1=cusp erosion; 2=no cusp erosion
-JON: 1=pess mass function (a la Gair 200x); 2=opt mass function (Barausse 2012)
-SPIN: 1=high spins; 2=flat spin dist; 3=Schwarzschild

#####################
standard models with cusp erosion (cusp1) enrico mass function (jon2) and spin distribution (spin1) 10 solar mass black holes (mbh10) 10 plunges per emri (npl1010) and default msigma (gultekin et al):

EMRICAT*_MBH10_SIGMA2_NPL1010_CUSP1_JON2_SPIN1.OUT

#####################
standard but with different msigma (affects t deplation, i.e. the 'duty cicle' or stady state of emris, and affects the cusp regrowht time):

EMRICAT*_MBH10_SIGMA1_NPL1010_CUSP1_JON2_SPIN1.OUT (PESSIMISTIC MSIGMA)
EMRICAT*_MBH10_SIGMA3_NPL1010_CUSP1_JON2_SPIN1.OUT (OPTIMISTIC MSIGMA)

#####################
standard but with 30 solar mass bhs:

EMRICAT*_MBH30_SIGMA2_NPL1010_CUSP1_JON2_SPIN1.OUT 

#####################
standard but with no cusp erosion:

EMRICAT*_MBH10_SIGMA2_NPL1010_CUSP2_JON2_SPIN1.OUT 

#####################
standard but with pessimistic mass function (jon1). in this case there is no cusp erosion:

EMRICAT*_MBH10_SIGMA2_NPL1010_CUSP2_JON1_SPIN1.OUT 

#####################
effect of nplunges (affects duty cicle) per emri on standard model:

EMRICAT*_ MBH10_SIGMA2_NPL1000_CUSP1_JON2_SPIN1.dat (NO PLUNGES)
EMRICAT*_ MBH10_SIGMA2_NPL1100_CUSP1_JON2_SPIN1.dat (100 PLUNGES PER EMRI)

#####################
effect of spins on standard model:

EMRICAT*_MBH10_SIGMA2_NPL1010_CUSP1_JON2_SPIN2.dat (RANDOM SPINS)
EMRICAT*_MBH10_SIGMA2_NPL1010_CUSP1_JON2_SPIN3.dat (SCHWARZSCHILD)

#####################
most pessimistic models: pessimistic mass function (jon1) and 100 plunges per emri

EMRICAT*_MBH10_SIGMA2_NPL1100_CUSP2_JON1_SPIN3.dat (NO SPINS)

#####################
most optimistic model: optimistic mass function (jon2) and no plunges, no cusp erosion, high spins:

EMRICAT*_MBH10_SIGMA2_NPL1000_CUSP2_JON2_SPIN1.dat 

