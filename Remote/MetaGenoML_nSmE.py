import fromGenoToMLData as fm
import pickle
import os
import genoMML as gm
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri

genosTrain = ["SPANISH", "PDBP", "HBS"]  # Spanish repo has to be in he first place
covsTrain = ["COVS_" + cov for cov in genosTrain]
workPath = "/home/edusal/MML-nSmE-4/"
predictors = ["DISEASE", "PHENO_PLINK", "PHENO_PLINK"]
genoTest = ["PPMI", "HBS", "PDBP"]
covsTest = ["COVS_" + cov for cov in genoTest]
path2Data = "/home/edusal/data/FINALES/"  # all data in the same folder (geno and covs)
path2packages = "/home/edusal/packages/"  # all packages in the same folder (prsice, gwas and plink)

try:
    os.mkdir(workPath)
except Exception:
    pass

pandas2ri.activate()

readRDS = robjects.r['readRDS']
pcaSET = readRDS('/home/edusal/OBTAIN_PCA/pcaSET.rds')

#######################
# metanSmE function
########################

handlersML = {}
experts_L1 = {}
spanishSNPs = None

for i in range(0, len(genosTrain)):

    lworkPath = workPath + "/Repo_" + genosTrain[i]
    try:
        os.mkdir(lworkPath)
    except Exception:
        pass

    repoKeyName = "Repo" + genosTrain[i]

    if (os.path.isfile(lworkPath + "/mldatahandler.pydat")):
        with open(lworkPath + '/mldatahandler.pydat', 'rb') as mldatahandler_file:
            handlersML[repoKeyName] = pickle.load(mldatahandler_file)
        print(lworkPath + '/mldatahandler.pydat' + " already exists.")
    else:
        handlersML[repoKeyName] = fm.fromGenoToMLdata(lworkPath,
                                                      path2Geno=path2Data + "/" + genosTrain[i],
                                                      path2Covs=path2Data + "/" + covsTrain[i],
                                                      predictor=predictors[i],
                                                      path2GWAS=path2packages,
                                                      path2PRSice=path2packages,
                                                      path2plink=path2packages,
                                                      snpsSpain=spanishSNPs,
                                                      iter=i + 1)
    if genosTrain[i] == "SPANISH":
        spanishSNPs = handlersML[repoKeyName]["snpsToPull"]

    # generate k folds and obtain their level 1 experts
    if (os.path.isfile(lworkPath + "/models.pydat")):
        with open(lworkPath + '/models.pydat', 'rb') as mldatahandler_file:
            experts_L1[repoKeyName] = pickle.load(mldatahandler_file)
        print(workPath + '/models.pydat' + " already exists.")
    else:
        experts_L1[repoKeyName] = gm.genModels_nSmE(workPath=lworkPath,
                                                    handlerMLdata=handlersML[repoKeyName],
                                                    k=5)

lworkPath = workPath + "/META/"
try:
    os.mkdir(lworkPath)
except Exception:
    pass

if (os.path.isfile(lworkPath + "/expert_L2.pydat")):
    with open(lworkPath + "/expert_L2.pydat", 'rb') as finalResult_file:
        expert_L2 = pickle.load(finalResult_file)
    print(lworkPath + "/expert_L2.pydat" + " already exists.")
else:
    expert_L2 = gm.trainAndTestMML_nSmE(experts_L1=experts_L1,
                                        genoTrain=genosTrain,
                                        handlersML=handlersML,
                                        workPath=lworkPath)

for geno, covs in zip(genoTest, covsTest):
    print()
    print("#" * 30)
    print("Final Test with " + geno + "\n")

    lworkPath = workPath + "/" + geno + "/"
    try:
        os.mkdir(lworkPath)
    except Exception:
        pass

    if (os.path.isfile(lworkPath + "/handler- " + geno + ".pydat")):
        with open(lworkPath + "/handler- " + geno + ".pydat", 'rb') as handlerTest_file:
            handlerTest = pickle.load(handlerTest_file)
        print(lworkPath + "/handler- " + geno + ".pydat" + " already exists.")
    else:
        # generate mldata for the test repository
        handlerTest = gm.prepareFinalTest(workPath=lworkPath,
                                          path2Geno=path2Data + "/" + geno,
                                          path2Covs=path2Data + "/" + covs,
                                          predictor="PHENO_PLINK",
                                          snpsToPull=spanishSNPs)

        with open(lworkPath + "/handler- " + geno + ".pydat", 'wb') as handlerTest_file:
            pickle.dump(handlerTest, handlerTest_file)

    # obtain final evaluation results
    finalResult = gm.finalTest_nSmE(lworkPath,
                                    expert_L2,
                                    handlerTest,
                                    experts_L1)

    with open(lworkPath + "/finalResults.pydat", 'wb') as finalResult_file:
        pickle.dump(finalResult, finalResult_file)
