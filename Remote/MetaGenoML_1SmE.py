import fromGenoToMLData as fm
import pickle
import os
import genoMML as gm
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri

genoTrain = "SPANISH"  # Cohorte a estudiar
covsTrain = "COVS_SPANISH"  # Covariancias de la cohorte
path2Data = "/home/edusal/data/FINALES"  # all data in the same folder (geno and covs)
path2packages = "/home/edusal/packages/"  # all packages in the same folder (prsice, gwas and plink)
workPath = "/home/edusal/MML-1SmE-4/"
# dir.create(workPath)  # Si ya esta creado genera un warning

pandas2ri.activate()

readRDS = robjects.r['readRDS']
pcaSET = readRDS('/home/edusal/OBTAIN_PCA/pcaSET.rds')

genoTest = ["PPMI", "HBS", "PDBP"]  # Cohortes para para probar
covsTest = ["COVS_" + cov for cov in genoTest]  # Covarianzas de las cohortes para probar


lworkPath = workPath + "dataRepo/"
# dir.create(lworkPath) # Genera un warning si ya esta creado

if (os.path.isfile(lworkPath + "mldatahandler.pydat")):
    with open(lworkPath + 'mldatahandler.pydat', 'rb') as mldatahandler_file:
        handlerML = pickle.load(mldatahandler_file)
    print(lworkPath + 'mldatahandler.pydat' + " already exists.")
else:
    handlerML = fm.fromGenoToMLdata(lworkPath,
                                    path2Geno=path2Data + "/" + genoTrain,
                                    path2Covs=path2Data + "/" + covsTrain,
                                    predictor="DISEASE",  # El predictor de la cohorte SPANISH es DISEASE
                                    path2GWAS=path2packages,  # GWAS mide la relación de las mutaciones con la enfermedad
                                    path2PRSice=path2packages,  # Usa GWAS para seleccionar variables para el modelo de ML
                                    path2plink=path2packages,  # Archivos binarios con información para el ML
                                    iter=1)  # Mide el numero de pliegues que hacer en el data frame

imputeMissingData = "median"
gridSearch = 3
ncores = 20
algsML = None
# generate k folds and obtain their level 1 experts
experts_L1 = gm.genModels_1SmE(workPath=lworkPath,
                               handlerMLdata=handlerML,
                               k=5)

algsMML = algsML

expert_L2 = gm.trainAndTestMML_1SmE(experts_L1,
                                    handlerML,
                                    lworkPath,)

for geno, covs in zip(genoTest, covsTest):
    print()
    print("#" * 30)
    print("Final Test with " + geno + "\n")

    lworkPath = workPath + "/" + covs + "/"
    try:
        os.mkdir(lworkPath)
    except Exception:
        pass

    # generate mldata for the test repository
    handlerTest = gm.prepareFinalTest(workPath=lworkPath,
                                      path2Geno=path2Data + "/" + geno,
                                      path2Covs=path2Data + "/" + covs,
                                      predictor="PHENO_PLINK",
                                      snpsToPull=handlerML["snpsToPull"])

    with open(lworkPath + "/handler- " + geno + ".pydat", 'wb') as handlerTest_file:
        pickle.dump(handlerTest, handlerTest_file)

    # obtain final evaluation results
    finalResult = gm.finalTest_1SmE(lworkPath,
                                    expert_L2,
                                    handlerTest,
                                    experts_L1)

    with open(lworkPath + "/finalResults.pydat", 'wb') as finalResult_file:
        pickle.dump(finalResult, finalResult_file)
