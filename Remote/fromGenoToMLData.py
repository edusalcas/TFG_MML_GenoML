import os
import sys
import subprocess
import re
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold


def xstr(s):
    if s is None:
        return ''
    return str(s)

#----------------------------------------#
#--------getHandlerToGenotypeData--------#
#----------------------------------------#


def getHandlerToGenotypeData(geno,
                             covs,
                             predictor,
                             id,
                             fid,
                             pheno,
                             verify=True):

    if verify and\
        (not os.path.isfile(geno + ".bed") or
         not os.path.isfile(geno + ".bim") or
         not os.path.isfile(geno + ".fam")):

        sys.exit("Files not found: " + geno)

    if not os.path.isfile(covs + ".cov"):
        sys.exit("Files not found: " + covs)

    cdata = pd.read_csv(covs + ".cov", header=0, delim_whitespace=True)

    phenofile = pheno + ".pheno"

    if not os.path.isfile(phenofile):
        if predictor not in cdata.columns:
            sys.exit("Error: " + predictor + " not in " + cdata.columns)

    if id not in cdata.columns:
        sys.exit("Error: " + id + " not in " + cdata.columns)

    if fid not in cdata.columns:
        sys.exit("Error: " + fid + " not in " + cdata.columns)

    if os.path.isfile(phenofile):
        phenoDat = pd.read_csv(phenofile, header=0, delim_whitespace=True)
    else:
        phenoDat = cdata[[fid, id, predictor]]

        # write it
        phenoDat.to_csv(phenofile, index=False, sep=' ')

    cdata = cdata[[c for c in cdata.columns if c != predictor]]

    genoHandler = {}
    genoHandler["geno"] = geno
    genoHandler["pheno"] = pheno
    genoHandler["covs"] = covs
    genoHandler["id"] = id
    genoHandler["fid"] = fid
    genoHandler["covsDat"] = cdata
    genoHandler["phenoDat"] = phenoDat
    genoHandler["Class"] = predictor

    # attr(genoHandler,"class") = "genohandler"

    return(genoHandler)

#----------------------------------------#
#--------getPartitionsFromHandler--------#
#----------------------------------------#


def getPartitionsFromHandler(genoHandler,
                             path2plink,
                             workPath=None,
                             how="k-fold cv",
                             k=10,
                             p=0.75):

    assert how == "k-fold cv" or how == "holdout"

    genoHandler["plan"] = how
    pred = genoHandler["Class"]  # accessing the column given by the user

    if how == "holdout":
        train, test = train_test_split(genoHandler["phenoDat"][pred], test_size=(1 - p))
        hoFold = {"train1": train, "test1": test}
        genoHandler["folds"] = hoFold
        genoHandler["nfolds"] = 1
        genoHandler["trainFolds"] = {}
        genoHandler["testFolds"] = {}
        genoHandler["trainFolds"]["train1"] = train
        genoHandler["testFolds"]["test1"] = test
    else:
        folds = KFold(n_splits=k)
        folds = None
        genoHandler["folds"] = folds
        genoHandler["nfolds"] = k

        trainFolds = {}  # list with the indexes used for train in each one of the k cross-validations
        testFolds = {}  # list with the indexes saved for test in each one of the k cross-validations

        i = 1
        for train_index, test_index in folds.split(genoHandler["phenoDat"][pred]):
            trainFolds["train" + i] = train_index
            testFolds["test" + i] = test_index
            i = i + 1

        genoHandler["trainFolds"] = trainFolds
        genoHandler["testFolds"] = testFolds

    genoHandler = genDataFileNames(genoHandler=genoHandler, path2plink=path2plink)

    return(genoHandler)


#----------------------------------------#
#-------------genDataFileNames-----------#
#----------------------------------------#


def genDataFileNames(genoHandler,
                     # workPath = NULL,
                     onlyFold=-1,
                     path2plink="~/genoml-core-master/otherPackages/",
                     which_to_create=["train", "test"]):

    workPath = os.path.dirname(genoHandler["pheno"])

    if onlyFold > 0:
        indexes = onlyFold
    else:
        indexes = range(1, genoHandler["nfolds"] + 1)

    for tt in which_to_create:
        key = tt + "FoldFiles"
        genoHandler[key] = {}

    for i in indexes:

        for tt in which_to_create:
            foldFiles = {}
            key = tt + "FoldFiles"
            dirFold = workPath + "/" + "Fold" + tt + str(i) + "/"

            try:
                os.mkdir(dirFold)
            except Exception:
                pass

            splitIdx = tt + str(i)

            # dataframe with the covariates of the individuals given by this fold
            covsFoldDT = genoHandler["covsDat"].iloc[genoHandler[tt + "Folds"][splitIdx].index, ]
            covsFold = "COVS_" + tt + str(i)
            covsFile = dirFold + "/" + covsFold + ".cov"

            # write it
            covsFoldDT.to_csv(covsFile, index=False, sep=' ')

            foldFiles["covsFile"] = covsFile

            genoFold = os.path.basename(genoHandler["geno"]) + tt + "." + str(i)
            phenoFold = os.path.basename(genoHandler["pheno"]) + str(i)

            # generation of the ids file
            idsDT = genoHandler["covsDat"].iloc[genoHandler[tt + "Folds"][splitIdx].index, ][[genoHandler["fid"], genoHandler["id"]]]
            idsfile = dirFold + "/" + os.path.basename(genoHandler["covs"]) + tt + ".ids"

            # write it
            idsDT.to_csv(idsfile, index=False, sep=' ')

            foldFiles["idsFile"] = idsfile

            # Generating the phenotype files
            phenoFoldDT = genoHandler["phenoDat"].iloc[genoHandler[tt + "Folds"][splitIdx].index, ]
            phenofile = dirFold + "/" + phenoFold + tt + ".pheno"

            # write it
            phenoFoldDT.to_csv(phenofile, index=False, sep=' ')

            foldFiles["phenoFile"] = phenofile

            # Generating the genotype files
            genofile = dirFold + "/" + genoFold
            command = path2plink + "plink --bfile " + genoHandler["geno"] + " --keep " + idsfile + " --make-bed --out " + genofile
            foldFiles["genoFile"] = genofile
            foldFiles["genoFileCommand"] = command
            foldFiles["filesCreated"] = False
            genoHandler[key][i] = foldFiles

    return(genoHandler)

#----------------------------------------#
#------------genDataFromHandler----------#
#----------------------------------------#


def genDataFromHandler(genoHandler,
                       onlyFold=-1,
                       which_to_create=["train", "test"],
                       lazy=False):

    if "father" in genoHandler.keys() and genoHandler["father"] is not None:
        genDataFromHandler(genoHandler["father"], onlyFold, which_to_create, lazy)

    if "nfolds" in genoHandler.keys() and genoHandler["nfolds"] is not None:
        if onlyFold > 0:
            indexes = onlyFold
        else:
            indexes = range(1, genoHandler["nfolds"] + 1)

        for i in indexes:
            for tt in which_to_create:
                print(str(i))

                key = tt + "FoldFiles"
                print(key)

                if key in genoHandler.keys() and i in genoHandler[key].keys() and genoHandler[key][i] is not None:
                    if lazy:
                        if not os.path.isfile(genoHandler[key][i]["genoFile"] + ".bim"):
                            print("Creating genotype data for fold " + str(i) + " and " + tt + " with command " + genoHandler[key][i]["genoFileCommand"] + "\n")
                            os.system(genoHandler[key][i]["genoFileCommand"])
                            genoHandler[key][i]["filesCreated"] = True
                        else:
                            print(genoHandler[key][i]["genoFile"] + " already created, skipping\n")
                    else:
                        if genoHandler[key][i]["genoFileCommand"] is not None:
                            print("Creating genotype data for fold " + str(i) + " and " + tt + " with command " + genoHandler[key][i]["genoFileCommand"] + "\n")
                            os.system(genoHandler[key][i]["genoFileCommand"])
                            genoHandler[key][i]["filesCreated"] = True

    return(genoHandler)

#----------------------------------------#
#-------------mostRelevantSNPs-----------#
#----------------------------------------#


def mostRelevantSNPs(handler,
                     path2GWAS,
                     path2plink,
                     reduce="PRSICE",
                     phenoScale="DISC",
                     gwas="RISK_noSpain.tab",
                     SNPcolumnatGWAS="SNP",
                     herit=None,
                     clumpField="p",
                     addit="NA",
                     path2gcta64=None,
                     cores=1,
                     prune_windowSize=10000,
                     prune_stepSize=1,
                     prune_r2=0.1,
                     path2PRSice=None,
                     PRSiceexe="PRSice_linux",
                     workPath=None,
                     force=False):

    if path2gcta64 is None:
        path2gcta64 = path2plink
    if path2PRSice is None:
        path2PRSice = path2plink

    verifyHandler(handler)

    if(path2plink != ""):
        assert os.path.isdir(path2plink)

    handler = genDataFromHandler(handler, lazy=True)
    geno = os.path.basename(handler["geno"])
    pheno = os.path.basename(handler["pheno"])

    if workPath is None:
        workPath = os.path.dirname(handler["pheno"]) + "/"

    path2Genotype = os.path.dirname(handler["geno"]) + "/"
    cov = os.path.basename(handler["covs"])

    # options passed from list on draftCommandOptions.txt
    prefix = "g-" + geno + "-p-" + pheno + "-c-" + cov + "-a-" + addit
    fprefix = workPath + "/" + prefix  # CAMBIAR

    if reduce == "PRUNE" or reduce == "DEFAULT":
        command = path2plink + "plink --bfile " + path2Genotype + "/" + geno + " --indep-pairwise " +\
            prune_windowSize + " " + prune_stepSize + " " + prune_r2 + " --out " + fprefix +\
            ".temp"
        os.system(command)

        command = path2plink + "plink --bfile " + path2Genotype + "/" + geno + " --extract " +\
            fprefix + ".temp.prune.in --recode A --out " + fprefix + ".reduced_genos"
        os.system(command)

        command = "cut -f 1 " + fprefix + ".temp.prune.in > " + fprefix + ".reduced_genos_snpList"
        os.system(command)

        handler["rgenosSnpList"] = fprefix + ".reduced_genos_snpList"
        handler["snpsToPull"] = handler["rgenosSnpList"]

        return(handler)

    elif reduce == "SBLUP" and gwas != "NA" and herit is not None:

        # if $reduce = SBLUP, $gwas is not NA and $herit is not NA
        command = "wc -l < " + path2Genotype + "/" + geno + ".bim | awk '{print $1}'"
        print("The command to run: " + command)
        nsnps = int(os.system(command))
        print("The number of starting SNPs is " + nsnps)
        sbluplambda = nsnps * (1 / herit) - 1
        print("We will use a sblup-lambda parameter of" + sbluplambda)

        if force or \
           not os.path.isfile(workPath + "/" + geno + ".forSblup.bim") or \
           not os.path.isfile(workPath + "/" + geno + ".forSblup.fam") or \
           not os.path.isfile(workPath + "/" + geno + ".forSblup.bed"):

            command = path2plink + "plink --bfile " + path2Genotype + "/" + geno + " --pheno " +\
                handler["pheno"] + ".pheno --make-bed --out " + workPath + "/" + geno + ".forSblup"
            os.system(command)

        # Note that the GWAS for cojo analisys must be in the form
        # SNP A1 A2 freq b se p N
        # rs1001 A G 0.8493 0.0024 0.0055 0.6653 129850
        # rs1002 C G 0.0306 0.0034 0.0115 0.7659 129799
        # rs1003 A C 0.5128 0.0045 0.0038 0.2319 129830
        command = path2gcta64 + "/gcta64 --bfile " + workPath + "/" + geno + ".forSblup --cojo-file " +\
            path2GWAS + "/" + gwas + " --cojo-sblup " + sbluplambda + " --cojo-wind 10000 --maf 0.01 --chr 22 --thread-num " +\
            cores + " --out " + fprefix + ".temp"
        os.system(command)

        # load SBLUP results
        sblupdata = pd.read_csv(fprefix + ".temp.sblup.cojo", header=None, delim_whitespace=True)
        # start filters for sign matching and abs > 1 in sblup estimates to get ~25% data
        sblupdata["match"] = 1 if np.sign(sblupdata["V3"]) == np.sign(sblupdata["V4"]) else 0
        sblupdata = sblupdata[sblupdata["match"] == 1 and abs(sblupdata["V4"]) > 1]
        sblupdata.columns = ["SNP", "effectAllele", "gwasBeta", "sblupBeta", "effectMatch"]
        # export list of SNPs to pull

        sblupdata.to_csv(fprefix + ".sblupToPull", index=False, sep='\t')

        command = path2plink + "plink --bfile " + workPath + "/" + geno + ".forSblup --extract " +\
            workPath + geno + ".sblupToPull --indep-pairwise 10000 1 0.1 --out " + fprefix + ".pruning"
        os.system(command)

        command = path2plink + "plink --bfile " + workPath + geno + ".forSblup --extract " + fprefix +\
            ".pruning.prune.in --recode A --out " + fprefix + ".reduced_genos"
        os.system(command)

        # exports SNP list for extraction in validataion set
        command = "cut -f 1 " + fprefix + ".pruning.prune.in > " + fprefix + ".reduced_genos_snpList; rm " +\
            workPath + "/" + geno + ".forSblup.*"
        #cat("The command",command,"\n")
        os.system(command)
        handler["rgenosSnpList"] = fprefix + ".reduced_genos_snpList"
        handler["snpsToPull"] = handler["rgenosSnpList"]

        return(handler)

    elif reduce == "PRSICE" and gwas != "NA":
        # if $reduce = PRSICE, $phenoScale is DISC, $gwas is not NA, $cov = NA

        covstr = " " if cov == "NA" else " --cov-file " + path2Genotype + "/" + cov + ".cov "
        # Second check on covariate file
        print("The covstr is" + covstr)

        if covstr != " ":
            covs = pd.read_csv(path2Genotype + "/" + cov + ".cov", header=0, delim_whitespace=True)

            # print(covs)
            print(covs.shape)

            if len(covs.columns) == 2:
                covstr = " "

        print("The covstr is" + covstr)

        binaryTarget = True if phenoScale == "DISC" else False

        command = genPRSiceCommand(geno,
                                   pheno,
                                   covstr,
                                   path2PRSice,
                                   PRSiceexe,
                                   cores,
                                   fprefix,
                                   path2Genotype,
                                   path2GWAS,
                                   gwas,
                                   binaryTarget,
                                   gwasDef=" --beta --snp MarkerName --A1 Allele1 --A2 Allele2 --stat Effect --se StdErr --pvalue P-value")

        print("The command to run: " + command)
        os.system(command)

        command = "cut -f 2 " + fprefix + ".temp.snp > " + fprefix + ".temp.snpsToPull"
        print("The command to run: " + command)
        os.system(command)

        command = "awk 'NR == 2 {print $3}' " + fprefix + ".temp.summary"
        print("The command to run: " + command)
        try:
            thresh = float(subprocess.check_output(["awk", 'NR == 2 {print $3}', fprefix + ".temp.summary"]))
        except subprocess.CalledProcessError as e:
            sys.exit(e.output)

        command = clumpCommand(geno, gwas, thresh, fprefix, path2plink, path2Genotype, path2GWAS, SNPcolumnatGWAS, clumpField)
        print("The command to run: " + command)
        os.system(command)

        command = "cut -f 3 " + fprefix + ".tempClumps.clumped > " + fprefix + ".temp.snpsToPull2"
        print("The command to run: " + command)
        os.system(command)

        command = path2plink + "plink --bfile " + path2Genotype + geno + " --extract " +\
            fprefix + ".temp.snpsToPull2 --recode A --out " + fprefix + ".reduced_genos"

        print("The COMMAND: " + command)
        os.system(command)

        # exports SNP list for extraction in validation set
        command = "cut -f 1 " + fprefix + ".temp.snpsToPull2 > " + fprefix + ".reduced_genos_snpList"
        print("The command to run: " + command)
        os.system(command)

        handler["rgenosSnpList"] = fprefix + ".reduced_genos_snpList"
        handler["snpsToPull"] = fprefix + ".temp.snpsToPull2"
        handler["snpsClumped"] = fprefix + ".tempClumps.clumped"

        return(handler)

    else:
        sys.exit("The combination of parameters is not right")


#----------------------------------------#
#-----------getHandlerFromFold-----------#
#----------------------------------------#


def getHandlerFromFold(handler, type="train", index=1):

    key = type + "FoldFiles"
    if type == "test":
        assert handler["testFolds"] is not None
        assert len(handler["testFolds"]) >= index

    if type == "train":
        assert handler["trainFolds"] is not None
        assert len(handler["testFolds"]) >= index

    assert handler[key] is not None
    files = handler[key][index]
    #genoFileCommand = handler[key][index]["genoFileCommand"]
    father = handler
    handler = getHandlerToGenotypeData(geno=files["genoFile"],
                                       covs=re.sub(".cov", "", files["covsFile"]),
                                       predictor=handler["Class"],
                                       id=handler["id"],
                                       fid=handler["fid"],
                                       pheno=re.sub(".pheno", "", files["phenoFile"]),
                                       verify=False)
    handler["father"] = father
    handler["fatherkey"] = key
    handler["fatherindex"] = index

    return(handler)


#----------------------------------------#
#-------------fromSNPs2MLdata------------#
#----------------------------------------#


def fromSNPs2MLdata(handler,
                    addit,
                    path2plink,
                    predictor,
                    fsHandler=None):

    # We must have done feature selection with handleSNPs before
    if "snpsToPull" not in handler.keys() or handler["snpsToPull"] is None:
        handler["snpsToPull"] = "void"
    if fsHandler is None:
        # Then we have to pull SNPs as default
        fsHandler = handler.copy()

    if "nfolds" in handler.keys() and handler["nfolds"] is not None:
        modes = ["train", "test"]
    else:
        modes = ["train"]
        handler["nfolds"] = 1

    print("Number of folds here " + str(handler["nfolds"]) + "\n")

    handler = genDataFromHandler(handler, lazy=True)

    for fold in range(1, handler["nfolds"] + 1):
        for mode in modes:
            if mode == "train" and len(modes) == 1:
                geno = os.path.basename(handler["geno"])
                pheno = os.path.basename(handler["pheno"])
                workPath = os.path.dirname(handler["pheno"]) + "/"
                path2Genotype = os.path.dirname(handler["geno"]) + "/"
                cov = os.path.basename(handler["covs"])
                prefix = "g-" + geno + "-p-" + pheno + "-c-" + cov + "-a-" + addit
            elif mode == "train" and len(modes) > 1:
                geno = os.path.basename(handler["trainFoldFiles"][fold]["genoFile"])
                pheno = os.path.basename(re.sub(".pheno", "", handler["trainFoldFiles"][fold]["phenoFile"]))
                workPath = os.path.dirname(handler["trainFoldFiles"][fold]["genoFile"]) + "/"
                # path2Genotype = workPath
                cov = os.path.basename(re.sub(".cov", "", handler["trainFoldFiles"][fold]["covsFile"]))
                prefix = "g-" + geno + "-p-" + pheno + "-c-" + cov + "-a-" + addit
            elif mode == "test" and len(modes) > 1:
                geno = os.path.basename(handler["testFoldFiles"][fold]["genoFile"])
                pheno = os.path.basename(re.sub(".pheno", "", handler["testFoldFiles"][fold]["phenoFile"]))
                workPath = os.path.dirname(handler["testFoldFiles"][fold]["genoFile"]) + "/"
                # path2Genotype = workPath
                cov = os.path.basename(re.sub(".cov", "", handler["testFoldFiles"][fold]["covsFile"]))

                if handler["snpsToPull"] == fsHandler["snpsToPull"]:
                    # We need the previous prefix
                    genotrn = os.path.basename(handler["trainFoldFiles"][fold]["genoFile"])
                    phenotrn = os.path.basename(re.sub(".pheno", "", handler["trainFoldFiles"][fold]["phenoFile"]))
                    covtrn = os.path.basename(re.sub(".cov", "", handler["trainFoldFiles"][fold]["covsFile"]))
                    workPathtrn = os.path.dirname(handler["trainFoldFiles"][fold]["genoFile"]) + "/"
                    previous = "g-" + genotrn + "-p-" + phenotrn + "-c-" + covtrn + "-a-" + addit
                    prefix = "g-" + geno + "-p-" + pheno + "-c-" + cov + "-a-" + addit

                    command = path2plink + "plink --bfile " + workPath + "/" + geno + " --keep " +\
                        workPath + "/" + cov + ".cov" + " --extract " + workPathtrn + "/" + previous +\
                        ".reduced_genos_snpList --recode A --out " + workPath + "/" + prefix + ".reduced_genos"

                    print("Running command " + command + "\n")
                    os.system(command)

            else:
                sys.exit()

            print(handler["snpsToPull"])
            print(fsHandler["snpsToPull"])

            if handler["snpsToPull"] != fsHandler["snpsToPull"]:
                print("We are going to generate a SNP list from a SNP pool selected outside this handler")
                command = path2plink + "plink --bfile " + workPath + geno + " --extract " +\
                    xstr(fsHandler["snpsToPull"]) + " --recode A --out " + workPath + prefix + ".reduced_genos"
                print("The command to run is: " + command)
                os.system(command)
                command = "cut -f 1 " + xstr(fsHandler["snpsToPull"]) + " > " + workPath + prefix + ".reduced_genos_snpList"
                print("The command to run is: " + command)
                os.system(command)
                handler["snpsToPull"] = fsHandler["snpsToPull"]

            # now decide what to merge
            genoPheno = 2
            addCov = 0 if cov == "NA" else 1
            addAddit = 0 if addit == "NA" else 1
            nFiles = genoPheno + addCov + addAddit  # this specifies the number of files to merge
            print("MERGING " + str(nFiles) + " FILES")
            genotypeInput = workPath + prefix + ".reduced_genos.raw"
            print(genotypeInput)
            phenoInput = workPath + pheno + ".pheno"
            covInput = workPath + cov + ".cov"
            additInput = workPath + addit + ".addit"

            genosRaw = pd.read_csv(genotypeInput, header=0, delim_whitespace=True)
            phenoRaw = pd.read_csv(phenoInput, header=0, delim_whitespace=True)

            genosRaw["ID"] = genosRaw["FID"] + "_" + genosRaw["IID"]
            phenoRaw["ID"] = phenoRaw["FID"] + "_" + phenoRaw["IID"]
            fname = workPath + prefix + ".dataForML"
            phenoRaw.drop(["FID", "IID"], 1, inplace=True)
            genosRaw.drop(["FID", "IID", "MAT", "PAT", "SEX", "PHENOTYPE"], 1, inplace=True)

            # run for only geno and pheno data, ie nFiles = 2
            if nFiles == 2:
                temp = pd.merge(phenoRaw, genosRaw, on="ID")
                temp.rename(columns={temp.columns[0]: "PHENO"}, inplace=True)
                print(temp.columns)
                temp.to_csv(fname, index=False, sep='\t')

            # run for studies that have all geno, pheno, cov and addit data availible, ie nFiles = 4
            elif nFiles == 4:
                covRaw = pd.read_csv(covInput, header=0, delim_whitespace=True)
                additRaw = pd.read_csv(additInput, header=0, delim_whitespace=True)
                covRaw["ID"] = covRaw["FID"] + "_" + covRaw["IID"]
                additRaw["ID"] = additRaw["FID"] + "_" + additRaw["IID"]

                covRaw.drop(["FID", "IID"], 1, inplace=True)
                additRaw.drop(["FID", "IID"], 1, inplace=True)

                temp1 = pd.merge(phenoRaw, covRaw, on="ID")
                temp2 = pd.merge(temp1, additRaw, on="ID")
                temp3 = pd.merge(temp2, genosRaw, on="ID")
                temp3.rename(columns={temp3.columns[0]: "PHENO"}, inplace=True)
                print(temp3.columns)
                fname = workPath + prefix + ".dataForML"
                temp3.to_csv(fname, index=False, sep='\t')
                temp = temp3

            # run for studies that have all geno, pheno and cov data availible (addit is missing), ie nFiles = 3
            elif nFiles == 3 and addit == "NA":
                otherRaw = pd.read_csv(covInput, header=0, delim_whitespace=True)
                otherRaw["ID"] = otherRaw["FID"] + "_" + otherRaw["IID"]

                otherRaw.drop(["FID", "IID"], 1, inplace=True)
                temp1 = pd.merge(phenoRaw, otherRaw, on="ID")
                temp2 = pd.merge(temp1, genosRaw, on="ID")
                temp2.rename(columns={temp2.columns[0]: "PHENO"}, inplace=True)
                print(temp2.columns)
                fname = workPath + prefix + ".dataForML"
                temp2.to_csv(fname, index=False, sep='\t')
                temp = temp2

            # run for studies that have all geno, pheno and addit data availible (cov is missing), ie nFiles = 3
            elif nFiles == 3 and cov == "NA":
                otherRaw = pd.read_csv(additInput, header=0, delim_whitespace=True)
                otherRaw["ID"] = otherRaw["FID"] + "_" + otherRaw["IID"]

                otherRaw.drop(["FID", "IID"], 1, inplace=True)

                temp1 = pd.merge(phenoRaw, otherRaw, on="ID")
                temp2 = pd.merge(temp1, genosRaw, on="ID")
                temp2.rename(columns={temp2.columns[0]: "PHENO"}, inplace=True)
                print(temp2.columns)
                fname = workPath + prefix + ".dataForML"
                temp2.to_csv(fname, index=False, sep='\t')
                temp = temp2

            handler[mode + str(fold) + "mldata"] = fname

            print("First 100 variable names for your file below, the rest are likely just more genotypes...")
            print(temp.columns[0:100])
            print("... and the last 100 variable names for your file below...")
            print(temp.columns[-100:0])
            print("Your final file has " + str(len(temp["ID"])) + " samples, and " + str(len(temp.columns)) + " predictors for analysis")

    return(handler)

#----------------------------------------#
#--------------verifyHandler-------------#
#----------------------------------------#


def verifyHandler(h,
                  level=1):
    if(level >= 1):
        assert os.path.isfile(h["geno"] + ".bed")
        assert os.path.isfile(h["geno"] + ".bim")
        assert os.path.isfile(h["geno"] + ".fam")
        assert os.path.isfile(h["covs"] + ".cov")
        cdata = pd.read_csv(h["covs"] + ".cov", header=0, delim_whitespace=True)
        assert h["id"] in cdata.columns
        assert os.path.isfile(h["pheno"] + ".pheno")

    if(level >= 2):
        assert h["trainFolds"] is not None
        assert h["testFolds"] is not None

#----------------------------------------#
#-------------genPRSiceCommand-----------#
#----------------------------------------#


def genPRSiceCommand(geno,
                     pheno,
                     covstr,
                     path2PRSice,
                     PRSiceexe,
                     cores,
                     fprefix,
                     path2Genotype,
                     path2GWAS,
                     gwas,
                     binaryTarget,
                     barLevels="5E-8,4E-8,3E-8,2E-8,1E-8,9E-7,8E-7,7E-7,6E-7,5E-7,4E-7,3E-7,2E-7,1E-7,9E-6,8E-6,7E-6,6E-6,5E-6,4E-6,3E-6,2E-6,1E-6,9E-5,8E-5,7E-5,6E-5,5E-5,4E-5,3E-5,2E-5,1E-5,9E-4,8E-4,7E-4,6E-4,5E-4,4E-4,3E-4,2E-4,1E-4,9E-3,8E-3,7E-3,6E-3,5E-3,4E-3,3E-3,2E-3,1E-3,9E-2,8E-2,7E-2,6E-2,5E-2,4E-2,3E-2,2E-2,1E-2,9E-1,8E-1,7E-1,6E-1,5E-1,4E-1,3E-1,2E-1,1E-1,1 ",
                     gwasDef=" --beta --snp SNP --A1 A1 --A2 A2 --stat b --se se --pvalue p"):

    return("Rscript " + path2PRSice + "PRSice.R --binary-target T --prsice " + path2PRSice + PRSiceexe +
           " -n " + xstr(cores) + " --out " + fprefix + ".temp --pheno-file " + path2Genotype + "/" +
           pheno + ".pheno -t " + path2Genotype + "/" + geno + " -b " + path2GWAS + "/" + gwas + covstr +
           " --print-snp --score std --perm 10000 " + " --bar-levels " + barLevels +
           " --fastscore --binary-target " + str(binaryTarget) + gwasDef)

#----------------------------------------#
#---------------clumpCommand-------------#
#----------------------------------------#


def clumpCommand(geno,
                 gwas,
                 thresh,
                 fprefix,
                 path2plink,
                 path2Genotype,
                 path2GWAS,
                 SNPcolumnatGWAS,
                 clumpField):

    return(path2plink + "plink --bfile " + path2Genotype + "/" + geno + " --extract " + fprefix +
           ".temp.snpsToPull --clump " + path2GWAS + "/" + gwas + " --clump-p1 " + str(thresh) + " --clump-p2 " +
           str(thresh) + " --clump-snp-field " + str(SNPcolumnatGWAS) + " --clump-field " + str(clumpField) +
           " --clump-r2 0.1 --clump-kb 250 --out " + str(fprefix) + ".tempClumps")

#----------------------------------------#
#-------------fromGenoToMLdata-----------#
#----------------------------------------#


def fromGenoToMLdata(workPath,
                     iter,
                     path2Geno="/home/users/gsit/juanbot/JUAN_SpanishGWAS/UNRELATED.SPAIN4.HARDCALLS.Rsq0.8",
                     path2Covs="/home/users/gsit/juanbot/JUAN_SpanishGWAS/COVS_SPAIN",
                     predictor="DISEASE",
                     path2GWAS="/home/users/gsit/juanbot/JUAN_SpanishGWAS/toJuanNov7th2018/",
                     path2PRSice="/home/users/gsit/juanbot/genoml-core/otherPackages/",
                     path2plink="",
                     path2Pheno=None,
                     snpsSpain=None):

    if path2Pheno is None:
        path2Pheno = workPath + "/MyPhenotype"

    h = getHandlerToGenotypeData(geno=path2Geno,
                                 covs=path2Covs,
                                 id="IID",
                                 fid="FID",
                                 predictor=predictor,
                                 # With this we assure everything will be written under workPath
                                 pheno=path2Pheno)

    holdout = getPartitionsFromHandler(genoHandler=h,
                                       workPath=workPath,
                                       path2plink=path2plink,
                                       how="holdout",
                                       p=0.75)

    holdout = genDataFromHandler(holdout, lazy=True)

    # Save holdout
    with open(workPath + '/holdout.pydat', 'wb') as holdout_file:
        pickle.dump(holdout, holdout_file)

    # Load holdout
    with open(workPath + '/holdout.pydat', 'rb') as holdout_file:
        holdout = pickle.load(holdout_file)

    # If the current repo is the spanish, apply PRSice to reduce variables
    if iter == 1:
        print("SPANISH Repo -> mostRelevantSNPs")
        handlerSNPs = mostRelevantSNPs(handler=getHandlerFromFold(handler=holdout, type="train", index=1),
                                       path2plink=path2plink,
                                       gwas="RISK_noSpain_MAF0.05.tab",
                                       path2GWAS=path2GWAS,
                                       PRSiceexe="PRSice_linux",
                                       path2PRSice=path2PRSice,
                                       clumpField="P-value",
                                       SNPcolumnatGWAS="MarkerName")
    else:  # If not, extract only the SNPs selected from the spanish repo
        print("Other Repo -> NOT mostRelevantSNPs")
        handlerSNPs = holdout.copy()
        addit = "NA"
        geno = os.path.basename(handlerSNPs["geno"])
        pheno = os.path.basename(handlerSNPs["pheno"])
        cov = os.path.basename(handlerSNPs["covs"])
        path2Genotype = os.path.dirname(handlerSNPs["geno"]) + "/"
        prefix = "g-" + geno + "-p-" + pheno + "-c-" + cov + "-a-" + addit
        fprefix = workPath + "/" + prefix
        handlerSNPs["snpsToPull"] = snpsSpain
        command = path2plink + "plink --bfile " + path2Genotype + geno + " --extract " +\
            xstr(handlerSNPs["snpsToPull"]) + " --recode A --out " + fprefix + ".reduced_genos"
        print("Command to run is (snpsSpain): " + command)
        os.system(command)

        # exports SNP list for extraction in validation set
        command = "cut -f 1 " + xstr(handlerSNPs["snpsToPull"]) + " > " + fprefix + ".reduced_genos_snpList"
        os.system(command)
        handlerSNPs["rgenosSnpList"] = fprefix + ".reduced_genos_snpList"

    # Save handlerSNPs
    with open(workPath + '/handlerSNPs.pydat', 'wb') as handlerSNPs_file:
        pickle.dump(handlerSNPs, handlerSNPs_file)

    # Load handlerSNPs
    with open(workPath + '/handlerSNPs.pydat', 'rb') as handlerSNPs_file:
        handlerSNPs = pickle.load(handlerSNPs_file)

    # generate mldata for the repository
    mldatahandler = fromSNPs2MLdata(handler=holdout, addit="NA", path2plink=path2plink, predictor=predictor, fsHandler=handlerSNPs)

    # Save handlerSNPs
    with open(workPath + "/mldatahandler.pydat", 'wb') as mldatahandler_file:
        pickle.dump(mldatahandler, mldatahandler_file)

    # Load handlerSNPs
    with open(workPath + '/mldatahandler.pydat', 'rb') as mldatahandler_file:
        mldatahandler = pickle.load(mldatahandler_file)

    # These are the two ML datasets generated from the genotype data and the
    # feature selection
    print("Your train dataset is at " + mldatahandler["train1mldata"])
    print("Your test dataset is at" + mldatahandler["test1mldata"])

    return mldatahandler
