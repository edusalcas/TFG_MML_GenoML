import os


def munge(prefix=None, datatype=None, geno=None, pheno=None, featureSelection=None):
    command = "GenoMLMunging"

    if prefix is not None:
        command += " --prefix " + prefix

    if datatype is not None:
        command += " --datatype " + datatype

    if geno is not None:
        command += " --geno " + geno

    if pheno is not None:
        command += " --pheno " + pheno

    if featureSelection is not None:
        command += " --featureSelection " + str(featureSelection)

    print("Command to run: " + command)

    os.system(command)


def train(prefix=None, metric_max=None, algs=None, seed=None):
    command = "GenoML discrete supervised train"

    if prefix is not None:
        command += " --prefix " + prefix

    if metric_max is not None:
        command += " --metric_max " + metric_max

    if algs is not None:
        command += " --alg " + algs

    if seed is not None:
        command += " --seed " + str(seed)

    print("Command to run: " + command)

    os.system(command)


def tune(prefix=None, metric_tune=None, max_tune=None, seed=None):
    command = "GenoML discrete supervised tune"

    if prefix is not None:
        command += " --prefix " + prefix

    if metric_tune is not None:
        command += " --metric_tune " + metric_tune

    if max_tune is not None:
        command += " --max_tune " + str(max_tune)

    if seed is not None:
        command += " --seed " + str(seed)

    print("Command to run: " + command)

    os.system(command)
