# il2Pred
A computational method to predict il2 inducing or il2 non-inducing peptides based on amino acid composition or motifs

## Introduction
il2Pred is a tool developed by Raghva-Lab in 2024. It is designed to predict whether a peptide is il2 inducer or not. It utilizes amino-acid compositions and motifs as features to make predictions using Random Forest. il2Pred is also available as web-server at https://webs.iiitd.edu.in/raghava/il2Pred. Please read/cite the content about the il2Pred for complete information including algorithm behind the approach.

## PIP Installation
PIP version is also available for easy installation and usage of this tool. The following command is required to install the package 
```
pip install il2Pred
```
To know about the available option for the pip package, type the following command:
```
il2Pred -h
```
# Standalone

Standalone version of il2Pred is written in python3 and the following libraries are necessary for a successful run:

- scikit-learn = 0.24.1
- Pandas
- Numpy
- python - 3.12.2 


## Minimum USAGE
To know about the available option for the standalone, type the following command:
```
python il2pred.py -h
```
To run the example, type the following command:
```
python il2Pred.py -i peptide.fa
```
This will predict the probability whether a submitted sequence will il2 inducer or not. It will use other parameters by default. It will save the output in "outfile.csv" in CSV (comma separated variables).

## Full Usage
```
usage: il2Pred.py [-h] -i INPUT [-o OUTPUT] [-t THRESHOLD] [-j {1,2,3}] [-m {1,2}] [-d {1,2}]
                    [-wd WORKING]
=======
```
To run the example, type the following command:
```

unzip model.zip

il2Pred.py -i peptide.fa

```
```
Please provide following arguments.
=======
Following is complete list of all options, you may get these options
usage: il2Pred.py [-h] 
                     [-i INPUT]
                     [-o OUTPUT]
		     [-j {1,2,3}] 
                     [-m {1,2}] 
```
```
Please provide following arguments

optional arguments:

options:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Input: protein or peptide sequence in FASTA format
  -o OUTPUT, --output OUTPUT
                        Output: File for saving results by default outfile.csv
  -t THRESHOLD, --threshold THRESHOLD
                        Threshold: Value between 0 to 1 by default 0.48
  -j {1,2,3}, --job {1,2,3}
			Job: 1: il2 vs non il2, 2: il2 vs mhc non binder il2, 3: il2 vs mixed 
  -m {1,2}, --model {1,2}
                        Model: 1: ET (feature_selection_model) , 2: hybrid (MERCI + ET)
                        feature based on ET, by default 1
  -d {1,2}, --display {1,2}
                        Display: 1: il2 inducing peptides, 2: All proteins, by default 2
  -wd WORKING, --working WORKING
                        Working Directory: Temporary directory to write files
```

**Input File:** It allow users to provide input in the FASTA format.

**Output File:** Program will save the results in the CSV format, in case user does not provide output file name, it will be stored in "outfile.csv".

**Threshold:** User should provide threshold between 0 and 1, by default its 0.5.

**Display type:** This option allow users to display only il2 inducing peptides or all the input peptides.

**Working Directory:** Directory where intermediate files as well as final results will be saved

il2Pred Package Files
=======================
It contains the following files, brief description of these files given below


LICENSE				      : License information

README.md			      : This file provide information about this package

model1               : First dataset model

model2               : Second dataset model

model3               : Third dataset model

merci               : This folder contains merci locator file

motifs1             : This folder contain motifs of main dataset

motifs2             : This folder contain motifs of second dataset

motifs3            : This folder contain motifs of third dataset

il2Pred.py     : Main python program


peptide.fa : Example file containing peptide sequences in FASTA format

output.csv	: Example output file for the program
