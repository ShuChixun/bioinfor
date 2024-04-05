rm(list=ls())
# setwd("your workspace")
# 设置工作目录
# BiocManager::install("ChemmineR")
# BiocManager::install("ChemmineOB")
# 参考文章：https://cloud.tencent.com/developer/article/1548365

library(ChemmineOB)
library(ChemmineR)

#######################################################
# use a as an example

a=read.SDFset("http://faculty.ucr.edu/~tgirke/Documents/R_BioCond/Samples/sdfsample.sdf")
# read sdf data
blockmatrix <- datablock2ma(datablocklist=datablock(a))
# 将SDF文件转化为矩阵数据
colnames(blockmatrix)
head(blockmatrix)
# numchar <-splitNumChar(blockmatrix=blockmatrix) #分割字符串和数值型的数据
# numchar$charMA[,"PUBCHEM_OPENEYE_ISO_SMILES"]
blockmatrix[,"PUBCHEM_OPENEYE_ISO_SMILES"]
blockmatrix[,"PUBCHEM_MOLECULAR_FORMULA"]
# SM <- blockmatrix[,c('PUBCHEM_COMPOUND_CID','PUBCHEM_MOLECULAR_FORMULA','PUBCHEM_OPENEYE_ISO_SMILES')]
SM <- blockmatrix[,c('PUBCHEM_COMPOUND_CID','PUBCHEM_OPENEYE_ISO_SMILES')]
write.table(SM,"sdfsample.csv",row.names=FALSE,col.names=TRUE,sep=",")

# plot(a[1:4], print=FALSE)
######################################################
file_name = '../../data/structures.sdf'
sdfset <- read.SDFset(file_name)
valid <- validSDF(sdfset); sdfset <- sdfset[valid]
blockmatrix <- datablock2ma(datablocklist=datablock(sdfset))
colnames(blockmatrix)
head(blockmatrix)
# numchar <-splitNumChar(blockmatrix=blockmatrix) #分割字符串和数值型的数据
# numchar$charMA[,"PUBCHEM_OPENEYE_ISO_SMILES"]
head(blockmatrix[,"DRUGBANK_ID"])
head(blockmatrix[,"GENERIC_NAME"])
head(blockmatrix[,"SMILES"])
SM <- blockmatrix[,c('DRUGBANK_ID','SMILES')]
Out_file = '../../data/structure.csv'
write.table(SM, Out_file, row.names=FALSE, col.names=TRUE, sep=",")
