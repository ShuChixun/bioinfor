####calculating CMAP function similarity between drugs (R software)
rm(list=ls())
library("cmapR")
library("argparser")
library("GeneExpressionSignature")
library('SummarizedExperiment')
###install.packages("readxl")
library(readxl)
file_name = "../../data/cmap_instances_02.xls"
SampleInfoFile = read_excel(file_name,sheet=1,range='A1:O6101')
ls <- SampleInfoFile$instance_id
#Load files 
MatrixPath = "../../data/rankMatrix.txt"
RankedMatrix <- read.delim(MatrixPath, header = T, row.names = 1)
dim(RankedMatrix)
colnames(RankedMatrix)<-1:6101
# colnames(RankedMatrix)<-ls
RM <-RankedMatrix[,-6101]

## eset
SampleInfoFile$instance_id <- 1:6100
ls1 <- which(SampleInfoFile$instance_id%in%colnames(RM))
# in fact,instance_id is equal to colnames(RM)
RankedMatrix1<- RM[, ls1]
drugname <- SampleInfoFile[ls1, 3]
DrugAFD = new("AnnotatedDataFrame", data = drugname)
eset = ExpressionSet(assayData = as.matrix(RankedMatrix1), phenoData = DrugAFD)
## merge
MergingSet <- RankMerging(eset,"Spearman",weighted=TRUE)
# 22min cost too much time
# load('../../data/MergingSet.RData')
# save(MergingSet,file = '../../data/MergingSet.RData')
load('../../data/Ms.RData')
# PRL 22283 * 1309
# save(Ms, file='../../data/Ms.RData')
# similarity compution
begin = Sys.time()
ds <- ScoreGSEA(MergingSet,250,"avg")
end = Sys.time()
print(end-begin)
# Time difference of 2.996954 hours
ds[1:5,1:5]
load('../../data/CMap.RData')
# save(ds, file='../../data/CMap.RData')

library(readr)
Out_file = '../../data/CMap.csv'
write.csv(ds, Out_file)