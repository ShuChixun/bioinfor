###calculating drug GO similarity (R software)
rm(list=ls())
# setwd("C:/Users/ASUS/Desktop")
# 设置工作目录
# if (!requireNamespace("BiocManager", quietly = TRUE))
#   install.packages("BiocManager")
# BiocManager::install("GOSemSim")
# BiocManager::install("org.Hs.eg.db")

library(GOSemSim)
library(org.Hs.eg.db)
# file_name = "../../data/NCG.xlsx"
file_name = "../../data/GOid.csv"
s = read.csv(file_name)  ##read data from Excel file
columns(org.Hs.eg.db)
MF <- godata('org.Hs.eg.db', ont="MF", computeIC=FALSE)
GOlist = s$GO_id[1:1000]
# GOlist = s$GO_id
# 数据不能太大 9min
# 错误: 无法分配大小为17.0 Gb的矢量

goMap = mgoSim(GOlist, GOlist, semData = MF, measure = 'Wang', combine = NULL)

Out_file = '../../data/GOidsim.csv'
write.table(goMap, file=Out_file,sep=",",row.names=F, col.names = F)
